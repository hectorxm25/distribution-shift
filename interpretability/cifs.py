"""
Corrected Channel Importance via Feature Selection (CIFS) Implementation
Following the paper's methodology with probe networks and proper gradient computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import os
import json
import matplotlib.pyplot as plt
from nnsight import NNsight
import argparse

from utils import (
    Config,
    load_model,
    load_baseline_dataloader,  # This loads two single images (baseline and counterfactual)
    load_class_specific_dataloader,
    get_layer_names,
)


class ProbeNetwork(nn.Module):
    """Probe network A^l for making predictions from intermediate representations."""

    def __init__(self, input_channels, num_classes=10, spatial_size=None):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        if spatial_size is not None:
            # For conv layers with spatial dimensions
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(input_channels, num_classes)
        else:
            # For FC layers
            self.pool = None
            self.fc = nn.Linear(input_channels, num_classes)

    def forward(self, x):
        if self.pool is not None:
            # x shape: [batch, channels, height, width]
            x = self.pool(x)  # [batch, channels, 1, 1]
            x = x.squeeze(-1).squeeze(-1)  # [batch, channels]
        # x shape: [batch, channels/features]
        return self.fc(x)


class ChannelPerturbation(nn.Module):
    """Module to apply channel-wise perturbations for gradient computation."""

    def __init__(self, num_channels):
        super().__init__()
        # Initialize perturbations at zero
        self.delta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        # Apply channel-wise perturbation
        # x shape: [batch, channels, height, width] or [batch, features]
        if len(x.shape) == 4:
            return x * (1 + self.delta)
        elif len(x.shape) == 2:
            delta_flat = self.delta.squeeze(-1).squeeze(-1)
            return x * (1 + delta_flat)
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")


class IMGF:
    """Importance Mask Generation Functions."""

    @staticmethod
    def sigmoid(relevance, alpha=10.0):
        """Sigmoid IMGF - acts as a switch for PR/NR channels."""
        return torch.sigmoid(alpha * relevance)

    @staticmethod
    def softplus(relevance, alpha=1.0):
        """Softplus IMGF - monotonic alignment with relevance."""
        return F.softplus(alpha * relevance)

    @staticmethod
    def softmax(relevance, temperature=1.0):
        """Softmax IMGF - sparse activation on most relevant channels."""
        return F.softmax(relevance / temperature, dim=0)


class CIFsCorrect:
    """Corrected CIFS implementation with probe networks and proper gradient computation."""

    def __init__(self, model: nn.Module, nnsight_model: NNsight, config: Config):
        self.model = model
        self.nnsight_model = nnsight_model
        self.config = config
        self.device = config.device

        # Get layer names and build layer mapping
        self.layer_names = get_layer_names()
        self._build_layer_mapping()

        # Initialize probe networks for each layer
        self.probe_networks = self._initialize_probe_networks()

        # CIFS hyperparameters
        self.k = 3  # Top-k predictions to consider
        self.imgf_type = "sigmoid"  # Can be 'sigmoid', 'softplus', or 'softmax'
        self.imgf_alpha = 10.0  # Alpha for sigmoid/softplus
        self.beta = 0.5  # Balance coefficient for training loss

        print(f"[CIFS] Initialized with {len(self.probe_networks)} probe networks")

    def _build_layer_mapping(self):
        """Build mapping from layer names to actual modules."""
        self.layer_modules = {}

        for layer_name in self.layer_names:
            parts = layer_name.split(".")
            module = self.model

            try:
                for part in parts:
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)

                self.layer_modules[layer_name] = module
                print(f"[CIFS] Mapped {layer_name} -> {type(module).__name__}")
            except (AttributeError, IndexError, KeyError) as e:
                print(f"[CIFS] Warning: Could not find module for {layer_name}: {e}")

    def _initialize_probe_networks(self):
        """Initialize probe networks for each layer."""
        probe_networks = {}

        for layer_name, module in self.layer_modules.items():
            if isinstance(module, nn.Conv2d):
                # Conv layer - need spatial pooling
                probe_networks[layer_name] = ProbeNetwork(
                    module.out_channels,
                    num_classes=10,  # Adjust based on your dataset
                    spatial_size=True,
                ).to(self.device)
            elif isinstance(module, nn.Linear):
                # FC layer - no spatial pooling needed
                probe_networks[layer_name] = ProbeNetwork(
                    module.out_features, num_classes=10, spatial_size=None
                ).to(self.device)
            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm layer
                probe_networks[layer_name] = ProbeNetwork(
                    module.num_features, num_classes=10, spatial_size=True
                ).to(self.device)

        return probe_networks

    def compute_channel_relevance(
        self,
        layer_name: str,
        activation: torch.Tensor,
        true_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute channel relevance g^l following Equation 1.

        g^l = ∇_δ [Σ_{i∈y^{l,k}} p^l(δ)[i]]|_{δ=0}
        """
        if layer_name not in self.probe_networks:
            return None

        probe = self.probe_networks[layer_name]
        batch_size = activation.size(0)

        # Determine number of channels
        if len(activation.shape) == 4:
            num_channels = activation.size(1)
        else:
            num_channels = activation.size(1)

        # Initialize perturbation module
        if len(activation.shape) == 4:
            perturb = ChannelPerturbation(num_channels).to(self.device)
        else:
            # For FC layers, we need a different perturbation shape
            perturb = nn.Parameter(torch.zeros(1, num_channels)).to(self.device)

        # Enable gradient computation
        activation = activation.detach().requires_grad_(True)

        if isinstance(perturb, ChannelPerturbation):
            # Apply perturbation (at δ=0, this is identity)
            perturbed_activation = perturb(activation)
        else:
            # For FC layers
            perturbed_activation = activation * (1 + perturb)

        # Get probe prediction p^l
        probe_logits = probe(perturbed_activation)

        # Get top-k indices
        if true_label is not None:
            # During training, use true label + top k-1 predictions
            top_values, top_indices = torch.topk(probe_logits, k=self.k, dim=1)
            # Ensure true label is included
            top_indices[:, 0] = true_label
        else:
            # During testing, use top-k predictions
            top_values, top_indices = torch.topk(probe_logits, k=self.k, dim=1)

        # Sum of top-k logits
        top_k_sum = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            for j in range(self.k):
                top_k_sum[i] += probe_logits[i, top_indices[i, j]]

        # Compute gradient with respect to perturbation
        if isinstance(perturb, ChannelPerturbation):
            grad = torch.autograd.grad(
                outputs=top_k_sum.sum(),
                inputs=perturb.delta,
                create_graph=True,
                retain_graph=True,
            )[0]
            # grad shape: [1, num_channels, 1, 1]
            channel_relevance = grad.squeeze(0).squeeze(-1).squeeze(-1)
        else:
            grad = torch.autograd.grad(
                outputs=top_k_sum.sum(),
                inputs=perturb,
                create_graph=True,
                retain_graph=True,
            )[0]
            # grad shape: [1, num_channels]
            channel_relevance = grad.squeeze(0)

        return channel_relevance

    def generate_importance_mask(self, relevance: torch.Tensor) -> torch.Tensor:
        """Generate importance mask using IMGF."""
        if self.imgf_type == "sigmoid":
            return IMGF.sigmoid(relevance, self.imgf_alpha)
        elif self.imgf_type == "softplus":
            return IMGF.softplus(relevance, self.imgf_alpha)
        elif self.imgf_type == "softmax":
            return IMGF.softmax(relevance, temperature=1.0)
        else:
            raise ValueError(f"Unknown IMGF type: {self.imgf_type}")

    def identify_pr_nr_channels(
        self, dataloader: torch.utils.data.DataLoader, num_images: int = 100
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Identify PR (Positive Relevance) and NR (Negative Relevance) channels.

        Returns:
            Dictionary with layer_name -> {
                'relevance': average channel relevance scores,
                'pr_channels': indices of PR channels (g^l[i] > 0),
                'nr_channels': indices of NR channels (g^l[i] <= 0),
                'importance_mask': generated importance mask
            }
        """
        print(f"[CIFS] Identifying PR/NR channels from {num_images} images...")

        self.model.eval()

        # Storage for relevance scores
        layer_relevances = defaultdict(list)

        images_processed = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            if images_processed >= num_images:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            # Hook storage for activations
            activations = {}
            handles = []

            def save_activation(name):
                def hook(module, input, output):
                    activations[name] = output

                return hook

            # Register hooks
            for layer_name, module in self.layer_modules.items():
                if layer_name in self.probe_networks:
                    handle = module.register_forward_hook(save_activation(layer_name))
                    handles.append(handle)

            # Forward pass
            with torch.no_grad():
                _ = self.model(images)

            # Compute relevance for each layer
            for layer_name in self.layer_names:
                if layer_name in activations and layer_name in self.probe_networks:
                    activation = activations[layer_name]

                    # Compute channel relevance
                    relevance = self.compute_channel_relevance(
                        layer_name, activation, true_label=labels
                    )

                    if relevance is not None:
                        layer_relevances[layer_name].append(relevance.detach().cpu())

            # Clean up hooks
            for handle in handles:
                handle.remove()

            images_processed += batch_size

            if images_processed % 20 == 0:
                print(f"[CIFS] Processed {images_processed}/{num_images} images")

        # Aggregate results
        results = {}
        for layer_name in self.layer_names:
            if layer_name in layer_relevances and layer_relevances[layer_name]:
                # Average relevance across all images
                avg_relevance = torch.stack(layer_relevances[layer_name]).mean(dim=0)

                # Identify PR and NR channels
                pr_channels = torch.where(avg_relevance > 0)[0].tolist()
                nr_channels = torch.where(avg_relevance <= 0)[0].tolist()

                # Generate importance mask
                importance_mask = self.generate_importance_mask(avg_relevance)

                results[layer_name] = {
                    "relevance": avg_relevance,
                    "pr_channels": pr_channels,
                    "nr_channels": nr_channels,
                    "importance_mask": importance_mask,
                    "num_pr": len(pr_channels),
                    "num_nr": len(nr_channels),
                }

                print(
                    f"[CIFS] {layer_name}: {len(pr_channels)} PR, {len(nr_channels)} NR channels"
                )

        return results

    def get_top_k_channels(
        self, results: Dict[str, Dict[str, torch.Tensor]], k: int = 10
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """
        Get top-k PR and NR channels globally across all layers.

        Returns:
            Tuple of (top_pr_channels, top_nr_channels) dictionaries
        """
        print(f"[CIFS] Extracting top {k} PR and NR channels...")

        # Collect all PR and NR scores
        all_pr_scores = []
        all_nr_scores = []

        for layer_name, layer_results in results.items():
            relevance = layer_results["relevance"]

            # PR channels (positive relevance)
            for ch_idx in layer_results["pr_channels"]:
                all_pr_scores.append(
                    {
                        "layer": layer_name,
                        "channel": ch_idx,
                        "score": relevance[ch_idx].item(),
                    }
                )

            # NR channels (negative relevance)
            for ch_idx in layer_results["nr_channels"]:
                all_nr_scores.append(
                    {
                        "layer": layer_name,
                        "channel": ch_idx,
                        "score": abs(
                            relevance[ch_idx].item()
                        ),  # Use absolute value for ranking
                    }
                )

        # Sort and get top-k
        all_pr_scores.sort(key=lambda x: x["score"], reverse=True)
        all_nr_scores.sort(key=lambda x: x["score"], reverse=True)

        # Initialize result dictionaries
        top_pr = {layer: [] for layer in results.keys()}
        top_nr = {layer: [] for layer in results.keys()}

        # Get top-k PR channels
        for item in all_pr_scores[:k]:
            top_pr[item["layer"]].append(item["channel"])

        # Get top-k NR channels
        for item in all_nr_scores[:k]:
            top_nr[item["layer"]].append(item["channel"])

        print(f"[CIFS] Top PR channels (first 5):")
        for item in all_pr_scores[:5]:
            print(
                f"  {item['layer']}: channel {item['channel']}, relevance={item['score']:.4f}"
            )

        print(f"[CIFS] Top NR channels (first 5):")
        for item in all_nr_scores[:5]:
            print(
                f"  {item['layer']}: channel {item['channel']}, relevance=-{item['score']:.4f}"
            )

        return top_pr, top_nr

    def train_with_cifs(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        lr: float = 0.001,
    ):
        """
        Train the model with CIFS using the combined loss (Equation 3).

        ℓ_β(x,y) = [1/(1+β)] · ℓ_ce(p,y) + [β/((1+β)|I|)] · Σ_{l∈I} ℓ_ce(p^l,y)
        """
        print(f"[CIFS] Training with CIFS for {epochs} epochs...")

        # Optimizer for both model and probe networks
        all_params = list(self.model.parameters())
        for probe in self.probe_networks.values():
            all_params.extend(list(probe.parameters()))

        optimizer = torch.optim.Adam(all_params, lr=lr)

        for epoch in range(epochs):
            self.model.train()
            for probe in self.probe_networks.values():
                probe.train()

            total_loss = 0
            num_batches = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Hook to capture activations
                activations = {}
                handles = []

                def save_activation(name):
                    def hook(module, input, output):
                        activations[name] = output

                    return hook

                # Register hooks
                for layer_name, module in self.layer_modules.items():
                    if layer_name in self.probe_networks:
                        handle = module.register_forward_hook(
                            save_activation(layer_name)
                        )
                        handles.append(handle)

                # Forward pass through main model
                final_output = self.model(images)

                # Compute main loss
                main_loss = F.cross_entropy(final_output, labels)

                # Compute probe losses
                probe_losses = []
                for layer_name, probe in self.probe_networks.items():
                    if layer_name in activations:
                        activation = activations[layer_name]

                        # Compute relevance and importance mask
                        relevance = self.compute_channel_relevance(
                            layer_name, activation, true_label=labels
                        )

                        if relevance is not None:
                            # Generate importance mask
                            mask = self.generate_importance_mask(relevance)

                            # Apply mask to activation (Equation 2)
                            if len(activation.shape) == 4:
                                masked_activation = activation * mask.view(1, -1, 1, 1)
                            else:
                                masked_activation = activation * mask.view(1, -1)

                            # Probe prediction with masked activation
                            probe_output = probe(masked_activation)
                            probe_loss = F.cross_entropy(probe_output, labels)
                            probe_losses.append(probe_loss)

                # Clean up hooks
                for handle in handles:
                    handle.remove()

                # Combine losses (Equation 3)
                if probe_losses:
                    probe_loss_sum = sum(probe_losses) / len(probe_losses)
                    combined_loss = (1 / (1 + self.beta)) * main_loss + (
                        self.beta / (1 + self.beta)
                    ) * probe_loss_sum
                else:
                    combined_loss = main_loss

                # Backward and optimize
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

                total_loss += combined_loss.item()
                num_batches += 1

                if batch_idx % 50 == 0:
                    print(
                        f"[CIFS] Epoch {epoch+1}, Batch {batch_idx}, Loss: {combined_loss.item():.4f}"
                    )

            avg_loss = total_loss / num_batches
            print(
                f"[CIFS] Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_loss:.4f}"
            )

    def save_results(self, results: Dict, save_dir: str):
        """Save CIFS analysis results."""
        os.makedirs(save_dir, exist_ok=True)

        # Convert tensors to lists for JSON serialization
        json_results = {}
        for layer_name, layer_data in results.items():
            json_results[layer_name] = {
                "relevance": layer_data["relevance"].tolist(),
                "pr_channels": layer_data["pr_channels"],
                "nr_channels": layer_data["nr_channels"],
                "importance_mask": layer_data["importance_mask"].tolist(),
                "num_pr": layer_data["num_pr"],
                "num_nr": layer_data["num_nr"],
            }

        # Save to JSON
        results_path = os.path.join(save_dir, "cifs_results.json")
        with open(results_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"[CIFS] Results saved to {results_path}")

    def visualize_results(self, results: Dict, save_dir: str):
        """Visualize CIFS analysis results."""
        os.makedirs(save_dir, exist_ok=True)

        num_layers = len(results)
        fig, axes = plt.subplots(num_layers, 2, figsize=(15, 4 * num_layers))

        if num_layers == 1:
            axes = axes.reshape(1, -1)

        for idx, (layer_name, layer_data) in enumerate(results.items()):
            relevance = layer_data["relevance"].numpy()
            importance_mask = layer_data["importance_mask"].numpy()

            # Plot relevance scores
            ax1 = axes[idx, 0]
            channels = range(len(relevance))
            colors = ["green" if r > 0 else "red" for r in relevance]
            ax1.bar(channels, relevance, color=colors, alpha=0.6)
            ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax1.set_title(
                f'{layer_name} - Channel Relevance (PR={layer_data["num_pr"]}, NR={layer_data["num_nr"]})'
            )
            ax1.set_xlabel("Channel Index")
            ax1.set_ylabel("Relevance Score")
            ax1.grid(True, alpha=0.3)

            # Plot importance mask
            ax2 = axes[idx, 1]
            ax2.bar(channels, importance_mask, color="blue", alpha=0.6)
            ax2.set_title(f"{layer_name} - Importance Mask ({self.imgf_type})")
            ax2.set_xlabel("Channel Index")
            ax2.set_ylabel("Importance Weight")
            ax2.set_ylim([0, max(importance_mask) * 1.1])
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, "cifs_visualization.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[CIFS] Visualization saved to {save_path}")
        plt.close()

    def compare_two_images(
        self,
        baseline_loader: torch.utils.data.DataLoader,
        counterfactual_loader: torch.utils.data.DataLoader,
        top_k: int = 50,
        save_dir: str = None,
    ) -> Dict:
        """
        Compare channel importance between EXACTLY TWO images (baseline and counterfactual).

        This method processes only single images from each DataLoader, NOT batches or
        multiple images. It's designed for fine-grained comparison between one clean
        and one perturbed image.

        Args:
            baseline_loader: DataLoader containing exactly ONE baseline image
            counterfactual_loader: DataLoader containing exactly ONE counterfactual image
            top_k: Number of top channels to extract for overlap analysis
            save_dir: Directory to save results

        Returns:
            Dictionary with comparison results including overlaps and importance scores
        """
        print(
            f"[CIFS] Comparing channel importance between TWO images (baseline vs counterfactual)..."
        )
        print(
            f"[CIFS] Will identify top-{top_k} channels for each image and compute overlap"
        )

        self.model.eval()

        def analyze_single_image(dataloader, image_type):
            """Analyze channel importance for exactly ONE image."""
            print(f"[CIFS] Analyzing {image_type} image (1 image only)...")

            # Get the single image from the dataloader
            images, labels = next(iter(dataloader))
            assert (
                images.size(0) == 1
            ), f"Expected single image, got batch of {images.size(0)}"
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Hook storage for activations
            activations = {}
            handles = []

            def save_activation(name):
                def hook(module, input, output):
                    activations[name] = output

                return hook

            # Register hooks
            for layer_name, module in self.layer_modules.items():
                if layer_name in self.probe_networks:
                    handle = module.register_forward_hook(save_activation(layer_name))
                    handles.append(handle)

            # Forward pass
            with torch.no_grad():
                output = self.model(images)
                predicted_class = output.argmax(dim=1).item()

            # Store relevance for each layer
            layer_relevances = {}

            # Compute relevance for each layer
            for layer_name in self.layer_names:
                if layer_name in activations and layer_name in self.probe_networks:
                    activation = activations[layer_name]

                    # Compute channel relevance
                    relevance = self.compute_channel_relevance(
                        layer_name, activation, true_label=labels
                    )

                    if relevance is not None:
                        layer_relevances[layer_name] = relevance.detach().cpu()

            # Clean up hooks
            for handle in handles:
                handle.remove()

            # Process results
            results = {}
            for layer_name, relevance in layer_relevances.items():
                pr_channels = torch.where(relevance > 0)[0].tolist()
                nr_channels = torch.where(relevance <= 0)[0].tolist()

                results[layer_name] = {
                    "relevance": relevance,
                    "pr_channels": pr_channels,
                    "nr_channels": nr_channels,
                    "num_pr": len(pr_channels),
                    "num_nr": len(nr_channels),
                }

            return results, predicted_class, labels.item()

        # Analyze both images
        baseline_results, baseline_pred, baseline_true = analyze_single_image(
            baseline_loader, "baseline"
        )
        counterfactual_results, counter_pred, counter_true = analyze_single_image(
            counterfactual_loader, "counterfactual"
        )

        print(f"[CIFS] Baseline: true={baseline_true}, pred={baseline_pred}")
        print(f"[CIFS] Counterfactual: true={counter_true}, pred={counter_pred}")

        # Extract top-k channels globally for each image
        def get_global_top_k(results, k):
            """Get top-k PR and NR channels globally across all layers."""
            all_pr = []
            all_nr = []

            for layer_name, layer_data in results.items():
                relevance = layer_data["relevance"]

                # PR channels
                for ch_idx in layer_data["pr_channels"]:
                    all_pr.append(
                        {
                            "layer": layer_name,
                            "channel": ch_idx,
                            "score": relevance[ch_idx].item(),
                        }
                    )

                # NR channels
                for ch_idx in layer_data["nr_channels"]:
                    all_nr.append(
                        {
                            "layer": layer_name,
                            "channel": ch_idx,
                            "score": abs(relevance[ch_idx].item()),
                        }
                    )

            # Sort and get top-k
            all_pr.sort(key=lambda x: x["score"], reverse=True)
            all_nr.sort(key=lambda x: x["score"], reverse=True)

            top_pr = all_pr[:k]
            top_nr = all_nr[:k]

            return top_pr, top_nr

        # Get top-k channels for both images
        baseline_top_pr, baseline_top_nr = get_global_top_k(baseline_results, top_k)
        counter_top_pr, counter_top_nr = get_global_top_k(counterfactual_results, top_k)

        # Compute overlap
        def compute_overlap(list1, list2):
            """Compute overlap between two lists of channels."""
            set1 = {(item["layer"], item["channel"]) for item in list1}
            set2 = {(item["layer"], item["channel"]) for item in list2}
            overlap = set1 & set2
            return list(overlap), (
                len(overlap) / min(len(set1), len(set2))
                if min(len(set1), len(set2)) > 0
                else 0
            )

        pr_overlap, pr_overlap_ratio = compute_overlap(baseline_top_pr, counter_top_pr)
        nr_overlap, nr_overlap_ratio = compute_overlap(baseline_top_nr, counter_top_nr)

        print(f"\n[CIFS] === Top-{top_k} Channel Overlap ===")
        print(
            f"PR overlap: {len(pr_overlap)}/{top_k} channels ({pr_overlap_ratio*100:.1f}%)"
        )
        print(
            f"NR overlap: {len(nr_overlap)}/{top_k} channels ({nr_overlap_ratio*100:.1f}%)"
        )

        # Show some example overlapping channels
        if pr_overlap:
            print(f"\n[CIFS] Example PR overlapping channels (first 5):")
            for layer, ch in pr_overlap[:5]:
                print(f"  - {layer}: channel {ch}")

        if nr_overlap:
            print(f"\n[CIFS] Example NR overlapping channels (first 5):")
            for layer, ch in nr_overlap[:5]:
                print(f"  - {layer}: channel {ch}")

        # Save results
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            # Save JSON results - top 20 for readability
            json_results = {
                "baseline": {
                    "predicted_class": baseline_pred,
                    "true_class": baseline_true,
                    "top_20_pr": [
                        {
                            "layer": item["layer"],
                            "channel": item["channel"],
                            "score": item["score"],
                        }
                        for item in baseline_top_pr[:20]
                    ],
                    "top_20_nr": [
                        {
                            "layer": item["layer"],
                            "channel": item["channel"],
                            "score": item["score"],
                        }
                        for item in baseline_top_nr[:20]
                    ],
                },
                "counterfactual": {
                    "predicted_class": counter_pred,
                    "true_class": counter_true,
                    "top_20_pr": [
                        {
                            "layer": item["layer"],
                            "channel": item["channel"],
                            "score": item["score"],
                        }
                        for item in counter_top_pr[:20]
                    ],
                    "top_20_nr": [
                        {
                            "layer": item["layer"],
                            "channel": item["channel"],
                            "score": item["score"],
                        }
                        for item in counter_top_nr[:20]
                    ],
                },
                "overlap": {
                    f"top_{top_k}_pr_overlap": pr_overlap,
                    f"top_{top_k}_nr_overlap": nr_overlap,
                    "pr_overlap_ratio": pr_overlap_ratio,
                    "nr_overlap_ratio": nr_overlap_ratio,
                },
            }

            json_path = os.path.join(save_dir, "two_image_comparison.json")
            with open(json_path, "w") as f:
                json.dump(json_results, f, indent=2)
            print(f"[CIFS] Saved comparison results to {json_path}")

            # Visualize channel importance for both images
            self._visualize_two_image_comparison(
                baseline_results,
                counterfactual_results,
                baseline_top_pr[:20],
                baseline_top_nr[:20],
                counter_top_pr[:20],
                counter_top_nr[:20],
                save_dir,
            )

        return {
            "baseline_results": baseline_results,
            "counterfactual_results": counterfactual_results,
            "baseline_top_pr": baseline_top_pr,
            "baseline_top_nr": baseline_top_nr,
            "counterfactual_top_pr": counter_top_pr,
            "counterfactual_top_nr": counter_top_nr,
            "pr_overlap": pr_overlap,
            "nr_overlap": nr_overlap,
            "pr_overlap_ratio": pr_overlap_ratio,
            "nr_overlap_ratio": nr_overlap_ratio,
        }

    def _visualize_two_image_comparison(
        self,
        baseline_results,
        counterfactual_results,
        baseline_top_pr,
        baseline_top_nr,
        counter_top_pr,
        counter_top_nr,
        save_dir,
    ):
        """Create visualization comparing two images."""
        # Select layers to visualize (first 3 layers with data)
        layers_to_viz = []
        for layer in self.layer_names[:5]:
            if layer in baseline_results and layer in counterfactual_results:
                layers_to_viz.append(layer)
            if len(layers_to_viz) >= 3:
                break

        if not layers_to_viz:
            print("[CIFS] No layers to visualize")
            return

        num_layers = len(layers_to_viz)
        fig, axes = plt.subplots(num_layers, 2, figsize=(16, 5 * num_layers))

        if num_layers == 1:
            axes = axes.reshape(1, -1)

        # Create sets of top channels for highlighting
        baseline_pr_set = {(item["layer"], item["channel"]) for item in baseline_top_pr}
        baseline_nr_set = {(item["layer"], item["channel"]) for item in baseline_top_nr}
        counter_pr_set = {(item["layer"], item["channel"]) for item in counter_top_pr}
        counter_nr_set = {(item["layer"], item["channel"]) for item in counter_top_nr}

        for idx, layer_name in enumerate(layers_to_viz):
            # Baseline image
            ax1 = axes[idx, 0]
            relevance_b = baseline_results[layer_name]["relevance"].numpy()
            channels = range(len(relevance_b))

            # Color based on sign and if in top-k
            colors_b = []
            for ch_idx, r in enumerate(relevance_b):
                if (layer_name, ch_idx) in baseline_pr_set:
                    colors_b.append("darkgreen")  # Top PR
                elif (layer_name, ch_idx) in baseline_nr_set:
                    colors_b.append("darkred")  # Top NR
                elif r > 0:
                    colors_b.append("lightgreen")  # PR but not top
                else:
                    colors_b.append("lightcoral")

            ax1.bar(channels, relevance_b, color=colors_b, alpha=0.7)
            ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax1.set_title(
                f'{layer_name} - Baseline (PR={baseline_results[layer_name]["num_pr"]}, NR={baseline_results[layer_name]["num_nr"]})'
            )
            ax1.set_xlabel("Channel Index")
            ax1.set_ylabel("Relevance Score")
            ax1.grid(True, alpha=0.3)

            # Counterfactual image
            ax2 = axes[idx, 1]
            relevance_c = counterfactual_results[layer_name]["relevance"].numpy()

            # Color based on sign and if in top-k
            colors_c = []
            for ch_idx, r in enumerate(relevance_c):
                if (layer_name, ch_idx) in counter_pr_set:
                    colors_c.append("darkgreen")  # Top PR
                elif (layer_name, ch_idx) in counter_nr_set:
                    colors_c.append("darkred")  # Top NR
                elif r > 0:
                    colors_c.append("lightgreen")  # PR but not top
                else:
                    colors_c.append("lightcoral")  # NR but not top

            ax2.bar(channels, relevance_c, color=colors_c, alpha=0.7)
            ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax2.set_title(
                f'{layer_name} - Counterfactual (PR={counterfactual_results[layer_name]["num_pr"]}, NR={counterfactual_results[layer_name]["num_nr"]})'
            )
            ax2.set_xlabel("Channel Index")
            ax2.set_ylabel("Relevance Score")
            ax2.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="darkgreen", label="Top-20 PR"),
            Patch(facecolor="darkred", label="Top-20 NR"),
            Patch(facecolor="lightgreen", label="Other PR"),
            Patch(facecolor="lightcoral", label="Other NR"),
        ]
        fig.legend(
            handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98)
        )

        plt.suptitle(
            "Channel Importance Comparison: Baseline vs Counterfactual", fontsize=14
        )
        plt.tight_layout()

        save_path = os.path.join(save_dir, "channel_importance_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[CIFS] Saved visualization to {save_path}")
        plt.close()


def main():
    """Main function to run corrected CIFS analysis."""

    parser = argparse.ArgumentParser(description="CIFS Analysis")
    parser.add_argument(
        "--compare_two",
        action="store_true",
        help="Compare channel importance between two images only",
    )
    args = parser.parse_args()

    print("[CIFS] =============== CORRECTED CIFS ANALYSIS START ===============")

    # Initialize configuration
    config = Config()

    # Load model
    model = load_model(config)
    nnsight_model = NNsight(model)

    # Initialize corrected CIFS
    cifs = CIFsCorrect(model, nnsight_model, config)

    if args.compare_two:
        # Two-image comparison mode - ONLY processes 2 images total
        print("\n[CIFS] === Two-Image Comparison Mode ===")

        # Load exactly two images: one baseline, one counterfactual
        # This returns DataLoaders with single images only
        dataloaders = load_baseline_dataloader(config)

        # Set save directory
        save_dir = os.path.join("temp", "cifs", "two_images")

        # Run comparison on just these two images
        comparison_results = cifs.compare_two_images(
            dataloaders["baseline"],  # Single baseline image
            dataloaders["counterfactual"],  # Single counterfactual image
            top_k=50,  # Compare top-50 channels as requested
            save_dir=save_dir,
        )

        print(f"\n[CIFS] Two-image comparison complete. Results saved to {save_dir}")

        return comparison_results

    else:
        raise
        # Original full analysis mode - processes 100 images
        print("\n[CIFS] === Full Analysis Mode (100 images) ===")

        # Load full dataset for multi-image analysis
        dataloaders = load_class_specific_dataloader(config, num_images_per_class=100)

        # Analyze PR/NR channels across 100 images
        print("\n[CIFS] === Analyzing PR/NR Channels from 100 images ===")
        results = cifs.identify_pr_nr_channels(
            dataloaders["baseline"],
            num_images=100,  # Processing 100 images for statistical significance
        )

        # Get top-k channels
        top_pr, top_nr = cifs.get_top_k_channels(results, k=20)

        # Save and visualize
        save_dir = os.path.join(config.results_dir, "cifs_correct")
        cifs.save_results(results, save_dir)
        cifs.visualize_results(results, save_dir)

        print(
            "\n[CIFS] =============== CORRECTED CIFS ANALYSIS COMPLETE ==============="
        )

        return results, top_pr, top_nr


if __name__ == "__main__":
    results = main()
