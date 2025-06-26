"""
Adversarial Activation Patching Analysis for ResNet18
Analyzes the impact of adversarial perturbations through layer-wise and channel-wise activation patching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nnsight import NNsight
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# Add parent directory for custom modules
sys.path.append(os.path.abspath(os.path.join("..")))
from models.resnet import ResNet18
from adversarial.utils import load_dataset


@dataclass
class Config:
    """Configuration for activation patching experiments"""

    datapath: str = "/u/yshi23/distribution-shift/datasets"
    model_path: str = "/u/yshi23/distribution-shift/adversarial/verify/best_model.pth"
    perturb_path: str = "/u/yshi23/distribution-shift/adversarial/path_to_save"
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    batch_size: int = 64
    results_dir: str = "results"

    # Transform parameters
    normalize_mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    normalize_std: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010)

    # Perturbation dataset mapping
    dataset_mapping: Dict[str, str] = None

    def __post_init__(self):
        """Initialize dataset mapping and create results directory"""
        self.dataset_mapping = {
            "normal_high_dataset": "linf_high_dataset",
            "normal_low_dataset": "linf_low_dataset",
            "l2_normal_high_dataset": "l2_high_dataset",
            "l2_normal_low_dataset": "l2_low_dataset",
        }
        Path(self.results_dir).mkdir(exist_ok=True)


# Standard CIFAR-10 transforms
transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.25, 0.25, 0.25),
        transforms.RandomRotation(2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def load_model(config):
    """Load pre-trained ResNet18 model"""
    model = ResNet18(num_classes=10)
    checkpoint = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(config.device)
    model.eval()
    return model


def load_datasets(config):
    """Load baseline CIFAR-10 dataset"""
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    baseline_dataset = ConcatDataset([trainset, testset])
    return DataLoader(baseline_dataset, batch_size=config.batch_size, shuffle=False)


def load_perturbation_data(pt_path, config):
    """Load adversarial perturbation data"""
    data = torch.load(f"{pt_path}/data.pt")
    labels = torch.load(f"{pt_path}/labels.pt")
    dataset = torch.utils.data.TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False)


def load_all_perturbations(config):
    """Load all perturbation datasets"""
    dataloader_dict = {"baseline": load_datasets(config)}

    # Map folder names to dict keys
    name_mapping = config.dataset_mapping

    for folder_name, key in name_mapping.items():
        path = os.path.join(config.perturb_path, folder_name)
        if os.path.exists(path):
            dataloader_dict[key] = load_perturbation_data(path, config)

    return dataloader_dict


def get_nested_module(model, name):
    """Get nested module by name"""
    parts = name.split(".")
    for part in parts:
        if part.isdigit():
            model = model[int(part)]
        else:
            model = getattr(model, part)
    return model


def compute_average_confidence_drop(
    nnsight_model,
    perturbations_dict,
    scenario="linf_high",
    num_images=100,
    patch_mode="layer",
    channel_indices=None,
    target_layer=None,
    config=None,
):
    """
    Computes the average confidence drop for a given scenario using activation patching.

    Args:
        nnsight_model: NNsight wrapped model
        perturbations_dict: Dict of dataloaders
        scenario: one of "linf_high", "linf_low", "l2_high", "l2_low", etc.
        num_images: Number of images to process
        patch_mode: "layer" for full layer patching, "channel" for channel-wise patching
        channel_indices: List of channel indices to patch (for channel mode)
        target_layer: Specific layer for channel patching (for channel mode)
        config: Config object with device information

    Returns:
        dict: {layer_name: average confidence drop}
    """
    # Map scenario to dataloader keys
    scenario_map = {
        "linf_high": ("baseline", "linf_high_dataset"),
        "linf_low": ("baseline", "linf_low_dataset"),
        "l2_high": ("baseline", "l2_high_dataset"),
        "l2_low": ("baseline", "l2_low_dataset"),
        "linf_low_vs_high": ("linf_low_dataset", "linf_high_dataset"),
        "l2_low_vs_high": ("l2_low_dataset", "l2_high_dataset"),
    }

    if scenario not in scenario_map:
        raise ValueError(f"Unknown scenario: {scenario}")

    loader_clean = perturbations_dict[scenario_map[scenario][0]]
    loader_adv = perturbations_dict[scenario_map[scenario][1]]

    # Get valid module names - following original code structure
    module_names = [
        name for name, module in nnsight_model._model.named_modules() if name
    ]

    # For channel mode, only patch the target layer
    if patch_mode == "channel":
        if not target_layer:
            raise ValueError("target_layer must be specified for channel mode")
        if target_layer not in module_names:
            raise ValueError(f"Target layer {target_layer} not found in model")
        if channel_indices is None or len(channel_indices) == 0:
            raise ValueError("channel_indices must be specified for channel mode")
        LAYER_NAMES = [target_layer]
    else:
        LAYER_NAMES = module_names

    # Initialize iterators and storage
    clean_iter = iter(loader_clean)
    adv_iter = iter(loader_adv)
    confidence_drops = {name: [] for name in LAYER_NAMES}

    # Process images
    for i in range(num_images):
        try:
            clean_img, _ = next(clean_iter)
            adv_img, _ = next(adv_iter)
        except StopIteration:
            break

        clean_img = clean_img[0:1].to(config.device)
        adv_img = adv_img[0:1].to(config.device)

        # Get clean logits - following original code pattern
        with nnsight_model.trace(clean_img):
            # For ResNet, we need to find the final FC layer
            # Try common names
            clean_logits_proxy = None
            for fc_name in ["fc", "linear", "classifier"]:
                try:
                    fc_module = get_nested_module(nnsight_model, fc_name)
                    clean_logits_proxy = fc_module.output.save()
                    break
                except AttributeError:
                    continue

            # If not found, use the model output directly
            if clean_logits_proxy is None:
                clean_logits_proxy = nnsight_model.output.save()

        clean_logits = clean_logits_proxy.detach().cpu()
        clean_label = torch.argmax(clean_logits, dim=1).item()

        # Get adv activations
        with nnsight_model.trace(adv_img):
            adv_activations_proxy_dict = {
                name: get_nested_module(nnsight_model, name).output.save()
                for name in LAYER_NAMES
            }

        # Extract adversarial activation values
        adv_activations = {}
        for k, v_proxy in adv_activations_proxy_dict.items():
            try:
                adv_activations[k] = v_proxy.value.detach()
            except AttributeError:
                adv_activations[k] = v_proxy.detach()

        # Validate channel indices if in channel mode
        if patch_mode == "channel":
            num_channels = adv_activations[target_layer].shape[1]
            invalid_indices = [
                idx for idx in channel_indices if idx >= num_channels or idx < 0
            ]
            if invalid_indices:
                raise ValueError(
                    f"Invalid channel indices {invalid_indices} for layer {target_layer} "
                    f"with {num_channels} channels. Valid range: [0, {num_channels-1}]"
                )

        # For each layer, patch and compute confidence drop
        for name in LAYER_NAMES:
            module_to_patch = get_nested_module(nnsight_model, name)

            with nnsight_model.trace(clean_img):
                if patch_mode == "layer":
                    # Full layer patching
                    module_to_patch.output = adv_activations[name]

                elif patch_mode == "channel" and channel_indices is not None:
                    # Channel-wise patching - more efficient single-trace approach
                    clean_act = module_to_patch.output

                    # Create patched activation
                    patched_act = clean_act.clone()

                    # Patch based on tensor dimensions
                    if (
                        len(adv_activations[name].shape) == 4
                    ):  # Conv layer: [B, C, H, W]
                        patched_act[:, channel_indices, :, :] = adv_activations[name][
                            :, channel_indices, :, :
                        ]
                    elif (
                        len(adv_activations[name].shape) == 3
                    ):  # BatchNorm or similar: [B, C, L]
                        patched_act[:, channel_indices, :] = adv_activations[name][
                            :, channel_indices, :
                        ]
                    elif len(adv_activations[name].shape) == 2:  # FC layer: [B, C]
                        patched_act[:, channel_indices] = adv_activations[name][
                            :, channel_indices
                        ]
                    else:
                        raise ValueError(
                            f"Unexpected tensor shape for layer {name}: {adv_activations[name].shape}"
                        )

                    # Replace output with patched activation
                    module_to_patch.output = patched_act

                # Get patched logits - same approach as clean logits
                patched_logits_proxy = None
                for fc_name in ["fc", "linear", "classifier"]:
                    try:
                        fc_module = get_nested_module(nnsight_model, fc_name)
                        patched_logits_proxy = fc_module.output.save()
                        break
                    except AttributeError:
                        continue

                if patched_logits_proxy is None:
                    patched_logits_proxy = nnsight_model.output.save()

            patched_logits = patched_logits_proxy.detach().cpu()
            drop = (
                clean_logits[0, clean_label] - patched_logits[0, clean_label]
            ).item()
            confidence_drops[name].append(drop)

    # Compute average drop per layer
    avg_confidence_drop = {
        name: float(np.mean(confidence_drops[name])) if confidence_drops[name] else 0.0
        for name in LAYER_NAMES
    }

    return avg_confidence_drop


def get_nested_module(model, module_name):
    """
    Get a nested module from the model by its full name.
    This is used to access modules through the nnsight wrapper.

    Args:
        model: The nnsight model
        module_name: Dot-separated module name (e.g., "layer1.0.conv1")

    Returns:
        The requested module (through nnsight wrapper)
    """
    # Start from the model root
    module = model

    # Navigate through the module hierarchy
    for part in module_name.split("."):
        module = getattr(module, part)

    return module


def compute_channel_importance(
    nnsight_model,
    perturbations_dict,
    layer_name,
    scenario="linf_high",
    num_images=100,
    config=None,
):
    """
    Compute importance of each channel in a specific layer.

    Args:
        layer_name: Name of layer to analyze
        scenario: Scenario name
        num_images: Number of images to use
        config: Config object

    Returns:
        dict: {channel_idx: average confidence drop}
    """
    # First, get the number of channels in this layer
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32).to(config.device)
        with nnsight_model.trace(dummy_input):
            module = get_nested_module(nnsight_model, layer_name)
            activation = module.output.save()

        act_shape = activation.shape
        if len(act_shape) == 4:  # Conv layer (N,C,H,W)
            num_channels = act_shape[1]
        elif len(act_shape) == 2:  # FC layer, (N,C)
            num_channels = act_shape[1]
        else:
            raise ValueError(f"Unsupported activation shape: {act_shape}")

    channel_drops = {}

    # Test channels individuallyz
    for ch_idx in tqdm(range(num_channels), desc=f"Testing channels in {layer_name}"):
        result = compute_average_confidence_drop(
            nnsight_model,
            perturbations_dict,
            scenario,
            num_images,
            patch_mode="channel",
            channel_indices=[ch_idx],
            target_layer=layer_name,
            config=config,
        )

        channel_drops[ch_idx] = result[layer_name]

    return channel_drops


def plot_stacked_confidence_drop_heatmaps(
    nnsight_model, perturbations_dict, num_images=10, save_path=None, config=None
):
    """Plot stacked heatmaps for different patching scenarios"""
    # Define scenarios and titles for plotting
    scenarios = [
        ("linf_low", "Clean Baseline vs. Linf Low Eps Perturbation"),
        ("linf_high", "Clean Baseline vs. Linf High Eps Perturbation"),
        ("linf_low_vs_high", "Linf Low Eps vs. Linf High Eps Perturbation"),
        ("l2_low", "Clean Baseline vs. L2 Low Eps Perturbation"),
        ("l2_high", "Clean Baseline vs. L2 High Eps Perturbation"),
        ("l2_low_vs_high", "L2 Low Eps vs. L2 High Eps Perturbation"),
    ]

    plot_data = []
    titles = []

    for scenario, title in scenarios:
        print(f"Computing: {title}...")
        try:
            results = compute_average_confidence_drop(
                nnsight_model,
                perturbations_dict,
                scenario=scenario,
                num_images=num_images,
                config=config,
            )
        except Exception as e:
            print(f"Error in scenario {scenario}: {e}")
            results = None
        plot_data.append(results)
        titles.append(title)

    # Check for at least one valid result
    if not any(plot_data):
        print("Error: No data to plot after computing all scenarios.")
        return

    # Use the first non-empty result to get layer names
    for res in plot_data:
        if res:
            layer_names = list(res.keys())
            break

    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(15, 16), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    all_values = []
    for res_dict in plot_data:
        if res_dict:
            all_values.extend(res_dict.values())

    if not all_values:
        print("Error: No data to plot after computing all scenarios.")
        return

    vmin = min(all_values)
    vmax = max(all_values)
    if vmin == vmax:
        vmin -= 0.1
        vmax += 0.1

    # Define a blue-purple-red colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        "blue_purple_red", ["blue", "purple", "red"]
    )

    for i, ax in enumerate(axes):
        current_drops_dict = plot_data[i]
        current_title = titles[i]

        if not current_drops_dict:
            print(f"Warning: No data for scenario: {current_title}")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        heatmap_values = np.array(list(current_drops_dict.values())).reshape(1, -1)
        im = ax.imshow(
            heatmap_values, cmap=custom_cmap, aspect="auto", vmin=vmin, vmax=vmax
        )

        ax.set_yticks([0])
        ax.text(
            -0.05,
            0.5,
            current_title,
            rotation=0,
            va="center",
            ha="right",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(len(layer_names)))

        if i == len(axes) - 1:
            ax.set_xticklabels(layer_names, rotation=90, ha="center")
        else:
            ax.set_xticklabels([])

    fig.subplots_adjust(left=0.2, right=0.80, top=0.95, bottom=0.10)
    cbar_ax = fig.add_axes([0.85, 0.10, 0.03, 0.85])
    fig.colorbar(
        im, cax=cbar_ax, label="Average Confidence Drop", orientation="vertical"
    )

    plt.suptitle(
        "Confidence Drop Per Layer for Different Patching Scenarios (Linf and L2)",
        fontsize=16,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved layer-wise patching heatmap to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_channel_importance(channel_drops, layer_name, scenario, save_path=None):
    """Plot channel importance as a bar chart"""
    channels = sorted(channel_drops.keys())
    drops = [channel_drops[ch] for ch in channels]

    # Create figure with appropriate size based on number of channels
    fig_width = max(12, len(channels) * 0.15)
    plt.figure(figsize=(fig_width, 6))

    # Create bar plot
    bars = plt.bar(channels, drops, width=0.8)

    # Color bars based on importance (gradient from blue to red)
    norm = plt.Normalize(vmin=min(drops), vmax=max(drops))
    cmap = LinearSegmentedColormap.from_list("importance", ["blue", "purple", "red"])
    for bar, drop in zip(bars, drops):
        bar.set_color(cmap(norm(drop)))

    plt.xlabel("Channel Index", fontsize=12)
    plt.ylabel("Average Confidence Drop", fontsize=12)
    plt.title(f"Channel Importance in {layer_name} ({scenario})", fontsize=14)

    # Add grid for better readability
    plt.grid(True, axis="y", alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.01)
    cbar.set_label("Importance", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved channel importance plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_channel_importance_comparison(all_channel_drops, layer_name, save_path=None):
    """Plot channel importance comparison across different scenarios"""
    scenarios = list(all_channel_drops.keys())
    if not scenarios:
        print("No data to plot")
        return

    # Get all unique channels
    all_channels = set()
    for drops in all_channel_drops.values():
        all_channels.update(drops.keys())
    channels = sorted(all_channels)

    # Prepare data for plotting
    n_scenarios = len(scenarios)
    n_channels = len(channels)
    x = np.arange(n_channels)
    width = 0.8 / n_scenarios

    plt.figure(figsize=(max(12, n_channels * 0.3), 8))

    colors = plt.cm.Set3(np.linspace(0, 1, n_scenarios))

    for i, (scenario, drops) in enumerate(all_channel_drops.items()):
        values = [drops.get(ch, 0) for ch in channels]
        offset = (i - n_scenarios / 2 + 0.5) * width
        plt.bar(x + offset, values, width, label=scenario, color=colors[i])

    plt.xlabel("Channel Index", fontsize=12)
    plt.ylabel("Average Confidence Drop", fontsize=12)
    plt.title(f"Channel Importance Comparison in {layer_name}", fontsize=14)
    plt.xticks(x, channels)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved channel comparison plot to {save_path}")
    else:
        plt.show()
    plt.close()


# Main execution
if __name__ == "__main__":
    print("Loading model and datasets...")

    # Initialize configuration
    config = Config()

    # Load model and wrap with NNsight
    model = load_model(config)
    nnsight_model = NNsight(model)

    # Load datasets
    dataloaders = load_all_perturbations(config)
    print(f"Loaded datasets: {list(dataloaders.keys())}")

    # ========== LAYER-WISE PATCHING EXAMPLE ==========
    print("\n" + "=" * 50)
    print("LAYER-WISE ACTIVATION PATCHING")
    print("=" * 50)

    # Run layer-wise patching and save heatmap
    layer_heatmap_path = os.path.join(config.results_dir, "patching_layer_heatmap.png")
    plot_stacked_confidence_drop_heatmaps(
        nnsight_model,
        dataloaders,
        num_images=1000,  # Use more images for stable results
        save_path=layer_heatmap_path,
        config=config,
    )

    # ========== CHANNEL-WISE PATCHING EXAMPLE ==========
    print("\n" + "=" * 50)
    print("CHANNEL-WISE ACTIVATION PATCHING")
    print("=" * 50)

    # Example: Analyze channels in layer3.0.conv1
    target_layer = "conv1"
    print(f"\nAnalyzing channel importance in {target_layer}...")

    # Compute channel importance for linf_high scenario
    channel_drops_linf_high = compute_channel_importance(
        nnsight_model,
        dataloaders,
        target_layer,
        scenario="linf_high",
        num_images=500,  # Use fewer images per channel for speed
        config=config,
    )

    # Save channel importance plot
    channel_heatmap_path = os.path.join(
        config.results_dir, "patching_channel_heatmap.png"
    )
    plot_channel_importance(
        channel_drops_linf_high,
        target_layer,
        "linf_high",
        save_path=channel_heatmap_path,
    )

    # ========== COMPARATIVE ANALYSIS ==========
    print("\n" + "=" * 50)
    print("COMPARATIVE CHANNEL ANALYSIS")
    print("=" * 50)

    # Compare channel importance across different perturbation types
    scenarios_to_compare = ["linf_low", "linf_high", "l2_low", "l2_high"]
    all_channel_drops = {}

    for scenario in scenarios_to_compare:
        print(f"\nComputing channel importance for {scenario}...")
        channel_drops = compute_channel_importance(
            nnsight_model,
            dataloaders,
            target_layer,
            scenario=scenario,
            num_images=100,  # Fewer images for quick comparison
            config=config,
        )
        all_channel_drops[scenario] = channel_drops

        # Print top 5 most important channels
        sorted_channels = sorted(
            channel_drops.items(), key=lambda x: x[1], reverse=True
        )[:5]
        print(f"\nTop 5 most important channels for {scenario}:")
        for ch_idx, drop in sorted_channels:
            print(f"  Channel {ch_idx}: {drop:.4f}")

    # Save comparison plot
    comparison_path = os.path.join(
        config.results_dir, "patching_channel_comparison.png"
    )
    plot_channel_importance_comparison(
        all_channel_drops, target_layer, save_path=comparison_path
    )

    # ========== SUMMARY STATISTICS ==========
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"\nAll results saved to '{config.results_dir}/' directory:")
    print(f"  - Layer-wise patching heatmap: {layer_heatmap_path}")
    print(f"  - Channel importance plot: {channel_heatmap_path}")
    print(f"  - Channel comparison plot: {comparison_path}")

    # Additional analysis: Find consistently important channels
    print(f"\n\nConsistently important channels in {target_layer}:")

    # Find channels that are in top 10 for all scenarios
    top_n = 10
    top_channels_per_scenario = {}
    for scenario, drops in all_channel_drops.items():
        sorted_ch = sorted(drops.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_channels_per_scenario[scenario] = set([ch for ch, _ in sorted_ch])

    # Find intersection
    consistent_channels = set.intersection(*top_channels_per_scenario.values())
    print(
        f"Channels in top {top_n} across all scenarios: {sorted(consistent_channels)}"
    )

    # Average importance across scenarios
    if consistent_channels:
        print("\nAverage importance of consistent channels:")
        for ch in sorted(consistent_channels):
            avg_importance = np.mean(
                [drops[ch] for drops in all_channel_drops.values()]
            )
            print(f"  Channel {ch}: {avg_importance:.4f}")
