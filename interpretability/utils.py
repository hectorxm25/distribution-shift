"""
Utility functions for adversarial activation patching analysis.
Contains helper functions for data loading, model operations, and visualization.
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
from typing import Dict, Tuple
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
    results_dir: str = "results/temp/"

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
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(0.25, 0.25, 0.25),
        # transforms.RandomRotation(2),
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


def load_baseline_dataloader(config):
    """Load baseline CIFAR-10 dataloader"""
    dl = load_datasets(config)
    # NOTE here we are just returning the first image as the baseline, and the second image as the counterfactual
    # This is a simplification for the sake of the example
    # outputs {baseline: DataLoader, counterfactual: DataLoader} where each DataLoader contains one image

    # Get the first two samples from the dataset
    data_iter = iter(dl)
    baseline_img, baseline_label = next(data_iter)
    counterfactual_img, counterfactual_label = next(data_iter)

    # Take only the first image from each batch
    baseline_img = baseline_img[0:1]  # Keep batch dimension
    baseline_label = baseline_label[0:1]
    counterfactual_img = counterfactual_img[0:1]
    counterfactual_label = counterfactual_label[0:1]

    # Create TensorDatasets for single images
    baseline_dataset = TensorDataset(baseline_img, baseline_label)
    counterfactual_dataset = TensorDataset(counterfactual_img, counterfactual_label)

    # Create DataLoaders with batch_size=1
    baseline_loader = DataLoader(baseline_dataset, batch_size=1, shuffle=False)
    counterfactual_loader = DataLoader(
        counterfactual_dataset, batch_size=1, shuffle=False
    )

    # save the two images as png in /visuals subfolder
    visuals_dir = os.path.join(config.results_dir, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)
    torchvision.utils.save_image(
        baseline_img,
        os.path.join(visuals_dir, f"baseline_image_{baseline_label.item()}.png"),
        normalize=True,
        scale_each=True,
    )
    torchvision.utils.save_image(
        counterfactual_img,
        os.path.join(
            visuals_dir, f"counterfactual_image_{counterfactual_label.item()}.png"
        ),
        normalize=True,
        scale_each=True,
    )

    return {"baseline": baseline_loader, "counterfactual": counterfactual_loader}


def load_class_specific_dataloader(config, num_images_per_class=100):
    """
    Load CIFAR-10 dataloader with specific classes for class flip analysis.
    Returns 100 images from class 0 as baseline and 100 images from class 1 as counterfactual.
    """
    # Load full CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    full_dataset = ConcatDataset([trainset, testset])
    
    # Separate images by class
    class_0_images = []
    class_0_labels = []
    class_1_images = []
    class_1_labels = []
    
    for img, label in full_dataset:
        if label == 0 and len(class_0_images) < num_images_per_class:
            class_0_images.append(img)
            class_0_labels.append(label)
        elif label == 1 and len(class_1_images) < num_images_per_class:
            class_1_images.append(img)
            class_1_labels.append(label)
        
        # Stop when we have enough images from both classes
        if len(class_0_images) >= num_images_per_class and len(class_1_images) >= num_images_per_class:
            break
    
    # Convert to tensors
    class_0_tensor = torch.stack(class_0_images)
    class_0_labels_tensor = torch.tensor(class_0_labels)
    class_1_tensor = torch.stack(class_1_images)
    class_1_labels_tensor = torch.tensor(class_1_labels)
    
    # Create datasets and dataloaders
    class_0_dataset = TensorDataset(class_0_tensor, class_0_labels_tensor)
    class_1_dataset = TensorDataset(class_1_tensor, class_1_labels_tensor)
    
    class_0_loader = DataLoader(class_0_dataset, batch_size=config.batch_size, shuffle=False)
    class_1_loader = DataLoader(class_1_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"Loaded {len(class_0_images)} images from class 0 and {len(class_1_images)} images from class 1")
    
    return {"baseline": class_0_loader, "counterfactual": class_1_loader}


def get_layer_names():
    """reads layer names in model_layer_names and returns a list of layer names"""
    # open the file and read lines
    layer_names_file = "model_layer_names.txt"
    if not os.path.exists(layer_names_file):
        raise FileNotFoundError(f"Layer names file {layer_names_file} not found.")
    with open(layer_names_file, "r") as f:
        lines = f.readlines()
    # split by linebreaks and strip whitespace
    # only include lines that contains the word "weight"
    layer_names = [
        line.strip().replace(".weight", "")
        for line in lines
        if line.strip() and "weight" in line
    ]
    return layer_names


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
    # Set model to evaluation mode - CRITICAL for deterministic behavior
    nnsight_model.eval()

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Map scenario to dataloader keys
    scenario_map = {
        "linf_high": ("baseline", "linf_high_dataset"),
        "linf_low": ("baseline", "linf_low_dataset"),
        "l2_high": ("baseline", "l2_high_dataset"),
        "l2_low": ("baseline", "l2_low_dataset"),
        "linf_low_vs_high": ("linf_low_dataset", "linf_high_dataset"),
        "l2_low_vs_high": ("l2_low_dataset", "l2_high_dataset"),
        "counterfactual": ("baseline", "counterfactual"),  # for the single image case
        "class_flip_analysis": ("baseline", "counterfactual"),  # for class 0 vs class 1 analysis
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

    # Process images - wrap in no_grad for efficiency and determinism
    with torch.no_grad():
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

            # Get adv activations with explicit cloning for safety
            adv_activations = {}
            with nnsight_model.trace(adv_img):
                for name in LAYER_NAMES:
                    module = get_nested_module(nnsight_model, name)
                    # Clone and detach immediately to ensure we capture the right values
                    act_proxy = module.output.save()

            # Extract adversarial activation values outside the trace context
            for name in LAYER_NAMES:
                module = get_nested_module(nnsight_model, name)
                with nnsight_model.trace(adv_img):
                    act_proxy = module.output.save()
                # Extract and clone the activation to ensure it's independent
                try:
                    adv_activations[name] = act_proxy.value.detach().clone()
                except AttributeError:
                    adv_activations[name] = act_proxy.detach().clone()

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
                        # Full layer patching - use clone to ensure no side effects
                        module_to_patch.output = adv_activations[name].clone()

                    elif patch_mode == "channel" and channel_indices is not None:
                        # Channel-wise patching - more efficient single-trace approach
                        clean_act = module_to_patch.output

                        # Create patched activation with explicit cloning
                        patched_act = clean_act.clone()

                        # Patch based on tensor dimensions
                        if (
                            len(adv_activations[name].shape) == 4
                        ):  # Conv layer: [B, C, H, W]
                            patched_act[:, channel_indices, :, :] = adv_activations[
                                name
                            ][:, channel_indices, :, :].clone()
                        elif (
                            len(adv_activations[name].shape) == 3
                        ):  # BatchNorm or similar: [B, C, L]
                            patched_act[:, channel_indices, :] = adv_activations[name][
                                :, channel_indices, :
                            ].clone()
                        elif len(adv_activations[name].shape) == 2:  # FC layer: [B, C]
                            patched_act[:, channel_indices] = adv_activations[name][
                                :, channel_indices
                            ].clone()
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


def average_class_flip(
    nnsight_model,
    perturbations_dict,
    scenario="linf_high",
    num_images=100,
    patch_mode="layer",
    channel_indices=None,
    target_layer=None,
    config=None,
    hit_rate_type="original",  # New parameter: "original" or "adversarial",
    base_class: int = None,  # Optional base class for targeted analysis
    flip_class: int = None,  # Optional flip class for targeted analysis
):
    """
    Computes the average class flip rate for a given scenario using activation patching.

    Args:
        nnsight_model: NNsight wrapped model
        perturbations_dict: Dict of dataloaders
        scenario: one of "linf_high", "linf_low", "l2_high", "l2_low", etc.
        num_images: Number of images to process
        patch_mode: "layer" for full layer patching, "channel" for channel-wise patching
        channel_indices: List of channel indices to patch (for channel mode)
        target_layer: Specific layer for channel patching (for channel mode)
        config: Config object with device information
        hit_rate_type:
            - "original": Measures class flip rate - percentage of cases where patching
                          causes the prediction to differ from the original clean prediction
            - "adversarial": Measures adversarial transfer rate - percentage of cases where
                             patching causes the prediction to match the adversarial prediction
        base_class: If specified, only consider images originally classified as this class as the "original" class
        flip_class: If specified, only consider images classified as this class after adversarial perturbation as the "adversarial" class

    Returns:
        dict: {layer_name: hit rate percentage (0-100)}
    """
    # Set model to evaluation mode - CRITICAL for deterministic behavior
    nnsight_model.eval()

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Map scenario to dataloader keys
    scenario_map = {
        "linf_high": ("baseline", "linf_high_dataset"),
        "linf_low": ("baseline", "linf_low_dataset"),
        "l2_high": ("baseline", "l2_high_dataset"),
        "l2_low": ("baseline", "l2_low_dataset"),
        "linf_low_vs_high": ("linf_low_dataset", "linf_high_dataset"),
        "l2_low_vs_high": ("l2_low_dataset", "l2_high_dataset"),
        "counterfactual": ("baseline", "counterfactual"),  # for the single image case
        "class_flip_analysis": ("baseline", "counterfactual"),  # for class 0 vs class 1 analysis
    }

    if scenario not in scenario_map:
        raise ValueError(f"Unknown scenario: {scenario}")

    if hit_rate_type not in ["original", "adversarial"]:
        raise ValueError(
            f"hit_rate_type must be 'original' or 'adversarial', got: {hit_rate_type}"
        )

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

    # Store hit counts instead of confidence drops
    hit_counts = {name: [] for name in LAYER_NAMES}

    # Process images - wrap in no_grad for efficiency and determinism
    with torch.no_grad():
        for i in range(num_images):
            try:
                clean_img, _ = next(clean_iter)
                adv_img, _ = next(adv_iter)

            except StopIteration:
                break

            clean_img = clean_img[0:1].to(config.device)  # [1, 3, 32, 32]
            adv_img = adv_img[0:1].to(config.device)  # [1, 3, 32, 32]

            # Step 1: Clean run - get clean logits and label
            with nnsight_model.trace(clean_img):
                # Find the final FC layer
                clean_logits_proxy = None
                for fc_name in ["fc", "linear", "classifier"]:
                    try:
                        fc_module = get_nested_module(nnsight_model, fc_name)
                        clean_logits_proxy = fc_module.output.save()
                        break
                    except AttributeError:
                        continue

                if clean_logits_proxy is None:
                    raise AttributeError(
                        "No final fully connected layer found in model. "
                        "Ensure the model has a final fully connected layer."
                    )

            clean_logits = clean_logits_proxy.detach().cpu()
            clean_label = torch.argmax(clean_logits, dim=1).item()

            # Step 2: Adversarial run - get adversarial activations and label
            with nnsight_model.trace(adv_img):
                # Get adversarial logits
                adv_logits_proxy = None
                for fc_name in ["fc", "linear", "classifier"]:
                    try:
                        fc_module = get_nested_module(nnsight_model, fc_name)
                        adv_logits_proxy = fc_module.output.save()
                        break
                    except AttributeError:
                        continue

                if adv_logits_proxy is None:
                    raise AttributeError(
                        "No final fully connected layer found in model. "
                        "Ensure the model has a final fully connected layer."
                    )

                # Save adversarial activations for layers we'll patch
                layers_to_extract = (
                    LAYER_NAMES if patch_mode == "layer" else [target_layer]
                )
                adv_activations = {}

                for name in layers_to_extract:
                    module = get_nested_module(nnsight_model, name)
                    adv_activations[name] = module.output.save()

            # Extract adversarial values outside trace
            adv_logits = adv_logits_proxy.detach().cpu()
            adv_label = torch.argmax(adv_logits, dim=1).item()

            # Extract saved adversarial activations
            for name in layers_to_extract:
                adv_activations[name] = adv_activations[name].detach()

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

            # Step 3: Activation Patching - patch each layer and measure effect
            for name in LAYER_NAMES:
                module_to_patch = get_nested_module(nnsight_model, name)

                # Run clean image with patched activations
                with nnsight_model.trace(clean_img):
                    if patch_mode == "layer":
                        # Full layer patching - directly assign the saved adversarial activation
                        module_to_patch.output = adv_activations[name]

                    elif patch_mode == "channel" and channel_indices is not None:
                        # Channel-wise patching - patch only specified channels
                        # Direct assignment to specific indices, following the LLM example pattern
                        if (
                            len(adv_activations[name].shape) == 4
                        ):  # Conv layer: [B, C, H, W]
                            module_to_patch.output[:, channel_indices, :, :] = (
                                adv_activations[name][:, channel_indices, :, :]
                            )
                        elif (
                            len(adv_activations[name].shape) == 3
                        ):  # BatchNorm or similar: [B, C, L]
                            module_to_patch.output[:, channel_indices, :] = (
                                adv_activations[name][:, channel_indices, :]
                            )
                        elif len(adv_activations[name].shape) == 2:  # FC layer: [B, C]
                            module_to_patch.output[:, channel_indices] = (
                                adv_activations[name][:, channel_indices]
                            )
                        else:
                            raise ValueError(
                                f"Unexpected tensor shape for layer {name}: {adv_activations[name].shape}"
                            )

                    # Get patched logits
                    patched_logits_proxy = None
                    for fc_name in ["fc", "linear", "classifier"]:
                        try:
                            fc_module = get_nested_module(nnsight_model, fc_name)
                            patched_logits_proxy = fc_module.output.save()
                            break
                        except AttributeError:
                            continue

                    if patched_logits_proxy is None:
                        raise AttributeError(
                            "No final fully connected layer found in model."
                        )

                # Extract patched results outside trace
                patched_logits = patched_logits_proxy.detach().cpu()
                patched_label = torch.argmax(patched_logits, dim=1).item()

                # Compute hit based on hit_rate_type
                if hit_rate_type == "original":
                    # Hit if patching causes class to flip away from original
                    hit = 1 if patched_label != clean_label else 0
                else:  # hit_rate_type == "adversarial"
                    # Hit if patching causes class to match adversarial
                    hit = 1 if patched_label == adv_label else 0

                hit_counts[name].append(hit)

                # Debug output for first few examples
                if i < 3 and patch_mode == "channel":
                    print(f"\nExample {i}, Layer {name}:")
                    print(
                        f"  Clean label: {clean_label}, Adv label: {adv_label}, Patched label: {patched_label}"
                    )
                    print(f"  Hit ({hit_rate_type}): {hit}")
                    print(f"  Clean logits (first 5): {clean_logits[0][:5].tolist()}")
                    print(
                        f"  Patched logits (first 5): {patched_logits[0][:5].tolist()}"
                    )
                    logit_diff = torch.abs(clean_logits - patched_logits).max().item()
                    print(f"  Max logit difference: {logit_diff:.6f}")

    # Compute average hit rate per layer (as percentage)
    avg_hit_rates = {
        name: float(np.mean(hit_counts[name]) * 100) if hit_counts[name] else 0.0
        for name in LAYER_NAMES
    }

    return avg_hit_rates


def compute_channel_importance(
    nnsight_model,
    perturbations_dict,
    layer_name,
    scenario="linf_high",
    num_images=100,
    config=None,
    class_flip=False,  # New parameter
    hit_rate_type="original",  # New parameter for class_flip mode
):
    """
    Compute importance of each channel in a specific layer.

    Args:
        nnsight_model: NNsight wrapped model
        perturbations_dict: Dict of dataloaders
        layer_name: Name of layer to analyze
        scenario: Scenario name
        num_images: Number of images to use
        config: Config object
        class_flip: If True, use class flip analysis instead of confidence drop
        hit_rate_type: "original" or "adversarial" (only used when class_flip=True)

    Returns:
        dict: {channel_idx: average confidence drop} or {channel_idx: hit rate percentage}
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

    channel_results = {}

    # Test channels individually
    analysis_type = "class flip" if class_flip else "confidence drop"
    for ch_idx in tqdm(
        range(num_channels), desc=f"Testing channels in {layer_name} ({analysis_type})"
    ):
        if class_flip:
            # Use class flip analysis
            result = average_class_flip(
                nnsight_model,
                perturbations_dict,
                scenario,
                num_images,
                patch_mode="channel",
                channel_indices=[ch_idx],
                target_layer=layer_name,
                config=config,
                hit_rate_type=hit_rate_type,
            )
        else:
            # Use confidence drop analysis (original behavior)
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

        channel_results[ch_idx] = result[layer_name]

    return channel_results


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


def plot_channel_importance(channel_drops, layer_name, scenario, save_path=None, analysis_type="confidence"):
    """Plot channel importance as a bar chart
    
    Args:
        channel_drops: Dictionary of channel importance values
        layer_name: Name of the layer being analyzed
        scenario: Scenario name (e.g., 'class_flip_analysis')
        save_path: Path to save the plot
        analysis_type: Type of analysis - 'confidence' for confidence drop or 'flip' for class flip rate
    """
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
    
    # Determine labels based on analysis type and scenario
    if scenario == "class_flip_analysis":
        if analysis_type == "flip":
            plt.ylabel("Average Class Flip Rate (%)", fontsize=12)
            plt.title(f"Channel Importance: Class Flip Rate in {layer_name}\n(Averaged over 100 images: Class 0 → Class 1)", fontsize=14)
            cbar_label = "Flip Rate (%)"
        else:
            plt.ylabel("Average Confidence Drop", fontsize=12)
            plt.title(f"Channel Importance: Confidence Drop in {layer_name}\n(Averaged over 100 images: Class 0 → Class 1)", fontsize=14)
            cbar_label = "Confidence Drop"
    else:
        plt.ylabel("Average Confidence Drop", fontsize=12)
        plt.title(f"Channel Importance in {layer_name} ({scenario})", fontsize=14)
        cbar_label = "Importance"

    # Add grid for better readability
    plt.grid(True, axis="y", alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.01)
    cbar.set_label(cbar_label, fontsize=10)

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


def find_closest_image(base_image, target_class, dataloader):
    """
    Find the closest image in the dataset to a given base image that belongs to a specific target class.
    Uses ssim (Structural Similarity Index) for comparison. Returns the closest image, its label, the ssim score, and its index
    """
    from skimage.metrics import structural_similarity as ssim
    import numpy as np

    # Convert base_image to numpy if it's a tensor
    if hasattr(base_image, "cpu"):  # PyTorch tensor
        base_image_np = base_image.cpu().numpy()
    else:  # Already numpy array
        base_image_np = base_image

    # Ensure proper format for SSIM: (H, W, C)
    if base_image_np.shape[0] == 3:  # If channels first (C, H, W)
        base_image_np = base_image_np.transpose(1, 2, 0)

    closest_image = None
    closest_label = None
    closest_ssim = -1.0
    closest_index = -1
    global_idx = 0

    for batch_imgs, batch_labels in dataloader:
        # Iterate through each image in the batch
        for i in range(batch_imgs.size(0)):
            img = batch_imgs[i]

            # Handle labels - check if one-hot encoded or scalar
            if batch_labels[i].numel() > 1:  # One-hot encoded
                label = torch.argmax(batch_labels[i]).item()
            else:  # Scalar labels
                label = batch_labels[i].item()

            if label != target_class:
                global_idx += 1
                continue  # Only consider images of the target class

            # Convert image to numpy for SSIM calculation
            if hasattr(img, "cpu"):  # PyTorch tensor
                img_np = img.cpu().numpy()
            else:  # Already numpy array
                img_np = img

            # Ensure proper format for SSIM: (H, W, C)
            if img_np.shape[0] == 3:  # If channels first (C, H, W)
                img_np = img_np.transpose(1, 2, 0)

            # Calculate SSIM - use channel_axis parameter for newer scikit-image versions
            # Determine data range based on the image values
            data_range = max(
                base_image_np.max() - base_image_np.min(), img_np.max() - img_np.min()
            )

            try:
                current_ssim = ssim(
                    base_image_np, img_np, channel_axis=2, data_range=data_range
                )
            except TypeError:
                # Fallback for older scikit-image versions
                current_ssim = ssim(
                    base_image_np, img_np, multichannel=True, data_range=data_range
                )

            if current_ssim > closest_ssim:
                closest_ssim = current_ssim
                closest_image = img
                closest_label = label
                closest_index = global_idx

            global_idx += 1

    return closest_image, closest_label, closest_ssim, closest_index
