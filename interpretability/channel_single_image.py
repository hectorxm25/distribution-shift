# script to perform channel-wise patching and analysis on two images
# to evaluate the channel importance in every layer of the ResNet18 model
# uses two-slope normalization for better visualization: last FC layer confidence drop is around 15, previous layers are < 1

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm
from nnsight import NNsight

# Import utilities from the utils module
from utils import (
    Config,
    load_model,
    load_baseline_dataloader,
    get_layer_names,
    compute_channel_importance,
    plot_channel_importance,
)

if __name__ == "__main__":
    print("Loading model and datasets...")

    # Initialize configuration
    config = Config()

    # Load model and wrap with NNsight
    model = load_model(config)
    nnsight_model = NNsight(model)

    # Load dataloader dic containing only two images: the first image is the baseline and the second image is the counterfactual
    dataloaders = load_baseline_dataloader(config)
    print(f"Loaded datasets: {list(dataloaders.keys())}")

    # obtain model layer names
    layer_names = get_layer_names()
    print(f"Model layer names: {layer_names}")

    # Store channel importance data for all layers
    all_layer_channel_importance = {}

    # perform channel-wise patching for every layer
    for layer_name in layer_names:
        # Compute channel-wise importance for this layer
        print(f"Computing channel-wise importance for {layer_name}...")
        channel_importance = compute_channel_importance(
            nnsight_model,
            dataloaders,
            layer_name,
            scenario="counterfactual",
            num_images=2,
            config=config,
        )

        # Store the channel importance data
        all_layer_channel_importance[layer_name] = channel_importance
        print(
            f"Computed importance for {len(channel_importance)} channels in {layer_name}"
        )

    # Create stacked heatmaps for all layers
    print("\nCreating stacked heatmaps for all layers...")

    # Create figure with subplots for each layer
    num_layers = len(layer_names)
    fig, axes = plt.subplots(
        nrows=num_layers, ncols=1, figsize=(15, 3 * num_layers), sharex=False
    )

    # Ensure axes is always a list for consistent indexing
    if num_layers == 1:
        axes = [axes]

    # IMPROVED COLOR SCALING: Two-slope normalization optimized for your data range
    # Early layers < 1, linear layer ~15

    all_values = []
    for layer_data in all_layer_channel_importance.values():
        all_values.extend(layer_data.values())

    if all_values:
        vmin = min(all_values)
        vmax = max(all_values)

        # Two-slope normalization with center at 1.0
        # This gives linear scaling for values < 1, compressed scaling for values > 1
        norm = TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)

        # Alternative center points you could try:
        # norm = TwoSlopeNorm(vcenter=0.5, vmin=vmin, vmax=vmax)  # Center at 0.5
        # norm = TwoSlopeNorm(vcenter=2.0, vmin=vmin, vmax=vmax)   # Center at 2.0

    else:
        vmin, vmax = 0, 1
        norm = None

    # Define colormap - using a more perceptually uniform colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        "blue_purple_red", ["blue", "purple", "red"]
    )
    # Alternative: use a perceptually uniform colormap
    # custom_cmap = plt.cm.viridis  # or plt.cm.plasma, plt.cm.inferno

    # Create heatmap for each layer
    for i, layer_name in enumerate(layer_names):
        ax = axes[i]
        channel_data = all_layer_channel_importance[layer_name]

        if not channel_data:
            ax.text(
                0.5,
                0.5,
                f"No data for {layer_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(layer_name)
            continue

        # Prepare data for heatmap
        channels = sorted(channel_data.keys())
        values = [channel_data[ch] for ch in channels]

        # Create heatmap data as a 2D array (1 row, multiple columns)
        heatmap_data = np.array([values])

        # Create the heatmap with improved normalization
        if norm is not None:
            im = ax.imshow(heatmap_data, cmap=custom_cmap, norm=norm, aspect="auto")
        else:
            im = ax.imshow(
                heatmap_data, cmap=custom_cmap, vmin=vmin, vmax=vmax, aspect="auto"
            )

        # Set labels and title with value range info
        value_range = f"[{min(values):.2e}, {max(values):.2e}]"
        ax.set_title(
            f"{layer_name} (Channels: {len(channels)}) {value_range}",
            fontsize=10,
            pad=5,
        )
        ax.set_ylabel("Layer", fontsize=8)

        # Set x-axis ticks to show channel indices
        if len(channels) <= 20:
            ax.set_xticks(range(len(channels)))
            ax.set_xticklabels(channels, fontsize=6)
        else:
            # Show every 10th channel for readability
            tick_indices = range(0, len(channels), max(1, len(channels) // 10))
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([channels[idx] for idx in tick_indices], fontsize=6)

        ax.set_yticks([])  # Remove y-axis ticks since we only have one row

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add a single colorbar for the entire figure
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)

    # Update colorbar label based on normalization used
    if isinstance(norm, TwoSlopeNorm):
        cbar.set_label(
            "Channel Importance (Two-Slope Scale, center=1.0)",
            rotation=270,
            labelpad=15,
        )
    elif isinstance(norm, PowerNorm):
        cbar.set_label(
            "Channel Importance (Square Root Scale)", rotation=270, labelpad=15
        )
    else:
        cbar.set_label(
            "Channel Importance (Confidence Drop)", rotation=270, labelpad=15
        )

    # Set overall title and labels
    fig.suptitle("Channel Importance Heatmaps for All Layers", fontsize=16, y=0.95)
    fig.text(0.5, 0.02, "Channel Index", ha="center", fontsize=12)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08, right=0.9)

    # Save the stacked heatmap
    save_path = os.path.join(
        config.results_dir, "stacked_channel_importance_heatmaps.png"
    )
    os.makedirs(config.results_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved stacked heatmaps to {save_path}")

    # BONUS: Create a separate figure with per-layer normalization for comparison
    print("\nCreating per-layer normalized heatmaps...")
    fig2, axes2 = plt.subplots(
        nrows=num_layers, ncols=1, figsize=(15, 3 * num_layers), sharex=False
    )

    if num_layers == 1:
        axes2 = [axes2]

    for i, layer_name in enumerate(layer_names):
        ax = axes2[i]
        channel_data = all_layer_channel_importance[layer_name]

        if not channel_data:
            continue

        channels = sorted(channel_data.keys())
        values = [channel_data[ch] for ch in channels]
        heatmap_data = np.array([values])

        # Use per-layer normalization (each layer gets its own color scale)
        layer_vmin, layer_vmax = min(values), max(values)
        if layer_vmin == layer_vmax:
            layer_vmin -= 0.1
            layer_vmax += 0.1

        im2 = ax.imshow(
            heatmap_data,
            cmap=custom_cmap,
            vmin=layer_vmin,
            vmax=layer_vmax,
            aspect="auto",
        )

        # Add individual colorbar for each layer
        divider = plt.gca().inset_axes([1.02, 0, 0.02, 1])
        plt.colorbar(im2, cax=divider)

        ax.set_title(f"{layer_name} (Per-layer normalized)", fontsize=10, pad=5)
        ax.set_ylabel("Layer", fontsize=8)

        # Set x-axis ticks
        if len(channels) <= 20:
            ax.set_xticks(range(len(channels)))
            ax.set_xticklabels(channels, fontsize=6)
        else:
            tick_indices = range(0, len(channels), max(1, len(channels) // 10))
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([channels[idx] for idx in tick_indices], fontsize=6)

        ax.set_yticks([])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig2.suptitle(
        "Per-Layer Normalized Channel Importance Heatmaps", fontsize=16, y=0.95
    )
    fig2.text(0.5, 0.02, "Channel Index", ha="center", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08, right=0.85)

    # Save per-layer normalized version
    save_path2 = os.path.join(
        config.results_dir, "per_layer_normalized_channel_importance_heatmaps.png"
    )
    plt.savefig(save_path2, dpi=300, bbox_inches="tight")
    print(f"Saved per-layer normalized heatmaps to {save_path2}")

    # Also save individual plots for each layer
    print("\nSaving individual heatmaps for each layer...")
    for layer_name in layer_names:
        channel_data = all_layer_channel_importance[layer_name]
        if channel_data:
            individual_save_path = os.path.join(
                config.results_dir, f"channel_importance_{layer_name}.png"
            )
            plot_channel_importance(
                channel_data,
                layer_name,
                "counterfactual",
                save_path=individual_save_path,
            )

    plt.show()
    print("Analysis complete!")
