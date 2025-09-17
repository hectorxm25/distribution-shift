# script to perform channel-wise patching and analysis on 100 images from each class
# to evaluate the channel importance in every layer of the ResNet18 model
# analyzes class flip rates between class 0 (baseline) and class 1 (counterfactual)
# uses two-slope normalization for better visualization: last FC layer confidence drop is around 15, previous layers are < 1

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm
from nnsight import NNsight
import argparse

# Import utilities from the utils module
from utils import (
    Config,
    load_model,
    load_baseline_dataloader,
    load_class_specific_dataloader,
    get_layer_names,
    compute_channel_importance,
    plot_channel_importance,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform channel-wise patching and analysis on two images"
    )

    parser.add_argument(
        "--class_flip",
        type=str,
        required=True,
        choices=["false", "original", "adversarial"],
        help=(
            "Control channel importance computation and visualization behavior:\n"
            "- 'false': Use class_flip=False, no hit_rate_type, apply 2-slope normalization\n"
            "- 'original': Use class_flip=True with hit_rate_type='original', no normalization\n"
            "- 'adversarial': Use class_flip=True with hit_rate_type='adversarial', no normalization\n"
            "Analysis performed on 100 images from class 0 vs 100 images from class 1"
        ),
    )

    return parser.parse_args()


def compute_importance_with_args(
    nnsight_model, dataloaders, layer_name, config, class_flip_arg
):
    """
    Compute channel importance based on class_flip argument value.

    Args:
        nnsight_model: NNsight wrapped model
        dataloaders: Data loaders dictionary
        layer_name: Name of the layer to analyze
        config: Configuration object
        class_flip_arg: The class_flip argument value ("false", "original", or "adversarial")

    Returns:
        Channel importance dictionary
    """
    print(f"[DEBUG] Starting compute_importance_with_args for layer: {layer_name}")
    print(f"[DEBUG] Class flip argument: {class_flip_arg}")
    
    if class_flip_arg == "false":
        # class_flip="false": Use class_flip=False, no hit_rate_type parameter
        print(f"[EXEC] Computing with class_flip=False (no hit_rate_type) for {layer_name}...")
        result = compute_channel_importance(
            nnsight_model,
            dataloaders,
            layer_name,
            scenario="class_flip_analysis",
            num_images=100,
            config=config,
            class_flip=False,
            # Note: hit_rate_type is NOT passed when class_flip="false"
        )
        print(f"[DEBUG] Completed confidence drop analysis for {layer_name}")
        return result
    else:
        # class_flip="original" or "adversarial": Use class_flip=True with corresponding hit_rate_type
        print(
            f"[EXEC] Computing with class_flip=True, hit_rate_type='{class_flip_arg}' for {layer_name}..."
        )
        result = compute_channel_importance(
            nnsight_model,
            dataloaders,
            layer_name,
            scenario="class_flip_analysis",
            num_images=100,
            config=config,
            class_flip=True,
            hit_rate_type=class_flip_arg,  # "original" or "adversarial"
        )
        print(f"[DEBUG] Completed class flip analysis for {layer_name}")
        return result
def create_heatmaps(all_layer_channel_importance, layer_names, config, class_flip_arg):
    """
    Create heatmaps with normalization based on class_flip argument.

    Args:
        all_layer_channel_importance: Dictionary of channel importance data
        layer_names: List of layer names
        config: Configuration object
        class_flip_arg: The class_flip argument value
    """
    print(f"\n[DEBUG] Entering create_heatmaps function")
    print(f"[DEBUG] Class flip argument: {class_flip_arg}")
    print(f"[DEBUG] Number of layers: {len(layer_names)}")
    print(f"[EXEC] Creating heatmaps with class_flip='{class_flip_arg}'...")

    # Create figure with subplots for each layer
    num_layers = len(layer_names)
    print(f"[DEBUG] Creating figure with {num_layers} subplots")
    fig, axes = plt.subplots(
        nrows=num_layers, ncols=1, figsize=(15, 3 * num_layers), sharex=False
    )

    # Ensure axes is always a list for consistent indexing
    if num_layers == 1:
        axes = [axes]

    # Determine normalization strategy based on class_flip argument
    print(f"[DEBUG] Determining normalization strategy for class_flip_arg: {class_flip_arg}")
    if class_flip_arg == "false":
        use_percentage_scale = False
        # Apply 2-slope normalization for class_flip="false"
        print("[EXEC] Applying 2-slope normalization (center=1.0)...")

        all_values = []
        for layer_data in all_layer_channel_importance.values():
            all_values.extend(layer_data.values())
        
        print(f"[DEBUG] Collected {len(all_values)} values across all layers")

        if all_values:
            vmin = min(all_values)
            vmax = max(all_values)
            print(f"[DEBUG] Value range: {vmin:.3f} to {vmax:.3f}")
            # Two-slope normalization with center at 1.0
            norm = TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)
            colorbar_label = "Channel Importance (Two-Slope Scale, center=1.0)"
        else:
            print("[WARNING] No values found, using default normalization")
            vmin, vmax = 0, 1
            norm = None
            colorbar_label = "Channel Importance"

        use_global_norm = True
        print("[DEBUG] Using global normalization")

    else:
        use_percentage_scale = True
        # No normalization for class_flip="original" or "adversarial"
        print("[EXEC] No normalization applied - plotting values as-is...")
        norm = None
        colorbar_label = "Channel Importance (Raw Values)"
        use_global_norm = False
        print("[DEBUG] Using individual layer normalization")

    # Define colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        "blue_purple_red", ["blue", "purple", "red"]
    )

    # Create heatmap for each layer
    print(f"[DEBUG] Creating individual heatmaps for {len(layer_names)} layers")
    for i, layer_name in enumerate(layer_names):
        print(f"[DEBUG] Processing layer {i+1}/{len(layer_names)}: {layer_name}")
        ax = axes[i]
        channel_data = all_layer_channel_importance[layer_name]

        if not channel_data:
            print(f"[WARNING] No data found for layer {layer_name}")
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
        print(f"[DEBUG] Layer {layer_name}: {len(channels)} channels, value range: {min(values):.3f} to {max(values):.3f}")

        # Create heatmap data as a 2D array (1 row, multiple columns)
        heatmap_data = np.array([values])

        # Create the heatmap with appropriate normalization
        if use_global_norm and norm is not None:
            # Use global normalization (for class_flip="false")
            im = ax.imshow(heatmap_data, cmap=custom_cmap, norm=norm, aspect="auto")
        elif not use_global_norm:
            # No normalization - use raw value ranges (for class_flip="original"/"adversarial")
            layer_vmin, layer_vmax = min(values), max(values)
            if layer_vmin == layer_vmax:
                layer_vmin -= 0.1
                layer_vmax += 0.1
            im = ax.imshow(
                heatmap_data,
                cmap=custom_cmap,
                vmin=layer_vmin,
                vmax=layer_vmax,
                aspect="auto",
            )
        else:
            # Fallback
            im = ax.imshow(heatmap_data, cmap=custom_cmap, aspect="auto")

        # Set labels and title with value range info
        value_range = f"[{min(values):.2f}, {max(values):.2f}]"
        if class_flip_arg in ["original", "adversarial"]:
            # For class flip modes, show percentage
            ax.set_title(
                f"{layer_name} (Channels: {len(channels)}) Avg Flip Rate: {value_range}%",
                fontsize=10,
                pad=5,
            )
        else:
            # For confidence drop mode
            ax.set_title(
                f"{layer_name} (Channels: {len(channels)}) Avg Confidence Drop: {value_range}",
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

    # Add colorbar
    if use_global_norm:
        # Single colorbar for global normalization
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(colorbar_label, rotation=270, labelpad=15)
    else:
        # Individual colorbars for each layer when no global normalization
        for i, layer_name in enumerate(layer_names):
            if (
                layer_name in all_layer_channel_importance
                and all_layer_channel_importance[layer_name]
            ):
                ax = axes[i]
                # Get the image from this specific axis
                im_layer = ax.images[0] if ax.images else None
                if im_layer:
                    divider = ax.inset_axes([1.02, 0, 0.02, 1])
                    cbar_layer = plt.colorbar(im_layer, cax=divider)
                    if class_flip_arg in ["original", "adversarial"]:
                        label = "Flip Rate %"
                    else:
                        label = "Conf. Drop"
                    cbar_layer.set_label(
                        label, rotation=270, labelpad=10, fontsize=8
                    )

    # Set overall title and labels
    if class_flip_arg in ["original", "adversarial"]:
        analysis_type = f"Class Flip Analysis (100 images each: Class 0→Class 1, mode='{class_flip_arg}')"
    else:
        analysis_type = f"Confidence Drop Analysis (100 images each: Class 0→Class 1, mode='{class_flip_arg}')"
    
    fig.suptitle(
        f"Channel Importance: {analysis_type}",
        fontsize=16,
        y=0.95,
    )
    fig.text(0.5, 0.02, "Channel Index", ha="center", fontsize=12)

    # Adjust layout - always use global layout since we always have single colorbar
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08, right=0.9)

    # Save the heatmap
    if use_percentage_scale:
        save_path = os.path.join(
            config.results_dir,
            f"stacked_channel_importance_heatmaps_{class_flip_arg}.png",
        )
    else:
        save_path = os.path.join(
            config.results_dir, f"channel_importance_heatmaps_{class_flip_arg}.png"
        )
    os.makedirs(config.results_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {'stacked ' if use_percentage_scale else ''}heatmaps to {save_path}")

    return fig


if __name__ == "__main__":
    print("[DEBUG] =============== SCRIPT START ===============")
    # Parse command line arguments
    args = parse_arguments()

    print(f"[EXEC] Running analysis with --class_flip='{args.class_flip}'")
    print("[EXEC] Loading model and datasets...")

    # Initialize configuration
    print("[DEBUG] Initializing configuration...")
    config = Config()
    print(f"[DEBUG] Config created - device: {config.device}, results_dir: {config.results_dir}")

    # Load model and wrap with NNsight
    print("[DEBUG] Loading model...")
    model = load_model(config)
    print("[DEBUG] Wrapping model with NNsight...")
    nnsight_model = NNsight(model)
    print("[DEBUG] Model loaded and wrapped successfully")

    # Load dataloader dic containing 100 images from class 0 (baseline) and 100 images from class 1 (counterfactual)
    print("[DEBUG] Loading class-specific dataloaders...")
    dataloaders = load_class_specific_dataloader(config, num_images_per_class=100)
    print(f"[EXEC] Loaded datasets: {list(dataloaders.keys())}")

    # obtain model layer names
    print("[DEBUG] Getting model layer names...")
    layer_names = get_layer_names()
    print(f"[EXEC] Model layer names: {layer_names}")
    print(f"[DEBUG] Found {len(layer_names)} layers to analyze")

    # Store channel importance data for all layers
    print("[DEBUG] Initializing storage for channel importance data...")
    all_layer_channel_importance = {}

    # perform channel-wise patching for every layer
    print(f"\n[DEBUG] =============== STARTING LAYER ANALYSIS ===============")
    print(f"[EXEC] Performing channel analysis across 100 images from each class (Class 0 vs Class 1)...")
    
    for layer_idx, layer_name in enumerate(layer_names):
        print(f"\n[DEBUG] ===== LAYER {layer_idx + 1}/{len(layer_names)}: {layer_name} =====")
        print(f"[EXEC] Analyzing layer: {layer_name}")
        
        # Compute channel-wise importance for this layer based on class_flip argument
        print(f"[DEBUG] Calling compute_importance_with_args for {layer_name}")
        channel_importance = compute_importance_with_args(
            nnsight_model, dataloaders, layer_name, config, args.class_flip
        )
        print(f"[DEBUG] Completed channel importance computation for {layer_name}")

        # Store the channel importance data
        all_layer_channel_importance[layer_name] = channel_importance
        print(f"[DEBUG] Stored results for {layer_name}")
        
        # Show summary statistics
        if channel_importance:
            values = list(channel_importance.values())
            analysis_type_str = "flip rate %" if args.class_flip in ["original", "adversarial"] else "confidence drop"
            print(f"[EXEC] Computed {analysis_type_str} for {len(channel_importance)} channels")
            print(f"[EXEC] Range: {min(values):.3f} to {max(values):.3f}")
            print(f"[EXEC] Mean: {np.mean(values):.3f}, Std: {np.std(values):.3f}")
        else:
            print(f"[WARNING] No data computed for {layer_name}")
    
    print(f"\n[DEBUG] =============== COMPLETED ALL LAYER ANALYSIS ===============")

    # Create heatmaps with appropriate normalization based on class_flip argument
    print(f"\n[DEBUG] =============== STARTING HEATMAP CREATION ===============")
    fig = create_heatmaps(
        all_layer_channel_importance, layer_names, config, args.class_flip
    )
    print(f"[DEBUG] Completed main heatmap creation")

    # Also save individual plots for each layer
    print(f"\n[DEBUG] =============== SAVING INDIVIDUAL PLOTS ===============")
    print("[EXEC] Saving individual heatmaps for each layer...")
    for layer_idx, layer_name in enumerate(layer_names):
        print(f"[DEBUG] Creating individual plot for layer {layer_idx + 1}/{len(layer_names)}: {layer_name}")
        channel_data = all_layer_channel_importance[layer_name]
        if channel_data:
            individual_save_path = os.path.join(
                config.results_dir,
                f"channel_importance_{layer_name}_{args.class_flip}.png",
            )
            # Determine analysis type for proper labeling
            analysis_type = "flip" if args.class_flip in ["original", "adversarial"] else "confidence"
            print(f"[DEBUG] Plotting with analysis_type='{analysis_type}' for {layer_name}")
            plot_channel_importance(
                channel_data,
                layer_name,
                "class_flip_analysis",
                save_path=individual_save_path,
                analysis_type=analysis_type,
            )
            print(f"[DEBUG] Saved individual plot for {layer_name}")
        else:
            print(f"[WARNING] No data to plot for {layer_name}")

    print(f"[DEBUG] Showing plots...")
    plt.show()
    
    # Print final summary
    print(f"\n[DEBUG] =============== FINAL SUMMARY ===============")
    analysis_type_str = "Class flip rate" if args.class_flip in ["original", "adversarial"] else "Confidence drop"
    print(f"[EXEC] === Analysis Summary ===")
    print(f"[EXEC] Analysis type: {analysis_type_str} (mode: {args.class_flip})")
    print(f"[EXEC] Images analyzed: 100 from Class 0, 100 from Class 1")
    print(f"[EXEC] Layers analyzed: {len(layer_names)}")
    total_channels = sum(len(data) for data in all_layer_channel_importance.values())
    print(f"[EXEC] Total channels tested: {total_channels}")
    print(f"[EXEC] Results saved to: {config.results_dir}")
    print(f"[EXEC] Analysis complete!")
    print(f"[DEBUG] =============== SCRIPT END ===============")
