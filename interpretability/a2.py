"""
Adversarial Activation Patching Analysis for ResNet18
Analyzes the impact of adversarial perturbations through layer-wise and channel-wise activation patching.
"""

import torch
import os
import numpy as np
from nnsight import NNsight

# Import utilities from the utils module
from utils import (
    Config,
    load_model,
    load_all_perturbations,
    compute_channel_importance,
    plot_stacked_confidence_drop_heatmaps,
    plot_channel_importance,
    plot_channel_importance_comparison,
)


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
