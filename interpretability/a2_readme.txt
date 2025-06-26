Adversarial Activation Patching Analysis - README
=====================================================

This document explains the functions and workflow of a2.py, which performs adversarial activation patching analysis on ResNet18 models to understand how adversarial perturbations affect different layers and channels.

OVERVIEW
========
The script analyzes the impact of adversarial perturbations through layer-wise and channel-wise activation patching. It uses the NNsight library to intercept and modify neural network activations, allowing researchers to understand which parts of the network are most affected by adversarial attacks.

MAIN COMPONENTS
===============

1. Configuration Class
----------------------
- Config: Dataclass that stores all configuration parameters including:
  * Data paths (datasets, model, perturbations)
  * Device settings (CUDA/CPU)
  * Batch size and results directory
  * Normalization parameters for CIFAR-10
  * Dataset mapping between folder names and experiment keys

2. Data Loading Functions
-------------------------
- load_model(config): Loads pre-trained ResNet18 model from checkpoint
- load_datasets(config): Loads baseline CIFAR-10 dataset with standard transforms
- load_perturbation_data(pt_path, config): Loads adversarial perturbation data from .pt files
- load_all_perturbations(config): Loads all perturbation datasets and maps them to experiment keys

3. Core Analysis Functions
--------------------------
- compute_average_confidence_drop(): Main function that performs activation patching
  * Supports both layer-wise and channel-wise patching modes
  * Compares clean vs adversarial activations
  * Measures confidence drop on correct class predictions
  * Handles multiple scenarios (linf_low, linf_high, l2_low, l2_high, etc.)

- compute_channel_importance(): Analyzes importance of individual channels within a layer
  * Tests each channel individually by patching it with adversarial activations
  * Returns importance scores based on confidence drop

4. Visualization Functions
--------------------------
- plot_stacked_confidence_drop_heatmaps(): Creates heatmaps showing layer-wise confidence drops
- plot_channel_importance(): Bar chart showing individual channel importance
- plot_channel_importance_comparison(): Compares channel importance across scenarios

5. Utility Functions
--------------------
- get_nested_module(): Navigates model hierarchy to access specific modules
- Standard CIFAR-10 transforms with data augmentation

WORKFLOW
========

1. Initialization
-----------------
- Load configuration parameters
- Initialize ResNet18 model and wrap with NNsight
- Load baseline CIFAR-10 dataset and adversarial perturbation datasets

2. Layer-wise Analysis (Optional)
---------------------------------
- For each perturbation scenario:
  * Compare clean baseline vs adversarial images
  * Patch entire layer activations from adversarial to clean images
  * Measure confidence drop on correct predictions
  * Generate heatmaps showing which layers are most affected

3. Channel-wise Analysis
------------------------
- Select target layer for detailed analysis
- For each channel in the layer:
  * Patch only that specific channel with adversarial activations
  * Measure resulting confidence drop
  * Create importance ranking of channels
- Generate visualizations showing channel-wise importance

4. Comparative Analysis (Optional)
----------------------------------
- Compare channel importance across different attack types
- Identify consistently important channels across scenarios
- Generate comparison plots and summary statistics

EXPERIMENT SCENARIOS
====================
The script supports multiple perturbation scenarios:
- linf_low: L-infinity norm, low epsilon
- linf_high: L-infinity norm, high epsilon  
- l2_low: L2 norm, low epsilon
- l2_high: L2 norm, high epsilon
- linf_low_vs_high: Compare low vs high epsilon L-infinity
- l2_low_vs_high: Compare low vs high epsilon L2

ACTIVATION PATCHING METHODOLOGY
===============================
1. Forward pass on clean image to get baseline activations and predictions
2. Forward pass on adversarial image to get adversarial activations
3. For each layer/channel:
   - Run clean image through network
   - Replace target layer/channel activations with adversarial ones
   - Measure change in confidence for correct class
4. Higher confidence drops indicate more important layers/channels

OUTPUT FILES
============
The script generates several visualization files in the results/ directory:
- patching_layer_heatmap.png: Layer-wise confidence drop heatmaps
- patching_channel_heatmap.png: Channel importance within target layer
- patching_channel_comparison.png: Channel importance across scenarios

TECHNICAL DETAILS
==================

Dependencies:
- torch, torchvision: PyTorch framework
- nnsight: For activation intervention
- matplotlib: Plotting and visualization
- numpy: Numerical computations
- tqdm: Progress bars

Model Architecture:
- ResNet18 trained on CIFAR-10
- 10 output classes
- Standard CIFAR-10 preprocessing

Patching Mechanism:
- Uses NNsight's trace context to intercept activations
- Supports different tensor shapes (Conv: [B,C,H,W], FC: [B,C])
- Preserves gradient computation for end-to-end analysis

USAGE EXAMPLE
=============
1. Ensure all dependencies are installed
2. Set correct paths in Config class
3. Run: python a2.py
4. Check results/ directory for generated plots

The script is designed to be modular - you can comment/uncomment different analysis sections based on your research needs.

INTERPRETATION
==============
- Higher confidence drops indicate layers/channels more sensitive to adversarial perturbations
- Consistent patterns across attack types suggest fundamental vulnerabilities
- Channel-wise analysis can reveal which feature detectors are most affected
- Results can inform defense strategies and model interpretability research

NOTES
=====
- Script uses relatively small sample sizes for demonstration (adjust num_images for production)
- Memory usage scales with model size and number of images processed
- Results may vary based on specific model training and adversarial attack parameters
- Channel analysis can be computationally intensive for layers with many channels
