# Adversarial Perturbation Grid Search Experiment

This module provides a comprehensive grid search experiment for analyzing the universality and perceptibility of adversarial perturbations across different epsilon and step size parameters for PGD L-infinity attacks.

## Overview

The experiment measures two key metrics for adversarial perturbations:
1. **Perceptibility (SSIM)**: How visually similar the perturbed images are to the original images
2. **Universality (Label Flip Rate)**: How effectively a perturbation mask transfers across different images

## Experiment Design

### Grid Search Parameters
- **Epsilon range**: 0.031 to 5×0.031 (0.155), evenly distributed into 5 values
- **Step size range**: 0.01 to 0.10, evenly distributed into 5 values
- **Attack configuration**: PGD L-infinity norm, 10 iterations, no random start
- **Total combinations**: 5 × 5 = 25 parameter pairs

### For Each Parameter Combination:
1. **Source Image Selection**: Randomly sample one image from the test loader
2. **Mask Generation**: Generate adversarial example using PGD attack with current epsilon/step_size
3. **Mask Extraction**: Extract pure perturbation by subtracting original from adversarial image
4. **Universality Testing**: Apply the mask to 100 random images from the loader
5. **Metric Calculation**:
   - Calculate SSIM between each masked image and its original (unmasked) version
   - Check if the model misclassifies each masked image (label flip)
   - Compute average SSIM and label flip rate percentage

## Usage

### Basic Usage

```python
from grid_search_experiment import run_adversarial_grid_search_experiment
import utils

# Load dataset
dataset, train_loader, test_loader = utils.load_dataset("/path/to/datasets")

# Run experiment
results = run_adversarial_grid_search_experiment(
    model_path="/path/to/model.pt",
    data_path="/path/to/datasets", 
    loader=test_loader,
    save_path="/path/to/results",
    num_random_images=100,
    verbose=True
)
```

### Testing the Setup

Use the provided test script to verify everything works:

```bash
cd distribution-shift/adversarial
python test_grid_search.py
```

## Results Structure

The function returns a comprehensive results dictionary:

```python
{
    'experiment_config': {
        'model_path': str,
        'data_path': str, 
        'save_path': str,
        'num_random_images': int,
        'epsilons': list,
        'step_sizes': list,
        'attack_config': dict
    },
    'grid_results': {
        'eps_0.0310_step_0.0100': {
            'epsilon': float,
            'step_size': float,
            'avg_ssim': float,           # Average SSIM across all random images
            'ssim_std': float,           # Standard deviation of SSIM scores
            'label_flip_rate': float,    # Percentage of images misclassified
            'label_flips': int,          # Number of images misclassified
            'total_tested': int,         # Total images tested
            'attack_success': bool,      # Whether attack succeeded on source image
            'original_image_info': dict, # Info about source image
            'mask_stats': dict          # Statistics about the mask
        },
        # ... results for all 25 parameter combinations
    }
}
```

## Generated Files

The experiment automatically generates:

1. **`grid_search_results.pt`**: Complete results dictionary
2. **`grid_search_summary.png`**: Visualization heatmaps showing:
   - Average SSIM scores across parameter combinations
   - Label flip rates across parameter combinations  
   - Attack success rates
   - SSIM vs Universality trade-off scatter plot
3. **`summary_statistics.txt`**: Detailed text summary with:
   - Overall statistics
   - Best/worst performing configurations
   - Detailed results for each parameter combination

## Key Implementation Details

### Robust Error Handling
- Handles failed adversarial attacks gracefully
- Continues experiment even if individual images fail
- Provides detailed error reporting

### Memory Management
- Processes images in batches to avoid GPU memory issues
- Efficient random sampling from large datasets
- Proper cleanup of intermediate tensors

### Accurate Metrics
- SSIM calculated using the existing `calculate_ssim` function from `visualization_experiments.py`
- Label flip detection compares original vs masked predictions
- Proper statistical aggregation with standard deviations

### Comprehensive Logging
- Detailed progress reporting when `verbose=True`
- Attack success tracking
- Mask statistics logging
- Timing and completion status

## Critical Implementation Notes

### SSIM Baseline Correctness
The SSIM is calculated comparing the **masked random image** to the **original random image** (not the source image used to generate the mask). This measures the perceptibility of the mask when applied to different images.

### Universality Measurement
Label flip rate measures how often the mask causes misclassification when transferred to random images. This directly quantifies the universality of the adversarial perturbation.

### Grid Search Completeness
The experiment tests all 25 combinations systematically, ensuring comprehensive coverage of the parameter space.

### Statistical Validity
Each parameter combination is tested on 100 random images (configurable) to ensure statistically meaningful results.

## Customization Options

- **`num_random_images`**: Adjust the number of random images tested per combination (default: 100)
- **`verbose`**: Control the amount of progress output (default: True)
- Parameter ranges can be modified by editing the epsilon and step_size arrays in the function

## Performance Considerations

- **Runtime**: Approximately 2-5 minutes per parameter combination (25 total = ~1-2 hours)
- **Memory**: Processes images in batches of 32 to manage GPU memory
- **Storage**: Results files are typically 1-10 MB depending on configuration

## Troubleshooting

1. **Path Issues**: Ensure model and dataset paths are correct
2. **Memory Errors**: Reduce batch size or num_random_images
3. **Import Errors**: Ensure all dependencies are available in Python path
4. **CUDA Errors**: Verify GPU availability and CUDA setup

Run the test script first to identify and resolve any setup issues before running the full experiment. 