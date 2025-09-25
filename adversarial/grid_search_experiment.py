"""
Grid Search Experiment for Adversarial Perturbation Universality and Perceptibility Analysis

This module performs a comprehensive grid search over epsilon and step size parameters
for PGD L-infinity attacks, measuring both the perceptibility (via SSIM) and universality 
(via label flip rate) of the generated adversarial perturbations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from robustness import model_utils, datasets
from robustness.datasets import CIFAR
import torchvision.utils as vutils

# Import local utilities
import utils
from visualization_experiments import calculate_ssim

def run_adversarial_grid_search_experiment(
    model_path, 
    data_path, 
    loader, 
    save_path, 
    num_random_images=100,
    verbose=True
):
    """
    Performs a comprehensive grid search experiment over epsilon and step size parameters
    for PGD L-infinity attacks, measuring both perceptibility (SSIM) and universality 
    (label flip rate) of adversarial perturbations.
    
    Args:
        model_path (str): Path to the trained model weights (.pt file)
        data_path (str): Path to the dataset directory  
        loader (DataLoader): Test data loader to sample images from
        save_path (str): Directory to save experiment results
        num_random_images (int): Number of random images to test each mask on (default: 100)
        verbose (bool): Whether to print detailed progress information
        
    Returns:
        dict: Comprehensive results dictionary containing:
            - experiment_config: Configuration parameters used
            - grid_results: Results for each (epsilon, step_size) pair containing:
                - avg_ssim: Average SSIM across all random images
                - label_flip_rate: Percentage of images that got misclassified
                - total_tested: Total number of images tested
                - original_image_info: Info about the source image used for mask generation
                - attack_success: Whether the attack succeeded on the original image
    
    Experiment Design:
        - Epsilon range: 0.031 to 5*0.031 (0.031 to 0.155), 5 evenly spaced values
        - Step size range: 0.01 to 0.10, 5 evenly spaced values  
        - Attack: PGD L-infinity with 10 iterations, no random start
        
    For each (epsilon, step_size) pair:
        1. Sample a random image from the loader
        2. Generate adversarial example using PGD attack with specified parameters
        3. Extract the pure perturbation mask (adv_image - original_image)
        4. Apply this mask to 100 random images from the loader
        5. For each masked image:
           - Calculate SSIM compared to the unmasked random image
           - Check if the model misclassifies the masked image (label flip)
        6. Compute average SSIM and label flip rate
    """
    
    if verbose:
        print("="*80)
        print("ADVERSARIAL PERTURBATION UNIVERSALITY AND PERCEPTIBILITY EXPERIMENT")
        print("="*80)
        print(f"Model path: {model_path}")
        print(f"Data path: {data_path}")
        print(f"Save path: {save_path}")
        print(f"Number of random images per experiment: {num_random_images}")
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Define grid search parameters
    # Epsilon: 0.031 to 5*0.031 (0.155), 5 evenly spaced values
    epsilon_min, epsilon_max = 0.031, 5 * 0.031
    epsilons = np.linspace(epsilon_min, epsilon_max, 5)
    
    # Step size: 0.01 to 0.10, 5 evenly spaced values  
    step_size_min, step_size_max = 0.01, 0.10
    step_sizes = np.linspace(step_size_min, step_size_max, 5)
    
    if verbose:
        print(f"\nGrid Search Parameters:")
        print(f"Epsilons: {epsilons}")
        print(f"Step sizes: {step_sizes}")
        print(f"Total combinations: {len(epsilons)} x {len(step_sizes)} = {len(epsilons) * len(step_sizes)}")
    
    # Load model and dataset
    if verbose:
        print(f"\nLoading model and dataset...")
    
    # Load dataset using the utility function
    dataset, train_loader, test_loader = utils.load_dataset(data_path)
    
    # Load model
    model, _ = model_utils.make_and_restore_model(
        arch='resnet18', 
        dataset=dataset, 
        resume_path=model_path
    )
    model.eval()
    model = model.cuda()
    
    if verbose:
        print("Model and dataset loaded successfully")
    
    # Initialize results storage
    experiment_config = {
        'model_path': model_path,
        'data_path': data_path,
        'save_path': save_path,
        'num_random_images': num_random_images,
        'epsilons': epsilons.tolist(),
        'step_sizes': step_sizes.tolist(),
        'attack_config': {
            'constraint': 'inf',
            'iterations': 10,
            'random_start': False
        }
    }
    
    grid_results = {}
    
    # Create a list of batches by iterating through the loader once
    # This avoids the 'in_order' attribute issue
    if verbose:
        print(f"\nLoading all batches from DataLoader...")
    
    loader_list = []
    try:
        for batch_idx, (images, labels) in enumerate(loader):
            loader_list.append((images, labels))
            if verbose and batch_idx % 100 == 0:
                print(f"Loaded {batch_idx + 1} batches...")
    except Exception as e:
        print(f"Error loading batches from DataLoader: {e}")
        print("This might be due to DataLoader compatibility issues.")
        raise e
    
    if verbose:
        print(f"\nStarting grid search experiment...")
        print(f"Loader contains {len(loader_list)} batches")
    
    # Grid search over all parameter combinations
    total_combinations = len(epsilons) * len(step_sizes)
    combination_count = 0
    
    for eps_idx, epsilon in enumerate(epsilons):
        for step_idx, step_size in enumerate(step_sizes):
            combination_count += 1
            
            if verbose:
                print(f"\n" + "-"*60)  
                print(f"Combination {combination_count}/{total_combinations}")
                print(f"Epsilon: {epsilon:.4f}, Step size: {step_size:.4f}")
                print("-"*60)
            
            # Create attack configuration for this combination
            attack_config = {
                'constraint': 'inf',
                'eps': epsilon,
                'step_size': step_size,
                'iterations': 10,
                'random_start': False
            }
            
            # Step 1: Get a random image from the loader for mask generation
            try:
                # Sample a random batch
                random_batch_idx = random.randint(0, len(loader_list) - 1)
                images_batch, labels_batch = loader_list[random_batch_idx]
                
                # Sample a random image from this batch
                random_img_idx = random.randint(0, images_batch.size(0) - 1)
                source_image = images_batch[random_img_idx:random_img_idx+1].cuda()
                source_label = labels_batch[random_img_idx:random_img_idx+1].cuda()
                
                if verbose:
                    print(f"Selected source image with label: {source_label.item()}")
                
            except Exception as e:
                print(f"Error sampling source image: {e}")
                continue
            
            # Step 2: Generate adversarial example and extract mask
            try:
                # Generate adversarial example
                with torch.no_grad():
                    # Get original prediction
                    original_logits, _ = model(source_image)
                    original_pred = torch.argmax(original_logits, dim=1)
                
                # Generate adversarial example
                _, adv_image = model(source_image, source_label, make_adv=True, **attack_config)
                
                # Extract the pure perturbation mask
                mask = adv_image - source_image
                
                # Check if attack was successful on source image
                with torch.no_grad():
                    adv_logits, _ = model(adv_image)
                    adv_pred = torch.argmax(adv_logits, dim=1)
                    attack_success = (original_pred != adv_pred).item()
                
                if verbose:
                    print(f"Original prediction: {original_pred.item()}")
                    print(f"Adversarial prediction: {adv_pred.item()}")
                    print(f"Attack successful: {attack_success}")
                    print(f"Mask statistics - Min: {mask.min().item():.6f}, Max: {mask.max().item():.6f}, Mean: {mask.mean().item():.6f}")
                
            except Exception as e:
                print(f"Error generating adversarial example: {e}")
                continue
            
            # Step 3: Apply mask to random images and measure metrics
            ssim_scores = []
            label_flips = 0
            total_tested = 0
            
            if verbose:
                print(f"Testing mask on {num_random_images} random images...")
            
            # We need to collect enough random images
            random_images_collected = []
            random_labels_collected = []
            
            # Collect random images from multiple batches if needed
            while len(random_images_collected) < num_random_images:
                # Sample a random batch
                random_batch_idx = random.randint(0, len(loader_list) - 1)
                images_batch, labels_batch = loader_list[random_batch_idx]
                
                # Add all images from this batch
                remaining_needed = num_random_images - len(random_images_collected)
                images_to_take = min(images_batch.size(0), remaining_needed)
                
                # Randomly select which images to take from this batch
                indices = torch.randperm(images_batch.size(0))[:images_to_take]
                
                random_images_collected.append(images_batch[indices])
                random_labels_collected.append(labels_batch[indices])
            
            # Concatenate all collected images
            random_images = torch.cat(random_images_collected, dim=0)[:num_random_images].cuda()
            random_labels = torch.cat(random_labels_collected, dim=0)[:num_random_images].cuda()
            
            if verbose:
                print(f"Collected {random_images.size(0)} random images for testing")
            
            # Process images in batches to avoid memory issues
            batch_size = 32  # Process 32 images at a time
            
            for batch_start in range(0, num_random_images, batch_size):
                batch_end = min(batch_start + batch_size, num_random_images)
                batch_images = random_images[batch_start:batch_end]
                batch_labels = random_labels[batch_start:batch_end]
                
                try:
                    # Apply mask to batch
                    masked_images = torch.clamp(batch_images + mask, 0, 1)
                    
                    # Calculate SSIM for each image in batch
                    for i in range(batch_images.size(0)):
                        original_img = batch_images[i:i+1]
                        masked_img = masked_images[i:i+1]
                        
                        try:
                            ssim_score = calculate_ssim(original_img, masked_img)
                            ssim_scores.append(ssim_score)
                        except Exception as e:
                            if verbose:
                                print(f"Error calculating SSIM for image {total_tested + i}: {e}")
                            # Use a default low SSIM score for failed calculations
                            ssim_scores.append(0.0)
                    
                    # Check label flips for batch
                    with torch.no_grad():
                        # Original predictions
                        original_logits, _ = model(batch_images)
                        original_preds = torch.argmax(original_logits, dim=1)
                        
                        # Predictions on masked images
                        masked_logits, _ = model(masked_images)
                        masked_preds = torch.argmax(masked_logits, dim=1)
                        
                        # Count label flips
                        flips = (original_preds != masked_preds).sum().item()
                        label_flips += flips
                    
                    total_tested += batch_images.size(0)
                    
                except Exception as e:
                    print(f"Error processing batch {batch_start}-{batch_end}: {e}")
                    # Skip this batch but continue with others
                    continue
            
            # Calculate final metrics
            avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
            label_flip_rate = (label_flips / total_tested * 100) if total_tested > 0 else 0.0
            
            # Store results
            result_key = f"eps_{epsilon:.4f}_step_{step_size:.4f}"
            grid_results[result_key] = {
                'epsilon': epsilon,
                'step_size': step_size,
                'avg_ssim': avg_ssim,
                'ssim_std': np.std(ssim_scores) if ssim_scores else 0.0,
                'label_flip_rate': label_flip_rate,
                'label_flips': label_flips,
                'total_tested': total_tested,
                'attack_success': attack_success,
                'original_image_info': {
                    'original_label': source_label.item(),
                    'original_pred': original_pred.item(),
                    'adv_pred': adv_pred.item()
                },
                'mask_stats': {
                    'min': mask.min().item(),
                    'max': mask.max().item(),  
                    'mean': mask.mean().item(),
                    'std': mask.std().item()
                }
            }
            
            if verbose:
                print(f"Results:")
                print(f"  Average SSIM: {avg_ssim:.4f} ± {np.std(ssim_scores):.4f}")
                print(f"  Label flip rate: {label_flip_rate:.2f}% ({label_flips}/{total_tested})")
                print(f"  Attack success on source: {attack_success}")
    
    # Compile final results
    final_results = {
        'experiment_config': experiment_config,
        'grid_results': grid_results
    }
    
    # Save results
    results_file = os.path.join(save_path, 'grid_search_results.pt')
    torch.save(final_results, results_file)
    
    # Generate and save summary plots
    _generate_summary_plots(final_results, save_path, verbose)
    
    # Generate and save summary statistics
    _generate_summary_statistics(final_results, save_path, verbose)
    
    if verbose:
        print(f"\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print(f"Results saved to: {results_file}")
        print("Summary plots and statistics generated")
        print("="*80)
    
    return final_results


def _generate_summary_plots(results, save_path, verbose=True):
    """Generate summary heatmaps and plots for the grid search results."""
    
    grid_results = results['grid_results']
    config = results['experiment_config']
    
    epsilons = np.array(config['epsilons'])
    step_sizes = np.array(config['step_sizes'])
    
    # Extract metrics into matrices
    ssim_matrix = np.zeros((len(epsilons), len(step_sizes)))
    flip_rate_matrix = np.zeros((len(epsilons), len(step_sizes)))
    attack_success_matrix = np.zeros((len(epsilons), len(step_sizes)))
    
    for eps_idx, epsilon in enumerate(epsilons):
        for step_idx, step_size in enumerate(step_sizes):
            result_key = f"eps_{epsilon:.4f}_step_{step_size:.4f}"
            if result_key in grid_results:
                result = grid_results[result_key]
                ssim_matrix[eps_idx, step_idx] = result['avg_ssim']
                flip_rate_matrix[eps_idx, step_idx] = result['label_flip_rate']
                attack_success_matrix[eps_idx, step_idx] = result['attack_success']
    
    # Create heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # SSIM heatmap
    im1 = axes[0, 0].imshow(ssim_matrix, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Average SSIM Score')
    axes[0, 0].set_xlabel('Step Size')
    axes[0, 0].set_ylabel('Epsilon')
    axes[0, 0].set_xticks(range(len(step_sizes)))
    axes[0, 0].set_xticklabels([f'{s:.3f}' for s in step_sizes])
    axes[0, 0].set_yticks(range(len(epsilons)))
    axes[0, 0].set_yticklabels([f'{e:.3f}' for e in epsilons])
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Add text annotations for SSIM
    for i in range(len(epsilons)):
        for j in range(len(step_sizes)):
            axes[0, 0].text(j, i, f'{ssim_matrix[i, j]:.3f}', 
                           ha='center', va='center', color='white', fontsize=8)
    
    # Label flip rate heatmap
    im2 = axes[0, 1].imshow(flip_rate_matrix, cmap='Reds', aspect='auto')
    axes[0, 1].set_title('Label Flip Rate (%)')
    axes[0, 1].set_xlabel('Step Size')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].set_xticks(range(len(step_sizes)))
    axes[0, 1].set_xticklabels([f'{s:.3f}' for s in step_sizes])
    axes[0, 1].set_yticks(range(len(epsilons)))
    axes[0, 1].set_yticklabels([f'{e:.3f}' for e in epsilons])
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Add text annotations for flip rate
    for i in range(len(epsilons)):
        for j in range(len(step_sizes)):
            axes[0, 1].text(j, i, f'{flip_rate_matrix[i, j]:.1f}%', 
                           ha='center', va='center', color='white', fontsize=8)
    
    # Attack success rate heatmap
    im3 = axes[1, 0].imshow(attack_success_matrix, cmap='Blues', aspect='auto')
    axes[1, 0].set_title('Attack Success on Source Image')
    axes[1, 0].set_xlabel('Step Size')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_xticks(range(len(step_sizes)))
    axes[1, 0].set_xticklabels([f'{s:.3f}' for s in step_sizes])
    axes[1, 0].set_yticks(range(len(epsilons)))
    axes[1, 0].set_yticklabels([f'{e:.3f}' for e in epsilons])
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Add text annotations for attack success
    for i in range(len(epsilons)):
        for j in range(len(step_sizes)):
            success_text = 'Yes' if attack_success_matrix[i, j] else 'No'
            axes[1, 0].text(j, i, success_text, 
                           ha='center', va='center', color='white', fontsize=8)
    
    # Scatter plot: SSIM vs Label Flip Rate
    ssim_flat = []
    flip_flat = []
    for result in grid_results.values():
        ssim_flat.append(result['avg_ssim'])
        flip_flat.append(result['label_flip_rate'])
    
    axes[1, 1].scatter(ssim_flat, flip_flat, alpha=0.7, s=50)
    axes[1, 1].set_xlabel('Average SSIM')
    axes[1, 1].set_ylabel('Label Flip Rate (%)')
    axes[1, 1].set_title('SSIM vs Universality Trade-off')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'grid_search_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Summary plots saved to: {plot_path}")


def _generate_summary_statistics(results, save_path, verbose=True):
    """Generate summary statistics text file."""
    
    grid_results = results['grid_results']
    config = results['experiment_config']
    
    stats_file = os.path.join(save_path, 'summary_statistics.txt')
    
    with open(stats_file, 'w') as f:
        f.write("ADVERSARIAL PERTURBATION GRID SEARCH EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("EXPERIMENT CONFIGURATION:\n")
        f.write(f"Model path: {config['model_path']}\n")
        f.write(f"Data path: {config['data_path']}\n")
        f.write(f"Number of random images tested per combination: {config['num_random_images']}\n")
        f.write(f"Epsilon range: {min(config['epsilons']):.4f} to {max(config['epsilons']):.4f}\n")
        f.write(f"Step size range: {min(config['step_sizes']):.4f} to {max(config['step_sizes']):.4f}\n")
        f.write(f"Total parameter combinations: {len(grid_results)}\n")
        f.write(f"Attack configuration: {config['attack_config']}\n\n")
        
        # Calculate overall statistics
        all_ssim = [r['avg_ssim'] for r in grid_results.values()]
        all_flip_rates = [r['label_flip_rate'] for r in grid_results.values()]
        all_attack_success = [r['attack_success'] for r in grid_results.values()]
        
        f.write("OVERALL STATISTICS:\n")
        f.write(f"SSIM - Mean: {np.mean(all_ssim):.4f}, Std: {np.std(all_ssim):.4f}, Range: [{np.min(all_ssim):.4f}, {np.max(all_ssim):.4f}]\n")
        f.write(f"Label Flip Rate - Mean: {np.mean(all_flip_rates):.2f}%, Std: {np.std(all_flip_rates):.2f}%, Range: [{np.min(all_flip_rates):.2f}%, {np.max(all_flip_rates):.2f}%]\n")
        f.write(f"Attack Success Rate: {np.mean(all_attack_success)*100:.1f}% ({sum(all_attack_success)}/{len(all_attack_success)})\n\n")
        
        # Find best and worst configurations
        best_ssim_key = max(grid_results.keys(), key=lambda k: grid_results[k]['avg_ssim'])
        worst_ssim_key = min(grid_results.keys(), key=lambda k: grid_results[k]['avg_ssim'])
        best_flip_key = max(grid_results.keys(), key=lambda k: grid_results[k]['label_flip_rate'])
        worst_flip_key = min(grid_results.keys(), key=lambda k: grid_results[k]['label_flip_rate'])
        
        f.write("EXTREME CONFIGURATIONS:\n")
        f.write(f"Highest SSIM: {best_ssim_key} (SSIM: {grid_results[best_ssim_key]['avg_ssim']:.4f})\n")
        f.write(f"Lowest SSIM: {worst_ssim_key} (SSIM: {grid_results[worst_ssim_key]['avg_ssim']:.4f})\n")
        f.write(f"Highest Flip Rate: {best_flip_key} (Rate: {grid_results[best_flip_key]['label_flip_rate']:.2f}%)\n")
        f.write(f"Lowest Flip Rate: {worst_flip_key} (Rate: {grid_results[worst_flip_key]['label_flip_rate']:.2f}%)\n\n")
        
        f.write("DETAILED RESULTS BY CONFIGURATION:\n")
        f.write("-"*80 + "\n")
        
        # Sort results by epsilon then step size for organized output
        sorted_keys = sorted(grid_results.keys(), 
                           key=lambda k: (grid_results[k]['epsilon'], grid_results[k]['step_size']))
        
        for key in sorted_keys:
            result = grid_results[key]
            f.write(f"\n{key}:\n")
            f.write(f"  Epsilon: {result['epsilon']:.4f}, Step Size: {result['step_size']:.4f}\n")
            f.write(f"  Average SSIM: {result['avg_ssim']:.4f} ± {result['ssim_std']:.4f}\n")
            f.write(f"  Label Flip Rate: {result['label_flip_rate']:.2f}% ({result['label_flips']}/{result['total_tested']})\n")
            f.write(f"  Attack Success on Source: {'Yes' if result['attack_success'] else 'No'}\n")
            f.write(f"  Source Image - Original Label: {result['original_image_info']['original_label']}, "
                   f"Original Pred: {result['original_image_info']['original_pred']}, "
                   f"Adv Pred: {result['original_image_info']['adv_pred']}\n")
    
    if verbose:
        print(f"Summary statistics saved to: {stats_file}")


if __name__ == "__main__":
    # Example usage
    MODEL_PATH = "/u/hectorxm/distribution-shift/models/natural/149_checkpoint.pt"
    DATA_PATH = "/u/hectorxm/distribution-shift/datasets"
    SAVE_PATH = "/u/hectorxm/distribution-shift/adversarial/grid_search_results"
    
    # Load dataset
    dataset, train_loader, test_loader = utils.load_dataset(DATA_PATH)
    
    # Run experiment
    results = run_adversarial_grid_search_experiment(
        model_path=MODEL_PATH,
        data_path=DATA_PATH, 
        loader=test_loader,
        save_path=SAVE_PATH,
        num_random_images=100,
        verbose=True
    )
    
    print("Experiment completed successfully!") 