from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR
import torch
import torch.nn.functional as F
# import cox
from tqdm import tqdm
import os
import re
import torchvision.utils as vutils
import random
import numpy as np
from TRADES.trades import trades_loss
# import argparse
import matplotlib.pyplot as plt
import sys
# local imports
import mask_generation as mask_gen
import test
import train
import utils

# Add path for UAP functionality
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sgd-uap-torch'))
from attacks import uap_sgd

def calculate_snr(original_image, perturbed_image):
    """
    Calculate the Signal-to-Noise Ratio (SNR) between an original and perturbed image.
    
    For images, SNR is defined as the ratio of the mean signal value to the standard 
    deviation of the noise (difference between images).
    
    Args:
        original_image (torch.Tensor): The reference/baseline image (natural image)
        perturbed_image (torch.Tensor): The perturbed/test image 
        
    Returns:
        float: SNR value. Higher values indicate less perceptible changes.
    """
    # Ensure tensors are on the same device and have the same shape
    assert original_image.shape == perturbed_image.shape, "Images must have the same shape"
    
    # Convert to float if needed
    original = original_image.float()
    perturbed = perturbed_image.float()
    
    # Calculate noise as the difference between original and perturbed
    noise = perturbed - original
    
    # Calculate signal power (mean of original image)
    signal_power = torch.mean(original)
    
    # Calculate noise power (standard deviation of noise)
    noise_power = torch.std(noise)
    
    # Avoid division by zero
    if noise_power == 0:
        return float('inf')  # Perfect similarity, no noise
    
    # SNR = signal_power / noise_power
    snr = signal_power / noise_power
    return float(snr)


def calculate_psnr(original_image, perturbed_image, max_pixel_value=1.0):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between an original and perturbed image.
    
    PSNR = 20 * log10(max_pixel_value / RMSE)
    where RMSE is the root mean square error between the images.
    
    Args:
        original_image (torch.Tensor): The reference/baseline image (natural image)
        perturbed_image (torch.Tensor): The perturbed/test image
        max_pixel_value (float): Maximum possible pixel value (1.0 for normalized images, 255 for 8-bit)
        
    Returns:
        float: PSNR value in decibels (dB). Higher values indicate less perceptible changes.
    """
    # Ensure tensors are on the same device and have the same shape
    assert original_image.shape == perturbed_image.shape, "Images must have the same shape"
    
    # Convert to float if needed
    original = original_image.float()
    perturbed = perturbed_image.float()
    
    # Calculate Mean Squared Error (MSE)
    mse = torch.mean((original - perturbed) ** 2)
    
    # Handle case where images are identical (MSE = 0)
    if mse == 0:
        return float('inf')  # Perfect similarity
    
    # Calculate PSNR
    psnr = 20 * torch.log10(torch.tensor(max_pixel_value)) - 10 * torch.log10(mse)
    return float(psnr)


def calculate_ssim(original_image, perturbed_image, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Calculate the Structural Similarity Index Measure (SSIM) between an original and perturbed image.
    
    SSIM measures similarity based on luminance, contrast, and structure.
    SSIM = (2*μx*μy + C1)(2*σxy + C2) / (μx² + μy² + C1)(σx² + σy² + C2)
    
    Args:
        original_image (torch.Tensor): The reference/baseline image (natural image)
        perturbed_image (torch.Tensor): The perturbed/test image
        window_size (int): Size of the sliding window for local statistics (default: 11)
        C1, C2 (float): Constants to stabilize division with weak denominators
        
    Returns:
        float: SSIM value between -1 and 1. Values closer to 1 indicate higher similarity.
    """
    # Ensure tensors are on the same device and have the same shape
    assert original_image.shape == perturbed_image.shape, "Images must have the same shape"
    
    # Convert to float if needed
    original = original_image.float()
    perturbed = perturbed_image.float()
    
    # If input is a batch, process each image separately and average
    if len(original.shape) == 4:  # (batch, channels, height, width)
        ssim_values = []
        for i in range(original.shape[0]):
            ssim_val = calculate_ssim(original[i], perturbed[i], window_size, C1, C2)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    
    # Add batch dimension if needed
    if len(original.shape) == 3:  # (channels, height, width)
        original = original.unsqueeze(0)
        perturbed = perturbed.unsqueeze(0)
    
    # Create Gaussian window
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g
    
    def create_window(window_size, channel):
        _1D_window = gaussian_window(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    channels = original.size(1)
    window = create_window(window_size, channels)
    
    if original.is_cuda:
        window = window.cuda(original.get_device())
    window = window.type_as(original)
    
    # Calculate local means
    mu1 = F.conv2d(original, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(perturbed, window, padding=window_size//2, groups=channels)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate local variances and covariance
    sigma1_sq = F.conv2d(original * original, window, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(perturbed * perturbed, window, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(original * perturbed, window, padding=window_size//2, groups=channels) - mu1_mu2
    
    # Calculate SSIM
    numerator1 = 2 * mu1_mu2 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2
    
    ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
    
    # Return mean SSIM over the entire image
    return float(ssim_map.mean())


def run_perceptibility_analysis(save_path, model_path, N, loader, visualize=False):
    """
    Analyzes perceptibility of adversarial perturbations using SNR, PSNR, and SSIM metrics.
    
    This function generates adversarial examples using 6 different attack configurations
    and calculates perceptibility metrics against natural images as baseline.
    
    Args:
        save_path (str): Directory to save results and visualizations
        model_path (str): Path to the .pt model weights file
        N (int): Number of images to process
        loader (DataLoader): DataLoader containing images and labels
        visualize (bool): Whether to save visualization images (default: False)
    
    Attack Configurations:
        1. Low epsilon PGD: eps=0.031, step=0.01
        2. Medium epsilon, high step: eps=4*0.031, step=0.1
        3. Medium epsilon, low step: eps=4*0.031, step=0.01
        4. High epsilon, high step: eps=25*0.031, step=0.1
        5. High epsilon, low step: eps=25*0.031, step=0.01
        6. Untargeted UAP
    
    For PGD attacks (1-5), also generates masks and applies them to random images.
    """
    # Create save directories
    os.makedirs(save_path, exist_ok=True)
    if visualize:
        os.makedirs(os.path.join(save_path, 'visualizations'), exist_ok=True)
    
    print(f"Starting perceptibility analysis with {N} images")
    print(f"Model path: {model_path}")
    print(f"Save path: {save_path}")
    
    # Define attack configurations
    attack_configs = {
        'low_eps_pgd': {
            'constraint': 'inf',
            'eps': 0.031,
            'step_size': 0.01,
            'iterations': 10,
            'random_start': False,
            'name': 'Low Epsilon PGD (eps=0.031, step=0.01)'
        },
        'mid_eps_high_step_pgd': {
            'constraint': 'inf',
            'eps': 4 * 0.031,
            'step_size': 0.1,
            'iterations': 10,
            'random_start': False,
            'name': 'Medium Epsilon, High Step PGD (eps=4*0.031, step=0.1)'
        },
        'mid_eps_low_step_pgd': {
            'constraint': 'inf',
            'eps': 4 * 0.031,
            'step_size': 0.01,
            'iterations': 10,
            'random_start': False,
            'name': 'Medium Epsilon, Low Step PGD (eps=4*0.031, step=0.01)'
        },
        'high_eps_high_step_pgd': {
            'constraint': 'inf',
            'eps': 25 * 0.031,
            'step_size': 0.1,
            'iterations': 10,
            'random_start': False,
            'name': 'High Epsilon, High Step PGD (eps=25*0.031, step=0.1)'
        },
        'high_eps_low_step_pgd': {
            'constraint': 'inf',
            'eps': 25 * 0.031,
            'step_size': 0.01,
            'iterations': 10,
            'random_start': False,
            'name': 'High Epsilon, Low Step PGD (eps=25*0.031, step=0.01)'
        }
    }
    
    # UAP configuration
    uap_config = {
        'nb_epoch': 10,
        'eps': 8/255,  # Standard epsilon for UAP
        'beta': 12,
        'step_decay': 0.8,
        'name': 'Untargeted UAP'
    }
    
    # Load model and dataset
    dataset = CIFAR('/afs/csail.mit.edu/u/h/hectorxm/distribution-shift/datasets')  # Temporary path, adjust as needed
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset, resume_path=model_path)
    model.eval()
    
    # Generate UAP first (needs full loader)
    print("Generating Untargeted UAP...")
    try:
        # Create a wrapper for the robustness model to work with UAP generation
        class ModelWrapper(torch.nn.Module):
            def __init__(self, robustness_model):
                super().__init__()
                self.model = robustness_model
            
            def forward(self, x):
                # Robustness models return (logits, representation), we only need logits
                output, _ = self.model(x)
                return output
        
        wrapped_model = ModelWrapper(model)
        
        uap_perturbation, uap_losses = uap_sgd(
            wrapped_model, loader, 
            nb_epoch=uap_config['nb_epoch'],
            eps=uap_config['eps'],
            beta=uap_config['beta'],
            step_decay=uap_config['step_decay']
        )
        print(f"UAP generation completed. Final loss: {uap_losses[-1]:.4f}")
    except Exception as e:
        print(f"Error generating UAP: {e}")
        uap_perturbation = None
    
    # Process N images
    all_results = []
    processed_count = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        if processed_count >= N:
            break
            
        batch_size = min(images.size(0), N - processed_count)
        images = images[:batch_size]
        labels = labels[:batch_size]
        
        # Move to GPU
        images = images.cuda()
        labels = labels.cuda()
        
        for img_idx in range(batch_size):
            if processed_count >= N:
                break
                
            single_image = images[img_idx:img_idx+1]
            single_label = labels[img_idx:img_idx+1]
            
            print(f"Processing image {processed_count + 1}/{N}")
            
            # Store results for this image
            image_results = {
                'image_id': processed_count,
                'original_label': single_label.cpu().item(),
                'attacks': {}
            }
            
            # Process PGD attacks
            for attack_name, config in attack_configs.items():
                try:
                    # Generate adversarial example
                    _, adv_image = model(single_image, single_label, make_adv=True, **{k: v for k, v in config.items() if k != 'name'})
                    
                    # Calculate perceptibility metrics vs natural image
                    snr = calculate_snr(single_image, adv_image)
                    psnr = calculate_psnr(single_image, adv_image)
                    ssim = calculate_ssim(single_image, adv_image)
                    
                    # Extract mask (perturbation)
                    mask = adv_image - single_image
                    
                    # Get a truly random image from the dataset (not from current batch)
                    # This ensures we always have a different image for mask application
                    random_batch_iter = iter(loader)
                    for _ in range(torch.randint(1, 4, (1,)).item()):  # Skip 1-3 batches randomly
                        try:
                            random_images, _ = next(random_batch_iter)
                        except StopIteration:
                            random_batch_iter = iter(loader)
                            random_images, _ = next(random_batch_iter)
                    
                    # Select a random image from this batch
                    random_idx = torch.randint(0, random_images.size(0), (1,)).item()
                    random_image = random_images[random_idx:random_idx+1].cuda()
                    masked_random_image = torch.clamp(random_image + mask, 0, 1)
                    
                    # Calculate metrics for mask applied to random image
                    mask_snr = calculate_snr(random_image, masked_random_image)
                    mask_psnr = calculate_psnr(random_image, masked_random_image)
                    mask_ssim = calculate_ssim(random_image, masked_random_image)
                    
                    # Store results
                    image_results['attacks'][attack_name] = {
                        'config': config,
                        'original_vs_adv': {
                            'snr': snr,
                            'psnr': psnr,
                            'ssim': ssim
                        },
                        'random_vs_masked': {
                            'snr': mask_snr,
                            'psnr': mask_psnr,
                            'ssim': mask_ssim
                        }
                    }
                    
                    # Visualization
                    if visualize:
                        attack_dir = os.path.join(save_path, 'visualizations', f'image_{processed_count:03d}', attack_name)
                        os.makedirs(attack_dir, exist_ok=True)
                        
                        # Save images
                        vutils.save_image(single_image, os.path.join(attack_dir, 'original.png'), normalize=True)
                        vutils.save_image(adv_image, os.path.join(attack_dir, 'adversarial.png'), normalize=True)
                        vutils.save_image(mask, os.path.join(attack_dir, 'mask.png'), normalize=True)
                        vutils.save_image(random_image, os.path.join(attack_dir, 'random_base.png'), normalize=True)
                        vutils.save_image(masked_random_image, os.path.join(attack_dir, 'random_masked.png'), normalize=True)
                        
                        # Save metrics text file
                        with open(os.path.join(attack_dir, 'metrics.txt'), 'w') as f:
                            f.write(f"Attack: {config['name']}\n")
                            f.write(f"Image ID: {processed_count}\n")
                            f.write(f"Original Label: {single_label.cpu().item()}\n\n")
                            
                            f.write("Metrics (Original vs Adversarial):\n")
                            f.write(f"SNR: {snr:.4f}\n")
                            f.write(f"PSNR: {psnr:.4f} dB\n")
                            f.write(f"SSIM: {ssim:.4f}\n\n")
                            
                            f.write("Metrics (Random Image vs Masked Random Image):\n")
                            f.write(f"SNR: {mask_snr:.4f}\n")
                            f.write(f"PSNR: {mask_psnr:.4f} dB\n")
                            f.write(f"SSIM: {mask_ssim:.4f}\n")
                
                except Exception as e:
                    print(f"Error processing {attack_name} for image {processed_count}: {e}")
                    continue
            
            # Process UAP attack
            if uap_perturbation is not None:
                try:
                    # Apply UAP to current image
                    uap_image = torch.clamp(single_image + uap_perturbation.cuda(), 0, 1)
                    
                    # Calculate perceptibility metrics vs natural image
                    uap_snr = calculate_snr(single_image, uap_image)
                    uap_psnr = calculate_psnr(single_image, uap_image)
                    uap_ssim = calculate_ssim(single_image, uap_image)
                    
                    # Store results
                    image_results['attacks']['uap'] = {
                        'config': uap_config,
                        'original_vs_adv': {
                            'snr': uap_snr,
                            'psnr': uap_psnr,
                            'ssim': uap_ssim
                        }
                    }
                    
                    # Visualization
                    if visualize:
                        uap_dir = os.path.join(save_path, 'visualizations', f'image_{processed_count:03d}', 'uap')
                        os.makedirs(uap_dir, exist_ok=True)
                        
                        # Save images
                        vutils.save_image(single_image, os.path.join(uap_dir, 'original.png'), normalize=True)
                        vutils.save_image(uap_image, os.path.join(uap_dir, 'uap_applied.png'), normalize=True)
                        vutils.save_image(uap_perturbation, os.path.join(uap_dir, 'uap_pattern.png'), normalize=True)
                        
                        # Save metrics text file
                        with open(os.path.join(uap_dir, 'metrics.txt'), 'w') as f:
                            f.write(f"Attack: {uap_config['name']}\n")
                            f.write(f"Image ID: {processed_count}\n")
                            f.write(f"Original Label: {single_label.cpu().item()}\n\n")
                            
                            f.write("Metrics (Original vs UAP Applied):\n")
                            f.write(f"SNR: {uap_snr:.4f}\n")
                            f.write(f"PSNR: {uap_psnr:.4f} dB\n")
                            f.write(f"SSIM: {uap_ssim:.4f}\n")
                
                except Exception as e:
                    print(f"Error processing UAP for image {processed_count}: {e}")
            
            all_results.append(image_results)
            processed_count += 1
    
    # Save comprehensive results
    results_summary = {
        'experiment_config': {
            'num_images': N,
            'model_path': model_path,
            'save_path': save_path,
            'visualize': visualize
        },
        'attack_configs': {**attack_configs, 'uap': uap_config},
        'results': all_results
    }
    
    # Save results as torch file
    torch.save(results_summary, os.path.join(save_path, 'perceptibility_results.pt'))
    
    # Generate summary statistics
    generate_summary_statistics(results_summary, save_path)
    
    print(f"Perceptibility analysis completed. Results saved to {save_path}")
    return results_summary


def generate_summary_statistics(results_summary, save_path):
    """
    Generate summary statistics from the perceptibility analysis results.
    
    Args:
        results_summary (dict): Complete results from run_perceptibility_analysis
        save_path (str): Directory to save summary files
    """
    all_results = results_summary['results']
    attack_names = list(results_summary['attack_configs'].keys())
    
    # Initialize statistics collectors
    stats = {}
    for attack in attack_names:
        stats[attack] = {
            'original_vs_adv': {'snr': [], 'psnr': [], 'ssim': []},
            'random_vs_masked': {'snr': [], 'psnr': [], 'ssim': []} if attack != 'uap' else None
        }
    
    # Collect all metric values
    for result in all_results:
        for attack_name, attack_data in result['attacks'].items():
            if 'original_vs_adv' in attack_data:
                for metric, value in attack_data['original_vs_adv'].items():
                    if not (np.isnan(value) or np.isinf(value)):
                        stats[attack_name]['original_vs_adv'][metric].append(value)
            
            if attack_name != 'uap' and 'random_vs_masked' in attack_data:
                for metric, value in attack_data['random_vs_masked'].items():
                    if not (np.isnan(value) or np.isinf(value)):
                        stats[attack_name]['random_vs_masked'][metric].append(value)
    
    # Calculate summary statistics
    with open(os.path.join(save_path, 'summary_statistics.txt'), 'w') as f:
        f.write("PERCEPTIBILITY ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of images processed: {len(all_results)}\n")
        f.write(f"Number of attack configurations: {len(attack_names)}\n\n")
        
        for attack_name in attack_names:
            config = results_summary['attack_configs'][attack_name]
            f.write(f"\nATTACK: {config['name']}\n")
            f.write("-" * 40 + "\n")
            
            # Original vs Adversarial metrics  
            f.write("Original vs Adversarial Image:\n")
            for metric in ['snr', 'psnr', 'ssim']:
                values = stats[attack_name]['original_vs_adv'][metric]
                if values:
                    f.write(f"  {metric.upper()}: mean={np.mean(values):.4f}, "
                           f"std={np.std(values):.4f}, "
                           f"min={np.min(values):.4f}, "
                           f"max={np.max(values):.4f}\n")
                else:
                    f.write(f"  {metric.upper()}: No valid values\n")
            
            # Random vs Masked metrics (for PGD attacks only)
            if attack_name != 'uap' and stats[attack_name]['random_vs_masked']:
                f.write("\nRandom Image vs Masked Random Image:\n")
                for metric in ['snr', 'psnr', 'ssim']:
                    values = stats[attack_name]['random_vs_masked'][metric]
                    if values:
                        f.write(f"  {metric.upper()}: mean={np.mean(values):.4f}, "
                               f"std={np.std(values):.4f}, "
                               f"min={np.min(values):.4f}, "
                               f"max={np.max(values):.4f}\n")
                    else:
                        f.write(f"  {metric.upper()}: No valid values\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Higher SNR, PSNR, and SSIM values indicate less perceptible changes.\n")
        f.write("SSIM values range from -1 to 1, with 1 indicating perfect similarity.\n")
    
    print(f"Summary statistics saved to {os.path.join(save_path, 'summary_statistics.txt')}")


if __name__ == "__main__":
    
    # Load dataset using utils functions
    DATA_PATH = '/afs/csail.mit.edu/u/h/hectorxm/distribution-shift/datasets'
    
    
    # Load the dataset and get loaders
    dataset, train_loader, test_loader = utils.load_dataset(DATA_PATH)
    print("Successfully loaded dataset")
    
    results = run_perceptibility_analysis(
        save_path ='/afs/csail.mit.edu/u/h/hectorxm/distribution-shift/adversarial/visualizations/perceptibility_analysis/experimentN=300',
        model_path='/afs/csail.mit.edu/u/h/hectorxm/distribution-shift/models/natural/149_checkpoint.pt',
        N=300,
        loader=test_loader,
        visualize=False
    )
    generate_summary_statistics(results, '/afs/csail.mit.edu/u/h/hectorxm/distribution-shift/adversarial/visualizations/perceptibility_analysis/experimentN=300')
    print("done")



