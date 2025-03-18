from robustness.datasets import CIFAR
import torch
import random
import numpy as np

def create_and_save_dataset(data_path):
    """
    Saves the dataset splits for later use in a way compatible with robustness library
    """
    # Set all random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Create dataset using robustness CIFAR class
    dataset = CIFAR(data_path)

    # Create train/test loaders with reproducible settings
    train_loader, test_loader = dataset.make_loaders(
        batch_size=128, 
        workers=8,
    )
    
    # Save the dataset and reproducibility info
    torch.save({
        'dataset': dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'seed': 42,
    }, f"{data_path}/dataset.pt")
    
    print(f"Successfully saved dataset to {data_path}/dataset.pt")
    return dataset

def load_dataset(data_path):
    """
    Loads previously saved dataset in a way compatible with robustness library
    """
    # Load the saved dataset and reproducibility info
    checkpoint = torch.load(f"{data_path}/dataset.pt")
    dataset = checkpoint['dataset']
    
    # Restore all random seeds (just in case)
    torch.manual_seed(checkpoint['seed'])
    random.seed(checkpoint['seed'])
    np.random.seed(checkpoint['seed'])
    
    # Create new loaders with same reproducibility settings
    train_loader = checkpoint['train_loader']
    test_loader = checkpoint['test_loader']
    
    print("Successfully loaded data loaders")
    return dataset, train_loader, test_loader

def create_random_images(num_images, epsilon=1):
    """
    Creates random images following the differential privacy noise injection framework.
    Returns tensor of shape (num_images, 3, 32, 32). 
    Taken from https://www.perplexity.ai/search/i-am-a-researcher-in-machine-l-my84HmbhQVmET3VCkjlCWA
    """
     # Generate base noise image
    base_noise = np.random.normal(0, 1, size=(num_images, 3, 32, 32))
    
    # Calculate sensitivity (assuming pixel values in [0, 1])
    sensitivity = 1.0
    
    # Calculate noise scale based on epsilon
    noise_scale = sensitivity / epsilon
    
    # Add calibrated noise
    dp_noise = base_noise + np.random.laplace(0, noise_scale, size=(num_images, 3, 32, 32))
    
    # Clip values to [0, 1] range
    dp_noise = np.clip(dp_noise, 0, 1)

    # Convert to PyTorch tensor
    dp_noise_tensor = torch.tensor(dp_noise, dtype=torch.float32)
    
    return dp_noise_tensor

def create_gaussian_noise(num_images, mu, sigma):
    """
    Creates gaussian noise.
    Returns tensor of shape (num_images, 3, 32, 32). 
    """
    base_noise = np.random.normal(mu, sigma, size=(num_images, 3, 32, 32))
    return torch.tensor(base_noise, dtype=torch.float32)

def load_pt_tensor(path, verbose=False):
    """
    loads a tensor from a .pt file.
    """
    tensor = torch.load(path)
    if verbose:
        print(f"Loaded tensor from {path}")
        print(f"Tensor shape: {tensor.shape}")
        print(f"Tensor: {tensor}")
    return tensor

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)[0]  # Only return the logits 
    

if __name__ == "__main__":
    load_pt_tensor("/home/gridsan/hmartinez/distribution-shift/adversarial/visualizations/mask_superimposed/experiment_5_test_set/superimposed/superimposed_large_4.pt", verbose=True)
