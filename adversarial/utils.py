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

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)[0]  # Only return the logits 