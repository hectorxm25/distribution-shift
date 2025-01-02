"""
This file will generate the PGD masks and implement our initial honeypot idea.
The main objective is to generate a perturbation PGD mask from a trained model
and then see if the trained model is able to classify this mask as the class it came
from. We will visualize both the natural image, the mask, and the fully perturbed image.
"""

import torch
import torchvision
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR
from train import load_dataset

# global configs
DATASET_PATH = "/home/gridsan/hmartinez/distribution-shift/datasets"
OUTPUT_DIR = "/home/gridsan/hmartinez/distribution-shift/adversarial/temp"

ATTACK_PARAMS = {
        'constraint': '2',      # Use L2 PGD attack
        'eps': 0.5,            # L2 radius/epsilon
        'step_size': 0.1,      # Changed from 'attack_lr' to 'step_size'
        'iterations': 7,      # Changed from 'attack_steps' to 'iterations'
        'random_start': False,  # Changed from 'random_restarts' to 'random_start'
    }

ADVERSARIAL_TRAINING_PARAMS = {
        'out_dir': OUTPUT_DIR,
        'adv_train': 1,        # Enable adversarial training
        'epochs': 150, # TODO: maybe change this in the future to see less overfitting in our experiments
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'step_lr': 50,
        'step_lr_gamma': 0.1,
        **ATTACK_PARAMS        # Add attack parameters
    }


def visualize_dataset_images(data_loader, image_path):
    """
    Visualize and save a single image from a data loader with improved quality.
    Specifically handles CIFAR-10 images from the robustness library.
    """
    # Get single image
    images, labels = next(iter(data_loader))
    image = images[0]
    label = labels[0].item()
    
    natural_image = image.cpu().numpy()
    natural_image = natural_image / 2 + 0.5
    
    natural_image_tensor = torch.tensor(natural_image)
    vutils.save_image(natural_image_tensor, image_path, normalize=True)

    # Map CIFAR-10 numeric labels to human readable class names
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
    human_readable = cifar10_classes[label]
    
    return natural_image_tensor.numpy(), label, human_readable



def create_mask(config, model_path, loader, image_path):
    """
    Creates a mask by generating an adversarial example and subtracting it from the original image.
    
    Args:
        config (dict): Attack configuration parameters
        model_path (str): Path to the trained model
        loader (robustness.datasets.DataLoader): Data loader for the dataset
        image_path (str): Path to save the images
        
    Returns:
        numpy.ndarray: The generated mask (difference between adversarial and original image) and the other related images
    """
    # Create dataset instance first
    dataset = CIFAR(DATASET_PATH)
    
    # Load dataset and model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset, resume_path=model_path)
    model.eval()

    images, labels = next(iter(loader))
    image = images[0]
    label = labels.item() if labels.dim() == 0 else labels[0].item()

    # Move to GPU
    image_tensor = image.unsqueeze(0).cuda()
    label_tensor = torch.tensor([label]).cuda()

    # Generate adversarial example
    _, adv_image = model(image_tensor, label_tensor, make_adv=True, **config)
    
    # Convert back to CPU and numpy
    original_image = image_tensor.cpu().numpy()
    adversarial_image = adv_image.cpu().detach().numpy()
    
    # Generate mask by taking the difference
    mask = adversarial_image - original_image
    gradient_mask = adversarial_image - 2*original_image

    # Save original, adversarial and mask images
    original_tensor = torch.from_numpy(original_image)
    adversarial_tensor = torch.from_numpy(adversarial_image) 
    mask_tensor = torch.from_numpy(mask)
    gradient_mask_tensor = torch.from_numpy(gradient_mask)

    vutils.save_image(original_tensor, image_path + '/original.png', normalize=True)
    vutils.save_image(adversarial_tensor, image_path + '/adversarial.png', normalize=True)
    vutils.save_image(mask_tensor, image_path + '/mask.png', normalize=True)
    vutils.save_image(gradient_mask_tensor, image_path + '/gradient_mask.png', normalize=True)
    return gradient_mask, mask, original_image, adversarial_image, label


def inference_mask(mask_image, model_path):
    """
    Inference the mask image through the model and see if it is able to classify the mask image
    """
    # Load dataset and model
    dataset = CIFAR(DATASET_PATH)
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset, resume_path=model_path)
    model.eval()

    # Convert mask image to tensor with correct dimensions
    mask_tensor = torch.from_numpy(mask_image)
    if len(mask_tensor.shape) == 3:  # If [channels, height, width]
        mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension -> [1, channels, height, width]
    
    # Move to GPU
    mask_tensor = mask_tensor.cuda()

    # Get model prediction
    with torch.no_grad():
        output, _ = model(mask_tensor)
        predicted_label = output.argmax(dim=1).item()

    return predicted_label


# NAIVE APPROACH, JUST CHECKS IF THE MODEL CAN CLASSIFY CORRECTLY ONLY GIVEN THE MASK OF THE NATURAL IMAGE
def check_mask_accuracy(loader, model_path):
    """
    Takes a data loader and model path, creates masks for all images and checks model accuracy on masks
    Args:
        loader: DataLoader containing images to create masks from
        model_path: Path to the trained model checkpoint
    Returns:
        float: Accuracy of model predictions on mask images
    """
    # Load dataset and model
    dataset = CIFAR(DATASET_PATH)
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset, resume_path=model_path)
    model.eval()

    correct = 0
    total = 0

    for images, labels in loader:
        # Process one image at a time
        for i in range(len(images)):
            image = images[i:i+1]  # Keep batch dimension but only one image
            label = labels[i]
            
            # Generate masks using create_mask function
            _, mask, _, _, _ = create_mask(ATTACK_PARAMS, model_path, [(image, label)], OUTPUT_DIR)
            
            # Get predictions on mask
            mask_pred = inference_mask(mask, model_path)
            
            # Check if prediction matches the true label
            if mask_pred == label.item():
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"Mask Inference Accuracy: {accuracy * 100:.2f}%")
    
    return accuracy

if __name__ == "__main__":
    _, _, test_loader = load_dataset(DATASET_PATH)
    accuracy = check_mask_accuracy(test_loader, "/home/gridsan/hmartinez/distribution-shift/models/natural/149_checkpoint.pt")
    print("accuracy: ", accuracy)