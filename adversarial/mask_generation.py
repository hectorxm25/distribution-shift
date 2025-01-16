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
        'constraint': 'inf',      # Use L2 PGD attack
        'eps': 0.031*25,            # L2 radius/epsilon
        'step_size': 0.1,      # Changed from 'attack_lr' to 'step_size'
        'iterations': 10,      # Changed from 'attack_steps' to 'iterations'
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


def create_masks_batch(config, model_path, images, labels):
    """
    Creates masks for a batch of images and returns both images and predictions
    
    Args:
        config (dict): Attack configuration parameters
        model_path (str): Path to the trained model
        images (torch.Tensor): Batch of images with shape [batch_size, channels, height, width]
        labels (torch.Tensor): Batch of labels with shape [batch_size]
        
    Returns:
        tuple: Original tensors, adversarial tensors, mask tensors, gradient mask tensors,
              original labels, and predicted labels for adversarial images
    """
    # Create dataset instance first
    dataset = CIFAR(DATASET_PATH)
    
    # Load dataset and model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset, resume_path=model_path)
    model.eval()

    # Move to GPU
    images = images.cuda()
    labels = labels.cuda()

    # Generate adversarial examples
    _, adv_images = model(images, labels, make_adv=True, **config)
    
    # Get model predictions for adversarial images
    with torch.no_grad():
        adv_output, _ = model(adv_images)
        adv_predicted_labels = adv_output.argmax(dim=1).cpu()

    # Convert back to CPU and numpy
    original_images = images.cpu().numpy()
    adversarial_images = adv_images.cpu().detach().numpy()
    
    # Generate masks by taking the difference
    masks = adversarial_images - original_images
    gradient_masks = adversarial_images - 1.5*original_images

    # Convert to tensors
    original_tensors = torch.from_numpy(original_images)
    adversarial_tensors = torch.from_numpy(adversarial_images)
    mask_tensors = torch.from_numpy(masks)
    gradient_mask_tensors = torch.from_numpy(gradient_masks)

    return (original_tensors, adversarial_tensors, mask_tensors, 
            gradient_mask_tensors, labels.cpu(), adv_predicted_labels)

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
    gradient_mask = adversarial_image - 1.5*original_image

    # Save original, adversarial and mask images
    original_tensor = torch.from_numpy(original_image)
    adversarial_tensor = torch.from_numpy(adversarial_image) 
    mask_tensor = torch.from_numpy(mask)
    gradient_mask_tensor = torch.from_numpy(gradient_mask)

    # FOR DEBUGGING ONLY 

    # vutils.save_image(original_tensor, image_path + '/original.png', normalize=True)
    # vutils.save_image(adversarial_tensor, image_path + '/adversarial.png', normalize=True)
    # vutils.save_image(mask_tensor, image_path + '/mask.png', normalize=True)
    # vutils.save_image(gradient_mask_tensor, image_path + '/gradient_mask.png', normalize=True)

    return gradient_mask, mask, original_image, adversarial_image, label

def inference_mask_batch(mask_images, model_path):
    """
    Inference a batch of mask images through the model and get predictions for the entire batch
    
    Args:
        mask_images (numpy.ndarray): Batch of mask images with shape [batch_size, channels, height, width]
        model_path (str): Path to the trained model checkpoint
        
    Returns:
        list: Predicted labels for each mask image in the batch
    """
    # Load dataset and model
    dataset = CIFAR(DATASET_PATH)
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset, resume_path=model_path)
    model.eval()


    # Move to GPU
    mask_tensor = mask_tensor.cuda()

    # Get model predictions for batch
    with torch.no_grad():
        output, _ = model(mask_tensor)
        predicted_labels = output.argmax(dim=1).cpu().tolist()

    return predicted_labels

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
# ACHIEVED ONLY 10% ACCURACY
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
            gradient_mask, mask, _, _, _ = create_mask(ATTACK_PARAMS, model_path, [(image, label)], OUTPUT_DIR)
            
            # Get predictions on mask or gradient mask
            mask_pred = inference_mask(mask, model_path)
            # mask_pred = inference_mask(gradient_mask, model_path)
            
            # Check if prediction matches the true label
            if mask_pred == label.item():
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"Mask Inference Accuracy: {accuracy * 100:.2f}%")
    
    return accuracy


def check_mask_accuracy_both_batch(loader, model_path):
    """
    Takes a data loader and model path, creates masks for all images and checks model accuracy on masks
    for both natural and adversarial labels, processing images in batches.
    Args:
        loader: DataLoader containing images to create masks from
        model_path: Path to the trained model checkpoint
    Returns:
        tuple: (natural_accuracy, adversarial_accuracy) - Accuracy of model predictions on mask images
               for natural and adversarial labels respectively
    """
    # Load dataset and model
    dataset = CIFAR(DATASET_PATH)
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset, resume_path=model_path)
    model.eval()

    natural_correct = 0
    adversarial_correct = 0
    total = 0

    for images, labels in loader:
        # Generate masks using create_masks_batch function
        _, _, masks, _, orig_labels, adv_labels = create_masks_batch(ATTACK_PARAMS, model_path, images, labels)
        print("orig_labels: ", orig_labels)
        print("adv_labels: ", adv_labels)
    
        # move masks to GPU
        masks = masks.cuda()

        # Get predictions on masks in batch
        with torch.no_grad():
            model_output, _ = model(masks)
            mask_preds = model_output.argmax(dim=1)
        
        # Move predictions to CPU for comparison
        mask_preds = mask_preds.cpu()
            
        # Compare predictions with both natural and adversarial labels
        natural_correct += (mask_preds == orig_labels).sum().item()
        adversarial_correct += (mask_preds == adv_labels).sum().item()
        total += len(labels)

    natural_accuracy = natural_correct / total
    adversarial_accuracy = adversarial_correct / total
    
    print(f"Natural Mask Inference Accuracy: {natural_accuracy * 100:.2f}%")
    print(f"Adversarial Mask Inference Accuracy: {adversarial_accuracy * 100:.2f}%")
    
    return natural_accuracy, adversarial_accuracy



if __name__ == "__main__":
    _, _, test_loader = load_dataset(DATASET_PATH)
    print("Loaded test loader of size: ", len(test_loader))
    # # quick one-batch test
    # for images, labels in test_loader:
    #     one_batch_loader = [(images, labels)]
    #     break
    # print("Created a one-batch loader of size: ", len(one_batch_loader))
    print("Running accuracy check...")
    natural_accuracy, adversarial_accuracy = check_mask_accuracy_both_batch(test_loader, "/home/gridsan/hmartinez/distribution-shift/models/natural/149_checkpoint.pt")
    print("natural accuracy: ", natural_accuracy)
    print("adversarial accuracy: ", adversarial_accuracy)
