from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR
import torch
import torch.nn.functional as F
import cox
from tqdm import tqdm
import os
import re
import torchvision.utils as vutils
import random
import numpy as np
from TRADES.trades import trades_loss
import argparse
# local imports
import mask_generation as mask_gen
import test
import train
import utils
NO_ADVERSARIAL_TRAINING_PARAMS = {
        'out_dir': "/home/gridsan/hmartinez/distribution-shift/models/mask_trained_0.031*25eps_0.1stepsize",
        'adv_train': 0,  # Set to 1 for adversarial training
        'epochs': 150,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'step_lr': 50,
        'step_lr_gamma': 0.1,
    }

ATTACK_PARAMS = {
        'constraint': 'inf',      # Use L2 PGD attack
        'eps': 0.031*25,            # large epsilon
        'step_size': 0.1,      # large step size
        'iterations': 10,      # standard iterations
        'random_start': False,  # standard random start
    }
# standard PGD attack
NORMAL_ATTACK_PARAMS = {
        'constraint': 'inf',      # Use L2 PGD attack
        'eps': 0.031,            # small epsilon
        'step_size': 0.01,      # small step size
        'iterations': 10,      # standard iterations
        'random_start': False,  # standard random start
    }

def run_mask_training_experiment(model_path, save_path):
    # first make the data to pass into batch
    _, train_loader, _ = train.load_dataset("/home/gridsan/hmartinez/distribution-shift/datasets")
    mask_tensors_list = []
    labels_list = []
    adv_labels_list = []
    print(f"Creating masks for training data, we have {len(train_loader)} batches")
    print(f"This might take a while...")
    for batch in train_loader:
        images, labels = batch
        _,_,mask_tensors,_, mask_labels, adv_predicted_labels = mask_gen.create_masks_batch(ATTACK_PARAMS, model_path, images, labels)
        mask_tensors_list.append(mask_tensors)
        labels_list.append(mask_labels)
        adv_labels_list.append(adv_predicted_labels)
    print(f"Successfully created masks for training data")
    print(f"Training on natural and adversarial labels")
    train.train_with_masks(NO_ADVERSARIAL_TRAINING_PARAMS, model_path, mask_tensors_list, labels_list, adv_labels_list, save_path)
    print(f"Successfully trained on natural and adversarial labels")

def run_random_noise_mask_confidence_experiment(model_path):
    """
    This experiment will generate a pertubation mask for N random (Gaussian) images. It will then then apply those masks to the test set
    and see if the adversarial confidence for the test set is high. This would tell us that the high-epsilon pertubation masks are 
    picking out adversarial features and making them very large such that they overshadow any image they are on top of (any features, natural or otherwise)

    Another experiment this would do is take those same pertubation masks from random noise and simply run a test inference on them to see
    what their adversarial and natural confidence is. Note: We will use the model that is not trained on the masks for this experiment.
    """

    # first load the test set
    _, test_loader, _ = train.load_dataset("/home/gridsan/hmartinez/distribution-shift/datasets")
    dataset = CIFAR('/home/gridsan/hmartinez/distribution-shift/datasets')
    # load the model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset, resume_path=model_path)
    model.cuda()

    # generate masks for the random noise images
    mask_tensors_list = [] # list of lists that have batch size number of elements
    labels_list = []
    adv_labels_list = []
    random_noise_images = []
    natural_images_list = []

    for batch in test_loader:
        images, labels = batch
        natural_images_list.append(images)
        # generate random noise images
        for i in range(len(images)):
            random_noise_images.append(torch.rand(images[i].shape))
        print(f"Shape of images is {images.shape}")
        print(f"Shape of random noise images is {random_noise_images[-1].shape}")
        _,_,mask_tensors,_, mask_labels, adv_predicted_labels = mask_gen.create_masks_batch(ATTACK_PARAMS, model_path, images, labels)
        mask_tensors_list.append(mask_tensors)
        labels_list.append(mask_labels)
        adv_labels_list.append(adv_predicted_labels)
        # comment this out when we want to do all of the test set
        # break


    # now run a test inference on the random noise images
     # eval
    model.eval()
    
    correct_natural = 0
    correct_adversarial = 0
    correct_natural_superimposed = 0
    correct_adversarial_superimposed = 0
    total = 0
    with torch.no_grad():
        for i in range(len(mask_tensors_list)):
            # first just the masks
            mask_tensor = mask_tensors_list[i]
            label = labels_list[i]
            adv_label = adv_labels_list[i]
            mask_tensor = mask_tensor.cuda()
            label = label.cuda()
            adv_label = adv_label.cuda()
            outputs, _ = model(mask_tensor)
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct_natural += predicted.eq(label).sum().item()
            correct_adversarial += predicted.eq(adv_label).sum().item()

            # now masks superimposed on natural images (in no particular order)
            natural_image = natural_images_list[i]
            natural_image = natural_image.cuda()
            superimposed_image = natural_image + mask_tensor
            outputs, _ = model(superimposed_image)
            _, predicted = outputs.max(1)
            correct_natural_superimposed += predicted.eq(label).sum().item()
            correct_adversarial_superimposed += predicted.eq(adv_label).sum().item()


    # compute and return accuracy
    natural_accuracy = 100 * correct_natural/total
    adversarial_accuracy = 100 * correct_adversarial/total
    natural_superimposed_accuracy = 100 * correct_natural_superimposed/total
    adversarial_superimposed_accuracy = 100 * correct_adversarial_superimposed/total
    print(f"Natural accuracy on test set is {natural_accuracy}")
    print(f"Adversarial accuracy on test set is {adversarial_accuracy}")
    print(f"Natural superimposed accuracy on test set is {natural_superimposed_accuracy}")
    print(f"Adversarial superimposed accuracy on test set is {adversarial_superimposed_accuracy}")
    return natural_accuracy, adversarial_accuracy, natural_superimposed_accuracy, adversarial_superimposed_accuracy

def run_mask_visualization_experiment(model_path, save_folder, loader, N=5, random_images=False):
    """
    This experiment will create a mask from the model_path (which should be a natural model) for both large and small epsilon (and step size)
    attack configurations. It will then save the following into save_folder:
    1. The masks (2), labeled with natural and adversarial confidence
    2. The masks superimposed on the natural images (2) labeled with natural and adversarial confidence
    3. The masks superimposed on random natural images (2) labeled with natural and adversarial confidence
    This will be repeated N times for each of the 2 attack configurations. Note: The natural images will be sampled from the test set, not training set.
    If you get directory does not exist error, make sure to create the directory first.

    If random_images is True, then we will create random images and use them as the masks instead.
    """
    # first load the test set
    dataset = CIFAR('/home/gridsan/hmartinez/distribution-shift/datasets')
    # load the model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset, resume_path=model_path)
    model.cuda()

    # list out natural images, just one batch
    natural_images = []
    natural_labels = []
    for batch in loader:
        images, labels = batch
        natural_images.append(images) # adds only one batch
        natural_labels.append(labels)
        break


    # now we can create the masks, for one batch
    # create large epsilon mask
    original_tensors_large, _, mask_tensors_large, _, labels_large, adv_predicted_labels_large = mask_gen.create_masks_batch(ATTACK_PARAMS, model_path, natural_images[0], natural_labels[0])
    # create small epsilon mask
    original_tensors_small, _, mask_tensors_small, _, labels_small, adv_predicted_labels_small = mask_gen.create_masks_batch(NORMAL_ATTACK_PARAMS, model_path, natural_images[0], natural_labels[0])

    if random_images:
        # create random images, both small and large epsilon
        random_images_large = utils.create_random_images(N, epsilon=3)
        random_images_small = utils.create_random_images(N, epsilon=0.5)
        mask_tensors_large = list(random_images_large)
        mask_tensors_small = list(random_images_small)
        
    # save N masks for each of the 2 attack configurations and get their natural and adversarial confidence
    for i in range(N):
        # large epsilon
        original_tensor = original_tensors_large[i]
        mask_tensor = mask_tensors_large[i]
        label = labels_large[i]
        adv_predicted_label = adv_predicted_labels_large[i]

        # save the mask as an image
        # Normalize the mask for better visualization
        normalized_mask = (mask_tensor - mask_tensor.min()) / (mask_tensor.max() - mask_tensor.min() + 1e-8)
        vutils.save_image(normalized_mask, f"{save_folder}/masks/mask_large_{i}.png", normalize=False)
        # Still save the tensor data for potential future use
        torch.save(mask_tensor, f"{save_folder}/masks/mask_large_{i}.pt")
        torch.save(label, f"{save_folder}/masks/label_large_{i}.pt")
        torch.save(adv_predicted_label, f"{save_folder}/masks/adv_predicted_label_large_{i}.pt")
    
        # Get natural and adversarial confidence for the mask itself, softmax probabilities
        with torch.no_grad():
            mask_input = mask_tensor.unsqueeze(0).cuda()
            mask_output, _ = model(mask_input)
            mask_probs = F.softmax(mask_output, dim=1)
            
            # Get the confidence for the true label and adversarial label
            natural_confidence = mask_probs[0, label].item()
            adv_confidence = mask_probs[0, adv_predicted_label].item()
            
            # Get the highest confidence class
            highest_conf_class = torch.argmax(mask_probs, dim=1).item()
            highest_conf_value = mask_probs[0, highest_conf_class].item()
            
            # Save the confidence values
            confidence_data = {
                'natural_confidence': natural_confidence,
                'adv_confidence': adv_confidence,
                'highest_conf_class': highest_conf_class,
                'highest_conf_value': highest_conf_value,
                'label': label,
                'adv_predicted_label': adv_predicted_label
            }
            
            # Save as a tensor for consistency
            torch.save(confidence_data, f"{save_folder}/masks/confidence_large_{i}.pt")
            
            print(f"Mask {i} (large epsilon):")
            print(f"  Natural confidence (class {label}): {natural_confidence:.4f}")
            print(f"  Adversarial confidence (class {adv_predicted_label}): {adv_confidence:.4f}")
            print(f"  Highest confidence (class {highest_conf_class}): {highest_conf_value:.4f}")

        # Save the confidence information to a text file
        with open(f"{save_folder}/masks/confidence_large_{i}.txt", 'w') as f:
            f.write(f"Mask {i} (large epsilon):\n")
            f.write(f"  Natural confidence (class {label}): {natural_confidence:.4f}\n")
            f.write(f"  Adversarial confidence (class {adv_predicted_label}): {adv_confidence:.4f}\n")
            f.write(f"  Highest confidence (class {highest_conf_class}): {highest_conf_value:.4f}\n")
            f.write(f"  Label: {label}\n")
            f.write(f"  Adversarial label: {adv_predicted_label}\n")

        # now experiment (2), superimpose mask on natural image
        superimposed_image = original_tensor + mask_tensor
        
        # Save the superimposed image
        vutils.save_image(superimposed_image, f"{save_folder}/superimposed/superimposed_large_{i}.png", normalize=True)
        torch.save(superimposed_image, f"{save_folder}/superimposed/superimposed_large_{i}.pt")
        
        # Get confidence for the superimposed image
        with torch.no_grad():
            superimposed_input = superimposed_image.unsqueeze(0).cuda()
            superimposed_output, _ = model(superimposed_input)
            superimposed_probs = F.softmax(superimposed_output, dim=1)
            
            # Get the confidence for the true label and adversarial label
            natural_confidence_superimposed = superimposed_probs[0, label].item()
            adv_confidence_superimposed = superimposed_probs[0, adv_predicted_label].item()
            
            # Get the highest confidence class
            highest_conf_class_superimposed = torch.argmax(superimposed_probs, dim=1).item()
            highest_conf_value_superimposed = superimposed_probs[0, highest_conf_class_superimposed].item()
            
            # Save the confidence values
            superimposed_confidence_data = {
                'natural_confidence': natural_confidence_superimposed,
                'adv_confidence': adv_confidence_superimposed,
                'highest_conf_class': highest_conf_class_superimposed,
                'highest_conf_value': highest_conf_value_superimposed,
                'label': label,
                'adv_predicted_label': adv_predicted_label
            }
            
            # Save as a tensor for consistency
            torch.save(superimposed_confidence_data, f"{save_folder}/superimposed/confidence_large_{i}.pt")
            
            print(f"Superimposed image {i} (large epsilon):")
            print(f"  Natural confidence (class {label}): {natural_confidence_superimposed:.4f}")
            print(f"  Adversarial confidence (class {adv_predicted_label}): {adv_confidence_superimposed:.4f}")
            print(f"  Highest confidence (class {highest_conf_class_superimposed}): {highest_conf_value_superimposed:.4f}")
            
        # Save the superimposed confidence information to a text file
        with open(f"{save_folder}/superimposed/confidence_large_{i}.txt", 'w') as f:
            f.write(f"Superimposed image {i} (large epsilon):\n")
            f.write(f"  Natural confidence (class {label}): {natural_confidence_superimposed:.4f}\n")
            f.write(f"  Adversarial confidence (class {adv_predicted_label}): {adv_confidence_superimposed:.4f}\n")
            f.write(f"  Highest confidence (class {highest_conf_class_superimposed}): {highest_conf_value_superimposed:.4f}\n")
            f.write(f"  Label: {label}\n")
            f.write(f"  Adversarial label: {adv_predicted_label}\n")
        

        # now experiment (3), superimpose mask on random natural image, gets natural image from last i-th element in batch
        random_original_tensor = original_tensors_large[-i]
        random_superimposed_image = random_original_tensor + mask_tensor
        
        # Save the random superimposed image
        vutils.save_image(random_superimposed_image, f"{save_folder}/random_superimposed/random_superimposed_large_{i}.png", normalize=True)
        torch.save(random_superimposed_image, f"{save_folder}/random_superimposed/random_superimposed_large_{i}.pt")
        
        # Get confidence for the random superimposed image
        with torch.no_grad():
            random_superimposed_input = random_superimposed_image.unsqueeze(0).cuda()
            random_superimposed_output, _ = model(random_superimposed_input)
            random_superimposed_probs = F.softmax(random_superimposed_output, dim=1)
            
            # Get the highest confidence class
            random_highest_conf_class = torch.argmax(random_superimposed_probs, dim=1).item()
            random_highest_conf_value = random_superimposed_probs[0, random_highest_conf_class].item()
            
            # Get the confidence for the adversarial label
            random_original_label = labels_large[-i]
            random_natural_confidence = random_superimposed_probs[0, random_original_label].item()
            random_adv_confidence = random_superimposed_probs[0, adv_predicted_label].item()
            
            # Save the confidence values
            random_superimposed_confidence_data = {
                'highest_conf_class': random_highest_conf_class,
                'highest_conf_value': random_highest_conf_value,
                'adv_confidence': random_adv_confidence,
                'adv_predicted_label': adv_predicted_label,
                'natural_confidence': random_natural_confidence,
                'label': random_original_label, 
                'adv_label': adv_predicted_label
            }
            
            # Save as a tensor for consistency
            torch.save(random_superimposed_confidence_data, f"{save_folder}/random_superimposed/confidence_large_{i}.pt")
            
            print(f"Random superimposed image {i} (large epsilon):")
            print(f"  Highest confidence (class {random_highest_conf_class}): {random_highest_conf_value:.4f}")
            print(f"  Adversarial confidence (class {adv_predicted_label}): {random_adv_confidence:.4f}")
            print(f"  Natural confidence (class {random_original_label}): {random_natural_confidence:.4f}")
            
        # Save the random superimposed confidence information to a text file
        with open(f"{save_folder}/random_superimposed/confidence_large_{i}.txt", 'w') as f:
            f.write(f"Random superimposed image {i} (large epsilon):\n")
            f.write(f"  Natural confidence (class {random_original_label}): {random_natural_confidence:.4f}\n")
            f.write(f"  Adversarial confidence (class {adv_predicted_label}): {random_adv_confidence:.4f}\n")
            f.write(f"  Adversarial label: {adv_predicted_label}\n")
            f.write(f"  Highest confidence (class {random_highest_conf_class}): {random_highest_conf_value:.4f}\n")


    # now do the same for small epsilon
    for i in range(N):
        # small epsilon
        original_tensor = original_tensors_small[i]
        mask_tensor = mask_tensors_small[i]
        label = labels_small[i]
        adv_predicted_label = adv_predicted_labels_small[i]

        # save the mask as an image
        # Normalize the mask for better visualization
        normalized_mask = (mask_tensor - mask_tensor.min()) / (mask_tensor.max() - mask_tensor.min() + 1e-8)
        vutils.save_image(normalized_mask, f"{save_folder}/masks/mask_small_{i}.png", normalize=False)
        # Still save the tensor data for potential future use
        torch.save(mask_tensor, f"{save_folder}/masks/mask_small_{i}.pt")
        torch.save(label, f"{save_folder}/masks/label_small_{i}.pt")
        torch.save(adv_predicted_label, f"{save_folder}/masks/adv_predicted_label_small_{i}.pt")
    
        # Get natural and adversarial confidence for the mask itself, softmax probabilities
        with torch.no_grad():
            mask_input = mask_tensor.unsqueeze(0).cuda()
            mask_output, _ = model(mask_input)
            mask_probs = F.softmax(mask_output, dim=1)
            
            # Get the confidence for the true label and adversarial label
            natural_confidence = mask_probs[0, label].item()
            adv_confidence = mask_probs[0, adv_predicted_label].item()
            
            # Get the highest confidence class
            highest_conf_class = torch.argmax(mask_probs, dim=1).item()
            highest_conf_value = mask_probs[0, highest_conf_class].item()
            
            # Save the confidence values
            confidence_data = {
                'natural_confidence': natural_confidence,
                'adv_confidence': adv_confidence,
                'highest_conf_class': highest_conf_class,
                'highest_conf_value': highest_conf_value,
                'label': label,
                'adv_predicted_label': adv_predicted_label
            }
            
            # Save as a tensor for consistency
            torch.save(confidence_data, f"{save_folder}/masks/confidence_small_{i}.pt")
            
            print(f"Mask {i} (small epsilon):")
            print(f"  Natural confidence (class {label}): {natural_confidence:.4f}")
            print(f"  Adversarial confidence (class {adv_predicted_label}): {adv_confidence:.4f}")
            print(f"  Highest confidence (class {highest_conf_class}): {highest_conf_value:.4f}")

        # Save the confidence information to a text file
        with open(f"{save_folder}/masks/confidence_small_{i}.txt", 'w') as f:
            f.write(f"Mask {i} (small epsilon):\n")
            f.write(f"  Natural confidence (class {label}): {natural_confidence:.4f}\n")
            f.write(f"  Adversarial confidence (class {adv_predicted_label}): {adv_confidence:.4f}\n")
            f.write(f"  Highest confidence (class {highest_conf_class}): {highest_conf_value:.4f}\n")
        # Now save the mask superimposed on the original image
        superimposed_image = original_tensor + mask_tensor
        vutils.save_image(superimposed_image, f"{save_folder}/superimposed/superimposed_small_{i}.png", normalize=True)
        torch.save(superimposed_image, f"{save_folder}/superimposed/superimposed_small_{i}.pt")
        
        # Get confidence for the superimposed image
        with torch.no_grad():
            superimposed_input = superimposed_image.unsqueeze(0).cuda()
            superimposed_output, _ = model(superimposed_input)
            superimposed_probs = F.softmax(superimposed_output, dim=1)
            
            # Get the confidence for the true label and adversarial label
            superimposed_natural_confidence = superimposed_probs[0, label].item()
            superimposed_adv_confidence = superimposed_probs[0, adv_predicted_label].item()
            
            # Get the highest confidence class
            superimposed_highest_conf_class = torch.argmax(superimposed_probs, dim=1).item()
            superimposed_highest_conf_value = superimposed_probs[0, superimposed_highest_conf_class].item()
            
            # Save the confidence values
            superimposed_confidence_data = {
                'natural_confidence': superimposed_natural_confidence,
                'adv_confidence': superimposed_adv_confidence,
                'highest_conf_class': superimposed_highest_conf_class,
                'highest_conf_value': superimposed_highest_conf_value,
                'label': label,
                'adv_predicted_label': adv_predicted_label
            }
            
            # Save as a tensor for consistency
            torch.save(superimposed_confidence_data, f"{save_folder}/superimposed/confidence_small_{i}.pt")
            
            print(f"Superimposed image {i} (small epsilon):")
            print(f"  Natural confidence (class {label}): {superimposed_natural_confidence:.4f}")
            print(f"  Adversarial confidence (class {adv_predicted_label}): {superimposed_adv_confidence:.4f}")
            print(f"  Highest confidence (class {superimposed_highest_conf_class}): {superimposed_highest_conf_value:.4f}")
            
        # Save the superimposed confidence information to a text file
        with open(f"{save_folder}/superimposed/confidence_small_{i}.txt", 'w') as f:
            f.write(f"Superimposed image {i} (small epsilon):\n")
            f.write(f"  Natural confidence (class {label}): {superimposed_natural_confidence:.4f}\n")
            f.write(f"  Adversarial confidence (class {adv_predicted_label}): {superimposed_adv_confidence:.4f}\n")
            f.write(f"  Highest confidence (class {superimposed_highest_conf_class}): {superimposed_highest_conf_value:.4f}\n")
            
        # Now save the mask superimposed on a random image
        random_original_tensor = original_tensors_small[-i]
        random_superimposed_image = random_original_tensor + mask_tensor
        
        # Save the random superimposed image
        vutils.save_image(random_superimposed_image, f"{save_folder}/random_superimposed/random_superimposed_small_{i}.png", normalize=True)
        torch.save(random_superimposed_image, f"{save_folder}/random_superimposed/random_superimposed_small_{i}.pt")
        
        # Get confidence for the random superimposed image
        with torch.no_grad():
            random_superimposed_input = random_superimposed_image.unsqueeze(0).cuda()
            random_superimposed_output, _ = model(random_superimposed_input)
            random_superimposed_probs = F.softmax(random_superimposed_output, dim=1)
            
            # Get the highest confidence class
            random_highest_conf_class = torch.argmax(random_superimposed_probs, dim=1).item()
            random_highest_conf_value = random_superimposed_probs[0, random_highest_conf_class].item()
            
            # Get the confidence for the adversarial label
            random_original_label = labels_small[-i]
            random_natural_confidence = random_superimposed_probs[0, random_original_label].item()
            random_adv_confidence = random_superimposed_probs[0, adv_predicted_label].item()
            
            # Save the confidence values
            random_superimposed_confidence_data = {
                'highest_conf_class': random_highest_conf_class,
                'highest_conf_value': random_highest_conf_value,
                'adv_confidence': random_adv_confidence,
                'adv_predicted_label': adv_predicted_label,
                'natural_confidence': random_natural_confidence,
                'label': random_original_label,
                'adv_label': adv_predicted_label
            }
            
            # Save as a tensor for consistency
            torch.save(random_superimposed_confidence_data, f"{save_folder}/random_superimposed/confidence_small_{i}.pt")
            
            print(f"Random superimposed image {i} (small epsilon):")
            print(f"  Highest confidence (class {random_highest_conf_class}): {random_highest_conf_value:.4f}")
            print(f"  Adversarial confidence (class {adv_predicted_label}): {random_adv_confidence:.4f}")
            print(f"  Natural confidence (class {random_original_label}): {random_natural_confidence:.4f}")
            
        # Save the random superimposed confidence information to a text file
        with open(f"{save_folder}/random_superimposed/confidence_small_{i}.txt", 'w') as f:
            f.write(f"Random superimposed image {i} (small epsilon):\n")
            f.write(f"  Natural confidence (class {random_original_label}): {random_natural_confidence:.4f}\n")
            f.write(f"  Adversarial confidence (class {adv_predicted_label}): {random_adv_confidence:.4f}\n")
            f.write(f"  Adversarial label: {adv_predicted_label}\n")
            f.write(f"  Highest confidence (class {random_highest_conf_class}): {random_highest_conf_value:.4f}\n")

def analyze_mask_superimposed_experiment(experiment_folder):
    """
    Will return some statistics about the experiments, writes to a text file in same experiment folder
    `experiment_folder` should have the structure generated by `run_mask_visualization_experiment`
    """    
    # # Create a file to write the analysis results
    # analysis_file_path = os.path.join(experiment_folder, "analysis_results.txt")
    # with open(analysis_file_path, 'w') as analysis_file:
    #     analysis_file.write(f"Analysis of experiment in {experiment_folder}\n")
    #     analysis_file.write(f"Model used: {model_path}\n\n")
        
        # Iterate through all subdirectories in the experiment folder
    for subdir in os.listdir(experiment_folder):
        subdir_path = os.path.join(experiment_folder, subdir)
        if os.path.isdir(subdir_path):
            print(f"Inside of {subdir} directory:")
                
            # List all files in the subdirectory
            highest_natural_count_large = 0
            highest_adv_count_large = 0
            disagree_count_large = 0
            total_count_large = 0
            for file in os.listdir(subdir_path):
                # first analyze the large epsilon files
                if file.endswith('.txt') and 'large' in file:
                    with open(os.path.join(subdir_path, file), 'r') as f:
                        total_count_large += 1
                        content = f.read()
                        # Extract natural class, adversarial class, and highest confidence class
                        natural_class = int(re.search(r"Natural confidence \(class (\d+)\)", content).group(1))
                        adv_class = int(re.search(r"Adversarial confidence \(class (\d+)\)", content).group(1))
                        highest_class = int(re.search(r"Highest confidence \(class (\d+)\)", content).group(1))
                        # Count matches
                        if highest_class == natural_class:
                            highest_natural_count_large += 1
                        elif highest_class == adv_class:
                            highest_adv_count_large += 1
                        else:
                            disagree_count_large += 1
            
            # write a textfile in this subdirectory with the results
            with open(os.path.join(subdir_path, 'analysis_results_large.txt'), 'w') as f:
                f.write(f"Analysis of large epsilon masks\n")
                f.write(f"  Total masks analyzed: {total_count_large}\n")
                f.write(f"  Masks where highest confidence class matches natural class: {highest_natural_count_large} ({highest_natural_count_large/total_count_large*100:.2f}%)\n")
                f.write(f"  Masks where highest confidence class matches adversarial class: {highest_adv_count_large} ({highest_adv_count_large/total_count_large*100:.2f}%)\n")
                f.write(f"  Masks where highest confidence class matches neither: {disagree_count_large} ({disagree_count_large/total_count_large*100:.2f}%)\n\n")
            
            # now do the same for the small epsilon files
            highest_natural_count_small = 0
            highest_adv_count_small = 0
            disagree_count_small = 0
            total_count_small = 0
            for file in os.listdir(subdir_path):
                if file.endswith('.txt') and 'small' in file:
                    with open(os.path.join(subdir_path, file), 'r') as f:
                        total_count_small += 1
                        content = f.read()
                        # Extract natural class, adversarial class, and highest confidence class
                        natural_class = int(re.search(r"Natural confidence \(class (\d+)\)", content).group(1))
                        adv_class = int(re.search(r"Adversarial confidence \(class (\d+)\)", content).group(1))
                        highest_class = int(re.search(r"Highest confidence \(class (\d+)\)", content).group(1))
                        # Count matches
                        if highest_class == natural_class:
                            highest_natural_count_small += 1
                        elif highest_class == adv_class:
                            highest_adv_count_small += 1
                        else:
                            disagree_count_small += 1
            
            # write a textfile in this subdirectory with the results
            with open(os.path.join(subdir_path, 'analysis_results_small.txt'), 'w') as f:
                f.write(f"Analysis of small epsilon masks\n")
                f.write(f"  Total masks analyzed: {total_count_small}\n")
                f.write(f"  Masks where highest confidence class matches natural class: {highest_natural_count_small} ({highest_natural_count_small/total_count_small*100:.2f}%)\n")
                f.write(f"  Masks where highest confidence class matches adversarial class: {highest_adv_count_small} ({highest_adv_count_small/total_count_small*100:.2f}%)\n")
                f.write(f"  Masks where highest confidence class matches neither: {disagree_count_small} ({disagree_count_small/total_count_small*100:.2f}%)\n\n")

    print("Done analyzing mask superimposed experiment")      






    
    
    

if __name__ == "__main__":
    MODEL_PATH = "/home/gridsan/hmartinez/distribution-shift/models/natural/149_checkpoint.pt"
    SAVE_PATH = "/home/gridsan/hmartinez/distribution-shift/adversarial/visualizations/mask_superimposed/experiment_7_randomNoiseMasks_test_set"
    EXPERIMENT_FOLDER = "/home/gridsan/hmartinez/distribution-shift/adversarial/visualizations/mask_superimposed/experiment_7_randomNoiseMasks_test_set"
    # Get the training loader
    _, train_loader, test_loader = train.load_dataset("/home/gridsan/hmartinez/distribution-shift/datasets")
    run_mask_visualization_experiment(MODEL_PATH, SAVE_PATH, test_loader, N=100, random_images=True)
    # run_mask_training_experiment(MODEL_PATH, SAVE_PATH)
    # run_random_noise_mask_confidence_experiment(MODEL_PATH)
    analyze_mask_superimposed_experiment(EXPERIMENT_FOLDER)