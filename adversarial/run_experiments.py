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

# from .TRADES.trades import trades_loss
# import argparse

# local imports
from . import mask_generation as mask_gen
from . import test
from . import train
from . import utils

NO_ADVERSARIAL_TRAINING_PARAMS = {
    "out_dir": "/home/gridsan/hmartinez/distribution-shift/models/mask_trained_0.031*25eps_0.1stepsize",
    "adv_train": 0,  # Set to 1 for adversarial training
    "epochs": 150,
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "step_lr": 50,
    "step_lr_gamma": 0.1,
}

ATTACK_PARAMS = {
    "constraint": "inf",  # Use L2 PGD attack
    "eps": 0.031 * 25,  # large epsilon
    "step_size": 0.1,  # large step size
    "iterations": 10,  # standard iterations
    "random_start": False,  # standard random start
}
# standard PGD attack
NORMAL_ATTACK_PARAMS = {
    "constraint": "inf",  # Use L2 PGD attack
    "eps": 0.031,  # small epsilon
    "step_size": 0.01,  # small step size
    "iterations": 10,  # standard iterations
    "random_start": False,  # standard random start
}


def run_mask_training_experiment(model_path, save_path):
    # first make the data to pass into batch
    _, train_loader, _ = train.load_dataset(
        "/home/gridsan/hmartinez/distribution-shift/datasets"
    )
    mask_tensors_list = []
    labels_list = []
    adv_labels_list = []
    print(f"Creating masks for training data, we have {len(train_loader)} batches")
    print(f"This might take a while...")
    for batch in train_loader:
        images, labels = batch
        _, _, mask_tensors, _, mask_labels, adv_predicted_labels = (
            mask_gen.create_masks_batch(ATTACK_PARAMS, model_path, images, labels)
        )
        mask_tensors_list.append(mask_tensors)
        labels_list.append(mask_labels)
        adv_labels_list.append(adv_predicted_labels)
    print(f"Successfully created masks for training data")
    print(f"Training on natural and adversarial labels")
    train.train_with_masks(
        NO_ADVERSARIAL_TRAINING_PARAMS,
        model_path,
        mask_tensors_list,
        labels_list,
        adv_labels_list,
        save_path,
    )
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
    _, test_loader, _ = train.load_dataset(
        "/home/gridsan/hmartinez/distribution-shift/datasets"
    )
    dataset = CIFAR("/home/gridsan/hmartinez/distribution-shift/datasets")
    # load the model
    model, _ = model_utils.make_and_restore_model(
        arch="resnet18", dataset=dataset, resume_path=model_path
    )
    model.cuda()

    # generate masks for the random noise images
    mask_tensors_list = []  # list of lists that have batch size number of elements
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
        _, _, mask_tensors, _, mask_labels, adv_predicted_labels = (
            mask_gen.create_masks_batch(ATTACK_PARAMS, model_path, images, labels)
        )
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
    natural_accuracy = 100 * correct_natural / total
    adversarial_accuracy = 100 * correct_adversarial / total
    natural_superimposed_accuracy = 100 * correct_natural_superimposed / total
    adversarial_superimposed_accuracy = 100 * correct_adversarial_superimposed / total
    print(f"Natural accuracy on test set is {natural_accuracy}")
    print(f"Adversarial accuracy on test set is {adversarial_accuracy}")
    print(
        f"Natural superimposed accuracy on test set is {natural_superimposed_accuracy}"
    )
    print(
        f"Adversarial superimposed accuracy on test set is {adversarial_superimposed_accuracy}"
    )
    return (
        natural_accuracy,
        adversarial_accuracy,
        natural_superimposed_accuracy,
        adversarial_superimposed_accuracy,
    )


def run_mask_visualization_experiment(
    model_path, save_folder, loader, N=5, random_images=False
):
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
    dataset = CIFAR("/home/gridsan/hmartinez/distribution-shift/datasets")
    # load the model
    model, _ = model_utils.make_and_restore_model(
        arch="resnet18", dataset=dataset, resume_path=model_path
    )
    model.cuda()

    # list out natural images, just one batch
    natural_images = []
    natural_labels = []
    for batch in loader:
        images, labels = batch
        natural_images.append(images)  # adds only one batch
        natural_labels.append(labels)
        break

    # now we can create the masks, for one batch
    # create large epsilon mask
    (
        original_tensors_large,
        _,
        mask_tensors_large,
        _,
        labels_large,
        adv_predicted_labels_large,
    ) = mask_gen.create_masks_batch(
        ATTACK_PARAMS, model_path, natural_images[0], natural_labels[0]
    )
    # create small epsilon mask
    (
        original_tensors_small,
        _,
        mask_tensors_small,
        _,
        labels_small,
        adv_predicted_labels_small,
    ) = mask_gen.create_masks_batch(
        NORMAL_ATTACK_PARAMS, model_path, natural_images[0], natural_labels[0]
    )

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
        normalized_mask = (mask_tensor - mask_tensor.min()) / (
            mask_tensor.max() - mask_tensor.min() + 1e-8
        )
        vutils.save_image(
            normalized_mask, f"{save_folder}/masks/mask_large_{i}.png", normalize=False
        )
        # Still save the tensor data for potential future use
        torch.save(mask_tensor, f"{save_folder}/masks/mask_large_{i}.pt")
        torch.save(label, f"{save_folder}/masks/label_large_{i}.pt")
        torch.save(
            adv_predicted_label, f"{save_folder}/masks/adv_predicted_label_large_{i}.pt"
        )

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
                "natural_confidence": natural_confidence,
                "adv_confidence": adv_confidence,
                "highest_conf_class": highest_conf_class,
                "highest_conf_value": highest_conf_value,
                "label": label,
                "adv_predicted_label": adv_predicted_label,
            }

            # Save as a tensor for consistency
            torch.save(confidence_data, f"{save_folder}/masks/confidence_large_{i}.pt")

            print(f"Mask {i} (large epsilon):")
            print(f"  Natural confidence (class {label}): {natural_confidence:.4f}")
            print(
                f"  Adversarial confidence (class {adv_predicted_label}): {adv_confidence:.4f}"
            )
            print(
                f"  Highest confidence (class {highest_conf_class}): {highest_conf_value:.4f}"
            )

        # Save the confidence information to a text file
        with open(f"{save_folder}/masks/confidence_large_{i}.txt", "w") as f:
            f.write(f"Mask {i} (large epsilon):\n")
            f.write(f"  Natural confidence (class {label}): {natural_confidence:.4f}\n")
            f.write(
                f"  Adversarial confidence (class {adv_predicted_label}): {adv_confidence:.4f}\n"
            )
            f.write(
                f"  Highest confidence (class {highest_conf_class}): {highest_conf_value:.4f}\n"
            )
            f.write(f"  Label: {label}\n")
            f.write(f"  Adversarial label: {adv_predicted_label}\n")

        # now experiment (2), superimpose mask on natural image
        superimposed_image = original_tensor + mask_tensor

        # Save the superimposed image
        vutils.save_image(
            superimposed_image,
            f"{save_folder}/superimposed/superimposed_large_{i}.png",
            normalize=True,
        )
        torch.save(
            superimposed_image, f"{save_folder}/superimposed/superimposed_large_{i}.pt"
        )

        # Get confidence for the superimposed image
        with torch.no_grad():
            superimposed_input = superimposed_image.unsqueeze(0).cuda()
            superimposed_output, _ = model(superimposed_input)
            superimposed_probs = F.softmax(superimposed_output, dim=1)

            # Get the confidence for the true label and adversarial label
            natural_confidence_superimposed = superimposed_probs[0, label].item()
            adv_confidence_superimposed = superimposed_probs[
                0, adv_predicted_label
            ].item()

            # Get the highest confidence class
            highest_conf_class_superimposed = torch.argmax(
                superimposed_probs, dim=1
            ).item()
            highest_conf_value_superimposed = superimposed_probs[
                0, highest_conf_class_superimposed
            ].item()

            # Save the confidence values
            superimposed_confidence_data = {
                "natural_confidence": natural_confidence_superimposed,
                "adv_confidence": adv_confidence_superimposed,
                "highest_conf_class": highest_conf_class_superimposed,
                "highest_conf_value": highest_conf_value_superimposed,
                "label": label,
                "adv_predicted_label": adv_predicted_label,
            }

            # Save as a tensor for consistency
            torch.save(
                superimposed_confidence_data,
                f"{save_folder}/superimposed/confidence_large_{i}.pt",
            )

            print(f"Superimposed image {i} (large epsilon):")
            print(
                f"  Natural confidence (class {label}): {natural_confidence_superimposed:.4f}"
            )
            print(
                f"  Adversarial confidence (class {adv_predicted_label}): {adv_confidence_superimposed:.4f}"
            )
            print(
                f"  Highest confidence (class {highest_conf_class_superimposed}): {highest_conf_value_superimposed:.4f}"
            )

        # Save the superimposed confidence information to a text file
        with open(f"{save_folder}/superimposed/confidence_large_{i}.txt", "w") as f:
            f.write(f"Superimposed image {i} (large epsilon):\n")
            f.write(
                f"  Natural confidence (class {label}): {natural_confidence_superimposed:.4f}\n"
            )
            f.write(
                f"  Adversarial confidence (class {adv_predicted_label}): {adv_confidence_superimposed:.4f}\n"
            )
            f.write(
                f"  Highest confidence (class {highest_conf_class_superimposed}): {highest_conf_value_superimposed:.4f}\n"
            )
            f.write(f"  Label: {label}\n")
            f.write(f"  Adversarial label: {adv_predicted_label}\n")

        # now experiment (3), superimpose mask on random natural image, gets natural image from last i-th element in batch
        random_original_tensor = original_tensors_large[-i]
        random_superimposed_image = random_original_tensor + mask_tensor

        # Save the random superimposed image
        vutils.save_image(
            random_superimposed_image,
            f"{save_folder}/random_superimposed/random_superimposed_large_{i}.png",
            normalize=True,
        )
        torch.save(
            random_superimposed_image,
            f"{save_folder}/random_superimposed/random_superimposed_large_{i}.pt",
        )

        # Get confidence for the random superimposed image
        with torch.no_grad():
            random_superimposed_input = random_superimposed_image.unsqueeze(0).cuda()
            random_superimposed_output, _ = model(random_superimposed_input)
            random_superimposed_probs = F.softmax(random_superimposed_output, dim=1)

            # Get the highest confidence class
            random_highest_conf_class = torch.argmax(
                random_superimposed_probs, dim=1
            ).item()
            random_highest_conf_value = random_superimposed_probs[
                0, random_highest_conf_class
            ].item()

            # Get the confidence for the adversarial label
            random_original_label = labels_large[-i]
            random_natural_confidence = random_superimposed_probs[
                0, random_original_label
            ].item()
            random_adv_confidence = random_superimposed_probs[
                0, adv_predicted_label
            ].item()

            # Save the confidence values
            random_superimposed_confidence_data = {
                "highest_conf_class": random_highest_conf_class,
                "highest_conf_value": random_highest_conf_value,
                "adv_confidence": random_adv_confidence,
                "adv_predicted_label": adv_predicted_label,
                "natural_confidence": random_natural_confidence,
                "label": random_original_label,
                "adv_label": adv_predicted_label,
            }

            # Save as a tensor for consistency
            torch.save(
                random_superimposed_confidence_data,
                f"{save_folder}/random_superimposed/confidence_large_{i}.pt",
            )

            print(f"Random superimposed image {i} (large epsilon):")
            print(
                f"  Highest confidence (class {random_highest_conf_class}): {random_highest_conf_value:.4f}"
            )
            print(
                f"  Adversarial confidence (class {adv_predicted_label}): {random_adv_confidence:.4f}"
            )
            print(
                f"  Natural confidence (class {random_original_label}): {random_natural_confidence:.4f}"
            )

        # Save the random superimposed confidence information to a text file
        with open(
            f"{save_folder}/random_superimposed/confidence_large_{i}.txt", "w"
        ) as f:
            f.write(f"Random superimposed image {i} (large epsilon):\n")
            f.write(
                f"  Natural confidence (class {random_original_label}): {random_natural_confidence:.4f}\n"
            )
            f.write(
                f"  Adversarial confidence (class {adv_predicted_label}): {random_adv_confidence:.4f}\n"
            )
            f.write(f"  Adversarial label: {adv_predicted_label}\n")
            f.write(
                f"  Highest confidence (class {random_highest_conf_class}): {random_highest_conf_value:.4f}\n"
            )

    # now do the same for small epsilon
    for i in range(N):
        # small epsilon
        original_tensor = original_tensors_small[i]
        mask_tensor = mask_tensors_small[i]
        label = labels_small[i]
        adv_predicted_label = adv_predicted_labels_small[i]

        # save the mask as an image
        # Normalize the mask for better visualization
        normalized_mask = (mask_tensor - mask_tensor.min()) / (
            mask_tensor.max() - mask_tensor.min() + 1e-8
        )
        vutils.save_image(
            normalized_mask, f"{save_folder}/masks/mask_small_{i}.png", normalize=False
        )
        # Still save the tensor data for potential future use
        torch.save(mask_tensor, f"{save_folder}/masks/mask_small_{i}.pt")
        torch.save(label, f"{save_folder}/masks/label_small_{i}.pt")
        torch.save(
            adv_predicted_label, f"{save_folder}/masks/adv_predicted_label_small_{i}.pt"
        )

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
                "natural_confidence": natural_confidence,
                "adv_confidence": adv_confidence,
                "highest_conf_class": highest_conf_class,
                "highest_conf_value": highest_conf_value,
                "label": label,
                "adv_predicted_label": adv_predicted_label,
            }

            # Save as a tensor for consistency
            torch.save(confidence_data, f"{save_folder}/masks/confidence_small_{i}.pt")

            print(f"Mask {i} (small epsilon):")
            print(f"  Natural confidence (class {label}): {natural_confidence:.4f}")
            print(
                f"  Adversarial confidence (class {adv_predicted_label}): {adv_confidence:.4f}"
            )
            print(
                f"  Highest confidence (class {highest_conf_class}): {highest_conf_value:.4f}"
            )

        # Save the confidence information to a text file
        with open(f"{save_folder}/masks/confidence_small_{i}.txt", "w") as f:
            f.write(f"Mask {i} (small epsilon):\n")
            f.write(f"  Natural confidence (class {label}): {natural_confidence:.4f}\n")
            f.write(
                f"  Adversarial confidence (class {adv_predicted_label}): {adv_confidence:.4f}\n"
            )
            f.write(
                f"  Highest confidence (class {highest_conf_class}): {highest_conf_value:.4f}\n"
            )
        # Now save the mask superimposed on the original image
        superimposed_image = original_tensor + mask_tensor
        vutils.save_image(
            superimposed_image,
            f"{save_folder}/superimposed/superimposed_small_{i}.png",
            normalize=True,
        )
        torch.save(
            superimposed_image, f"{save_folder}/superimposed/superimposed_small_{i}.pt"
        )

        # Get confidence for the superimposed image
        with torch.no_grad():
            superimposed_input = superimposed_image.unsqueeze(0).cuda()
            superimposed_output, _ = model(superimposed_input)
            superimposed_probs = F.softmax(superimposed_output, dim=1)

            # Get the confidence for the true label and adversarial label
            superimposed_natural_confidence = superimposed_probs[0, label].item()
            superimposed_adv_confidence = superimposed_probs[
                0, adv_predicted_label
            ].item()

            # Get the highest confidence class
            superimposed_highest_conf_class = torch.argmax(
                superimposed_probs, dim=1
            ).item()
            superimposed_highest_conf_value = superimposed_probs[
                0, superimposed_highest_conf_class
            ].item()

            # Save the confidence values
            superimposed_confidence_data = {
                "natural_confidence": superimposed_natural_confidence,
                "adv_confidence": superimposed_adv_confidence,
                "highest_conf_class": superimposed_highest_conf_class,
                "highest_conf_value": superimposed_highest_conf_value,
                "label": label,
                "adv_predicted_label": adv_predicted_label,
            }

            # Save as a tensor for consistency
            torch.save(
                superimposed_confidence_data,
                f"{save_folder}/superimposed/confidence_small_{i}.pt",
            )

            print(f"Superimposed image {i} (small epsilon):")
            print(
                f"  Natural confidence (class {label}): {superimposed_natural_confidence:.4f}"
            )
            print(
                f"  Adversarial confidence (class {adv_predicted_label}): {superimposed_adv_confidence:.4f}"
            )
            print(
                f"  Highest confidence (class {superimposed_highest_conf_class}): {superimposed_highest_conf_value:.4f}"
            )

        # Save the superimposed confidence information to a text file
        with open(f"{save_folder}/superimposed/confidence_small_{i}.txt", "w") as f:
            f.write(f"Superimposed image {i} (small epsilon):\n")
            f.write(
                f"  Natural confidence (class {label}): {superimposed_natural_confidence:.4f}\n"
            )
            f.write(
                f"  Adversarial confidence (class {adv_predicted_label}): {superimposed_adv_confidence:.4f}\n"
            )
            f.write(
                f"  Highest confidence (class {superimposed_highest_conf_class}): {superimposed_highest_conf_value:.4f}\n"
            )

        # Now save the mask superimposed on a random image
        random_original_tensor = original_tensors_small[-i]
        random_superimposed_image = random_original_tensor + mask_tensor

        # Save the random superimposed image
        vutils.save_image(
            random_superimposed_image,
            f"{save_folder}/random_superimposed/random_superimposed_small_{i}.png",
            normalize=True,
        )
        torch.save(
            random_superimposed_image,
            f"{save_folder}/random_superimposed/random_superimposed_small_{i}.pt",
        )

        # Get confidence for the random superimposed image
        with torch.no_grad():
            random_superimposed_input = random_superimposed_image.unsqueeze(0).cuda()
            random_superimposed_output, _ = model(random_superimposed_input)
            random_superimposed_probs = F.softmax(random_superimposed_output, dim=1)

            # Get the highest confidence class
            random_highest_conf_class = torch.argmax(
                random_superimposed_probs, dim=1
            ).item()
            random_highest_conf_value = random_superimposed_probs[
                0, random_highest_conf_class
            ].item()

            # Get the confidence for the adversarial label
            random_original_label = labels_small[-i]
            random_natural_confidence = random_superimposed_probs[
                0, random_original_label
            ].item()
            random_adv_confidence = random_superimposed_probs[
                0, adv_predicted_label
            ].item()

            # Save the confidence values
            random_superimposed_confidence_data = {
                "highest_conf_class": random_highest_conf_class,
                "highest_conf_value": random_highest_conf_value,
                "adv_confidence": random_adv_confidence,
                "adv_predicted_label": adv_predicted_label,
                "natural_confidence": random_natural_confidence,
                "label": random_original_label,
                "adv_label": adv_predicted_label,
            }

            # Save as a tensor for consistency
            torch.save(
                random_superimposed_confidence_data,
                f"{save_folder}/random_superimposed/confidence_small_{i}.pt",
            )

            print(f"Random superimposed image {i} (small epsilon):")
            print(
                f"  Highest confidence (class {random_highest_conf_class}): {random_highest_conf_value:.4f}"
            )
            print(
                f"  Adversarial confidence (class {adv_predicted_label}): {random_adv_confidence:.4f}"
            )
            print(
                f"  Natural confidence (class {random_original_label}): {random_natural_confidence:.4f}"
            )

        # Save the random superimposed confidence information to a text file
        with open(
            f"{save_folder}/random_superimposed/confidence_small_{i}.txt", "w"
        ) as f:
            f.write(f"Random superimposed image {i} (small epsilon):\n")
            f.write(
                f"  Natural confidence (class {random_original_label}): {random_natural_confidence:.4f}\n"
            )
            f.write(
                f"  Adversarial confidence (class {adv_predicted_label}): {random_adv_confidence:.4f}\n"
            )
            f.write(f"  Adversarial label: {adv_predicted_label}\n")
            f.write(
                f"  Highest confidence (class {random_highest_conf_class}): {random_highest_conf_value:.4f}\n"
            )


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
                if file.endswith(".txt") and "large" in file:
                    with open(os.path.join(subdir_path, file), "r") as f:
                        total_count_large += 1
                        content = f.read()
                        # Extract natural class, adversarial class, and highest confidence class
                        natural_class = int(
                            re.search(
                                r"Natural confidence \(class (\d+)\)", content
                            ).group(1)
                        )
                        adv_class = int(
                            re.search(
                                r"Adversarial confidence \(class (\d+)\)", content
                            ).group(1)
                        )
                        highest_class = int(
                            re.search(
                                r"Highest confidence \(class (\d+)\)", content
                            ).group(1)
                        )
                        # Count matches
                        if highest_class == natural_class:
                            highest_natural_count_large += 1
                        elif highest_class == adv_class:
                            highest_adv_count_large += 1
                        else:
                            disagree_count_large += 1

            # write a textfile in this subdirectory with the results
            with open(
                os.path.join(subdir_path, "analysis_results_large.txt"), "w"
            ) as f:
                f.write(f"Analysis of large epsilon masks\n")
                f.write(f"  Total masks analyzed: {total_count_large}\n")
                f.write(
                    f"  Masks where highest confidence class matches natural class: {highest_natural_count_large} ({highest_natural_count_large/total_count_large*100:.2f}%)\n"
                )
                f.write(
                    f"  Masks where highest confidence class matches adversarial class: {highest_adv_count_large} ({highest_adv_count_large/total_count_large*100:.2f}%)\n"
                )
                f.write(
                    f"  Masks where highest confidence class matches neither: {disagree_count_large} ({disagree_count_large/total_count_large*100:.2f}%)\n\n"
                )

            # now do the same for the small epsilon files
            highest_natural_count_small = 0
            highest_adv_count_small = 0
            disagree_count_small = 0
            total_count_small = 0
            for file in os.listdir(subdir_path):
                if file.endswith(".txt") and "small" in file:
                    with open(os.path.join(subdir_path, file), "r") as f:
                        total_count_small += 1
                        content = f.read()
                        # Extract natural class, adversarial class, and highest confidence class
                        natural_class = int(
                            re.search(
                                r"Natural confidence \(class (\d+)\)", content
                            ).group(1)
                        )
                        adv_class = int(
                            re.search(
                                r"Adversarial confidence \(class (\d+)\)", content
                            ).group(1)
                        )
                        highest_class = int(
                            re.search(
                                r"Highest confidence \(class (\d+)\)", content
                            ).group(1)
                        )
                        # Count matches
                        if highest_class == natural_class:
                            highest_natural_count_small += 1
                        elif highest_class == adv_class:
                            highest_adv_count_small += 1
                        else:
                            disagree_count_small += 1

            # write a textfile in this subdirectory with the results
            with open(
                os.path.join(subdir_path, "analysis_results_small.txt"), "w"
            ) as f:
                f.write(f"Analysis of small epsilon masks\n")
                f.write(f"  Total masks analyzed: {total_count_small}\n")
                f.write(
                    f"  Masks where highest confidence class matches natural class: {highest_natural_count_small} ({highest_natural_count_small/total_count_small*100:.2f}%)\n"
                )
                f.write(
                    f"  Masks where highest confidence class matches adversarial class: {highest_adv_count_small} ({highest_adv_count_small/total_count_small*100:.2f}%)\n"
                )
                f.write(
                    f"  Masks where highest confidence class matches neither: {disagree_count_small} ({disagree_count_small/total_count_small*100:.2f}%)\n\n"
                )

    print("Done analyzing mask superimposed experiment")


# NOTE: This function will be very similar to `run_mask_visualization_experiment`, but it will not be saving images and will use another way to test accuracy
def run_mask_superimposed_random_experiment(
    model_path,
    save_folder,
    loader,
    attack_config=ATTACK_PARAMS,
    privacy_allocation=0.5,
    verbose=False,
    use_gaussian=False,
    mu=None,
    sigma=None,
    save_images=False,
):
    """
    NOTE: REQUIRES TO BE RUN ON GPU
    Will run the mask superimposed experiment on a random set of images from the test set. It will be done in this exact manner.
    `loader` can be either the test or train loader
    `attack_config` is the attack config to use for the attack
    `privacy_allocation` is the privacy allocation to use for the attack (epsilon), the higher the epsilon, the less noise, another tuning parameter
    `save_folder` is the folder to save the results to, in a text file
    `model_path` is the path to the model to use for the attack

    If `use_gaussian` is True, then the mask will be generated using a Gaussian distribution with mean `mu` and standard deviation `sigma`.
    If `save_images` is True, then some (few) images will be saved to the save_folder to visualize the added gaussian noise (not much else)

    Experiment Description: Will do four different tasks.
    a. Create a mask on the entire loader's images, then calculate the accuracy of the model on those masks, both the natural, adversarial, and disgreement accuracy.
        Note: Disagreement accuracy is the accuracy of the model on the masks where the highest confidence class does not match the natural or adversarial class
    b. Superimpose the mask on the image in the loader it was generated from, then do the same as above
    c. Superimpose the mask on a random image from the loader, then do the same as above
    d. Superimpose a random mask (generated by the differential privacy noise injection framework) on a natural image, then do so the same as above
    """
    # load model and dataset
    dataset = CIFAR("/home/gridsan/hmartinez/distribution-shift/datasets")
    model, _ = model_utils.make_and_restore_model(
        arch="resnet18", dataset=dataset, resume_path=model_path
    )
    model.eval()

    # create masks
    (
        original_images_list,
        adv_images_list,
        masks_list,
        labels_list,
        adv_predicted_labels_list,
    ) = ([], [], [], [], [])
    for images, labels in loader:
        original_images, adv_images, masks, _, labels, adv_predicted_labels = (
            mask_gen.create_masks_batch(attack_config, model_path, images, labels)
        )
        original_images_list.append(original_images)
        adv_images_list.append(adv_images)
        masks_list.append(masks)
        labels_list.append(labels)
        adv_predicted_labels_list.append(adv_predicted_labels)
    # keep as list and move all to cuda
    original_images = [tensor.cuda() for tensor in original_images_list]
    adv_images = [tensor.cuda() for tensor in adv_images_list]
    masks = [tensor.cuda() for tensor in masks_list]
    labels = [tensor.cuda() for tensor in labels_list]
    adv_predicted_labels = [tensor.cuda() for tensor in adv_predicted_labels_list]
    model.to("cuda")

    if verbose:
        print("Sanity Check -- Shapes of the tensors:")
        print(f"Original images len: {len(original_images)}")
        print(f"Adversarial images len: {len(adv_images)}")
        print(f"Masks len: {len(masks)}")
        print(f"Labels len: {len(labels)}")
        print(f"Adversarial predicted labels len: {len(adv_predicted_labels)}")

    # now to perform experiment a as per the docstring
    if verbose:
        print("Starting experiment (a)")
    correct_adversarial = 0
    correct_natural = 0
    total = 0
    disagreement_count = 0
    # iterate through the batches
    for i in range(len(original_images)):
        # get the natural, adversarial, and disagreement accuracy
        outputs, _ = model(masks[i])
        _, predicted = outputs.max(1)
        correct_adversarial += predicted.eq(adv_predicted_labels[i]).sum().item()
        total += labels[i].size(0)
        correct_natural += predicted.eq(labels[i]).sum().item()
        disagreement_count += (
            labels[i].size(0)
            - predicted.eq(adv_predicted_labels[i]).sum().item()
            - predicted.eq(labels[i]).sum().item()
        )  # the rest are disagreements

    # write the results to the text file
    # Calculate accuracy percentages
    accuracy_natural = 100 * correct_natural / total
    accuracy_adversarial = 100 * correct_adversarial / total
    disagreement_percentage = 100 * disagreement_count / total
    os.makedirs(save_folder, exist_ok=True)
    # Write results to a text file
    with open(f"{save_folder}/results_masks.txt", "w") as f:
        f.write(f"Experiment (a) Results\n")
        f.write(
            "Taking the three accuracies from the simple masks, no superimposition\n"
        )
        f.write(f"=================\n\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Natural accuracy: {accuracy_natural:.2f}%\n")
        f.write(f"Adversarial accuracy: {accuracy_adversarial:.2f}%\n")
        f.write(f"Disagreement percentage: {disagreement_percentage:.2f}%\n")
        f.write(f"\nAttack configuration:\n")
        for key, value in attack_config.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nPrivacy allocation: {privacy_allocation}\n")
        f.write(f"Model path: {model_path}\n")
        if len(original_images) > 100:
            f.write(f"Used training set for experiment\n")
        else:
            f.write(f"Used test set for experiment\n")
        f.write(f"==============================================\n\n")

    if save_images:
        # save the images to the save_folder
        os.makedirs(f"{save_folder}/images", exist_ok=True)
        for i in range(3):
            # save the original image
            vutils.save_image(
                original_images[0][i], f"{save_folder}/images/original_image_{i}.png"
            )
            vutils.save_image(
                adv_images[0][i], f"{save_folder}/images/adv_image_{i}.png"
            )
            vutils.save_image(masks[0][i], f"{save_folder}/images/mask_{i}.png")

    # now to perform experiment b as per the docstring
    if verbose:
        print("Starting experiment (b)")
    correct_adversarial = 0
    correct_natural = 0
    total = 0
    disagreement_count = 0
    # iterate through the batches
    for i in range(len(original_images)):
        # get the natural, adversarial, and disagreement accuracy
        outputs, _ = model(
            masks[i] + original_images[i]
        )  # superimpose the mask on the original image, non-random
        _, predicted = outputs.max(1)
        correct_adversarial += predicted.eq(adv_predicted_labels[i]).sum().item()
        total += labels[i].size(0)
        correct_natural += predicted.eq(labels[i]).sum().item()
        disagreement_count += (
            labels[i].size(0)
            - predicted.eq(adv_predicted_labels[i]).sum().item()
            - predicted.eq(labels[i]).sum().item()
        )  # the rest are disagreements

    # write the results to the text file
    # Calculate accuracy percentages
    accuracy_natural = 100 * correct_natural / total
    accuracy_adversarial = 100 * correct_adversarial / total
    disagreement_percentage = 100 * disagreement_count / total
    # Write results to a text file
    with open(f"{save_folder}/results_superimposed.txt", "w") as f:
        f.write(f"Experiment (b) Results\n")
        f.write(
            "Taking the three accuracies from the simple masks superimposed on the original image, non-random\n"
        )
        f.write(f"=================\n\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Natural accuracy: {accuracy_natural:.2f}%\n")
        f.write(f"Adversarial accuracy: {accuracy_adversarial:.2f}%\n")
        f.write(f"Disagreement percentage: {disagreement_percentage:.2f}%\n")
        f.write(f"==============================================\n\n")

    # now to perform experiment c as per the docstring
    if verbose:
        print("Starting experiment (c)")
    correct_adversarial = 0
    correct_natural = 0
    total = 0
    disagreement_count = 0
    # iterate through the batches
    for i in range(len(original_images)):
        # get the random image from the loader
        random_batch = original_images[random.randint(0, len(original_images) - 1)]
        random_image = random_batch[random.randint(0, len(random_batch) - 1)]
        random_image.cuda()
        # get the natural, adversarial, and disagreement accuracy
        outputs, _ = model(
            masks[i] + random_image
        )  # superimpose the mask on the adversarial image, random
        if save_images and i < 3:
            # save the random image
            vutils.save_image(
                outputs[0], f"{save_folder}/images/random_superimposed_image_{i}.png"
            )
        _, predicted = outputs.max(1)
        correct_adversarial += predicted.eq(adv_predicted_labels[i]).sum().item()
        total += labels[i].size(0)
        correct_natural += predicted.eq(labels[i]).sum().item()
        disagreement_count += (
            labels[i].size(0)
            - predicted.eq(adv_predicted_labels[i]).sum().item()
            - predicted.eq(labels[i]).sum().item()
        )  # the rest are disagreements
    # write the results to the text file
    # Calculate accuracy percentages
    accuracy_natural = 100 * correct_natural / total
    accuracy_adversarial = 100 * correct_adversarial / total
    disagreement_percentage = 100 * disagreement_count / total
    # Write results to a text file
    with open(f"{save_folder}/results_superimposed_random.txt", "w") as f:
        f.write(f"Experiment (c) Results\n")
        f.write(
            "Taking the three accuracies from the simple masks superimposed on a random image from the loader\n"
        )
        f.write(
            "NOTE: The natural, adversarial, and disagreement accuracies are relative to the adv/natural labels associated with the mask, not the random image\n"
        )
        f.write(f"=================\n\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Natural accuracy: {accuracy_natural:.2f}%\n")
        f.write(f"Adversarial accuracy: {accuracy_adversarial:.2f}%\n")
        f.write(f"Disagreement percentage: {disagreement_percentage:.2f}%\n")
        f.write(f"==============================================\n\n")

    # now to perform experiment d as per the docstring
    if verbose:
        print("Starting experiment (d)")
    correct_adversarial = 0
    correct_natural = 0
    total = 0
    disagreement_count = 0
    # iterate through the batches
    for i in range(len(original_images)):
        # create the random image using the differential privacy noise injection framework
        if use_gaussian:
            random_mask = utils.create_gaussian_noise(
                len(original_images[i]), mu, sigma
            )
        else:
            random_mask = utils.create_random_images(
                len(original_images[i]), epsilon=privacy_allocation
            )
        random_mask = random_mask.cuda()
        # get the natural, adversarial, and disagreement accuracy
        outputs, _ = model(torch.clamp(original_images[i] + random_mask, 0, 1))
        if save_images and i < 3:
            # save the random mask
            vutils.save_image(
                outputs[0],
                f"{save_folder}/images/random_noise_superimposed_image_{i}.png",
            )
            vutils.save_image(
                random_mask[0], f"{save_folder}/images/random_noise_mask_{i}.png"
            )
        _, predicted = outputs.max(1)
        correct_adversarial += predicted.eq(adv_predicted_labels[i]).sum().item()
        total += labels[i].size(0)
        correct_natural += predicted.eq(labels[i]).sum().item()
        disagreement_count += (
            labels[i].size(0)
            - predicted.eq(adv_predicted_labels[i]).sum().item()
            - predicted.eq(labels[i]).sum().item()
        )  # the rest are disagreements
    # write the results to the text file
    # Calculate accuracy percentages
    accuracy_natural = 100 * correct_natural / total
    accuracy_adversarial = 100 * correct_adversarial / total
    disagreement_percentage = 100 * disagreement_count / total
    # Write results to a text file
    with open(f"{save_folder}/results_superimposed_random_noise.txt", "w") as f:
        f.write(f"Experiment (d) Results\n")
        f.write(
            "Taking the three accuracies from random masks (generated by the differential privacy noise injection framework) superimposed on a natural image\n"
        )
        f.write(f"=================\n\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Natural accuracy: {accuracy_natural:.2f}%\n")
        f.write(f"Adversarial accuracy: {accuracy_adversarial:.2f}%\n")
        f.write(f"Disagreement percentage: {disagreement_percentage:.2f}%\n")
        if use_gaussian:
            f.write(f"Gaussian noise used\n")
            f.write(f"Gaussian noise mean: {mu}\n")
            f.write(f"Gaussian noise standard deviation: {sigma}\n")
        else:
            f.write(f"Privacy allocation: {privacy_allocation}\n")
        f.write(f"==============================================\n\n")

    print("Done with all experiments")
    return


def check_adversarial_labels(
    model_path,
    save_folder,
    loader,
    small_eps_config=NORMAL_ATTACK_PARAMS,
    large_eps_config=ATTACK_PARAMS,
    verbose=False,
):
    """
    This function will check the adversarial labels of the same images on two different attack configurations, and save
    them on a .pt file that saves the tensors for later comparisons. Will do it on `loader` images using
    `model_path` as the model to use.
    """
    if verbose:
        print(f"Checking adversarial labels for model at {model_path}")
        print(f"Small epsilon config: {small_eps_config}")
        print(f"Large epsilon config: {large_eps_config}")
        print(f"Save folder: {save_folder}")
        print(f"Len of loader: {len(loader)}")

    # load model
    dataset = CIFAR("/home/gridsan/hmartinez/distribution-shift/datasets")
    model, _ = model_utils.make_and_restore_model(
        arch="resnet18", dataset=dataset, resume_path=model_path
    )
    model.eval()
    model.to("cuda")

    # store adv_large and small labels
    adv_large_labels = []
    adv_small_labels = []
    total = 0
    disagree_count = 0
    for images, labels in loader:
        # get the adversarial images
        images = images.cuda()
        labels = labels.cuda()
        _, adv_images = model(images, labels, make_adv=True, **small_eps_config)
        _, adv_images_large = model(images, labels, make_adv=True, **large_eps_config)

        # get the adversarial labels
        adv_output, _ = model(adv_images)
        adv_predicted_labels = adv_output.argmax(dim=1).cpu()
        adv_output_large, _ = model(adv_images_large)
        adv_predicted_labels_large = adv_output_large.argmax(dim=1).cpu()

        adv_large_labels.append(adv_predicted_labels_large)
        adv_small_labels.append(adv_predicted_labels)

        # count the number of disagreements
        disagree_count += abs(
            labels.size(0)
            - adv_predicted_labels.eq(adv_predicted_labels_large).sum().item()
        )
        total += labels.size(0)

    # write the results to the text file
    with open(f"{save_folder}/results_adversarial_labels.txt", "w") as f:
        f.write(f"Model path: {model_path}\n")
        f.write(f"Small epsilon config: {small_eps_config}\n")
        f.write(f"Large epsilon config: {large_eps_config}\n")
        f.write(f"Len of loader: {len(loader)}\n")
        f.write("============================================= \n\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Disagreement percentage: {100 * disagree_count / total:.2f}%\n")
    if verbose:
        print(f"Total samples: {total}")
        print(f"Disagreement percentage: {100 * disagree_count / total:.2f}%")

    # Convert lists of tensors to single tensors
    adv_large_labels_tensor = torch.cat(adv_large_labels, dim=0)
    adv_small_labels_tensor = torch.cat(adv_small_labels, dim=0)

    # Save the tensors to files
    torch.save(adv_large_labels_tensor, f"{save_folder}/adv_large_labels.pt")
    torch.save(adv_small_labels_tensor, f"{save_folder}/adv_small_labels.pt")

    if verbose:
        print(
            f"Saved adversarial labels to {save_folder}/adv_large_labels.pt and {save_folder}/adv_small_labels.pt"
        )
        print(f"Large labels tensor shape: {adv_large_labels_tensor.shape}")
        print(f"Small labels tensor shape: {adv_small_labels_tensor.shape}")

    return disagree_count / total


def visualize_masks_and_natural_images(
    model_path,
    data_path,
    save_folder,
    loader,
    N=3,
    low_epsilon_config=NORMAL_ATTACK_PARAMS,
    high_epsilon_config=ATTACK_PARAMS,
):
    """
    This function will visualize the masks and natural images of the first N images in the loader.
    And save them to the save_folder. Will save low and high epsilon masks, natural images, and their superimposed versions.
    N must be less than 128 (batch size)
    """
    # load model
    # dpath = "/u/yshi23/distribution-shift/adversarial/datasets"
    # dataset = CIFAR('/home/gridsan/hmartinez/distribution-shift/datasets')
    dataset = CIFAR(data_path)
    model, _ = model_utils.make_and_restore_model(
        arch="resnet18", dataset=dataset, resume_path=model_path
    )
    model.eval()
    model.to("cuda")

    for images, labels in loader:
        # get the adversarial images
        images = images.cuda()
        labels = labels.cuda()
        _, adv_images = model(images, labels, make_adv=True, **low_epsilon_config)
        _, adv_images_large = model(
            images, labels, make_adv=True, **high_epsilon_config
        )
        break  # only do the first batch

    images = images[:N]
    adv_images = adv_images[:N]
    adv_images_large = adv_images_large[:N]
    labels = labels[:N]

    # calculate the masks
    masks_large = adv_images_large - images
    masks_small = adv_images - images

    # get the labels for all the images and masks
    adv_output, _ = model(adv_images)
    adv_predicted_labels = adv_output.argmax(dim=1).cpu()
    adv_output_large, _ = model(adv_images_large)
    adv_predicted_labels_large = adv_output_large.argmax(dim=1).cpu()
    mask_output, _ = model(masks_small)
    mask_predicted_labels = mask_output.argmax(dim=1).cpu()
    mask_output_large, _ = model(masks_large)
    mask_predicted_labels_large = mask_output_large.argmax(dim=1).cpu()

    labels = labels.cpu()

    # save the images
    os.makedirs(save_folder, exist_ok=True)
    for i in range(N):
        # save the natural image
        vutils.save_image(images[i], f"{save_folder}/natural_image_{i}.png")
        vutils.save_image(
            adv_images[i], f"{save_folder}/adv_image_small_epsilon_{i}.png"
        )
        vutils.save_image(
            adv_images_large[i], f"{save_folder}/adv_image_large_epsilon_{i}.png"
        )
        vutils.save_image(masks_small[i], f"{save_folder}/mask_small_epsilon_{i}.png")
        vutils.save_image(masks_large[i], f"{save_folder}/mask_large_epsilon_{i}.png")

    # save the labels to a text file
    with open(f"{save_folder}/labels.txt", "w") as f:
        f.write(f"Model path: {model_path}\n")
        f.write(f"Low epsilon config: {low_epsilon_config}\n")
        f.write(f"High epsilon config: {high_epsilon_config}\n")
        f.write(f"Natural labels: {labels}\n")
        f.write(f"Adversarial labels (small epsilon): {adv_predicted_labels}\n")
        f.write(f"Adversarial labels (large epsilon): {adv_predicted_labels_large}\n")
        f.write(f"Mask labels (small epsilon): {mask_predicted_labels}\n")
        f.write(f"Mask labels (large epsilon): {mask_predicted_labels_large}\n")

    print(f"Saved images to {save_folder}")
    return


if __name__ == "__main__":
    MODEL_PATH = "/u/yshi23/distribution-shift/adversarial/out/checkpoint.pt.best"
    SAVE_PATH = (
        "/u/yshi23/distribution-shift/adversarial/visualizations/phoebe_visualizations"
    )
    DATA_PATH = "/u/yshi23/distribution-shift/adversarial/datasets"
    # EXPERIMENT_FOLDER = "/home/gridsan/hmartinez/distribution-shift/adversarial/visualizations/mask_superimposed/experiment_7_randomNoiseMasks_test_set"
    # # Get the training loader
    _, train_loader, test_loader = train.load_dataset(DATA_PATH)
    # run_mask_visualization_experiment(MODEL_PATH, SAVE_PATH, test_loader, N=100, random_images=True)
    # # run_mask_training_experiment(MODEL_PATH, SAVE_PATH)
    # # run_random_noise_mask_confidence_experiment(MODEL_PATH)
    # analyze_mask_superimposed_experiment(EXPERIMENT_FOLDER)
    # run_mask_superimposed_random_experiment(MODEL_PATH, SAVE_PATH, test_loader, attack_config=ATTACK_PARAMS, privacy_allocation=10, verbose=True, use_gaussian=True, mu=0, sigma=0.1)
    # check_adversarial_labels(MODEL_PATH, SAVE_PATH, test_loader, small_eps_config=NORMAL_ATTACK_PARAMS, large_eps_config=ATTACK_PARAMS, verbose=True)
    visualize_masks_and_natural_images(
        MODEL_PATH,
        DATA_PATH,
        SAVE_PATH,
        test_loader,
        N=10,
        low_epsilon_config=NORMAL_ATTACK_PARAMS,
        high_epsilon_config=ATTACK_PARAMS,
    )
