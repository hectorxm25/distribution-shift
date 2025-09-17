#  exp where channel patching across all model layers is performed for a given image,
# its adversarial perturbation, and an image in the adversarial class that is most similar to it

import torch
import os
import numpy as np
from nnsight import NNsight
from matplotlib import pyplot as plt

from utils import (
    Config,
    load_model,
    load_all_perturbations,
    compute_channel_importance,
    plot_channel_importance,
    get_layer_names,
    find_closest_image,
)

if __name__ == "__main__":
    # load all dataloaders
    config = Config(results_dir="results/temp/")  # specify results directory
    model = load_model(config)
    nnsight_model = NNsight(model)
    dataloaders = load_all_perturbations(config)  # dictionary of dataloaders
    print(f"Loaded datasets: {list(dataloaders.keys())}")

    # just look at baseline and linf_high for now
    baseline_loader, linf_high_loader = (
        dataloaders["baseline"],
        dataloaders["linf_high_dataset"],
    )

    # Get first image from each batch
    baseline_batch_imgs, baseline_batch_labels = next(iter(baseline_loader))
    linf_high_batch_imgs, linf_high_batch_labels = next(iter(linf_high_loader))

    # Extract first image and label from each batch
    baseline_image = baseline_batch_imgs[0:1]  # Keep batch dimension for model
    linf_high_image = linf_high_batch_imgs[0:1]  # Keep batch dimension for model

    # Handle labels - check if they are one-hot encoded or scalar
    # Check if labels are one-hot encoded (size > 1) or scalar
    if baseline_batch_labels[0].numel() > 1:  # One-hot encoded
        baseline_true_label = torch.argmax(baseline_batch_labels[0]).item()
    else:  # Scalar labels
        baseline_true_label = baseline_batch_labels[0].item()

    if linf_high_batch_labels[0].numel() > 1:  # One-hot encoded
        linf_high_true_label = torch.argmax(linf_high_batch_labels[0]).item()
    else:  # Scalar labels
        linf_high_true_label = linf_high_batch_labels[0].item()

    # save baseline and linf_high images for sanity check
    baseline_image_path = os.path.join(config.results_dir, "baseline_image.png")
    linf_high_image_path = os.path.join(config.results_dir, "linf_high_image.png")

    # Convert images to numpy and transpose, then normalize to 0-1 range
    baseline_img_save = baseline_image[0].cpu().numpy().transpose(1, 2, 0)
    linf_high_img_save = linf_high_image[0].cpu().numpy().transpose(1, 2, 0)

    # Normalize to 0-1 range using min-max normalization
    baseline_img_save = (baseline_img_save - baseline_img_save.min()) / (
        baseline_img_save.max() - baseline_img_save.min()
    )
    linf_high_img_save = (linf_high_img_save - linf_high_img_save.min()) / (
        linf_high_img_save.max() - linf_high_img_save.min()
    )

    plt.imsave(baseline_image_path, baseline_img_save)
    plt.imsave(linf_high_image_path, linf_high_img_save)

    # find closest image in the baseline dataset to the linf_high image, with the linf_high predicted label
    closest_img, closest_lab, ssim, index = find_closest_image(
        linf_high_image[0], linf_high_true_label, baseline_loader
    )

    print(
        f"Closest image found at index {index} with SSIM: {ssim:.4f}"
        f" and label {closest_lab}"
    )
    # save closest image
    closest_image_path = os.path.join(
        config.results_dir, f"closest_image_{closest_lab}.png"
    )
    closest_img_np = closest_img.cpu().numpy().transpose(1, 2, 0)
    closest_img_np = (closest_img_np - closest_img_np.min()) / (
        closest_img_np.max() - closest_img_np.min()
    )
    plt.imsave(closest_image_path, closest_img_np)
