"""quick script to verify the population statistics of the adversarial dataset"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomCrop

from resnet import resnet18
from robustness import attacker

# local imports
import mask_generation as mask_gen

HIGH_EPS = {
    "constraint": "inf",  # Use L2 PGD attack
    "eps": 0.031 * 25,  # large epsilon
    "step_size": 0.1,  # large step size
    "iterations": 10,  # standard iterations
    "random_start": False,  # standard random start
}
# standard PGD attack
LOW_EPS = {
    "constraint": "inf",  # Use L2 PGD attack
    "eps": 0.031,  # small epsilon
    "step_size": 0.01,  # small step size
    "iterations": 10,  # standard iterations
    "random_start": False,  # standard random start
}


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        """Init with empty data and labels."""
        self.data = torch.empty((0, 3, 32, 32), dtype=torch.float32)
        self.labels = torch.empty((0,), dtype=torch.long)

    def append(self, data, labels):
        self.data = torch.cat((self.data, data), dim=0)
        self.labels = torch.cat((self.labels, labels), dim=0)

    def save(self, path):
        """dump data and labels to path"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.data, os.path.join(path, "data.pt"))
        torch.save(self.labels, os.path.join(path, "labels.pt"))


class DummyCifar10(torch.utils.data.Dataset):
    """Dummy dataset to get mean and std of CIFAR-10."""

    def __init__(self):
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.2023, 0.1994, 0.2010])


def get_cifar10_dataloaders(batch_size=128, val_split=0.1, seed=42):
    """
    Get CIFAR-10 dataloaders for training, validation, and testing.

    Args:
        batch_size (int): Batch size for dataloaders.
        val_split (float): Fraction of training set to use as validation.
        seed (int): Random seed for reproducibility.

    Returns:
        train_loader (DataLoader): Dataloader for training subset.
        val_loader (DataLoader): Dataloader for validation subset.
        test_loader (DataLoader): Dataloader for test set.
    """
    # Transforms
    transform_train = Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = Compose(
        [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Datasets
    full_train_dataset = CIFAR10(
        root="data", train=True, download=True, transform=transform_train
    )
    test_dataset = CIFAR10(
        root="data", train=False, download=True, transform=transform_test
    )

    # Create train/val split
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))

    np.random.seed(seed)
    np.random.shuffle(indices)
    val_indices, train_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        full_train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2
    )
    val_loader = DataLoader(
        full_train_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader


def train_model(train_loader, val_loader, epochs=30):
    """Train ResNet-18 on CIFAR-10"""

    # Model
    model = resnet18(num_classes=10)
    model = model.to("cuda")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs, targets = inputs.to("cuda"), targets.to("cuda")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        train_acc = correct / total
        test_acc = evaluate(model, val_loader)
        scheduler.step()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                model.state_dict(), "verify/best_model.pth"
            )  # Uncomment to save best model

        print(
            f"Epoch [{epoch}/50] | Loss: {running_loss / len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
        )

        # Save model checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"verify/model_epoch_{epoch}.pth")
            print(f"Model saved at epoch {epoch}")
    # Save final model
    torch.save(model.state_dict(), "verify/final_model.pth")
    # return model
    return model


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return correct / total


def test_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return correct / total


def main():
    # Load CIFAR-10 dataloaders
    train_loader, val_loader, test_loader = get_cifar10_dataloaders()

    # Train the model
    model = train_model(train_loader, val_loader)

    # Load the best model (if saved)
    # model.load_state_dict(torch.load("best_model.pth"))

    # Evaluate the model on the test set
    test_acc = test_model(model, test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")


def kl_divergence(p_logits, q_logits):
    p = F.softmax(p_logits, dim=1)
    q = F.softmax(q_logits, dim=1)
    return F.kl_div(p.log(), q, reduction="none").sum(dim=1)  # (N,)


def tv_distance(p_logits, q_logits):
    p = F.softmax(p_logits, dim=1)
    q = F.softmax(q_logits, dim=1)
    return 0.5 * (p - q).abs().sum(dim=1)  # (N,)


def flip_rate(p_logits, q_logits):
    p_pred = p_logits.argmax(dim=1)
    q_pred = q_logits.argmax(dim=1)
    return (p_pred != q_pred).float().mean()


def confidence_drop(p_logits, q_logits):
    p_conf = F.softmax(p_logits, dim=1).max(dim=1).values
    q_conf = F.softmax(q_logits, dim=1).max(dim=1).values
    return p_conf - q_conf


def generate_adversarial_examples(device="cuda", save_path="path_to_save"):

    # Load datasets
    trainset = CIFAR10(root="data", train=True, download=True, transform=ToTensor())
    testset = CIFAR10(root="data", train=False, download=True, transform=ToTensor())
    combined_dataset = torch.utils.data.ConcatDataset([trainset, testset])
    dataloader = DataLoader(
        combined_dataset, batch_size=128, shuffle=False, num_workers=2
    )

    # Load model
    model = resnet18(num_classes=10)
    model.load_state_dict(torch.load("verify/best_model.pth"))
    model.to(device)
    model.eval()

    dummy_cifar = DummyCifar10()
    attack_model = attacker.AttackerModel(model, dummy_cifar)
    attack_model.to(device)

    # Full input tensor
    full_inputs = torch.cat([x[0].unsqueeze(0) for x in combined_dataset], dim=0)

    # Define modes
    modes = {
        "normal": {"dataset_low": CustomDataset(), "dataset_high": CustomDataset()},
        "mask": {"dataset_low": CustomDataset(), "dataset_high": CustomDataset()},
        "mask_plus_random": {
            "dataset_low": CustomDataset(),
            "dataset_high": CustomDataset(),
        },
    }

    stats = {  # Initialize stats dictionaries
        mode: {
            stat: []
            for stat in [
                "kl_low",
                "kl_high",
                "tvd_low",
                "tvd_high",
                "flip_low",
                "flip_high",
                "conf_drop_low",
                "conf_drop_high",
            ]
        }
        for mode in modes
    }

    # Helper function
    def compute_stats(orig_logits, new_logits):
        return {
            "kl": kl_divergence(orig_logits, new_logits),
            "tvd": tv_distance(orig_logits, new_logits),
            "flip": flip_rate(orig_logits, new_logits),
            "conf_drop": confidence_drop(orig_logits, new_logits),
        }

    for inputs, labels in tqdm(dataloader, desc="Generating adversarial examples"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            orig_logits = model(inputs).detach().cpu()

        # Low and high adversarial examples
        low_logits, low_inputs = attack_model(inputs, labels, make_adv=True, **LOW_EPS)
        high_logits, high_inputs = attack_model(
            inputs, labels, make_adv=True, **HIGH_EPS
        )

        low_logits = low_logits.detach().cpu()
        high_logits = high_logits.detach().cpu()
        low_inputs = low_inputs.detach().cpu()
        high_inputs = high_inputs.detach().cpu()
        inputs_cpu = inputs.detach().cpu()

        # NORMAL mode
        for eps_inputs, eps_logits, eps in [
            (low_inputs, low_logits, "low"),
            (high_inputs, high_logits, "high"),
        ]:
            modes["normal"][f"dataset_{eps}"].append(
                eps_inputs, eps_logits.softmax(dim=1)
            )
            s = compute_stats(orig_logits, eps_logits)
            stats["normal"][f"kl_{eps}"].append(s["kl"])
            stats["normal"][f"tvd_{eps}"].append(s["tvd"])
            stats["normal"][f"flip_{eps}"].append(s["flip"])
            stats["normal"][f"conf_drop_{eps}"].append(s["conf_drop"])

        # MASK mode
        mask_low = torch.clamp(low_inputs - inputs_cpu, -1.0, 1.0)
        mask_high = torch.clamp(high_inputs - inputs_cpu, -1.0, 1.0)

        for mask_inputs, eps_logits, eps in [
            (mask_low, low_logits, "low"),
            (mask_high, high_logits, "high"),
        ]:
            modes["mask"][f"dataset_{eps}"].append(
                mask_inputs, eps_logits.softmax(dim=1)
            )
            mask_logits = model(mask_inputs.to(device)).detach().cpu()
            s = compute_stats(orig_logits, mask_logits)
            stats["mask"][f"kl_{eps}"].append(s["kl"])
            stats["mask"][f"tvd_{eps}"].append(s["tvd"])
            stats["mask"][f"flip_{eps}"].append(s["flip"])
            stats["mask"][f"conf_drop_{eps}"].append(s["conf_drop"])

        # MASK+RANDOM mode
        rand_indices = torch.randint(0, full_inputs.size(0), (inputs.size(0),))
        random_images = full_inputs[rand_indices]

        mask_plus_random_low = torch.clamp(mask_low + random_images, 0.0, 1.0)
        mask_plus_random_high = torch.clamp(mask_high + random_images, 0.0, 1.0)

        for mpr_inputs, eps_logits, eps in [
            (mask_plus_random_low, low_logits, "low"),
            (mask_plus_random_high, high_logits, "high"),
        ]:
            modes["mask_plus_random"][f"dataset_{eps}"].append(
                mpr_inputs, eps_logits.softmax(dim=1)
            )
            mpr_logits = model(mpr_inputs.to(device)).detach().cpu()
            s = compute_stats(orig_logits, mpr_logits)
            stats["mask_plus_random"][f"kl_{eps}"].append(s["kl"])
            stats["mask_plus_random"][f"tvd_{eps}"].append(s["tvd"])
            stats["mask_plus_random"][f"flip_{eps}"].append(s["flip"])
            stats["mask_plus_random"][f"conf_drop_{eps}"].append(s["conf_drop"])

        # Clean up
        del inputs
        torch.cuda.empty_cache()

    # Save datasets
    for mode_name, datasets in modes.items():
        for eps in ["low", "high"]:
            datasets[f"dataset_{eps}"].save(f"{save_path}/{mode_name}_{eps}_dataset")

    # Format plotting data
    def cat(l):
        return torch.cat(l).numpy()

    plotting_data = {}
    for mode, mode_stats in stats.items():
        for stat_name, values in mode_stats.items():
            if "flip" in stat_name:
                plotting_data[f"{stat_name}_{mode}"] = np.array(values)
            else:
                plotting_data[f"{stat_name}_{mode}"] = cat(values)

    return plotting_data


if __name__ == "__main__":
    # main()
    plot_dat = generate_adversarial_examples()
    np.savez("path_to_save/plotting_data.npz", **plot_dat)
