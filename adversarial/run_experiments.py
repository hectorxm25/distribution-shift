from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR
import torch
import torch.nn.functional as F
import cox
from tqdm import tqdm
import random
import numpy as np
from TRADES.trades import trades_loss
import argparse
# local imports
import mask_generation as mask_gen
import test
import train

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
        'eps': 0.031*25,            # L2 radius/epsilon
        'step_size': 0.1,      # Changed from 'attack_lr' to 'step_size'
        'iterations': 10,      # Changed from 'attack_steps' to 'iterations'
        'random_start': False,  # Changed from 'random_restarts' to 'random_start'
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

if __name__ == "__main__":
    MODEL_PATH = "/home/gridsan/hmartinez/distribution-shift/models/natural/149_checkpoint.pt"
    SAVE_PATH = "/home/gridsan/hmartinez/distribution-shift/models/mask_trained_0.031*25eps_0.1stepsize"
    run_mask_training_experiment(MODEL_PATH, SAVE_PATH)
