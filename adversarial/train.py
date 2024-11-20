from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR
import torch
import cox

OUTPUT_DIR = '/home/gridsan/hmartinez/distribution-shift/models/adversarial'
NO_ADVERSARIAL_TRAINING_PARAMS = {
    'out_dir': OUTPUT_DIR,
    'adv_train': 0,  # Set to 1 for adversarial training
    'epochs': 150,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'step_lr': 50,
    'step_lr_gamma': 0.1,
}

ADVERSARIAL_TRAINING_PARAMS = {
    'out_dir': OUTPUT_DIR,
    'adv_train': 1,  # Set to 1 for adversarial training, #TODO: DO THIS AT 1
    'epochs': 150,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'step_lr': 50,
    'step_lr_gamma': 0.1,
}
# taken from https://robustness.readthedocs.io/en/latest/api/robustness.defaults.html

ATTACK_PARAMS = {
    'constraint': '2',      # Use L2 PGD attack
    'eps': 0.5,            # L2 radius/epsilon
    'attack_lr': 0.1,      # Step size for PGD
    'attack_steps': 7,     # Number of PGD steps
    'random_restarts': 0   # Number of random restarts
}

ADVERSARIAL_TRAINING_PARAMS = {
    'out_dir': OUTPUT_DIR,
    'adv_train': 1,        # Enable adversarial training
    'epochs': 150,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'step_lr': 50,
    'step_lr_gamma': 0.1,
    **ATTACK_PARAMS        # Add attack parameters
}

def train_no_adversarial(config):
    # reference local dataset
    # NOTE: if doing there is no `datasets/cifar-10-python.tar.gz` file, then make sure to run this on a node with download network access
    dataset = CIFAR('/home/gridsan/hmartinez/distribution-shift/datasets') # change this as you like
    print("successfully pulled dataset")

    # create the model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset) # apparently, this changes the resnet18 architecture so that it's compatible with CIFAR10
    print("successfully created barebones model arch")

    # set up train/val loaders
    train_loader, val_loader = dataset.make_loaders(batch_size=128, workers=8)
    print("successfully created data loaders")
    
    # set up training
    # train_args = defaults.check_and_fill_args(config, defaults.TRAINING_ARGS, ds_class="CIFAR")
    config = cox.utils.Parameters(config)
    train_args = defaults.check_and_fill_args(config, defaults.TRAINING_ARGS, ds_class="CIFAR")
    # train_args = defaults.check_and_fill_args(train_args, defaults.PGD_ARGS, ds_class="CIFAR")
    print("successfully created train_args, now proceeding to train the model")

    # train the model
    train.train_model(config, model, (train_loader, val_loader))

    return model

def train_adversarial(config):
    # NOTE: if doing there is no `datasets/cifar-10-python.tar.gz` file, then make sure to run this on a node with download network access
    dataset = CIFAR('/home/gridsan/hmartinez/distribution-shift/datasets') # change this as you like
    print("successfully pulled dataset")

    # create the model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset) # apparently, this changes the resnet18 architecture so that it's compatible with CIFAR10
    print("successfully created barebones model arch")

    # set up train/val loaders
    train_loader, val_loader = dataset.make_loaders(batch_size=128, workers=8)
    print("successfully created data loaders")

    # set up training with both standard and PGD attack parameters
    config = cox.utils.Parameters(config)
    train_args = defaults.check_and_fill_args(config, defaults.TRAINING_ARGS, ds_class="CIFAR")
    train_args = defaults.check_and_fill_args(train_args, defaults.PGD_ARGS, ds_class="CIFAR")
    print("successfully created train_args, now proceeding to train the model adversarially")

    # train the model
    model = train.train_model(train_args, model, (train_loader, val_loader))
    return model

# TODO: add a function that will train a model with the first half of the dataset
# naturally and the second half adversarially, each with an equal number of gradient
# updates per image.




if __name__ == "__main__":
    train_adversarial(ADVERSARIAL_TRAINING_PARAMS)