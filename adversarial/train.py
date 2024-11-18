from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR
import torch
import cox

OUTPUT_DIR = '/home/gridsan/hmartinez/distribution-shift/models'
NO_ADVERSARIAL_TRAINING_PARAMS = {
    'out_dir': OUTPUT_DIR,
    'adv_train': 0,  # Set to 1 for adversarial training, #TODO: DO THIS AT 1
    'epochs': 150,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'step_lr': 50,
    'step_lr_gamma': 0.1,
}
# taken from https://robustness.readthedocs.io/en/latest/api/robustness.defaults.html

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

    return None

if __name__ == "__main__":
    train_no_adversarial(NO_ADVERSARIAL_TRAINING_PARAMS)