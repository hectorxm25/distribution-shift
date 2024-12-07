from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR
import torch
import cox
from tqdm import tqdm
import random
import numpy as np
OUTPUT_DIR = '/home/gridsan/hmartinez/distribution-shift/models/twice_natural'
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

# taken from https://robustness.readthedocs.io/en/latest/api/robustness.defaults.html

ATTACK_PARAMS = {
    'constraint': '2',      # Use L2 PGD attack
    'eps': 0.5,            # L2 radius/epsilon
    'attack_lr': 0.1,      # Step size for PGD
    'attack_steps': 7,     # Number of PGD steps
    'random_restarts': 0   # Number of random restarts
}

NON_ATTACK_PARAMS = {
    'constraint': '2',      
    'eps': 0,            # has 0 eps, so no attack, just repeats natural training twice per image
    'attack_lr': 0.1,      
    'attack_steps': 7,     
    'random_restarts': 0   
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

TWICE_NATURAL_TRAINING_PARAMS = {
    'out_dir': OUTPUT_DIR,
    'adv_train': 1,        # Enable adversarial training
    'epochs': 150,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'step_lr': 50,
    'step_lr_gamma': 0.1,
    **NON_ATTACK_PARAMS        # Add (NON)-attack parameters
}

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

def train_twice_natural(config, train_loader, val_loader):

    dataset = CIFAR('/home/gridsan/hmartinez/distribution-shift/datasets') # change this as you like
    print("successfully pulled dataset")

    # create the model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset) # apparently, this changes the resnet18 architecture so that it's compatible with CIFAR10
    print("successfully created barebones model arch")
    
    config = cox.utils.Parameters(config)
    train_args = defaults.check_and_fill_args(config, defaults.TRAINING_ARGS, ds_class="CIFAR")
    train_args = defaults.check_and_fill_args(train_args, defaults.PGD_ARGS, ds_class="CIFAR")
    print("successfully created train_args, now proceeding to train the model")

    # train the model
    model = train.train_model(train_args, model, (train_loader, val_loader))
    return model

def train_no_adversarial(config, train_loader, val_loader):
    # reference local dataset
    # NOTE: if doing there is no `datasets/cifar-10-python.tar.gz` file, then make sure to run this on a node with download network access
    dataset = CIFAR('/home/gridsan/hmartinez/distribution-shift/datasets') # change this as you like
    print("successfully pulled dataset")

    # create the model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset) # apparently, this changes the resnet18 architecture so that it's compatible with CIFAR10
    print("successfully created barebones model arch")

    # set up training
    # train_args = defaults.check_and_fill_args(config, defaults.TRAINING_ARGS, ds_class="CIFAR")
    config = cox.utils.Parameters(config)
    train_args = defaults.check_and_fill_args(config, defaults.TRAINING_ARGS, ds_class="CIFAR")
    print("successfully created train_args, now proceeding to train the model")

    # train the model
    train.train_model(train_args, model, (train_loader, val_loader))

    return model

def train_adversarial(config, train_loader, val_loader):
    # NOTE: if doing there is no `datasets/cifar-10-python.tar.gz` file, then make sure to run this on a node with download network access
    dataset = CIFAR('/home/gridsan/hmartinez/distribution-shift/datasets') # change this as you like
    print("successfully pulled dataset")

    # create the model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset) # apparently, this changes the resnet18 architecture so that it's compatible with CIFAR10
    print("successfully created barebones model arch")

    # set up training with both standard and PGD attack parameters
    config = cox.utils.Parameters(config)
    train_args = defaults.check_and_fill_args(config, defaults.TRAINING_ARGS, ds_class="CIFAR")
    train_args = defaults.check_and_fill_args(train_args, defaults.PGD_ARGS, ds_class="CIFAR")
    print("successfully created train_args, now proceeding to train the model adversarially")

    # train the model
    model = train.train_model(train_args, model, (train_loader, val_loader))
    return model

# TODO: re-work this completely once dataset properly indexed
def train_hybrid(natural_config, adversarial_config, save_path):
    """
    Trains a model with first half of dataset naturally and second half adversarially.
    Each half gets equal number of gradient updates per image.
    """
    # Set up dataset
    dataset = CIFAR('/home/gridsan/hmartinez/distribution-shift/datasets')
    print("Successfully pulled dataset")

    # Create model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset)
    print("Successfully created model architecture")

    # Get train/val loaders
    torch.manual_seed(42) # for reproducibility
    train_loader, val_loader = dataset.make_loaders(batch_size=128, workers=8)
    
    # Split training and validation data in half
    train_data = list(train_loader.dataset)
    val_data = list(val_loader.dataset)
    
    n_train = len(train_data)
    n_val = len(val_data)
    
    # Create new datasets with first/second halves
    train_natural = torch.utils.data.DataLoader(
        train_data[:n_train//2], 
        batch_size=128,
        shuffle=True,
        num_workers=8
    )
    train_adversarial = torch.utils.data.DataLoader(
        train_data[n_train//2:],
        batch_size=128, 
        shuffle=True,
        num_workers=8
    )
    val_natural = torch.utils.data.DataLoader(
        val_data[:n_val//2],
        batch_size=128,
        shuffle=False,
        num_workers=8
    )
    val_adversarial = torch.utils.data.DataLoader(
        val_data[n_val//2:],
        batch_size=128,
        shuffle=False,
        num_workers=8
    )

    # Set up training parameters
    natural_config = cox.utils.Parameters(natural_config)
    adversarial_config = cox.utils.Parameters(adversarial_config)
    train_args = defaults.check_and_fill_args(natural_config, defaults.TRAINING_ARGS, ds_class="CIFAR")
    adv_args = defaults.check_and_fill_args(adversarial_config, defaults.PGD_ARGS, ds_class="CIFAR")

    # Training loop
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=train_args.lr, momentum=0.9, weight_decay=train_args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Starting hybrid training...")
    for epoch in range(train_args.epochs):
        # Natural training phase
        for images, labels in tqdm(train_natural, desc=f"Epoch {epoch} - Natural"):
            images, labels = images.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # does this twice because we want to do two gradient updates per image
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            
        # Adversarial training phase  
        for images, labels in tqdm(train_adversarial, desc=f"Epoch {epoch} - Adversarial"):
            images, labels = images.cuda(), labels.cuda()

            # first do a normal pass and gradient update step with the natural image (to emulate standard training loop of adversarial training)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Generate adversarial examples using the model's built-in attack functionality
            # switch to eval mode to generate adversarial examples
            model.eval()
            # Add this line to enable gradients for the input images
            images = images.clone().detach().requires_grad_(True)
            images = images.cuda()
            _, adv_images = model(images, labels, make_adv=True, 
                                constraint=adv_args.constraint,
                                eps=adv_args.eps,
                                step_size=adv_args.attack_lr,
                                iterations=adv_args.attack_steps,
                                random_start=False,
                                targeted=False)
            
            # switch back to train mode to do the gradient update step with the adversarial image
            model.train()
            optimizer.zero_grad()
            outputs, _ = model(adv_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        nat_correct = 0
        adv_correct = 0
        total_nat = 0
        total_adv = 0
        
        for images, labels in val_natural:
            images, labels = images.cuda(), labels.cuda()
            outputs, _ = model(images)
            _, predicted = outputs.max(1)
            nat_correct += predicted.eq(labels).sum().item()
            total_nat += labels.size(0)
                
        # Adversarial validation    
        for images, labels in val_adversarial:
            images, labels = images.cuda(), labels.cuda()
            # Add this line to enable gradients for the input images
            images = images.clone().detach().requires_grad_(True)
            images = images.cuda()
            _, adv_images = model(images, labels, make_adv=True, 
                                constraint=adv_args.constraint,
                                eps=adv_args.eps,
                                step_size=adv_args.attack_lr,
                                iterations=adv_args.attack_steps,
                                random_start=False,
                                targeted=False)
            outputs, _ = model(adv_images)
            _, predicted = outputs.max(1)
            adv_correct += predicted.eq(labels).sum().item()
            total_adv += labels.size(0)
        
        print(f"Epoch {epoch}")
        print(f"Natural Validation Accuracy: {100.*nat_correct/total_nat:.2f}%")
        print(f"Adversarial Validation Accuracy: {100.*adv_correct/total_adv:.2f}%")
        
    # save the model
    torch.save(model.state_dict(), save_path)
    print(f"Successfully saved model to {save_path}")

    return model





if __name__ == "__main__":
    DATA_PATH = '/home/gridsan/hmartinez/distribution-shift/datasets'
    # load the dataset loaders
    dataset, train_loader, test_loader = load_dataset(DATA_PATH)
    print("successfully loaded dataset, starting to train twice natural model")
    # train the twice natural model
    model = train_twice_natural(TWICE_NATURAL_TRAINING_PARAMS, train_loader, test_loader)
    print("successfully trained twice natural model")
    
