from robustness import model_utils, datasets
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset, ModelWrapper
import mask_generation as mask_gen

## Global configs
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

def test_mask_accuracy(model_path, loader, attack_config=ATTACK_PARAMS):
    """
    Gets the accuracy of applying masks to the images in the loader. The config used to create the masks is the attack config.
    Returns natural and adversarial accuracies. 
    """
    # load dataset
    dataset = datasets.CIFAR("/home/gridsan/hmartinez/distribution-shift/datasets")
    # load model 
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset, resume_path=model_path)
    # eval
    model.eval()

    correct_natural, correct_adversarial = 0, 0
    total = 0
    # create masks outside of torch_nograd()
    masks_batches = []
    adv_labels_batches = []
    for images, labels in loader:
        _, _, masks, _, _, adv_labels = mask_gen.create_masks_batch(attack_config, model_path, images, labels)
        masks_batches.append(masks)
        adv_labels_batches.append(adv_labels)

    with torch.no_grad():
        batch_index = 0
        for images, labels in tqdm(loader): # batch by batch
            masks = masks_batches[batch_index]
            adv_labels = adv_labels_batches[batch_index]
            masks = masks.cuda()
            adv_labels = adv_labels.cuda()
            labels = labels.cuda()
            model.to('cuda')
            
            outputs, _ = model(masks)
            _, predicted = outputs.max(1)
            correct_adversarial += predicted.eq(adv_labels).sum().item()
            total += labels.size(0)
            correct_natural += predicted.eq(labels).sum().item()
            batch_index += 1
    # compute and return accuracy
    accuracy_natural = 100 * correct_natural/total
    accuracy_adversarial = 100 * correct_adversarial/total
    print(f"Natural accuracy: {accuracy_natural}, Adversarial accuracy: {accuracy_adversarial}")
    # print(f"Size of dataset {len(loader) * len(loader[0][0])}")
    return accuracy_natural, accuracy_adversarial
    

def test(model_path, loader):
    # set up dataset again
    dataset = datasets.CIFAR("/home/gridsan/hmartinez/distribution-shift/datasets")
    # load trained model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset, resume_path=model_path)
    # eval
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.cuda(), labels.cuda()
            outputs, _ = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # compute and return accuracy
    accuracy = 100 * correct/total
    print(f"Accuracy on test set is {accuracy}")
    return accuracy

def test_mask_model(model_path, loader):
    """
    this is the same as the above test function, but for the mask model since there was an issue in 
    loading the model correctly (mainly due to ModelWrapper issues)

    returns accuracy of the mask model on the loader set
    """

     # set up dataset again
    dataset = datasets.CIFAR("/home/gridsan/hmartinez/distribution-shift/datasets")

    print(f"Loading mask model from {model_path}, special load")
    # create base model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    # Wrap the model
    model = ModelWrapper(model)
    # Set device and eval mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            print(f"Outputs shape: {outputs.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Outputs: {outputs}")
            print(f"Labels: {labels}")
            _, predicted = outputs.max(1)
            print(f"Predicted: {predicted}")
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # compute and return accuracy
    accuracy = 100 * correct/total
    print(f"Accuracy on test set is {accuracy}")
    return accuracy

def compare_losses(natural_model_path, adversarial_model_path):
    """
    compares losses between natural and adversarial models
    """
    # set up dataset
    dataset = datasets.CIFAR("/home/gridsan/hmartinez/distribution-shift/datasets")
    train_loader, test_loader = dataset.make_loaders(batch_size=128, workers=8)
    
    # load both models
    natural_model, _ = model_utils.make_and_restore_model(
        arch='resnet18', 
        dataset=dataset,
        resume_path=natural_model_path
    )
    adversarial_model, _ = model_utils.make_and_restore_model(
        arch='resnet18',
        dataset=dataset, 
        resume_path=adversarial_model_path
    )
    
    # set both models to eval mode
    natural_model.eval()
    adversarial_model.eval()
    
    # store losses for each datapoint
    natural_losses_train = []
    natural_losses_test = []
    adversarial_losses_train = []
    adversarial_losses_test = []
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.cuda(), labels.cuda()
            
            # compute losses for natural model
            nat_outputs, _ = natural_model(images)
            nat_loss = criterion(nat_outputs, labels)
            natural_losses_test.extend(nat_loss.cpu().tolist())
            
            # compute losses for adversarial model
            adv_outputs, _ = adversarial_model(images)
            adv_loss = criterion(adv_outputs, labels)
            adversarial_losses_test.extend(adv_loss.cpu().tolist())

    with torch.no_grad():
        for images, labels in tqdm(train_loader):
            images, labels = images.cuda(), labels.cuda()
            nat_outputs, _ = natural_model(images)
            nat_loss = criterion(nat_outputs, labels)
            natural_losses_train.extend(nat_loss.cpu().tolist())

            adv_outputs, _ = adversarial_model(images)
            adv_loss = criterion(adv_outputs, labels)
            adversarial_losses_train.extend(adv_loss.cpu().tolist())
    
    return natural_losses_train, natural_losses_test, adversarial_losses_train, adversarial_losses_test

# NOTE: This function is no longer in use
def plot_loss_histogram(natural_losses, adversarial_losses, save_path):
    # Normalize losses to [0,1] range for both models
    def normalize_and_logit(losses):
        # Min-max normalization
        p = (losses - np.min(losses)) / (np.max(losses) - np.min(losses))
        # Clip to avoid infinity in logit
        p = np.clip(p, 0.001, 0.999)
        # Calculate logit
        return np.log(p / (1 - p))
    
    nat_logits = normalize_and_logit(natural_losses)
    adv_logits = normalize_and_logit(adversarial_losses)
    
    plt.figure(figsize=(12, 7))
    
    # Plot logit-transformed losses
    plt.hist(nat_logits, bins=50, alpha=0.6, label='Natural Model', color='blue', density=True, range=(-7, 2))
    plt.hist(adv_logits, bins=50, alpha=0.6, label='Adversarial Model', color='red', density=True, range=(-7, 2))
    
    plt.xlabel('log(p/(1-p)) where p is normalized loss', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Logit-Transformed Losses: Natural vs Adversarial Model', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add mean lines with labels
    nat_mean = np.mean(nat_logits)
    adv_mean = np.mean(adv_logits)
    plt.axvline(nat_mean, color='blue', linestyle='dashed', alpha=0.8, label='Natural Mean')
    plt.axvline(adv_mean, color='red', linestyle='dashed', alpha=0.8, label='Adversarial Mean')
    
    # Update statistics text with transformed values
    stats_text = f'Natural Model Stats (logit):\n'
    stats_text += f'Mean: {nat_mean:.3f}\n'
    stats_text += f'Std: {np.std(nat_logits):.3f}\n'
    stats_text += f'Median: {np.median(nat_logits):.3f}\n\n'
    stats_text += f'Adversarial Model Stats (logit):\n'
    stats_text += f'Mean: {adv_mean:.3f}\n'
    stats_text += f'Std: {np.std(adv_logits):.3f}\n'
    stats_text += f'Median: {np.median(adv_logits):.3f}'
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return nat_logits, adv_logits

# NOTE: This function is no longer in use
def plot_single_loss_histogram(losses, save_path, title):
    """
    Plots histogram of losses for a single model's training set.
    
    Args:
        losses: Array of loss values
        save_path: Path to save the plot
        title: Title for the plot
    """
    
    plt.figure(figsize=(12, 7))
    
    # Plot logit-transformed losses
    plt.hist(losses, bins=50, alpha=0.6, color='blue', density=True, range=(min(losses), max(losses)))
    
    plt.xlabel('Cross-Entropy Loss', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    
    # Add mean line
    mean = np.mean(losses)
    plt.axvline(mean, color='red', linestyle='dashed', alpha=0.8, label=f'Mean: {mean:.3f}')
    plt.legend(fontsize=10)
    
    # Add statistics text
    stats_text = f'Statistics (logit):\n'
    stats_text += f'Mean: {mean:.3f}\n'
    stats_text += f'Std: {np.std(losses):.3f}\n'
    stats_text += f'Median: {np.median(losses):.3f}'
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return losses

def calculate_losses(model_path, data_loader):
    # set up dataset
    dataset = datasets.CIFAR("/home/gridsan/hmartinez/distribution-shift/datasets")
    # load model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset, resume_path=model_path)
    # set model to eval mode
    model.eval()

    losses = []
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            losses.extend(loss.cpu().tolist())

    return losses

def get_natural_and_mask_confidence(model_path, images, labels, masks, adv_labels, apply_softmax=False):
    """
    Gets the confidence for the natural and mask images along both the natural and adversarial labels
    apply_softmax: if True, applies softmax to the outputs
    """
    # set up dataset
    dataset = datasets.CIFAR("/home/gridsan/hmartinez/distribution-shift/datasets")
    if "mask" in model_path:
        print(f"Loading mask model from {model_path}, special load")
        # create base model
        model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset)

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])

    else:
        # create base model
        print(f"Loading natural model from {model_path}, normal load")
        model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset, resume_path=model_path)

    # wrap model
    model = ModelWrapper(model)
    # Set device and eval mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.cuda()
    model.eval()

    natural_confidence = []
    # get natural confidence
    with torch.no_grad():
        # NOTE: CHANGE THIS SO THAT WE CAN GO THROUGH MANY BATCHES
        outputs = model(images.to(device))
        if apply_softmax:
            outputs = torch.nn.functional.softmax(outputs, dim=1)
        # outputs now is a list of lists, each inner list is the confidence for each class, either raw logits or softmaxed
        outputs = outputs.cpu()
        for i in range(len(outputs)):
            natural_confidence.append(outputs[i][labels[i]]) # now natural confidence is a list of confidence values for the natural label with the natural images
    
    mask_confidence = []
    # get mask confidence
    with torch.no_grad():
        outputs = model(masks.to(device))
        if apply_softmax:
            outputs = torch.nn.functional.softmax(outputs, dim=1)
        outputs = outputs.cpu()
        for i in range(len(outputs)):
            mask_confidence.append(outputs[i][labels[i]])
    
    mask_adv_confidence = []
    # get mask confidence for the adversarial labels
    with torch.no_grad():
        outputs = model(masks.to(device))
        if apply_softmax:
            outputs = torch.nn.functional.softmax(outputs, dim=1)
        outputs = outputs.cpu()
        for i in range(len(outputs)):
            mask_adv_confidence.append(outputs[i][adv_labels[i]]) # now mask adversarial confidence is a list of confidence values for adversarial label with the mask images


    # NOTE: Might not provide much information, but kept for completeness
    natural_adv_confidence = []
    # get natural confidence for the adversarial labels
    with torch.no_grad():
        outputs = model(images.to(device))
        if apply_softmax:
            outputs = torch.nn.functional.softmax(outputs, dim=1)
        outputs = outputs.cpu()
        for i in range(len(outputs)):
            natural_adv_confidence.append(outputs[i][adv_labels[i]]) # now natural adversarial confidence is a list of confidence values for adversarial label with the natural images

    return natural_confidence, mask_confidence, mask_adv_confidence, natural_adv_confidence

def calculate_losses_wrapped(model_path, data_loader):
    # set up dataset
    dataset = datasets.CIFAR("/home/gridsan/hmartinez/distribution-shift/datasets")
    # create base model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset)
    # wrap model
    model = ModelWrapper(model)
    # load weights
    model.load_state_dict(torch.load(model_path))
    # set model to eval mode and cuda
    model = model.cuda()
    model.eval()

    losses = []
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.extend(loss.cpu().tolist())

    return losses

def plot_loss_histogram_test_vs_train(train_losses, test_losses, save_path, title):
    plt.figure(figsize=(10, 6))

    # calculate normalized phi for each with clipping to prevent infinities
    def safe_phi(loss):
        p = np.exp(-loss)
        # Clip probabilities to avoid infinite logits
        p = np.clip(p, 1e-15, 1-1e-15)
        return np.log(p/(1-p))
    
    train_phi = [safe_phi(loss) for loss in train_losses]
    test_phi = [safe_phi(loss) for loss in test_losses]
    
    # Plot histograms
    plt.hist(train_phi, bins=50, alpha=0.5, density=True, label='Trained', color='blue')
    plt.hist(test_phi, bins=50, alpha=0.5, density=True, label='Untrained', color='red')
    
    plt.xlabel('phi(p)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add statistics text for both distributions
    stats_text = f'Statistics:\n'
    stats_text += f'Train:\n'
    stats_text += f'Mean: {np.mean(train_phi):.3f}\n'
    stats_text += f'Std: {np.std(train_phi):.3f}\n'
    stats_text += f'Median: {np.median(train_phi):.3f}\n\n'
    stats_text += f'Test:\n'
    stats_text += f'Mean: {np.mean(test_phi):.3f}\n'
    stats_text += f'Std: {np.std(test_phi):.3f}\n'
    stats_text += f'Median: {np.median(test_phi):.3f}'
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return None

def plot_loss_histogram_test_vs_train_multiple_batches(train_losses, test_losses, save_path, title):
    plt.figure(figsize=(10, 6))

    # flatten the losses when they are lists of lists
    import itertools

    train_losses = list(itertools.chain.from_iterable(train_losses))
    test_losses = list(itertools.chain.from_iterable(test_losses))

    # calculate normalized phi for each with clipping to prevent infinities
    def safe_phi(loss):
        p = np.exp(-loss)
        # Clip probabilities to avoid infinite logits
        p = np.clip(p, 1e-7, 1-1e-7)
        return np.log(p/(1-p))
    
    train_phi = [safe_phi(loss) for loss in train_losses]
    test_phi = [safe_phi(loss) for loss in test_losses]
    
    # Plot histograms
    plt.hist(train_phi, bins=50, alpha=0.5, density=True, label='Trained', color='blue')
    plt.hist(test_phi, bins=50, alpha=0.5, density=True, label='Untrained', color='red')
    
    plt.xlabel('phi(p)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add statistics text for both distributions
    stats_text = f'Statistics:\n'
    stats_text += f'Natural:\n'
    stats_text += f'Mean: {np.mean(train_phi):.3f}\n'
    stats_text += f'Std: {np.std(train_phi):.3f}\n'
    stats_text += f'Median: {np.median(train_phi):.3f}\n\n'
    stats_text += f'Mask:\n'
    stats_text += f'Mean: {np.mean(test_phi):.3f}\n'
    stats_text += f'Std: {np.std(test_phi):.3f}\n'
    stats_text += f'Median: {np.median(test_phi):.3f}'
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return None

def plot_loss_histogram_natural_vs_mask(natural_losses, mask_losses, save_path, title):
    """
    Plots histogram of losses for natural and mask models. Uses the phi function to normalize the losses.
    Takes in the losses from both the natural and mask images (not maskED images)
    The loss could be from both the natural and adversarial labels and it could come from models
    trained or not trained on the mask images. Will do experiments for both. 
    """

    plt.figure(figsize=(10, 6))

    # calculate normalized phi for each with more aggressive clipping to prevent infinities
    def safe_phi(loss):
        p = np.exp(-loss)
        # Clip probabilities to avoid infinite logits
        p = np.clip(p, 1e-7, 1-1e-7)
        return np.log(p/(1-p))
    
    natural_phi = [safe_phi(loss) for loss in natural_losses]
    mask_phi = [safe_phi(loss) for loss in mask_losses]
    
    # Plot histograms
    plt.hist(natural_phi, bins=50, alpha=0.5, density=True, label='Natural', color='blue')
    plt.hist(mask_phi, bins=50, alpha=0.5, density=True, label='Mask', color='red')
    
    plt.xlabel('phi(p)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add statistics text for both distributions
    stats_text = f'Statistics:\n'
    stats_text += f'Natural:\n'
    stats_text += f'Mean: {np.mean(natural_phi):.3f}\n'
    stats_text += f'Std: {np.std(natural_phi):.3f}\n'
    stats_text += f'Median: {np.median(natural_phi):.3f}\n\n'
    stats_text += f'Mask:\n'
    stats_text += f'Mean: {np.mean(mask_phi):.3f}\n'
    stats_text += f'Std: {np.std(mask_phi):.3f}\n'
    stats_text += f'Median: {np.median(mask_phi):.3f}'
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return None

if __name__ == "__main__":

    # test mask accuracy
    MODEL_PATH= "/home/gridsan/hmartinez/distribution-shift/models/natural/149_checkpoint.pt"
    _, training_loader, test_loader = load_dataset("/home/gridsan/hmartinez/distribution-shift/datasets")
    test_mask_accuracy(MODEL_PATH, training_loader)

    # MODEL_PATH_NATURAL_TRAINING = "/home/gridsan/hmartinez/distribution-shift/models/natural/149_checkpoint.pt"
    # _, training_loader, test_loader = load_dataset("/home/gridsan/hmartinez/distribution-shift/datasets")
    # # create a list of the training set images
    # training_images = []
    # for images, labels in training_loader:
    #     training_images.append(images) # list of tensors
    # print(f"Created {len(training_images)} training image batches")
    # # now create masks from the training set
    # mask_tensors_list, labels_list, adv_labels_list = mask_gen.create_masks_loader(ATTACK_PARAMS, MODEL_PATH_NATURAL_TRAINING, training_loader)
    # print(f"Created {len(mask_tensors_list)} mask, labels, adv_labels batches")

    # # find the confidences of the natural and mask images respective to each model, should be four items
    # # natural image confidence on the untrained model, mask image ADVERSARIAL confidence on the untrained model, natural image confidence on the trained model, mask image ADVERSARIAL confidence on the trained model
    # # pairs: natural image confidence on untrained vs trained model, mask adv confidence on untrained vs trained model, mask natural confidence on untrained vs trained model, mask natural trained vs natural trained

    # # get untrained model confidences
    # untrained_confidence_list = {
    #     "natural": [],
    #     "mask": [],
    #     "mask_adv": [],
    #     "natural_adv": []
    # }
    # # print(f"Sanity Check: This is the untrained model confidence list: {untrained_confidence_list}")
    
    # for i in range(len(training_images)):
    #     natural_untrained_confidence, mask_untrained_confidence, mask_adv_untrained_confidence, natural_adv_untrained_confidence = get_natural_and_mask_confidence(MODEL_PATH_NATURAL_TRAINING, training_images[i], labels_list[i], mask_tensors_list[i], adv_labels_list[i], apply_softmax=True)
    #     untrained_confidence_list["natural"].append(natural_untrained_confidence) # list of lists, with each inner list being the confidence for a batch of images
    #     untrained_confidence_list["mask"].append(mask_untrained_confidence)
    #     untrained_confidence_list["mask_adv"].append(mask_adv_untrained_confidence)
    #     untrained_confidence_list["natural_adv"].append(natural_adv_untrained_confidence)

    # # print(f"Sanity Check: This is the untrained_confidence_list['natural']: {untrained_confidence_list['natural']}")
    # # print(f"Sanity Check: This is the size of the untrained_confidence_list['natural']: {len(untrained_confidence_list['natural'])}")
    # # print(f"Sanity Check This is the first item in the untrained_confidence_list['natural']: {untrained_confidence_list['natural'][0]}")
    
    # # get trained model confidences
    # trained_confidence_list = {
    #     "natural": [],
    #     "mask": [],
    #     "mask_adv": [],
    #     "natural_adv": []
    # }
    # for i in range(len(training_images)):
    #     natural_trained_confidence, mask_trained_confidence, mask_adv_trained_confidence, natural_adv_trained_confidence = get_natural_and_mask_confidence(MODEL_PATH_MASK_TRAINING, training_images[i], labels_list[i], mask_tensors_list[i], adv_labels_list[i], apply_softmax=True)
    #     trained_confidence_list["natural"].append(natural_trained_confidence)
    #     trained_confidence_list["mask"].append(mask_trained_confidence)
    #     trained_confidence_list["mask_adv"].append(mask_adv_trained_confidence)
    #     trained_confidence_list["natural_adv"].append(natural_adv_trained_confidence)

    # # Make plots

    # # # Plot natural untrained vs natural trained to check utility loss
    # # plot_loss_histogram_test_vs_train_multiple_batches(trained_confidence_list["natural"], untrained_confidence_list["natural"], save_path="/home/gridsan/hmartinez/distribution-shift/adversarial/visualizations/maskLossPlots/maskedTrainedResults/natural_untrained_vs_trained_1_epoch_adv_trained.png",
    # #                                                     title="Natural Softmaxed Confidence on (Adv-Label) Trained vs Untrained on Training Set: N = 40k, 1 Epoch")

    # # # Plot Mask Adv on Untrained vs Trained, for sanity check to make sure trained has lower adv loss
    # # plot_loss_histogram_test_vs_train_multiple_batches(trained_confidence_list["mask_adv"], untrained_confidence_list["mask_adv"], save_path="/home/gridsan/hmartinez/distribution-shift/adversarial/visualizations/maskLossPlots/maskedTrainedResults/mask_adv_untrained_vs_trained_1_epoch_adv_trained.png",
    # #                                                     title="Mask Adv Softmaxed Confidence on (Adv-Label) Trained vs Untrained on Training Set: N = 40k, 1 Epoch")

    # # # Plot natural mask confidence on trained vs untrained, to check that there is a large gap between the two
    # # plot_loss_histogram_test_vs_train_multiple_batches(trained_confidence_list["mask"], untrained_confidence_list["mask"], save_path="/home/gridsan/hmartinez/distribution-shift/adversarial/visualizations/maskLossPlots/maskedTrainedResults/mask_untrained_vs_trained_1_epoch_adv_trained.png",
    # #                                                     title="Mask Softmaxed Confidence on (Adv-Label) Trained vs Untrained on Training Set: N = 40k, 1 Epoch")

    # # Plot mask natural confidence on trained vs natural image confidence on trained to see if there is a large loss/confidence gap between the two
    # plot_loss_histogram_test_vs_train_multiple_batches(trained_confidence_list["natural"], trained_confidence_list["mask"], save_path="/home/gridsan/hmartinez/distribution-shift/adversarial/visualizations/maskLossPlots/maskedTrainedResults/natural_vs_mask_trained_1_epoch_adv_trained.png",
    #                                                     title="Natural Softmaxed Confidence on (Adv-Label) Trained vs Mask Softmaxed Confidence on (Adv-Label) Trained: N = 40k, 1 Epoch")


    # # # get loss of untrained and trained model on test set
    # # untrained_accuracy = test(MODEL_PATH_NATURAL_TRAINING, test_loader)
    # # trained_accuracy = test_mask_model(MODEL_PATH_MASK_TRAINING, test_loader)
    # # print(f"Untrained Accuracy: {untrained_accuracy}")
    # # print(f"Trained Accuracy: {trained_accuracy}")