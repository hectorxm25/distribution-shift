from robustness import model_utils, datasets
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from train import load_dataset

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

def compare_losses(natural_model_path, adversarial_model_path):
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
    plt.hist(train_phi, bins=50, alpha=0.5, density=True, label='Train', color='blue')
    plt.hist(test_phi, bins=50, alpha=0.5, density=True, label='Test', color='red')
    
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

if __name__ == "__main__":
    ADVERSARIAL_MODEL_PATH = "/home/gridsan/hmartinez/distribution-shift/models/adversarial/0.031epsNormInf/149_checkpoint.pt"
    FIGURE_PATH = "/home/gridsan/hmartinez/distribution-shift/adversarial/visualizations/adv_0.031epsNormInf_test_vs_train.png"
    _, train_loader, test_loader = load_dataset("/home/gridsan/hmartinez/distribution-shift/datasets")
    train_losses = calculate_losses(ADVERSARIAL_MODEL_PATH, train_loader)
    test_losses = calculate_losses(ADVERSARIAL_MODEL_PATH, test_loader)
    plot_loss_histogram_test_vs_train(train_losses, test_losses, FIGURE_PATH, "Adversarial Model 0.031 eps NormInf")
   