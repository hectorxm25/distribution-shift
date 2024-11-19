from robustness import model_utils, datasets
import torch
from tqdm import tqdm

def test(model_path):
    # set up dataset again
    dataset = datasets.CIFAR("/home/gridsan/hmartinez/distribution-shift/datasets")
    # load trained model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset, resume_path=model_path)
    # create test loaders
    _, test_loader = dataset.make_loaders(batch_size=128, workers=8, only_val=True)
    # eval
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
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
    _, test_loader = dataset.make_loaders(batch_size=128, workers=8, only_val=True)
    
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
    natural_losses = []
    adversarial_losses = []
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.cuda(), labels.cuda()
            
            # compute losses for natural model
            nat_outputs, _ = natural_model(images)
            nat_loss = criterion(nat_outputs, labels)
            natural_losses.extend(nat_loss.cpu().tolist())
            
            # compute losses for adversarial model
            adv_outputs, _ = adversarial_model(images)
            adv_loss = criterion(adv_outputs, labels)
            adversarial_losses.extend(adv_loss.cpu().tolist())
    
    return natural_losses, adversarial_losses

def plot_loss_histogram(natural_losses, adversarial_losses, save_path):
    """
    Creates a histogram comparing losses between natural and adversarial models
    Args:
        natural_losses: List of losses from natural model
        adversarial_losses: List of losses from adversarial model
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(12, 7))  # Slightly larger figure
    
    # Fewer bins (25 instead of 50) and increased transparency
    plt.hist(natural_losses, bins=50, range=(0, 7.5), alpha=0.6, label='Natural Model', color='blue', density=True)
    plt.hist(adversarial_losses, bins=50, range=(0, 7.5), alpha=0.6, label='Adversarial Model', color='red', density=True)
    
    plt.xlabel('Cross Entropy Loss', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Losses: Natural vs Adversarial Model', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add mean lines with labels
    nat_mean = np.mean(natural_losses)
    adv_mean = np.mean(adversarial_losses)
    plt.axvline(nat_mean, color='blue', linestyle='dashed', alpha=0.8, label='Natural Mean')
    plt.axvline(adv_mean, color='red', linestyle='dashed', alpha=0.8, label='Adversarial Mean')
    
    # Enhanced statistics text
    stats_text = f'Natural Model Stats:\n'
    stats_text += f'Mean: {nat_mean:.3f}\n'
    stats_text += f'Std: {np.std(natural_losses):.3f}\n'
    stats_text += f'Median: {np.median(natural_losses):.3f}\n\n'
    stats_text += f'Adversarial Model Stats:\n'
    stats_text += f'Mean: {adv_mean:.3f}\n'
    stats_text += f'Std: {np.std(adversarial_losses):.3f}\n'
    stats_text += f'Median: {np.median(adversarial_losses):.3f}'
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Save plot with high resolution
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    MODEL_PATH = "/home/gridsan/hmartinez/distribution-shift/models/natural/149_checkpoint.pt"
    ADVERSARIAL_MODEL_PATH = "/home/gridsan/hmartinez/distribution-shift/models/adversarial/149_checkpoint.pt"
    FIGURE_PATH = "/home/gridsan/hmartinez/distribution-shift/adversarial/visualizations/adversarial_vs_natural_loss_150epochs_CIFAR10_resnet18.png"
    natural_losses, adversarial_losses = compare_losses(MODEL_PATH, ADVERSARIAL_MODEL_PATH)
    plot_loss_histogram(natural_losses, adversarial_losses, FIGURE_PATH)
