"""
this is to generate adversarial examples a la robustness package to be later used in training/evaling RESNET18
"""
import torch
from robustness import model_utils, datasets
from robustness.datasets import CIFAR
import cox

DEFAULT_ATTACK_CONFIG = {
    'constraint': 'inf',  # Use L-inf PGD attack
    'eps': 8/255,  # Epsilon for L-inf norm constraint
    'step_size': 2/255,  # Step size for PGD
    'iterations': 10,  # Number of PGD steps
    'do_tqdm': True  # Show progress bar
}

def generate_adv_examples(attack_config, model_path, data_path):
    """
    generate adversarial examples given a trained model
    returns adversarial_examples (x-values)
    """
    # load dataset
    dataset = CIFAR(data_path)

    # loading model
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=dataset, resume_path=model_path)
    model.eval()
    # create data loader
    train_loader, test_loader = dataset.make_loaders(batch_size=128, workers=8)

    # create both training and test data adversarial examples
    training_adv_examples = []
    training_original_images = []
    training_labels = []
    for images, targets in train_loader:
        images, targets = images.cuda(), targets.cuda()
        _, adv_images = model(images, targets, make_adv=True, **attack_config)
        training_adv_examples.append(adv_images.cpu())
        training_original_images.append(images.cpu())
        training_labels.append(targets.cpu())
    
    print(f"Finished generating training adv examples. Made this many: {len(training_adv_examples)}")

    training_adv_examples = torch.cat(training_adv_examples)
    training_original_images = torch.cat(training_original_images)
    training_labels = torch.cat(training_labels)

    # now do the same for test set
    test_adv_examples = []
    test_original_images = []
    test_labels = []
    for images, targets in test_loader:
        images, targets = images.cuda(), targets.cuda()
        _, adv_images = model(images, targets, make_adv=True, **attack_config)
        test_adv_examples.append(adv_images.cpu())
        test_original_images.append(images.cpu())
        test_labels.append(targets.cpu())

    print(f"Finished generating testing adv examples. Made this many: {len(test_adv_examples)}")


    test_adv_examples = torch.cat(test_adv_examples)
    test_original_images = torch.cat(test_original_images)
    test_labels = torch.cat(test_labels)

    target_map = {"TestAdvExamples": test_adv_examples, "TestOriginal": test_original_images, "TestLabels": test_labels,
                  "TrainAdvExamples": training_adv_examples, "TrainOriginal": training_original_images, "TrainLabels": training_labels}
    
    return target_map, model

def evaluate_robustness(model, adv_examples, labels):
    model.eval()
    # move to cpu
    labels = labels.cpu()

    # get output from adv_examples
    with torch.no_grad():
        adv_outputs, _ = model(adv_examples.cuda())
        adv_preds = adv_outputs.argmax(dim=1)
        adv_accuracy = (adv_preds.cpu() == labels).float().mean().item()

    # get accuracy
    print(f"Adversarial Accuracy is {adv_accuracy*100}")
    return adv_accuracy



if __name__ == "__main__":
    MODEL_PATH = "/home/gridsan/hmartinez/distribution-shift/models/149_checkpoint.pt"
    DATA_PATH = "/home/gridsan/hmartinez/distribution-shift/datasets"
    examples, model = generate_adv_examples(DEFAULT_ATTACK_CONFIG, MODEL_PATH, DATA_PATH)
    accuracy = evaluate_robustness(model, examples["TestAdvExamples"], examples["TestLabels"])



