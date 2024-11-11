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


if __name__ == "__main__":
    MODEL_PATH = "/home/gridsan/hmartinez/distribution-shift/models/149_checkpoint.pt"
    test(MODEL_PATH)