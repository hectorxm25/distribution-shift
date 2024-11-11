"""
This file is just to play around with and make sure the `robustness` package works well.
Will attempt to generate a few adversarial examples given an already-trained model
"""

import torch
import torchvision
import numpy as np
from robustness import model_utils, datasets
from robustness.datasets import CIFAR

DATASET_PATH = "/home/gridsan/hmartinez/distribution-shift/datasets"

def main():
    # attempt to pull CIFAR10 dataset
    Dataset = CIFAR(DATASET_PATH)
    print("successfully loaded dataset")
    # loading pre-trained model resnet18
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=Dataset, resume_path='/home/gridsan/hmartinez/distribution-shift/models/149_checkpoint.pt') # will download it if not present
    # specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create loader
    _,test_loader = Dataset.make_loaders(batch_size=1, workers=8)
    x, y = next(iter(test_loader))
    # make pred
    with torch.no_grad():
        output, _ = model(x.to(device))
    # get predicted class
    pred_class = output.argmax(1).item()
    print(f"Class is {pred_class}")



if __name__ == "__main__":
    main()