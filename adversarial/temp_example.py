"""
This file is just to play around with and make sure the `robustness` package works well.
Will attempt to generate a few adversarial examples given an already-trained model
"""

import torch
import torchvision
import numpy as np
import robustness.datasets

def main():
    # attempt to pull CIFAR10 dataset
    Dataset = robustness.datasets.CIFAR(data_path='/tmp/')
    print("successfully loaded dataset")



if __name__ == "__main__":
    main()