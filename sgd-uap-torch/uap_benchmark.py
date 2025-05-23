import numpy as np
import torch
import os
import sys
import time
from attacks import uap_sgd
from utils import loader_cifar, model_cifar, evaluate


def untargeted_uap(model_path, save_path, nb_epoch=10, eps=10/255, beta=10):
    """
    Performs an untargeted UAP attack on a model, look at notebook in `sgd-uap-torch` for more details.
    """
    print(f"Performing untargeted UAP attack with eps={eps} and beta={beta}")
    # load model
    print(f"Loading model from {model_path}")
    model, best_acc = model_cifar("resnet18", ckpt_path=model_path)
    # load test loader
    print(f"Loading test loader")
    test_loader = loader_cifar(dir_data="/u/hectorxm/distribution-shift/dataset", train=False)
    # perform attack
    print(f"Performing UAP attack")
    
    start_time = time.time()
    uap, losses = uap_sgd(model, test_loader, nb_epoch=nb_epoch, eps=eps, beta=beta)
    end_time = time.time()
    
    attack_duration = end_time - start_time
    print(f"UAP attack finished in {attack_duration:.2f} seconds")

    # save uap
    print(f"Saving UAP to {save_path}")
    torch.save(uap, save_path+"/untargeted_uap.pt")
    torch.save(losses, save_path+"/untargeted_losses.pt")
    # evaluate
    print(f"Evaluating UAP")
    _,_,_,_,outputs, labels = evaluate(model, test_loader, uap=uap)
    # calculate accuracy
    print(f"Calculating accuracy")
    accuracy = np.sum(outputs == labels) / len(labels)
    print(f"Accuracy: {accuracy}")
    return accuracy

if __name__ == "__main__":
    untargeted_uap(model_path="/u/hectorxm/distribution-shift/models/149_checkpoint.pt", save_path="/u/hectorxm/distribution-shift/adversarial/results/uap")
    