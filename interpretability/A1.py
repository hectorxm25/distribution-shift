import robustness
from robustness import model_utils
import torch.nn as nn
import torch
import nnsight
from nnsight import NNsight
import numpy as np


def load_dataset(data_path):
    """
    Loads previously saved dataset in a way compatible with robustness library
    """
    # Load the saved dataset and reproducibility info
    checkpoint = torch.load(f"{data_path}/dataset.pt")
    dataset = checkpoint['dataset']
    
    # Restore all random seeds (just in case)
    torch.manual_seed(checkpoint['seed'])
    np.random.seed(checkpoint['seed'])
    
    # Create new loaders with same reproducibility settings
    train_loader = checkpoint['train_loader']
    test_loader = checkpoint['test_loader']
    
    print("Successfully loaded data loaders")
    return dataset, train_loader, test_loader


def connect_layer_to_output(model_path, layer_name, image_batch, verbose=False):
    """
    Uses NNsight to connect `layer_name`'s output, applies a Global
    Average Pooling, flattens it, then connects it to the output layer,
    and returns the outputs of the GPA layer, flatten layer, and the output layer.
    """
    # load the model and dataset
    dataset, _, _ = load_dataset(data_path="/home/gridsan/hmartinez/distribution-shift/datasets")
    model = model_utils.make_and_restore_model(
        arch='resnet18',
        dataset=dataset,
        resume_path=model_path,
    )[0]
    # set nnsight model
    nnsight_model = NNsight(model)
    
    with nnsight_model.trace(image_batch) as trace:
        # get intermediate layer output
        intermediate_layer = getattr(nnsight_model, layer_name)
        
        # save output
        intermediate_activation = intermediate_layer.output.save()

    if verbose:
        print(f"Intermediate layer {layer_name} output saved")
        print(f"Intermediate layer shape: {intermediate_activation.shape}")
    # retrieve intermediate activate, apply GPA and flatten
    intermediate_tensor = intermediate_activation.value
    pooled_tensor = nn.AdaptiveAvgPool2d((1,1))(intermediate_tensor)
    flattened_tensor = torch.flatten(pooled_tensor, 1)

    # logging
    if verbose:
        print(f"Pooled tensor shape: {pooled_tensor.shape}")
        print(f"Flattened tensor shape: {flattened_tensor.shape}")

    # return GPA, flatten, and output
    return pooled_tensor, flattened_tensor



if __name__ == "__main__":
    MODEL_PATH = "/home/gridsan/hmartinez/distribution-shift/models/natural/149_checkpoint.pt"
    _, train_loader, _ = load_dataset(data_path="/home/gridsan/hmartinez/distribution-shift/datasets")
    image_batch = next(iter(train_loader))[0]
    connect_layer_to_output(MODEL_PATH, "layer1", image_batch, verbose=True)