import robustness
from robustness import model_utils
import torch.nn as nn
import torch
import nnsight
from nnsight import NNsight
import numpy as np
from collections import defaultdict 
import itertools 


# GLOBAL ATTACK CONFIGS
LARGE_EPS_ATTACK_PARAMS = {
        'constraint': 'inf',      # Use L2 PGD attack
        'eps': 0.031*25,            # large epsilon
        'step_size': 0.1,      # large step size
        'iterations': 10,      # standard iterations
        'random_start': False,  # standard random start
    }
# standard PGD attack
SMALL_EPS_ATTACK_PARAMS = {
        'constraint': 'inf',      # Use L2 PGD attack
        'eps': 0.031,            # small epsilon
        'step_size': 0.01,      # small step size
        'iterations': 10,      # standard iterations
        'random_start': False,  # standard random start
    }

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

# NOTE: Deprecated, do not use
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
    model.eval() # Ensure model is in eval mode
    # set nnsight model
    nnsight_model = NNsight(model)
    
    with nnsight_model.trace(image_batch) as trace:
        # get intermediate layer output
        # Use .fetch_module to handle potential nested modules if needed
        intermediate_layer = nnsight_model.fetch_module(layer_name)
        
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

def get_intermediate_layer_representations(model_path, loader, save_path, num_batches=10, verbose=False):
    """
    Gets the intermediate layer representations of the loader.
    Gets `num_batches` representations for each intermediate layer in the model. In this case, these will be the representations of each Resnet18 block.
    Saves the representations as tensors in `save_path`.
    NOTE: Must be done on a GPU
    """

    LAYERS_TO_INVESTIGATE = [
        "model.layer1",
        "model.layer2",
        "model.layer3",
        "model.layer4",
        "model.linear", # last layer?
    ]

    # load model and dataset first
    dataset, _, _ = load_dataset(data_path="/home/gridsan/hmartinez/distribution-shift/datasets")
    model = model_utils.make_and_restore_model(
        arch='resnet18',
        dataset=dataset,
        resume_path=model_path,
    )[0].to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # print layer names
    if verbose:
        print("--- Model Layer Names ---")
        # Print all named modules to see the full structure
        for name, module in model.named_modules():
             print(f"Name: {name}, Type: {type(module).__name__}")
        print("-------------------------")


    # wrap the NNsight model and get intermediate layer representations
    nnsight_model = NNsight(model)

    # will be of form {layer_name : {natural_images : [batch_1, batch_2, ...], low_eps_images : [batch_1, batch_2, ...], high_eps_images : [batch_1, batch_2, ...]}}
    representations = {layer_name : {
        "natural_images" : [],
        "low_eps_images" : [],
        "high_eps_images" : []
    } for layer_name in LAYERS_TO_INVESTIGATE}

    # iterate over the loader
    for i, (images, labels) in enumerate(loader):
        if i == num_batches:
            break
        if verbose:
            print(f"Processing batch {i} of {num_batches}")
        # move to GPU
        images = images.to("cuda" if torch.cuda.is_available() else "cpu")
        labels = labels.to("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(f"Images shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
        # create the small and large epsilon adversarial images to pass in
        _, small_eps_images = model(images, labels, make_adv=True, **SMALL_EPS_ATTACK_PARAMS)
        _, large_eps_images = model(images, labels, make_adv=True, **LARGE_EPS_ATTACK_PARAMS)
        if verbose:
            print(f"Small epsilon adversarial images shape: {small_eps_images.shape}")
            print(f"Large epsilon adversarial images shape: {large_eps_images.shape}")

        # get the intermediate layer representations for natural images
        with nnsight_model.trace(images) as trace:
            natural_save_proxy = {}
            for layer_path in LAYERS_TO_INVESTIGATE:
                # Start with the nnsight_model proxy
                module_proxy = nnsight_model
                # Navigate the nested path dynamically
                for part in layer_path.split('.'):
                    module_proxy = getattr(module_proxy, part)
                # Now module_proxy is the proxy for the target layer
                natural_save_proxy[layer_path] = module_proxy.output.save()
            if verbose:
                print(f"Natural save proxy: {natural_save_proxy}")

        # get the intermediate layer representations for small epsilon adversarial images
        with nnsight_model.trace(small_eps_images) as trace:
            small_eps_save_proxy = {}
            for layer_path in LAYERS_TO_INVESTIGATE:
                module_proxy = nnsight_model
                for part in layer_path.split('.'):
                    module_proxy = getattr(module_proxy, part)
                small_eps_save_proxy[layer_path] = module_proxy.output.save()
            if verbose:
                print(f"Small epsilon save proxy: {small_eps_save_proxy}")

        # get the intermediate layer representations for large epsilon adversarial images
        with nnsight_model.trace(large_eps_images) as trace:
            large_eps_save_proxy = {}
            for layer_path in LAYERS_TO_INVESTIGATE:
                module_proxy = nnsight_model
                for part in layer_path.split('.'):
                    module_proxy = getattr(module_proxy, part)
                large_eps_save_proxy[layer_path] = module_proxy.output.save()
            if verbose:
                print(f"Large epsilon save proxy: {large_eps_save_proxy}")
        # save proxy representations into the representations dictionary
        for layer in LAYERS_TO_INVESTIGATE:
            representations[layer]["natural_images"].append(natural_save_proxy[layer].value.detach().cpu())
            representations[layer]["low_eps_images"].append(small_eps_save_proxy[layer].value.detach().cpu())
            representations[layer]["high_eps_images"].append(large_eps_save_proxy[layer].value.detach().cpu())

        # if verbose:
        #     print(f"Representations: {representations}")

    if verbose:
        # print(f"After All Loops, Representations: {representations}")
        print(f"After All Loops, Representations shape layer1 natural_images: {representations['model.layer1']['natural_images'][0].shape}")
        print(f"After All Loops, Representations shape layer1 low_eps_images: {representations['model.layer1']['low_eps_images'][0].shape}")
        print(f"After All Loops, Representations shape layer1 high_eps_images: {representations['model.layer1']['high_eps_images'][0].shape}")
        print(f"After All Loops, Len of representations layer1 natural_images: {len(representations['model.layer1']['natural_images'])}")
        print(f"After All Loops, Len of representations layer1 low_eps_images: {len(representations['model.layer1']['low_eps_images'])}")
        print(f"After All Loops, Len of representations layer1 high_eps_images: {len(representations['model.layer1']['high_eps_images'])}")
        print(f"After All Loops, natural_images_layer_1_0: {representations['model.layer1']['natural_images'][0]}")
    # save the representations
    torch.save(representations, save_path)

    return representations

if __name__ == "__main__":
    MODEL_PATH = "/home/gridsan/hmartinez/distribution-shift/models/natural/149_checkpoint.pt"
    _, train_loader, _ = load_dataset(data_path="/home/gridsan/hmartinez/distribution-shift/datasets")
    # image_batch = next(iter(train_loader))[0]
    # connect_layer_to_output(MODEL_PATH, "layer1", image_batch, verbose=True)
    representations = get_intermediate_layer_representations(MODEL_PATH, train_loader, save_path="/home/gridsan/hmartinez/distribution-shift/interpretability/representations.pt", num_batches=10, verbose=False)
    # print(f"After Main, representations: {representations}")