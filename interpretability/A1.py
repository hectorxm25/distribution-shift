import robustness
from robustness import model_utils
import torch.nn as nn
import torch
import nnsight
from nnsight import NNsight
import numpy as np
import os
from collections import defaultdict 
import itertools 
from torch.utils.data import TensorDataset, DataLoader

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

DEFAULT_CONFIG = {
    "lr": 0.001,
    "epochs": 5,
    "batch_size": 128,
}

LAYERS_TO_INVESTIGATE = [
        "model.layer1",
        "model.layer2",
        "model.layer3",
        "model.layer4",
        "model.linear", # last layer?
    ]

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

def create_labels_for_representations(list_of_factuals, list_of_counterfactuals, shuffle=True, verbose=False):
    """
    Creates labels given the list of factuals and counterfactuals, as returned in the 
    `get_intermediate_layer_representations` function. Returns a data loader with the representations attached
    to their corresponding labels. This will be used to train a small linear classifier to predict the label of a given representation.
    To be used for probing. 

    NOTE: Assumes a 128 batch size.
    """
    # raise NotImplementedError("Needs to be fixed and have shapes work out properly")
    torch.manual_seed(42)

    # create labels and fix data size
    all_factuals = torch.cat(list_of_factuals, dim=0)
    all_counterfactuals = torch.cat(list_of_counterfactuals, dim=0)
    all_data = torch.cat([all_factuals, all_counterfactuals], dim=0)
    # Ensure labels are created as Long type for CrossEntropyLoss
    all_labels = torch.cat([
        torch.zeros(len(all_factuals), dtype=torch.long),
        torch.ones(len(all_counterfactuals), dtype=torch.long)
    ], dim=0)
    if verbose:
        print(f"All factuals shape: {all_factuals.shape}")
        print(f"All counterfactuals shape: {all_counterfactuals.shape}")
        print(f"All data shape: {all_data.shape}")
        print(f"All labels shape: {all_labels.shape}")

    # create dataset and dataloader
    dataset = TensorDataset(all_data, all_labels)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=shuffle)

    return dataloader

def get_representation(path_to_representations, layer_name, rep_type, verbose=False):
    """
    Gets the representation of a given layer and type from the representations dictionary.
    Layers are of form "model.layer1", "model.layer2", etc.
    Types are "natural_images", "low_eps_images", "high_eps_images", none else
    Returns a single tensor of shape (N, k) where N is the total number of samples
    across all batches and k is the flattened feature dimension.
    """
    assert rep_type in ["natural_images", "low_eps_images", "high_eps_images"], "Must be one of the three of natural, low, or high epsilon"
    representations = torch.load(path_to_representations)

    list_of_batches = representations[layer_name][rep_type]

    if not list_of_batches:
        print(f"Warning: No representations found for layer '{layer_name}', type '{rep_type}'. Returning empty tensor.")
        return torch.empty((0, 0))

    # Concatenate all batches
    all_reps_raw = torch.cat(list_of_batches, dim=0)

    # Flatten the representations
    all_reps_flattened = torch.flatten(all_reps_raw, start_dim=1)

    if verbose:
        print(f"Retrieving representations for layer: {layer_name}, type: {rep_type}")
        print(f"  Shape of a single batch: {list_of_batches[0].shape}")
        print(f"  Number of batches: {len(list_of_batches)}")
        print(f"  Shape after concatenating batches (raw): {all_reps_raw.shape}")
        print(f"  Shape after flattening (returned): {all_reps_flattened.shape}")

    return all_reps_flattened

class ProbingModel(nn.Module):
    """
    A simple linear classifier to probe the representations.
    No non-linearity.
    """
    def __init__(self, num_features):
        super(ProbingModel, self).__init__()
        self.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.fc(x)

def train_probing_model(dataloader, save_path, config=DEFAULT_CONFIG, verbose=False):
    """
    Trains a probing model on the given dataloader.
    """
    model = ProbingModel(num_features=dataloader.dataset[0][0].shape[0]) # should be 64*32*32 = 65536
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(config["epochs"]):
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if verbose:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    # save the model
    torch.save(model.state_dict(), save_path)

def train_all_probes(representations_path, save_folder, config=DEFAULT_CONFIG, verbose=False):
    """
    Trains a probing model on all layers and types of representations.
    """
    # create save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    # first for all layers
    for layer in LAYERS_TO_INVESTIGATE:
        if verbose:
            print(f"Training probing model for layer: {layer}")
        # now iterate over all three combinations of representations
        natural_rep = get_representation(representations_path, layer, "natural_images", verbose)
        low_eps_rep = get_representation(representations_path, layer, "low_eps_images", verbose)
        high_eps_rep = get_representation(representations_path, layer, "high_eps_images", verbose)
        # create dataloader
        natural_high_eps_loader = create_labels_for_representations([natural_rep], [high_eps_rep], shuffle=True, verbose=verbose)
        low_high_eps_loader = create_labels_for_representations([low_eps_rep], [high_eps_rep], shuffle=True, verbose=verbose)
        natural_low_eps_loader = create_labels_for_representations([natural_rep], [low_eps_rep], shuffle=True, verbose=verbose)

        # --- Perform 80/20 Train/Test Split on the Datasets --- 
        def split_dataloader(loader):
            dataset = loader.dataset
            total_size = len(dataset)
            train_size = int(0.8 * total_size)
            test_size = total_size - train_size
            if verbose:
                print(f"  Splitting dataset of size {total_size} into train ({train_size}) and test ({test_size})")
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
            # Use the original loader's batch size or a default
            batch_size = loader.batch_size if hasattr(loader, 'batch_size') and loader.batch_size else config['batch_size']
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # No shuffle for test
            return train_loader, test_loader

        natural_high_eps_train_loader, natural_high_eps_test_loader = split_dataloader(natural_high_eps_loader)
        low_high_eps_train_loader, low_high_eps_test_loader = split_dataloader(low_high_eps_loader)
        natural_low_eps_train_loader, natural_low_eps_test_loader = split_dataloader(natural_low_eps_loader)
        # ------------------------------------------------------

        # train the model using the *training* loaders
        train_probing_model(natural_high_eps_train_loader, save_folder+f"/{layer}_natural_high_eps.pt", config, verbose)
        train_probing_model(low_high_eps_train_loader, save_folder+f"/{layer}_low_high_eps.pt", config, verbose)
        train_probing_model(natural_low_eps_train_loader, save_folder+f"/{layer}_natural_low_eps.pt", config, verbose)

        # save the dataloaders
        # Save the test loaders for evaluation
        os.makedirs(save_folder+"/loaders/test", exist_ok=True) # Ensure loaders directory exists
        os.makedirs(save_folder+"/loaders/train", exist_ok=True) # Ensure loaders directory exists
        torch.save(natural_high_eps_test_loader, save_folder+f"/loaders/test/{layer}_natural_high_eps_test_loader.pt")
        torch.save(low_high_eps_test_loader, save_folder+f"/loaders/test/{layer}_low_high_eps_test_loader.pt")
        torch.save(natural_low_eps_test_loader, save_folder+f"/loaders/test/{layer}_natural_low_eps_test_loader.pt")
        # save train loaders
        torch.save(natural_high_eps_train_loader, save_folder+f"/loaders/train/{layer}_natural_high_eps_train_loader.pt")
        torch.save(low_high_eps_train_loader, save_folder+f"/loaders/train/{layer}_low_high_eps_train_loader.pt")
        torch.save(natural_low_eps_train_loader, save_folder+f"/loaders/train/{layer}_natural_low_eps_train_loader.pt")
    
    if verbose:
        print(f"Finished training probes for all layers.")

    return

def test_probing_model(dataloader, model_path, verbose=False):
    """
    Tests a trained probing model on the given test dataloader.

    Args:
        dataloader (DataLoader): The test data loader.
        model_path (str): Path to the saved model state_dict.
        verbose (bool): If True, print evaluation results.

    Returns:
        float: The accuracy of the model on the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine input features from the dataset
    try:
        print(f"dataloader.dataset[0][0].shape: {dataloader.dataset[0][0].shape}")
        num_features = dataloader.dataset[0][0].shape[0] # Assuming the dataset is not empty
    except IndexError:
        print(f"Error: DataLoader's dataset seems empty for {model_path}. Cannot determine features.")
        return 0.0

    if verbose:
        print(f"Loading classifier from {model_path} \n with num_features: {num_features}")
    model = ProbingModel(num_features=num_features)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return 0.0
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        return 0.0

    model.to(device)
    model.eval()  # Set model to evaluation mode

    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculations
        loss_fn = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            print(f"outputs shape: {outputs.shape}")
            _, predicted = torch.max(outputs.data, 1)
            print(f"predicted shape: {predicted.shape}")
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Calculate and accumulate loss
            loss = loss_fn(outputs, target)
            print(f"Loss: {loss.item()}")
            print(f"loss shape is {loss.shape}")
            total_loss += loss.item() * target.size(0)
    accuracy = 100 * correct / total if total > 0 else 0

    if verbose:
        print(f'Accuracy of the model {model_path.split("/")[-1]} on the test data: {accuracy:.2f}%')
        print(f"Total loss: {total_loss}")
    return accuracy, total_loss

def test_all_probes(save_folder, loaders_path, classifiers_path, verbose=False):
    """
    Tests all probes on the test loaders and train loaders.
    Assumes standard naming conventions for classifier files and for loader files.
    """
    # load the test and train loaders, CHANGE NAMES IF NECESSARY
    natural_high_eps_test_loader = torch.load(loaders_path+"/natural_high_eps_test_loader.pt")
    low_high_eps_test_loader = torch.load(loaders_path+"/low_high_eps_test_loader.pt")
    natural_low_eps_test_loader = torch.load(loaders_path+"/natural_low_eps_test_loader.pt")

    natural_high_eps_train_loader = torch.load(loaders_path+"/natural_high_eps_train_loader.pt")
    low_high_eps_train_loader = torch.load(loaders_path+"/low_high_eps_train_loader.pt")
    natural_low_eps_train_loader = torch.load(loaders_path+"/natural_low_eps_train_loader.pt")

    # set up natural, low, and high configurations
    correlate_configurations = ["natural_high_eps", "low_high_eps", "natural_low_eps"]

    # store results as dictionary
    results = {}

    for layer in LAYERS_TO_INVESTIGATE:
        for config in correlate_configurations:
            model_path = classifiers_path+"/"+layer+"_"+config+".pt"
            accuracy, total_loss = test_probing_model(dataloader=natural_high_eps_test_loader, model_path=model_path, verbose=verbose)


if __name__ == "__main__":
    # MODEL_PATH = "/home/gridsan/hmartinez/distribution-shift/models/natural/149_checkpoint.pt"
    # _, train_loader, _ = load_dataset(data_path="/home/gridsan/hmartinez/distribution-shift/datasets")
    # # image_batch = next(iter(train_loader))[0]
    # # connect_layer_to_output(MODEL_PATH, "layer1", image_batch, verbose=True)
    # representations = get_intermediate_layer_representations(MODEL_PATH, train_loader, save_path="/home/gridsan/hmartinez/distribution-shift/interpretability/representations.pt", num_batches=10, verbose=False)
    # # print(f"After Main, representations: {representations}")

    # natural_layer1 = get_representation(path_to_representations="/home/gridsan/hmartinez/distribution-shift/interpretability/representations.pt", layer_name="model.layer1", rep_type="natural_images", verbose=True)
    # high_eps_layer1 = get_representation(path_to_representations="/home/gridsan/hmartinez/distribution-shift/interpretability/representations.pt", layer_name="model.layer1", rep_type="high_eps_images", verbose=True)
    # # create dataloader
    # dataloader = create_labels_for_representations([natural_layer1], [high_eps_layer1], shuffle=True, verbose=True)
    
    # test_all_probes(save_folder="/home/gridsan/hmartinez/distribution-shift/interpretability/probes/test_results", verbose=True)
    # train_all_probes(representations_path="/home/gridsan/hmartinez/distribution-shift/interpretability/representations.pt", save_folder="/home/gridsan/hmartinez/distribution-shift/interpretability/probes_all_classes", config=DEFAULT_CONFIG, verbose=True)

    print("Testing singular probe")
    test_probing_model(dataloader=torch.load("/home/gridsan/hmartinez/distribution-shift/interpretability/probes_all_classes/loaders/test/model.layer1_low_high_eps_test_loader.pt"), model_path="/home/gridsan/hmartinez/distribution-shift/interpretability/probes_all_classes/model.layer1_low_high_eps.pt", verbose=True)

    # debugging_loader = torch.load("/home/gridsan/hmartinez/distribution-shift/interpretability/probes_all_classes/loaders/test/model.layer1_low_high_eps_test_loader.pt")
    # print(f"Debugging loader: {debugging_loader}")
    # print(f"len of debugging loader: {len(debugging_loader)}")

    # for data, target in debugging_loader:
    #     print(f"Data shape: {data.shape}")
    #     print(f" data[0]: {data[0]}")
    #     print(f"data[0] shape: {data[0].shape}")
    #     print(f"Target shape: {target.shape}")
    #     break
