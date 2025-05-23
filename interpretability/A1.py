import robustness
from robustness import model_utils
import torch.nn as nn
import torch
import nnsight
from nnsight import NNsight
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict 
import itertools 
from torch.utils.data import TensorDataset, DataLoader

# --- Specify Target GPU ID ---
TARGET_GPU_ID = 1
_DEVICE = torch.device(f"cuda:{TARGET_GPU_ID}" if torch.cuda.is_available() and TARGET_GPU_ID < torch.cuda.device_count() else "cpu")
print(f"Using device: {_DEVICE}")
if _DEVICE.type == 'cuda':
    torch.cuda.set_device(_DEVICE) # Set default CUDA device
# ---------------------------

# GLOBAL ATTACK CONFIGS
LARGE_EPS_ATTACK_PARAMS = {
        'constraint': 'inf',      # Use Linf PGD attack
        'eps': 0.031*25,            # large epsilon
        'step_size': 0.1,      # large step size
        'iterations': 10,      # standard iterations
        'random_start': False,  # standard random start
    }
# standard PGD attack
SMALL_EPS_ATTACK_PARAMS = {
        'constraint': 'inf',      # Use Linf PGD attack
        'eps': 0.031,            # small epsilon
        'step_size': 0.01,      # small step size
        'iterations': 10,      # standard iterations
        'random_start': False,  # standard random start
    }

# standard PGD L-2 attack
L2_SMALL_EPS_ATTACK_PARAMS = {
        'constraint': '2',      # Use L2 PGD attack
        'eps': 0.15,            # small epsilon
        'step_size': 0.01,      # small step size
        'iterations': 10,      # standard iterations
        'random_start': False,  # standard random start
}

L2_LARGE_EPS_ATTACK_PARAMS = {
        'constraint': '2',      # Use L2 PGD attack
        'eps': 0.15*25,            # large epsilon
        'step_size': 0.1,      # NOTE: This is a bit larger than the default step size for large eps attacks that we used for inf attacks
        'iterations': 10,      # standard iterations
        'random_start': False,  # standard random start
    }

DEFAULT_CONFIG = {
    "lr": 0.001,
    "epochs": 3,
    "batch_size": 128,
}
# Full granular layers of conv weights and skip layers
LAYERS_TO_INVESTIGATE = [
        "model.layer1", # Output of the entire first ResNet block
        "model.layer1.0.conv1",
        "model.layer1.0.conv2",
        "model.layer1.1.conv1",
        "model.layer1.1.conv2",
        "model.layer2", # Output of the entire second ResNet block
        "model.layer2.0.conv1",
        "model.layer2.0.conv2",
        "model.layer2.0.shortcut.0", # Conv layer in shortcut
        "model.layer2.0.shortcut.1", # BatchNorm layer in shortcut
        "model.layer2.1.conv1",
        "model.layer2.1.conv2",
        "model.layer3", # Output of the entire third ResNet block
        "model.layer3.0.conv1",
        "model.layer3.0.conv2",
        "model.layer3.0.shortcut.0", # Conv layer in shortcut
        "model.layer3.0.shortcut.1", # BatchNorm layer in shortcut
        "model.layer3.1.conv1",
        "model.layer3.1.conv2",
        "model.layer4", # Output of the entire fourth ResNet block
        "model.layer4.0.conv1",
        "model.layer4.0.conv2",
        "model.layer4.0.shortcut.0", # Conv layer in shortcut
        "model.layer4.0.shortcut.1", # BatchNorm layer in shortcut
        "model.layer4.1.conv1",
        "model.layer4.1.conv2",
        "model.linear", # Output of the final linear layer
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

def get_intermediate_layer_representations(model_path, loader, save_path, num_batches=10, verbose=False, batch_skip_offset=0):
    """
    Gets the intermediate layer representations of the loader for a specific chunk of batches.
    Gets `num_batches` representations for each intermediate layer in the model,
    starting after `batch_skip_offset` batches.
    Saves the representations as tensors in `save_path`.
    NOTE: Must be done on a GPU
    """

    # load model and dataset first
    dataset, _, _ = load_dataset(data_path="/u/hectorxm/distribution-shift/dataset")
    model = model_utils.make_and_restore_model(
        arch='resnet18',
        dataset=dataset,
        resume_path=model_path,
    )[0].to(_DEVICE)
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
        if i < batch_skip_offset:
            if verbose and i == 0: # Print only once per call if skipping
                print(f"Skipping batches until offset {batch_skip_offset}...")
            continue

        if i >= batch_skip_offset + num_batches:
            if verbose:
                print(f"Processed {num_batches} batches starting from offset {batch_skip_offset}. Stopping.")
            break

        if verbose:
            print(f"Processing batch {i} (effective batch index {i - batch_skip_offset} for this chunk) of {num_batches} for this chunk.")
        # move to GPU
        images = images.to(_DEVICE)
        labels = labels.to(_DEVICE)
        if verbose:
            print(f"Images shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
        # create the small and large epsilon adversarial images to pass in
        _, small_eps_images = model(images, labels, make_adv=True, **L2_SMALL_EPS_ATTACK_PARAMS)
        _, large_eps_images = model(images, labels, make_adv=True, **L2_LARGE_EPS_ATTACK_PARAMS)
        
        # Ensure adversarial images are on the correct device
        small_eps_images = small_eps_images.to(_DEVICE)
        large_eps_images = large_eps_images.to(_DEVICE)

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

        # get the intermediate layer representations for small epsilon adversarial images
        with nnsight_model.trace(small_eps_images) as trace:
            small_eps_save_proxy = {}
            for layer_path in LAYERS_TO_INVESTIGATE:
                module_proxy = nnsight_model
                for part in layer_path.split('.'):
                    module_proxy = getattr(module_proxy, part)
                small_eps_save_proxy[layer_path] = module_proxy.output.save()
        
        # clear cache
        torch.cuda.empty_cache()

        # get the intermediate layer representations for large epsilon adversarial images
        with nnsight_model.trace(large_eps_images) as trace:
            large_eps_save_proxy = {}
            for layer_path in LAYERS_TO_INVESTIGATE:
                module_proxy = nnsight_model
                for part in layer_path.split('.'):
                    module_proxy = getattr(module_proxy, part)
                large_eps_save_proxy[layer_path] = module_proxy.output.save()
        # save proxy representations into the representations dictionary
        for layer in LAYERS_TO_INVESTIGATE:
            representations[layer]["natural_images"].append(natural_save_proxy[layer].value.detach().cpu())
            representations[layer]["low_eps_images"].append(small_eps_save_proxy[layer].value.detach().cpu())
            representations[layer]["high_eps_images"].append(large_eps_save_proxy[layer].value.detach().cpu())

    # save the representations
    torch.save(representations, save_path)

    if verbose:
        print(f"Saved representations for batches {batch_skip_offset} to {batch_skip_offset + num_batches -1} at {save_path}")

    return representations

def create_labels_for_representations(list_of_factuals, list_of_counterfactuals, shuffle=True, verbose=False):
    """
    Creates labels given the list of factuals and counterfactuals, as returned in the 
    `get_intermediate_layer_representations` function. Returns a data loader with the representations attached
    to their corresponding labels. This will be used to train a small linear classifier to predict the label of a given representation.
    To be used for probing. 

    NOTE: Assumes a 128 batch size.
    """
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

    for batch_idx, (data, target) in enumerate(dataloader):
        print(f"Batch {batch_idx} shape: {data.shape}")
        print(f"Batch {batch_idx} labels shape: {target.shape}")
        print(f"Batch {batch_idx} labels: {target}")
        break

    return dataloader

def sanity_check(dataloader, shuffle=False, verbose=False):
    """
    Creates two sanity-check dataloaders from an input dataloader:
    1. Flipped Labels: Same data, but labels are flipped (0->1, 1->0).
    2. Random Labels: Same data, but labels are randomly assigned (0 or 1).

    Args:
        dataloader (DataLoader): The input data loader (expects data, label pairs).
        shuffle (bool): Whether to shuffle the new dataloaders.
        verbose (bool): If True, print shapes and info.

    Returns:
        tuple(DataLoader, DataLoader): A tuple containing the flipped_labels_loader
                                       and the random_labels_loader.
    """
    torch.manual_seed(42) # for reproducible random labels

    all_data = []
    original_labels = []

    # Extract all data and labels from the input dataloader
    for data, labels in dataloader:
        all_data.append(data)
        original_labels.append(labels)

    # Concatenate batches
    all_data_tensor = torch.cat(all_data, dim=0)
    original_labels_tensor = torch.cat(original_labels, dim=0)

    if verbose:
        print(f"[Sanity Check] Input data shape: {all_data_tensor.shape}")
        print(f"[Sanity Check] Input labels shape: {original_labels_tensor.shape}")
        print(f"[Sanity Check] Original first 10 labels: {original_labels_tensor[:10]}")

    # 1. Create flipped labels
    # Assuming binary classification with labels 0 and 1
    flipped_labels_tensor = 1 - original_labels_tensor
    if verbose:
        print(f"[Sanity Check] Flipped first 10 labels: {flipped_labels_tensor[:10]}")

    # 2. Create random labels
    random_labels_tensor = torch.randint(0, 2, size=original_labels_tensor.shape, dtype=torch.long)
    if verbose:
        print(f"[Sanity Check] Random first 10 labels: {random_labels_tensor[:10]}")


    # Get batch size from original loader or use a default
    batch_size = dataloader.batch_size if hasattr(dataloader, 'batch_size') and dataloader.batch_size else 128

    # Create new datasets and dataloaders
    flipped_dataset = TensorDataset(all_data_tensor, flipped_labels_tensor)
    random_dataset = TensorDataset(all_data_tensor, random_labels_tensor)

    flipped_loader = DataLoader(flipped_dataset, batch_size=batch_size, shuffle=shuffle)
    random_loader = DataLoader(random_dataset, batch_size=batch_size, shuffle=shuffle)

    return flipped_loader, random_loader

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

    # just returns the list of all batches, not flattened or concatenated, since dataloader will do that
    return list_of_batches

    # NOTE: Potential bug here, going to change and see if results are different
    # # Concatenate all batches
    # all_reps_raw = torch.cat(list_of_batches, dim=0)

    # # Flatten the representations
    # all_reps_flattened = torch.flatten(all_reps_raw, start_dim=1)

    # if verbose:
    #     print(f"Retrieving representations for layer: {layer_name}, type: {rep_type}")
    #     print(f"  Shape of a single batch: {list_of_batches[0].shape}")
    #     print(f"  Number of batches: {len(list_of_batches)}")
    #     print(f"  Shape after concatenating batches (raw): {all_reps_raw.shape}")
    #     print(f"  Shape after flattening (returned): {all_reps_flattened.shape}")

    # return all_reps_flattened

def consolidate_chunked_representations(chunk_file_paths, final_save_path, verbose=False):
    """
    Consolidates multiple chunked representation files into a single file.

    Args:
        chunk_file_paths (list[str]): List of paths to the .pt files for each chunk.
        final_save_path (str): Path to save the final consolidated representations .pt file.
        verbose (bool): If True, print progress.
    """
    if not chunk_file_paths:
        print("Warning: No chunk file paths provided for consolidation.")
        return

    # Initialize the consolidated representations dictionary based on the structure of the first chunk
    # This assumes all chunks have the same layers and representation types.
    first_chunk_data = torch.load(chunk_file_paths[0])
    consolidated_representations = {layer_name: {
        "natural_images": [],
        "low_eps_images": [],
        "high_eps_images": []
    } for layer_name in first_chunk_data.keys()}

    if verbose:
        print(f"Initialized consolidated_representations with layers: {list(consolidated_representations.keys())}")

    for i, file_path in enumerate(chunk_file_paths):
        if verbose:
            print(f"Processing chunk {i+1}/{len(chunk_file_paths)}: {file_path}")
        try:
            chunk_data = torch.load(file_path)
            for layer_name in consolidated_representations.keys():
                if layer_name in chunk_data:
                    for rep_type in consolidated_representations[layer_name].keys():
                        if rep_type in chunk_data[layer_name] and chunk_data[layer_name][rep_type]:
                            consolidated_representations[layer_name][rep_type].extend(chunk_data[layer_name][rep_type])
                        elif verbose:
                            print(f"  Warning: Rep type '{rep_type}' missing or empty in layer '{layer_name}' of chunk {file_path}")
                elif verbose:
                    print(f"  Warning: Layer '{layer_name}' missing in chunk {file_path}")
        except FileNotFoundError:
            print(f"Error: Chunk file not found: {file_path}. Skipping this file.")
            continue
        except Exception as e:
            print(f"Error loading or processing chunk file {file_path}: {e}. Skipping this file.")
            continue

    if verbose:
        for layer_name in consolidated_representations.keys():
            for rep_type in consolidated_representations[layer_name].keys():
                num_batches_total = len(consolidated_representations[layer_name][rep_type])
                print(f"  Layer '{layer_name}', Rep_type '{rep_type}': Total batches after consolidation = {num_batches_total}")
                if num_batches_total > 0:
                    print(f"    Shape of first batch: {consolidated_representations[layer_name][rep_type][0].shape}")

    try:
        torch.save(consolidated_representations, final_save_path)
        if verbose:
            print(f"Successfully saved consolidated representations to {final_save_path}")
    except Exception as e:
        print(f"Error saving consolidated representations to {final_save_path}: {e}")

    return consolidated_representations

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

def train_probing_model(dataloader, test_loader, save_path, config=DEFAULT_CONFIG, verbose=False):
    """
    Trains a probing model on the given dataloader, evaluates on test_loader,
    and returns training and validation loss histories.

    Args:
        dataloader (DataLoader): Training data loader.
        test_loader (DataLoader): Validation (test) data loader.
        save_path (str): Path to save the trained model state_dict.
        config (dict): Training configuration (lr, epochs, batch_size).
        verbose (bool): If True, print training progress.

    Returns:
        tuple(list, list): train_loss_history, val_loss_history (epoch-wise average losses).
    """
    # --- FIX 1: Calculate correct flattened feature size ---
    sample_data = dataloader.dataset[0][0] # Get one sample tensor
    # Calculate flattened size (e.g., 64*32*32 = 65536)
    # Handle potential scalar features (e.g., from model.linear output if you probe that later)
    num_features = sample_data.numel() # .numel() calculates the total number of elements
    if verbose:
        print(f"  Initializing ProbingModel with num_features: {num_features}") # Should print 65536 for layer 1
    # ---------------------------------------------------------

    # Initialize model with the *correct* number of features
    model = ProbingModel(num_features=num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    model.to(_DEVICE)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(config["epochs"]):
        model.train() # Set model to training mode
        running_train_loss = 0.0
        num_train_batches = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(_DEVICE), target.to(_DEVICE)

            # Flatten the batch before model
            original_shape = data.shape
            try:
                data = data.view(data.size(0), -1) # Flattens dimensions after the first (batch)
            except RuntimeError as e:
                 print(f"Error reshaping data in train_probing_model (batch {batch_idx}). Original shape: {original_shape}. Error: {e}")
                 if data.numel() == 0:
                     continue
                 else:
                     raise e
            # if verbose and batch_idx == 0 and epoch == 0: # Print shape info once per training run
            #      print(f"  Flattened data shape from {original_shape} to {data.shape}")

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            num_train_batches += 1

            # Optional: Print batch loss if very verbose needed
            # if verbose:
            #     print(f"Epoch {epoch}, Batch {batch_idx}, Batch Loss: {loss.item():.4f}")

        # Calculate average training loss for the epoch
        avg_epoch_train_loss = running_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        train_loss_history.append(avg_epoch_train_loss)

        # Evaluate on validation set at the end of the epoch
        avg_epoch_val_loss, val_accuracy = evaluate_probe(model, test_loader, criterion, _DEVICE, verbose=False)
        val_loss_history.append(avg_epoch_val_loss)

        if verbose:
            print(f"Epoch {epoch}: Avg Train Loss: {avg_epoch_train_loss:.4f}, Avg Val Loss: {avg_epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # save the model
    try:
        torch.save(model.state_dict(), save_path)
        if verbose:
            print(f"Saved trained model to {save_path}")
    except Exception as e:
        print(f"Error saving model to {save_path}: {e}")

    return train_loss_history, val_loss_history

def plot_losses(train_losses, val_losses, save_path):
    """
    Plots training and validation losses on the same graph and saves it.

    Args:
        train_losses (list): List of average training losses per epoch.
        val_losses (list): List of average validation losses per epoch.
        save_path (str): Path to save the plot image.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure() # Create a new figure
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss - {os.path.basename(save_path).replace("_losses.png", "")}')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(save_path)
        print(f"Saved loss plot to {save_path}")
    except Exception as e:
        print(f"Error saving plot to {save_path}: {e}")
    plt.close() # Close the figure to free memory

def train_all_probes(representations_path, save_folder, config=DEFAULT_CONFIG, verbose=False, plot_loss=False):
    """
    Trains a probing model on all layers and types of representations.
    Optionally plots and saves the training/validation loss curves.
    """
    # create save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    # first for all layers
    for layer in LAYERS_TO_INVESTIGATE:
        if verbose:
            print(f"Training probing model for layer: {layer}")
        # now iterate over all three combinations of representations, note that now these are lists of batches, not tensors that have been flattened
        natural_rep = get_representation(representations_path, layer, "natural_images", verbose)
        low_eps_rep = get_representation(representations_path, layer, "low_eps_images", verbose)
        high_eps_rep = get_representation(representations_path, layer, "high_eps_images", verbose)
        # create dataloader
        natural_high_eps_loader = create_labels_for_representations(natural_rep, high_eps_rep, shuffle=True, verbose=verbose)
        low_high_eps_loader = create_labels_for_representations(low_eps_rep, high_eps_rep, shuffle=True, verbose=verbose)
        natural_low_eps_loader = create_labels_for_representations(natural_rep, low_eps_rep, shuffle=True, verbose=verbose)

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
        train_loss_history_natural_high_eps, val_loss_history_natural_high_eps = train_probing_model(natural_high_eps_train_loader, natural_high_eps_test_loader, save_folder+f"/{layer}_natural_high_eps.pt", config, verbose)
        train_loss_history_low_high_eps, val_loss_history_low_high_eps = train_probing_model(low_high_eps_train_loader, low_high_eps_test_loader, save_folder+f"/{layer}_low_high_eps.pt", config, verbose)
        train_loss_history_natural_low_eps, val_loss_history_natural_low_eps = train_probing_model(natural_low_eps_train_loader, natural_low_eps_test_loader, save_folder+f"/{layer}_natural_low_eps.pt", config, verbose)

        if plot_loss:
            plot_losses(train_loss_history_natural_high_eps, val_loss_history_natural_high_eps, save_folder+f"/{layer}_natural_high_eps_losses.png")
            plot_losses(train_loss_history_low_high_eps, val_loss_history_low_high_eps, save_folder+f"/{layer}_low_high_eps_losses.png")
            plot_losses(train_loss_history_natural_low_eps, val_loss_history_natural_low_eps, save_folder+f"/{layer}_natural_low_eps_losses.png")

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

def evaluate_probe(model, dataloader, criterion, device, verbose=False):
    """Helper function to evaluate a probing model on a dataloader."""
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():  # Disable gradient calculations
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            # Flatten the batch before model
            original_shape = data.shape
            try:
                data = data.view(data.size(0), -1) # Flattens dimensions after the first (batch)
            except RuntimeError as e:
                print(f"Error reshaping data in evaluate_probe. Original shape: {original_shape}, Target shape: ({data.size(0)}, -1). Error: {e}")
                # Handle potential empty batches or other issues
                if data.numel() == 0: # If batch is empty, skip
                    continue
                else: # Re-raise if it's another issue
                    raise e

            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            num_batches += 1

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    # It's good practice to set model back to train mode if called during training loop
    # model.train() # Caller should handle this if needed.
    return average_loss, accuracy

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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Remove old device logic

    # Determine input features from the dataset
    try:
        sample_data = dataloader.dataset[0][0] # Get one sample tensor
        # Calculate flattened size (e.g., 64*32*32 = 65536)
        # Handle potential scalar features (e.g., from model.linear output if you probe that later)
        num_features = sample_data.numel() # .numel() calculates the total number of elements
    except IndexError:
        print(f"Error: DataLoader's dataset seems empty for {model_path}. Cannot determine features.")
        return 0.0

    if verbose:
        print(f"Loading classifier from {model_path} \n with num_features: {num_features}")
    model = ProbingModel(num_features=num_features)
    try:
        model.load_state_dict(torch.load(model_path, map_location=_DEVICE)) # Use _DEVICE for map_location
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return 0.0
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        return 0.0

    model.to(_DEVICE) # Use _DEVICE
    model.eval()  # Set model to evaluation mode

    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculations
        loss_fn = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        for data, target in dataloader:
            data, target = data.to(_DEVICE), target.to(_DEVICE) # Use _DEVICE

            # --- FIX 2: Flatten the batch before model ---
            # Reshape data from [BatchSize, C, H, W] to [BatchSize, Features]
            original_shape = data.shape
            data = data.view(data.size(0), -1) # Flattens dimensions after the first (batch)
            # ----------------------------------------------

            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Calculate and accumulate loss
            loss = loss_fn(outputs, target)
            print(f"Loss: {loss.item()}")
            total_loss += loss.item() * target.size(0)
    accuracy = 100 * correct / total if total > 0 else 0

    if verbose:
        print(f'Accuracy of the model {model_path.split("/")[-1]} on the test data: {accuracy:.2f}%')
        print(f"Total loss: {total_loss}")
    return accuracy, total_loss

# TODO: Finish this for a complete test of all probes
def test_all_probes(save_folder, loaders_path, classifiers_path, verbose=False, save_plots=False):
    """
    Tests all probes on the test loaders and train loaders.
    Assumes standard naming conventions for classifier files and for loader files.
    """
    

    # # currently not used, but could be used if we expect different train/test behaviors
    # natural_high_eps_train_loader = torch.load(loaders_path+"/natural_high_eps_train_loader.pt")
    # low_high_eps_train_loader = torch.load(loaders_path+"/low_high_eps_train_loader.pt")
    # natural_low_eps_train_loader = torch.load(loaders_path+"/natural_low_eps_train_loader.pt")

    # set up natural, low, and high configurations
    correlate_configurations = ["natural_high_eps", "low_high_eps", "natural_low_eps"]

    # store results as dictionary
    results = {}

    print(f"Testing all probes. Results will be saved in: {save_folder}")
    for layer in LAYERS_TO_INVESTIGATE:
        print(f"Testing layer: {layer}")
        # load the test and train loaders, CHANGE NAMES IF NECESSARY
        natural_high_eps_test_loader = torch.load(loaders_path+f"/{layer}_natural_high_eps_test_loader.pt")
        low_high_eps_test_loader = torch.load(loaders_path+f"/{layer}_low_high_eps_test_loader.pt")
        natural_low_eps_test_loader = torch.load(loaders_path+f"/{layer}_natural_low_eps_test_loader.pt")

        for config in correlate_configurations:
            print(f"Testing config: {config}")
            model_path = classifiers_path+"/"+layer+"_"+config+".pt"
            if config == "natural_high_eps":
                accuracy, total_loss = test_probing_model(dataloader=natural_high_eps_test_loader, model_path=model_path, verbose=verbose)
            elif config == "low_high_eps":
                accuracy, total_loss = test_probing_model(dataloader=low_high_eps_test_loader, model_path=model_path, verbose=verbose)
            elif config == "natural_low_eps":
                accuracy, total_loss = test_probing_model(dataloader=natural_low_eps_test_loader, model_path=model_path, verbose=verbose)
            
            results[layer+"_"+config] = {"accuracy": accuracy, "total_loss": total_loss}

    # save results as a .pt file
    torch.save(results, save_folder+"/probing_test_results.pt")

    if save_plots:
        # plot the results, makes 6 plots, two plots for each of the three configuration: one for accuracy and one for loss
        # each plot will be a bar graph that shows every layer given a certain configuration. 
        
        layers_for_plot = [l.replace("model.", "") for l in LAYERS_TO_INVESTIGATE]
        x_indices = np.arange(len(layers_for_plot))

        for config_name in correlate_configurations:
            accuracies = [results[f"{layer}_{config_name}"]["accuracy"] for layer in LAYERS_TO_INVESTIGATE]
            losses = [results[f"{layer}_{config_name}"]["total_loss"] for layer in LAYERS_TO_INVESTIGATE]
            
            config_title_name = config_name.replace("_", " ").title()

            # --- Accuracy Plot for current configuration ---
            fig_acc, ax_acc = plt.subplots(figsize=(18, 10)) # Wider figure for more layers
            bars_acc = ax_acc.bar(x_indices, accuracies, color='skyblue')

            ax_acc.set_ylabel('Accuracy (%)')
            ax_acc.set_xlabel('Layer')
            ax_acc.set_title(f'Probing Model Accuracy for {config_title_name} by Layer')
            ax_acc.set_xticks(x_indices)
            ax_acc.set_xticklabels(layers_for_plot, rotation=60, ha="right") # Rotate labels
            ax_acc.grid(True, linestyle='--', alpha=0.7)

            # Helper to add labels to bars
            def autolabel(rects, ax):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.1f}', # Display one decimal place
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

            autolabel(bars_acc, ax_acc)
            fig_acc.tight_layout()
            acc_plot_path = os.path.join(save_folder, f"accuracy_plot_{config_name}.png")
            plt.savefig(acc_plot_path)
            print(f"Saved accuracy plot to {acc_plot_path}")
            plt.close(fig_acc)

            # --- Loss Plot for current configuration ---
            fig_loss, ax_loss = plt.subplots(figsize=(18, 10)) # Wider figure
            bars_loss = ax_loss.bar(x_indices, losses, color='salmon')

            ax_loss.set_ylabel('Total Loss')
            ax_loss.set_xlabel('Layer')
            ax_loss.set_title(f'Probing Model Total Loss for {config_title_name} by Layer')
            ax_loss.set_xticks(x_indices)
            ax_loss.set_xticklabels(layers_for_plot, rotation=60, ha="right") # Rotate labels
            ax_loss.grid(True, linestyle='--', alpha=0.7)

            autolabel(bars_loss, ax_loss)
            fig_loss.tight_layout()
            loss_plot_path = os.path.join(save_folder, f"loss_plot_{config_name}.png")
            plt.savefig(loss_plot_path)
            print(f"Saved total loss plot to {loss_plot_path}")
            plt.close(fig_loss)

    return results


if __name__ == "__main__":
    MODEL_PATH = "/u/hectorxm/distribution-shift/models/149_checkpoint.pt"
    DATASET_PATH = "/u/hectorxm/distribution-shift/dataset"
    BASE_SAVE_DIR = "/u/hectorxm/distribution-shift/interpretability"
    
    # --- Configuration for Batch Processing ---
    CHUNK_SIZE = 10  # Number of batches to process at a time
    DESIRED_TOTAL_BATCHES = 100 # Target total number of batches
    
    CHUNKED_REPS_SAVE_DIR = os.path.join(BASE_SAVE_DIR, "L-2_representations_chunked")
    CONSOLIDATED_REPS_FILENAME = f"L-2_representations_{DESIRED_TOTAL_BATCHES}batches_fullLayers.pt"
    CONSOLIDATED_REPS_SAVE_PATH = os.path.join(BASE_SAVE_DIR, CONSOLIDATED_REPS_FILENAME)
    
    # Define base directory for this run's probes and results based on total batches
    RUN_SPECIFIC_DIR = os.path.join(BASE_SAVE_DIR, f"L-2_probes_all_layers_{DESIRED_TOTAL_BATCHES}batches")
    PROBES_SAVE_DIR = os.path.join(RUN_SPECIFIC_DIR, "probes") # Models and loaders subdir will be here
    RESULTS_SAVE_DIR = os.path.join(RUN_SPECIFIC_DIR, "test_results")

    # os.makedirs(CHUNKED_REPS_SAVE_DIR, exist_ok=True)
    # os.makedirs(PROBES_SAVE_DIR, exist_ok=True) 
    # os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

    # _, train_loader, _ = load_dataset(data_path=DATASET_PATH)

    # # --- 1. Generate Representations in Chunks ---
    # chunk_file_paths = []
    # num_loops = (DESIRED_TOTAL_BATCHES + CHUNK_SIZE - 1) // CHUNK_SIZE # Ceiling division

    # print(f"Starting representation generation for {DESIRED_TOTAL_BATCHES} batches in chunks of {CHUNK_SIZE}...")
    # for i in range(num_loops):
    #     current_offset = i * CHUNK_SIZE
    #     # Determine the number of batches for this specific chunk
    #     batches_in_this_chunk = min(CHUNK_SIZE, DESIRED_TOTAL_BATCHES - current_offset)
            
    #     chunk_save_path = os.path.join(CHUNKED_REPS_SAVE_DIR, f"representations_chunk_{i}_offset_{current_offset}_nbatches_{batches_in_this_chunk}.pt")
    #     print(f"--- Processing Chunk {i+1}/{num_loops} (Offset: {current_offset}, Batches: {batches_in_this_chunk}) ---")
    #     print(f"Saving chunk to: {chunk_save_path}")
        
    #     # Potentially, only run if file doesn't exist to allow resuming
    #     if not os.path.exists(chunk_save_path):
    #         get_intermediate_layer_representations(
    #             MODEL_PATH,
    #             train_loader,
    #             save_path=chunk_save_path,
    #             num_batches=batches_in_this_chunk,
    #             batch_skip_offset=current_offset,
    #             verbose=True
    #         )
    #     else:
    #         print(f"Chunk file {chunk_save_path} already exists. Skipping generation for this chunk.")
    #     chunk_file_paths.append(chunk_save_path)
    
    # # Ensure all expected chunk files are present before consolidation
    # actual_chunk_files_present = [p for p in chunk_file_paths if os.path.exists(p)]
    # if len(actual_chunk_files_present) != num_loops:
    #     print(f"Warning: Expected {num_loops} chunk files, but only found {len(actual_chunk_files_present)}. Proceeding with available files.")
    
    # print("Finished generating all representation chunks.")

    # # --- 2. Consolidate Chunked Representations ---
    # if not os.path.exists(CONSOLIDATED_REPS_SAVE_PATH) or len(actual_chunk_files_present) > 0 : # Only consolidate if target doesn't exist or new chunks were processed
    #     print(f"Consolidating {len(actual_chunk_files_present)} representation chunks into {CONSOLIDATED_REPS_SAVE_PATH}...")
    #     consolidate_chunked_representations(
    #         actual_chunk_files_present, # Use only files that were actually created/found
    #         CONSOLIDATED_REPS_SAVE_PATH,
    #         verbose=True
    #     )
    #     print("Finished consolidating representations.")
    # else:
    #     print(f"Consolidated file {CONSOLIDATED_REPS_SAVE_PATH} already exists and no new chunks processed. Skipping consolidation.")


    # --- 3. Train All Probes using Consolidated Representations ---
    print(f"Training all probes using consolidated representations from: {CONSOLIDATED_REPS_SAVE_PATH}")
    print(f"Probes and related files will be saved in: {PROBES_SAVE_DIR}")
    print(f"Training all probes with config: {DEFAULT_CONFIG}")
    # Check if probes already exist for this configuration to avoid retraining.
    # A simple check could be if the directory PROBES_SAVE_DIR already has .pt model files for layers.
    # This is a basic check; more robust would be to check for specific expected output files.
    # For now, we proceed to train_all_probes, which itself might have internal checks or overwrite.
    train_all_probes(
        representations_path=CONSOLIDATED_REPS_SAVE_PATH,
        save_folder=PROBES_SAVE_DIR, 
        config=DEFAULT_CONFIG,
        verbose=True,
        plot_loss=True
    )
    print("Finished training all probes.")

    # --- 4. Test All Probes ---
    # Loaders for testing are saved by train_all_probes inside PROBES_SAVE_DIR/loaders/test
    loaders_for_testing_path = os.path.join(PROBES_SAVE_DIR, "loaders", "test")
    # Probes (classifiers) are saved by train_all_probes directly in PROBES_SAVE_DIR
    classifiers_for_testing_path = PROBES_SAVE_DIR 

    print(f"Testing all probes. Results will be saved in: {RESULTS_SAVE_DIR}")
    if not os.path.exists(loaders_for_testing_path):
        print(f"Error: Test loaders directory not found at {loaders_for_testing_path}. Cannot run tests. Ensure train_all_probes ran successfully.")
    else:
        results = test_all_probes(
            save_folder=RESULTS_SAVE_DIR, 
            loaders_path=loaders_for_testing_path,
            classifiers_path=classifiers_for_testing_path,
            verbose=False, 
            save_plots=True
        )
        print(f"Finished testing all probes. Results summary saved in {RESULTS_SAVE_DIR}.")
        # print(f"Detailed results: {results}") # Optionally print full results dictionary

    # Example of how you might have previously run parts (for reference, now handled by the new workflow)
    # Original test_all_probes call:
    # results = test_all_probes(save_folder="/u/hectorxm/distribution-shift/interpretability/probes_all_layers/test_results", 
    #                           loaders_path="/u/hectorxm/distribution-shift/interpretability/probes_all_layers/loaders/test", 
    #                           classifiers_path="/u/hectorxm/distribution-shift/interpretability/probes_all_layers/probes", 
    #                           verbose=True, save_plots=True)
    # Original train_all_probes call:
    # train_all_probes(representations_path="/u/hectorxm/distribution-shift/interpretability/representations_10batches_fullLayers.pt", 
    #                  save_folder="/u/hectorxm/distribution-shift/interpretability/probes_all_layers", 
    #                  config=DEFAULT_CONFIG, verbose=True, plot_loss=True)
