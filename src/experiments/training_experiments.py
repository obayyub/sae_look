import torch
from tqdm import tqdm
from collections import defaultdict
from torch import nn
from src.models.sae_model import SAE, BatchedSAE, BatchedSAE_Updated
from src.data.make_sparse_data import generate_hidden_data
import os
import pandas as pd
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Find the project root directory
current_file = Path(__file__).resolve()
project_root = current_file.parent

# Go up directory levels until we find the project root
# (assuming the project root is where src/ is located)
while project_root.name != "src" and project_root.parent != project_root:
    project_root = project_root.parent

# If we found 'src', go one level up to get the actual project root
if project_root.name == "src":
    project_root = project_root.parent

# Define global paths
MODELS_DIR = project_root / "models"
DATA_DIR = project_root / "data"
CSV_PATH = DATA_DIR / "model_results.csv"


def run_experiment():
    sparsity = 50
    hidden_dim = 512
    width_factor = 4

    data = generate_hidden_data(dim=hidden_dim, sparsity=sparsity)
    train_size = int(0.8 * len(data))
    train_data, test_data = data[:train_size], data[train_size:]

    # Use global DEVICE
    if isinstance(train_data, torch.Tensor):
        train_data = train_data.to(DEVICE)
        test_data = test_data.to(DEVICE)

    relu_model = SAE(hidden_dim, width_factor, nn.ReLU()).to(DEVICE)
    print("Training ReLU model...")
    result = relu_model.train(train_data, test_data, output_epoch=True)


def run_DOE():
    sparsities = [5, 10, 20, 30, 40, 50]
    l1_lams = [2e-5, 3e-5, 4e-5, 5e-5, 6e-5]
    hidden_dim = 128
    results = defaultdict(list)
    width_factor = 4

    for sparsity in tqdm(sparsities, position=0, leave=False):
        for l1_lam in l1_lams:
            data = generate_hidden_data(
                dim=hidden_dim, n_samples=2**10, sparsity=sparsity
            )

            indices = torch.randperm(len(data))
            data = data[indices]

            train_size = int(0.8 * len(data))
            train_data, test_data = data[:train_size], data[train_size:]

            # Use global DEVICE
            if isinstance(train_data, torch.Tensor):
                train_data = train_data.to(DEVICE)
                test_data = test_data.to(DEVICE)

            relu_model = SAE(hidden_dim, width_factor, nn.ReLU()).to(DEVICE)
            results[sparsity].append(
                relu_model.train(train_data, test_data, l1_lam=l1_lam)
            )

    return results


def run_batched_DOE(model_type=BatchedSAE):
    sparsities = [5, 10, 20, 30, 40, 50]
    l1_lams = [2e-5, 3e-5, 4e-5, 5e-5, 6e-5]
    hidden_dim = 128
    results = defaultdict(list)
    width_factor = 4
    n_models = len(sparsities)

    # Generate data for all sparsities at once
    train_datasets = []
    test_datasets = []

    n_samples = 2**15  # 32768 samples
    print(f"n_samples: {n_samples}")
    batch_size = 128  # Make sure batch_size < n_samples

    for sparsity in sparsities:
        data, _ = generate_hidden_data(
            dim=hidden_dim, n_samples=n_samples, sparsity=sparsity
        )
        indices = torch.randperm(len(data))
        data = data[indices]
        train_size = int(0.8 * len(data))

        if isinstance(data, torch.Tensor):
            data = data.to(DEVICE)

        train_datasets.append(data[:train_size])
        test_datasets.append(data[train_size:])

    # Stack datasets and ensure they have the correct shape
    train_data_stacked = torch.stack(
        train_datasets
    )  # [n_models, n_samples, hidden_dim]
    test_data_stacked = torch.stack(test_datasets)  # [n_models, n_samples, hidden_dim]

    # Train models for all sparsities simultaneously
    for l1_lam in tqdm(l1_lams, position=0, leave=False):
        batched_model = model_type(
            hidden_dim, n_models=n_models, width_ratio=width_factor
        ).to(DEVICE)
        batch_results = batched_model.train(
            train_data_stacked,
            test_data_stacked,
            batch_size=batch_size,  # Explicitly pass batch_size
            l1_lam=l1_lam,
        )

        # Store results for each sparsity
        for sparsity, result in zip(sparsities, batch_results):
            results[sparsity].append(result)

    return results


def run_batched_DOE_with_features(
    model_type=BatchedSAE_Updated,
    sparsities=[5, 10, 20, 30, 40, 50],
    l1_lams=[2e-5, 3e-5, 4e-5, 5e-5, 6e-5],
    width_factor=4,
    save_to_csv=False,
):
    hidden_dim = 128
    n_basis = 512
    results = defaultdict(list)

    n_models = len(sparsities)
    features_list = []

    # Generate data for all sparsities at once
    train_datasets = []
    test_datasets = []

    n_samples = 2**12  # 32768 samples
    batch_size = 256  # Make sure batch_size < n_samples

    # Create directories if they don't exist
    if save_to_csv:
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)

    for sparsity in sparsities:
        data, features = generate_hidden_data(
            dim=hidden_dim, n_samples=n_samples, sparsity=sparsity
        )
        indices = torch.randperm(len(data))
        data = data[indices]
        train_size = int(0.8 * len(data))

        if isinstance(data, torch.Tensor):
            data = data.to(DEVICE)

        train_datasets.append(data[:train_size])
        test_datasets.append(data[train_size:])
        features_list.append(features)

    # Stack datasets and ensure they have the correct shape
    train_data_stacked = torch.stack(
        train_datasets
    )  # [n_models, n_samples, hidden_dim]
    test_data_stacked = torch.stack(test_datasets)  # [n_models, n_samples, hidden_dim]

    # Train models for all sparsities simultaneously
    for l1_lam in tqdm(l1_lams, position=0, leave=False):
        batched_model = model_type(
            hidden_dim, n_models=n_models, width_ratio=width_factor
        ).to(DEVICE)
        batch_results = batched_model.train(
            train_data_stacked, test_data_stacked, batch_size=batch_size, l1_lam=l1_lam
        )

        # Store results for each sparsity
        for i, sparsity in enumerate(sparsities):
            # Generate a unique model ID
            model_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create result dictionary
            result_dict = {**batch_results[i]}
            
            # Save model if requested
            if save_to_csv:
                model_filename = f"{model_id}.pt"
                model_path = MODELS_DIR / model_filename
                
                # Save the full batched model state dict and metadata
                save_dict = {
                    'model_state_dict': batched_model.state_dict(),
                    'features': features_list[i],
                    'model_index': i,  # Save which model in the batch this is
                    'sparsity': sparsity,
                    'l1_lam': l1_lam,
                    'n_basis': n_basis,
                    'hidden_dim': hidden_dim,
                    'width_factor': width_factor,
                    'model_class': model_type.__name__
                }
                torch.save(save_dict, model_path)
                
                # Add model path and metadata to results
                result_dict["model_path"] = str(model_path)
                result_dict["model_id"] = model_id
                result_dict["timestamp"] = timestamp
                result_dict["n_basis"] = n_basis
                result_dict["sparsity"] = sparsity
                result_dict["model_class"] = model_type.__name__
                result_dict["hidden_dim"] = hidden_dim
                result_dict["width_factor"] = width_factor
                
                # Convert to DataFrame row - exclude tensors, features, weights, and biases
                csv_dict = {k: v for k, v in result_dict.items() 
                          if not isinstance(v, torch.Tensor) 
                          and k != "features"
                          and k not in ["weights", "biases"]}
                df_row = pd.DataFrame([csv_dict])
                
                # Check if CSV exists and append or create new
                if os.path.exists(CSV_PATH):
                    df_row.to_csv(CSV_PATH, mode='a', header=False, index=False)
                else:
                    df_row.to_csv(CSV_PATH, index=False)
            
            # Add features to results dictionary for the return value
            result_dict["features"] = features_list[i]
            
            # Store in results
            results[sparsity].append(result_dict)

    return results


# Example of how to load a model and its features
def load_model_and_features(model_path):
    # Load the saved dictionary
    saved_data = torch.load(model_path)
    
    # Get the model class
    model_class_name = saved_data["model_class"]
    model_classes = {
        "BatchedSAE": BatchedSAE,
        "BatchedSAE_Updated": BatchedSAE_Updated,
    }
    if model_class_name in model_classes:
        model_class = model_classes[model_class_name]
    else:
        raise ValueError(f"Model class {model_class_name} not found")
    
    # Create a new batched model with the same parameters
    batched_model = model_class(
        input_dim=saved_data["hidden_dim"],
        n_models=1,  # We only need one model when loading
        width_ratio=saved_data["width_factor"]
    )
    
    # Extract the specific model's weights from the state dict
    model_index = saved_data["model_index"]
    full_state_dict = saved_data["model_state_dict"]
    
    # Create a new state dict with just the weights for the model we want
    single_model_state_dict = {}
    for k, v in full_state_dict.items():
        if k in ['W_e', 'b_e', 'W_d', 'b_d']:
            # Extract just this model's parameters
            single_model_state_dict[k] = v[model_index:model_index+1]
    
    # Load the state dict
    batched_model.load_state_dict(single_model_state_dict)
    
    return batched_model, saved_data["features"], saved_data
