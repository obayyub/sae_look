import torch
from tqdm import tqdm
from collections import defaultdict
from torch import nn
from src.models.sae_model import SAE
from src.data.make_sparse_data import generate_hidden_data

DEVICE = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")

def run_experiment():
    sparsity = 50
    hidden_dim = 128
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
            data = generate_hidden_data(dim=hidden_dim, n_samples=2**10, sparsity=sparsity)
            
            indices = torch.randperm(len(data))
            data = data[indices]
            
            train_size = int(0.8 * len(data))
            train_data, test_data = data[:train_size], data[train_size:]
            
            # Use global DEVICE
            if isinstance(train_data, torch.Tensor):
                train_data = train_data.to(DEVICE)
                test_data = test_data.to(DEVICE)
            
            relu_model = SAE(hidden_dim, width_factor, nn.ReLU()).to(DEVICE)
            results[sparsity].append(relu_model.train(train_data, test_data, l1_lam=l1_lam))
    
    return results