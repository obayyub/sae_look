import torch
import numpy as np
from tqdm import tqdm

# Import DEVICE if defined elsewhere, or define it here
DEVICE = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")

def generate_hidden_data(dim=128, n_features=512, 
                        n_samples=(2**14), sparsity=10):
    #basically want features Y times random vector w where w is sparse, then sum resulting vectors for hidden state
    #overcomplete feature basis?
    features = np.random.randn(n_features, dim)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    #init sparsity weights
    weights = np.zeros((n_samples, n_features))
    #generate sparsity weights
    for i in range(n_samples):
        active_feats = np.random.choice(n_features, size=sparsity, replace=False)
        weights[i, active_feats] = np.ones(sparsity)
    #make hidden data via sum of sparse features
    hidden_data = weights @ features

    # Convert to tensor and move to device
    return torch.tensor(hidden_data, dtype=torch.float32, device=DEVICE), features