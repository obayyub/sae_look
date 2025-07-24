import torch
import numpy as np
from tqdm import tqdm

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def generate_hidden_data(
    dim=128,
    n_features=512,
    n_samples=(2**14),
    sparsity=10,
    importance_base=1.0,
    noise_scale=0.0,
    seed=None,
):
    """Generate synthetic hidden data with sparse overcomplete basis.
    Args:
        dim (int): Dimensionality of the hidden data.
        n_features (int): Number of features in the overcomplete basis.
        n_samples (int): Number of samples to generate.
        sparsity (int): Number of active features per sample.
        importance_base (float): Base for exponential decay.
        noise_scale (float): Scale for noise added to the data.
        seed (int, optional): Random seed for reproducibility.
    """

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # generate random normalized features
    features = np.random.randn(n_features, dim)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # init sparsity weights
    weights = np.zeros((n_samples, n_features))

    feature_importance = importance_base ** np.arange(n_features)

    # generate sparsity weights
    for i in range(n_samples):
        active_feats = np.random.choice(n_features, size=sparsity, replace=False)
        weights[i, active_feats] = feature_importance[active_feats]

    # make hidden data via sum of sparse features
    hidden_data = weights @ features

    # add noise
    if noise_scale > 0:
        noise = np.random.normal(0, noise_scale, hidden_data.shape)
        hidden_data += noise

    # Convert to tensor and move to device
    return torch.tensor(hidden_data, dtype=torch.float32, device=DEVICE), features
