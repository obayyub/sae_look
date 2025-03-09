import torch
from tqdm import tqdm
from collections import defaultdict
from torch import nn
from src.models.sae_model import SAE, BatchedSAE, BatchedSAE_Updated
from src.data.make_sparse_data import generate_hidden_data
import os
import sqlite3
from datetime import datetime
import pickle

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DB_DIR = PROJECT_ROOT / "db"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
DEFAULT_DB_PATH = DB_DIR / "experiments.db"

def save_model(model, model_id):
    model_path = MODELS_DIR / f"model_{model_id}.pt"
    model_cpu = model.cpu()
    torch.save(model_cpu.state_dict(), model_path)
    return str(model_path.relative_to(PROJECT_ROOT))

def load_model(model_path, model_class, input_dim, width_ratio):
    full_path = PROJECT_ROOT / model_path
    model = model_class(input_dim, width_ratio, nn.ReLU()).to(DEVICE)
    model.load_state_dict(torch.load(full_path), map_location=DEVICE)
    return model

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

def init_database(db_path=DEFAULT_DB_PATH):
    '''Initialize the sqlite database'''
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        model_class TEXT,
        model_path TEXT,
        input_dim INTEGER,
        width_ratio INTEGER,
        batch_size INTEGER,
        l1_lam REAL,
        sparsity INTEGER,
        n_basis_vectors INTEGER,
        n_samples INTEGER,
        mse REAL,
        l0 REAL,
        train_data BLOB
    )''')

    conn.commit()
    conn.close()

def save_experiment(results_dict, model, db_path=DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    if torch.is_tensor(results_dict['train_data']):
        train_data = results_dict['train_data'].detach().cpu()
    else:
        train_data = results_dict['train_data']

    c.execute('''INSERT INTO experiments (
        timestamp,
        model_class,
        input_dim,
        width_ratio,
        batch_size,
        l1_lam,
        sparsity,
        n_basis_vectors,
        n_samples,
        mse,
        l0,
        train_data
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        model.__class__.__name__,
        results_dict['input_dim'],
        results_dict['width_ratio'],
        results_dict['batch_size'],
        results_dict['l1_lam'],
        results_dict['sparsity'],
        results_dict['n_basis_vectors'],
        results_dict['n_samples'],
        results_dict['mse'],
        results_dict['l0'],
        sqlite3.Binary(pickle.dumps(train_data))
    ))

    experiment_id = c.lastrowid

    model_path = save_model(model, experiment_id)
    c.execute('''UPDATE experiments SET model_path = ? WHERE id = ?''', 
              (model_path, experiment_id))

    conn.commit()
    conn.close()

def load_experiment(experiment_id, db_path=DEFAULT_DB_PATH, load_model=True):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('SELECT * FROM experiments WHERE id = ?', (experiment_id,))
    row = c.fetchone()
    if row is None:
        return None
    
    columns = [desc[0] for desc in c.description]
    result_dict = dict(zip(columns, row))

    train_data_shape = (result_dict['n_basis_vectors'], result_dict['n_samples'])
    result_dict['train_data'] = torch.from_numpy(
        pickle.loads(result_dict['train_data'])
    ).reshape(train_data_shape)
    
    if load_model:
        model_class = globals()[result_dict['model_class']]
        result_dict['model'] = load_model(
            result_dict['model_path'], 
            model_class, 
            result_dict['input_dim'], 
            result_dict['width_ratio']
        )

    conn.close()
    return result_dict

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
):
    hidden_dim = 128
    results = defaultdict(list)

    n_models = len(sparsities)
    features_list = []

    # Generate data for all sparsities at once
    train_datasets = []
    test_datasets = []

    n_samples = 2**12  # 32768 samples
    batch_size = 256  # Make sure batch_size < n_samples

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
            # batch_results is a list of dictionaries, one per model
            # Append both the model result and corresponding features
            results[sparsity].append(
                {
                    **batch_results[i],  # Unpack the model result dictionary
                    "features": features_list[i],  # Add the features
                }
            )

    return results
