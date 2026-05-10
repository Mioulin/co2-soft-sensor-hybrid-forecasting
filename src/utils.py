import pickle, random
from pathlib import Path
import numpy as np
import torch
from src.config import SEED

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def save_scaler(scaler, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f: pickle.dump(scaler, f)

def load_scaler(path):
    with open(path, "rb") as f: return pickle.load(f)

def checkpoint_name(arch, horizon): return f"{arch}_h{horizon:02d}.pt"

def inverse_transform_predictions(arr, scaler):
    """Inverse-transform a scaled (n, 6) array back to CO2 fraction units."""
    return scaler.inverse_transform(arr)
