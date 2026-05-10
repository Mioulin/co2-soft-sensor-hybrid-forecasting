"""
Script 03: Train Mini-Transformer for all forecast horizons using SDAE latent features.

Requires outputs/checkpoints/sdae.pt. Same interface as 02_train_gru.py.

Usage:
  python scripts/03_train_transformer.py [--horizon 1]
  python scripts/03_train_transformer.py
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src import config as cfg
from src.data_loading import load_all_runs
from src.target_reconstruction import reconstruct_targets
from src.splitting import split_runs
from src.windowing import build_dataset
from src.autoencoder import SDAE
from src.models import build_model
from src.training import train_model, make_loader
from src.utils import set_seed, load_scaler, checkpoint_name

set_seed()
parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=None)
args = parser.parse_args()
horizons = [args.horizon] if args.horizon else cfg.HORIZONS

print("Loading data...")
runs = load_all_runs(cfg.ALL_RUNS)
profiles, masks = {}, {}
for run_id, run_data in runs.items():
    df = run_data["df"]
    _, obs_mask, y_profile = reconstruct_targets(
        df["at400_frac"].values, df["label"].values,
        kinetic=run_data["kinetic"], method="kinetic_residual",
    )
    profiles[run_id] = y_profile
    masks[run_id]    = obs_mask

train_runs, val_runs, _ = split_runs(runs)
feat_scaler   = load_scaler(cfg.OUT_METRICS / "feature_scaler.pkl")
target_scaler = load_scaler(cfg.OUT_METRICS / "target_scaler.pkl")

# Load trained SDAE encoder
sdae_ckpt = cfg.OUT_CHECKPOINTS / "sdae.pt"
if not sdae_ckpt.exists():
    raise FileNotFoundError(
        f"SDAE checkpoint not found: {sdae_ckpt}\n"
        "Run scripts/run_pipeline.py Phase 2 first."
    )
sdae = SDAE(n_features=95)  # 89 numeric + 6 one-hot
sdae.load_state_dict(torch.load(sdae_ckpt, weights_only=True))
sdae.eval()
latent_dim = cfg.SDAE_LATENT_DIM
print(f"  Loaded SDAE -> latent_dim={latent_dim}")

for h in horizons:
    print(f"\n=== Transformer | h={h} ===")
    X_tr, y_tr, m_tr, k_tr = build_dataset(
        train_runs, profiles, masks, h,
        feature_scaler=feat_scaler, target_scaler=target_scaler,
        sdae_encoder=sdae,
    )
    X_va, y_va, m_va, k_va = build_dataset(
        val_runs, profiles, masks, h,
        feature_scaler=feat_scaler, target_scaler=target_scaler,
        sdae_encoder=sdae,
    )
    print(f"  Train samples: {len(X_tr)}, Val samples: {len(X_va)}, Feature dim: {X_tr.shape[2]}")

    model = build_model("transformer", latent_dim, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Transformer params: {n_params:,}")

    ckpt = cfg.OUT_CHECKPOINTS / checkpoint_name("transformer", h)
    history = train_model(
        model,
        make_loader(X_tr, y_tr, m_tr, k_tr, shuffle=True),
        make_loader(X_va, y_va, m_va, k_va, shuffle=False),
        checkpoint_path=ckpt,
        verbose=True,
    )
    print(f"  Best val loss: {min(history['val_loss']):.6f}")
    np.save(cfg.OUT_METRICS / f"history_transformer_h{h:02d}.npy", history)

print("\nDone. Run 04_evaluate_models.py next.")
