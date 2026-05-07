"""
Script 03: Train Mini-Transformer for all forecast horizons.
Same interface as 02_train_gru.py.
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src import config as cfg
from src.data_loading import load_all_runs
from src.target_reconstruction import reconstruct_targets
from src.splitting import split_runs
from src.windowing import build_dataset
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
    _, obs_mask, y_profile = reconstruct_targets(df["at400_frac"].values, df["label"].values)
    profiles[run_id] = y_profile
    masks[run_id]    = obs_mask

train_runs, val_runs, _ = split_runs(runs)
feat_scaler   = load_scaler(cfg.OUT_METRICS / "feature_scaler.pkl")
target_scaler = load_scaler(cfg.OUT_METRICS / "target_scaler.pkl")

for h in horizons:
    print(f"\n=== Transformer | h={h} ===")
    X_tr, y_tr, m_tr, k_tr = build_dataset(
        train_runs, profiles, masks, h,
        feature_scaler=feat_scaler, target_scaler=target_scaler
    )
    X_va, y_va, m_va, k_va = build_dataset(
        val_runs, profiles, masks, h,
        feature_scaler=feat_scaler, target_scaler=target_scaler
    )
    print(f"  Train samples: {len(X_tr)}, Val samples: {len(X_va)}")

    n_features = X_tr.shape[2]
    model = build_model("transformer", n_features, cfg)
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
