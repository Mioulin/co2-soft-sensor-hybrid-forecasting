"""
Script 01: build processed dataset.

Loads all runs, reconstructs targets, fits scalers on train,
saves profiles/masks/scalers to outputs/.
"""

import pickle, sys
from pathlib import Path

# Make src importable from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config as cfg
from src.data_loading import load_all_runs, run_summary
from src.target_reconstruction import reconstruct_targets
from src.splitting import split_runs
from src.utils import set_seed, fit_feature_scaler, fit_target_scaler, save_scaler

set_seed()

print("Loading all runs...")
runs = load_all_runs(cfg.ALL_RUNS)

print("\nRun summary:")
print(run_summary(runs).to_string())

# Reconstruct targets for all runs
profiles, masks, sparses = {}, {}, {}
for run_id, run_data in runs.items():
    df = run_data["df"]
    y_sparse, obs_mask, y_profile = reconstruct_targets(
        df["at400_frac"].values, df["label"].values
    )
    profiles[run_id] = y_profile
    masks[run_id]    = obs_mask
    sparses[run_id]  = y_sparse
    print(f"  {run_id}: profile {y_profile.shape}, obs_mask {obs_mask.shape}")

train_runs, val_runs, test_runs = split_runs(runs)

print("\nFitting scalers on train runs...")
feat_scaler   = fit_feature_scaler(train_runs, use_label_onehot=cfg.USE_LABEL_ONEHOT)
target_scaler = fit_target_scaler({r: profiles[r] for r in train_runs})

# Save
out = cfg.OUT_METRICS
out.mkdir(parents=True, exist_ok=True)
save_scaler(feat_scaler,   out / "feature_scaler.pkl")
save_scaler(target_scaler, out / "target_scaler.pkl")

# Save profiles and masks
np.save(out / "profiles.npy",  {r: profiles[r] for r in cfg.ALL_RUNS})
np.save(out / "masks.npy",     {r: masks[r]    for r in cfg.ALL_RUNS})

print(f"\nSaved scalers and processed arrays to {out}")
print("Done. Run 02_train_gru.py next.")
