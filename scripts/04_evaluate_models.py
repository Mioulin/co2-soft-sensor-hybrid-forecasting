"""
Script 04: Evaluate all models on held-out test run (140207_1).

Outputs:
  outputs/metrics/results_table.csv  - full comparison table
  outputs/predictions/*.npy          - raw predictions per model/horizon
  outputs/figures/                   - evaluation plots
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src import config as cfg
from src.data_loading import load_all_runs
from src.target_reconstruction import reconstruct_targets
from src.splitting import split_runs
from src.windowing import build_dataset
from src.models import build_model
from src.training import predict, make_loader
from src.metrics import (
    compute_all_metrics, per_point_metrics,
    persistence_predictions, kinetic_predictions,
)
from src.plotting import (
    plot_predictions_per_point, plot_scatter_pred_vs_target,
    plot_rmse_by_horizon, plot_per_point_rmse_heatmap, plot_loss_curves,
)
from src.utils import set_seed, load_scaler, checkpoint_name, inverse_transform_predictions

set_seed()

print("Loading data...")
runs     = load_all_runs(cfg.ALL_RUNS)
profiles, masks = {}, {}
for run_id, run_data in runs.items():
    df = run_data["df"]
    _, obs_mask, y_profile = reconstruct_targets(df["at400_frac"].values, df["label"].values)
    profiles[run_id] = y_profile
    masks[run_id]    = obs_mask

train_runs, val_runs, test_runs = split_runs(runs)
feat_scaler   = load_scaler(cfg.OUT_METRICS / "feature_scaler.pkl")
target_scaler = load_scaler(cfg.OUT_METRICS / "target_scaler.pkl")

all_results     = []
per_point_store = {}
device = torch.device("cpu")

for h in cfg.HORIZONS:
    print(f"\n--- Horizon h={h} ---")

    # Build test windows
    X_te, y_te, m_te, k_te = build_dataset(
        test_runs, profiles, masks, h,
        feature_scaler=feat_scaler, target_scaler=target_scaler
    )
    # Inverse-transform targets back to fractional CO2 for evaluation
    y_te_real = inverse_transform_predictions(y_te, target_scaler)
    k_te_real = k_te  # kinetic never scaled

    # --- Persistence baseline ---
    p_pers, t_pers, m_pers = persistence_predictions(
        test_runs, profiles, masks, h, cfg.WINDOW_LENGTH
    )
    all_results.append(compute_all_metrics(p_pers, t_pers, m_pers, "persistence", h))

    # --- Kinetic baseline ---
    p_kn, t_kn, m_kn = kinetic_predictions(
        test_runs, profiles, masks, h, cfg.WINDOW_LENGTH
    )
    all_results.append(compute_all_metrics(p_kn, t_kn, m_kn, "kinetic", h))

    # --- Neural models ---
    for arch in ["gru", "transformer"]:
        ckpt = cfg.OUT_CHECKPOINTS / checkpoint_name(arch, h)
        if not ckpt.exists():
            print(f"  Checkpoint not found: {ckpt}, skipping")
            continue

        n_features = X_te.shape[2]
        model = build_model(arch, n_features, cfg)
        model.load_state_dict(torch.load(ckpt, weights_only=True))

        loader = make_loader(X_te, y_te, m_te, k_te, shuffle=False)
        preds_sc, _, _ = predict(model, loader, device)

        # Inverse-transform to fractional CO2
        preds_real = inverse_transform_predictions(preds_sc, target_scaler)

        # Save raw predictions
        cfg.OUT_PREDICTIONS.mkdir(parents=True, exist_ok=True)
        np.save(cfg.OUT_PREDICTIONS / f"pred_{arch}_h{h:02d}.npy", preds_real)
        np.save(cfg.OUT_PREDICTIONS / f"target_h{h:02d}.npy",      y_te_real)
        np.save(cfg.OUT_PREDICTIONS / f"mask_h{h:02d}.npy",        m_te)
        np.save(cfg.OUT_PREDICTIONS / f"kinetic_h{h:02d}.npy",     k_te_real)

        all_results.append(compute_all_metrics(preds_real, y_te_real, m_te, arch, h))

        # Per-point metrics (h=1 only for heatmap)
        if h == 1:
            per_point_store[arch] = per_point_metrics(preds_real, y_te_real, m_te)

        # Loss curves
        hist_path = cfg.OUT_METRICS / f"history_{arch}_h{h:02d}.npy"
        if hist_path.exists():
            hist = np.load(hist_path, allow_pickle=True).item()
            plot_loss_curves(hist, arch, h)

        # Prediction plots
        plot_predictions_per_point(preds_real, y_te_real, m_te, arch, h)
        plot_scatter_pred_vs_target(preds_real, y_te_real, arch, h)

        print(f"  {arch}: full_rmse={all_results[-1]['full_rmse']:.5f}  "
              f"obs_rmse={all_results[-1]['obs_rmse']:.5f}")

# Summary table
results_df = pd.DataFrame(all_results)
results_df.to_csv(cfg.OUT_METRICS / "results_table.csv", index=False)
print(f"\n=== Results table ===")
print(results_df.pivot_table(index="label", columns="horizon",
      values="obs_rmse").to_string())

# Plots
plot_rmse_by_horizon(results_df)
if per_point_store:
    plot_per_point_rmse_heatmap(per_point_store)

print(f"\nSaved results to {cfg.OUT_METRICS}/results_table.csv")
print("Run 05_run_fusion.py next.")
