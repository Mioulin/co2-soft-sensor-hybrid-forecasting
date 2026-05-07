"""
Script 05: Inverse-variance and Kalman-like fusion.

Fusion weights estimated on validation run (140313_1).
Final fusion evaluated on test run (140207_1).
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
from src.metrics import compute_all_metrics, kinetic_predictions
from src.fusion import (
    estimate_residual_variances, estimate_residual_covariance,
    inverse_variance_fusion, kalman_fusion, fusion_weights,
    localization_matrix,
)
from src.plotting import (
    plot_fusion_weights, plot_fusion_comparison, plot_covariance_heatmaps,
)
from src.utils import set_seed, load_scaler, checkpoint_name, inverse_transform_predictions

set_seed()

print("Loading data...")
runs = load_all_runs(cfg.ALL_RUNS)
profiles, masks = {}, {}
for run_id, run_data in runs.items():
    df = run_data["df"]
    _, obs_mask, y_profile = reconstruct_targets(df["at400_frac"].values, df["label"].values)
    profiles[run_id] = y_profile
    masks[run_id]    = obs_mask

train_runs, val_runs, test_runs = split_runs(runs)
feat_scaler   = load_scaler(cfg.OUT_METRICS / "feature_scaler.pkl")
target_scaler = load_scaler(cfg.OUT_METRICS / "target_scaler.pkl")
device        = torch.device("cpu")

fusion_results = []

for h in cfg.HORIZONS:
    print(f"\n--- Fusion | h={h} ---")

    # --- Validation windows (for fusion weight estimation) ---
    X_va, y_va, m_va, k_va = build_dataset(
        val_runs, profiles, masks, h,
        feature_scaler=feat_scaler, target_scaler=target_scaler
    )
    y_va_real = inverse_transform_predictions(y_va, target_scaler)
    k_va_real = k_va

    # Kinetic val predictions (for variance estimation)
    p_kn_va, t_kn_va, _ = kinetic_predictions(val_runs, profiles, masks, h, cfg.WINDOW_LENGTH)
    var_kinetic = estimate_residual_variances(p_kn_va, t_kn_va)
    B = estimate_residual_covariance(p_kn_va, t_kn_va, cfg.KALMAN_REG_EPS)

    # --- Test windows ---
    X_te, y_te, m_te, k_te = build_dataset(
        test_runs, profiles, masks, h,
        feature_scaler=feat_scaler, target_scaler=target_scaler
    )
    y_te_real = inverse_transform_predictions(y_te, target_scaler)
    k_te_real = k_te

    for arch in ["gru", "transformer"]:
        ckpt = cfg.OUT_CHECKPOINTS / checkpoint_name(arch, h)
        if not ckpt.exists():
            print(f"  Checkpoint not found: {ckpt}, skipping {arch}")
            continue

        n_features = X_te.shape[2]
        model = build_model(arch, n_features, cfg)
        model.load_state_dict(torch.load(ckpt, weights_only=True))

        # Val predictions (for variance estimation)
        loader_va = make_loader(X_va, y_va, m_va, k_va, shuffle=False)
        preds_va_sc, _, _ = predict(model, loader_va, device)
        preds_va_real = inverse_transform_predictions(preds_va_sc, target_scaler)
        var_model = estimate_residual_variances(preds_va_real, y_va_real)
        R = estimate_residual_covariance(preds_va_real, y_va_real, cfg.KALMAN_REG_EPS)

        # Test predictions
        loader_te = make_loader(X_te, y_te, m_te, k_te, shuffle=False)
        preds_te_sc, _, _ = predict(model, loader_te, device)
        preds_te_real = inverse_transform_predictions(preds_te_sc, target_scaler)

        # ── Inverse-variance fusion ──────────────────────────────────────────
        fused_iv = inverse_variance_fusion(k_te_real, preds_te_real, var_kinetic, var_model)
        m_iv = compute_all_metrics(fused_iv, y_te_real, m_te, f"{arch}_fused_iv", h)
        fusion_results.append(m_iv)

        w_k, w_m = fusion_weights(var_kinetic, var_model)
        plot_fusion_weights(w_k, w_m, arch, h)
        plot_fusion_comparison(k_te_real, preds_te_real, fused_iv, y_te_real, m_te, arch, h)

        print(f"  {arch} IV-fusion obs_rmse: {m_iv['obs_rmse']:.5f}  "
              f"(kinetic var: {var_kinetic.mean():.2e}, model var: {var_model.mean():.2e})")

        # ── Kalman-like covariance fusion ────────────────────────────────────
        L = localization_matrix(decay=0.5)
        B_loc = B * L
        R_loc = R * L

        fused_kf = kalman_fusion(k_te_real, preds_te_real, B_loc, R_loc)
        m_kf = compute_all_metrics(fused_kf, y_te_real, m_te, f"{arch}_fused_kalman", h)
        fusion_results.append(m_kf)

        if h == 1:
            plot_covariance_heatmaps(B, R, arch)

        print(f"  {arch} Kalman-fusion obs_rmse: {m_kf['obs_rmse']:.5f}")

        # Save fused predictions
        cfg.OUT_FUSION.mkdir(parents=True, exist_ok=True)
        np.save(cfg.OUT_FUSION / f"fused_iv_{arch}_h{h:02d}.npy",     fused_iv)
        np.save(cfg.OUT_FUSION / f"fused_kalman_{arch}_h{h:02d}.npy", fused_kf)

# Full fusion results table
fdf = pd.DataFrame(fusion_results)
fdf.to_csv(cfg.OUT_METRICS / "fusion_results.csv", index=False)
print(f"\n=== Fusion results ===")
print(fdf.pivot_table(index="label", columns="horizon", values="obs_rmse").to_string())
print(f"\nSaved to {cfg.OUT_METRICS}/fusion_results.csv")
