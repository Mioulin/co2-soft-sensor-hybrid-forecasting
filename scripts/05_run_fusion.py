"""
Script 05: Inverse-variance and Kalman-GC fusion.

Fusion weights estimated on validation run (140313_1) per horizon.
Final fusion evaluated on test run (140207_1) only.

Requires: outputs/checkpoints/sdae.pt, gru_h*.pt, transformer_h*.pt
Run after 02, 03, 04 (or after run_pipeline.py).
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
from src.autoencoder import SDAE
from src.models import build_model
from src.training import predict, make_loader
from src.metrics import compute_all_metrics, kinetic_predictions
from src.fusion import (
    estimate_residual_variances, estimate_residual_covariance,
    inverse_variance_fusion, kalman_fusion_gc, fusion_weights,
    gaspari_cohn_matrix,
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
    _, obs_mask, y_profile = reconstruct_targets(
        df["at400_frac"].values, df["label"].values,
        kinetic=run_data["kinetic"], method="kinetic_residual",
    )
    profiles[run_id] = y_profile
    masks[run_id]    = obs_mask

train_runs, val_runs, test_runs = split_runs(runs)
feat_scaler   = load_scaler(cfg.OUT_METRICS / "feature_scaler.pkl")
target_scaler = load_scaler(cfg.OUT_METRICS / "target_scaler.pkl")

# Load trained SDAE encoder
sdae_ckpt = cfg.OUT_CHECKPOINTS / "sdae.pt"
if not sdae_ckpt.exists():
    raise FileNotFoundError(f"SDAE checkpoint not found: {sdae_ckpt}")
sdae = SDAE(n_features=95)  # 89 numeric + 6 one-hot
sdae.load_state_dict(torch.load(sdae_ckpt, weights_only=True))
sdae.eval()
latent_dim = cfg.SDAE_LATENT_DIM
device = torch.device("cpu")
print(f"  Loaded SDAE -> latent_dim={latent_dim}")

fusion_results = []

for h in cfg.HORIZONS:
    print(f"\n--- Fusion | h={h} ---")

    # --- Validation windows (calibration only, never test) ---
    X_va, y_va, m_va, k_va = build_dataset(
        val_runs, profiles, masks, h,
        feature_scaler=feat_scaler, target_scaler=target_scaler,
        sdae_encoder=sdae,
    )
    y_va_real = inverse_transform_predictions(y_va, target_scaler)

    # Kinetic residual covariance estimated on val at the same horizon h
    p_kn_va, t_kn_va, _ = kinetic_predictions(val_runs, profiles, masks, h, cfg.WINDOW_LENGTH)
    var_kinetic = estimate_residual_variances(p_kn_va, t_kn_va)
    B = estimate_residual_covariance(p_kn_va, t_kn_va, cfg.KALMAN_REG_EPS)

    # --- Test windows (evaluation only) ---
    X_te, y_te, m_te, k_te = build_dataset(
        test_runs, profiles, masks, h,
        feature_scaler=feat_scaler, target_scaler=target_scaler,
        sdae_encoder=sdae,
    )
    y_te_real = inverse_transform_predictions(y_te, target_scaler)
    k_te_real = k_te  # kinetic never scaled

    for arch in ["gru", "transformer"]:
        ckpt = cfg.OUT_CHECKPOINTS / checkpoint_name(arch, h)
        if not ckpt.exists():
            print(f"  Checkpoint not found: {ckpt}, skipping {arch}")
            continue

        model = build_model(arch, latent_dim, cfg)
        model.load_state_dict(torch.load(ckpt, weights_only=True))

        # Val predictions -> model residual covariance (calibration only)
        loader_va = make_loader(X_va, y_va, m_va, k_va, shuffle=False)
        preds_va_sc, _, _ = predict(model, loader_va, device)
        preds_va_real = inverse_transform_predictions(preds_va_sc, target_scaler)
        var_model = estimate_residual_variances(preds_va_real, y_va_real)
        R = estimate_residual_covariance(preds_va_real, y_va_real, cfg.KALMAN_REG_EPS)

        # Test predictions (no calibration on test)
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

        # ── Kalman + Gaspari-Cohn fusion ─────────────────────────────────────
        fused_kgc = kalman_fusion_gc(k_te_real, preds_te_real, B, R, L=cfg.GC_CORR_LENGTH)
        m_kgc = compute_all_metrics(fused_kgc, y_te_real, m_te, f"{arch}_fused_kgc", h)
        fusion_results.append(m_kgc)

        if h == 1:
            plot_covariance_heatmaps(B, R, arch)

        print(f"  {arch} Kalman-GC obs_rmse: {m_kgc['obs_rmse']:.5f}")

        # Save fused predictions
        cfg.OUT_FUSION.mkdir(parents=True, exist_ok=True)
        np.save(cfg.OUT_FUSION / f"fused_iv_{arch}_h{h:02d}.npy",  fused_iv)
        np.save(cfg.OUT_FUSION / f"fused_kgc_{arch}_h{h:02d}.npy", fused_kgc)

# Full fusion results table
fdf = pd.DataFrame(fusion_results)
fdf.to_csv(cfg.OUT_METRICS / "fusion_results.csv", index=False)
print(f"\n=== Fusion results (Observed-point RMSE) ===")
print(fdf.pivot_table(index="label", columns="horizon", values="obs_rmse").to_string())
print(f"\nSaved to {cfg.OUT_METRICS}/fusion_results.csv")
