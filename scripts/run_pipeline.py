"""
Full pipeline: dataset -> SDAE -> GRU/Transformer -> evaluation -> fusion.
Run this single script to reproduce all results.
"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import config as cfg
from src.data_loading import load_all_runs, run_summary
from src.target_reconstruction import reconstruct_targets
from src.splitting import split_runs
from src.windowing import build_dataset, _get_feature_matrix
from src.autoencoder import SDAE, train_sdae, encode_dataset
from src.models import build_model
from src.training import train_model, make_loader, predict
from src.metrics import (compute_all_metrics, per_point_metrics,
                          persistence_predictions, kinetic_predictions)
from src.fusion import (estimate_residual_variances, estimate_residual_covariance,
                        inverse_variance_fusion, kalman_fusion_gc, fusion_weights,
                        gaspari_cohn_matrix)
from src.utils import set_seed, load_scaler, save_scaler
from sklearn.preprocessing import StandardScaler

set_seed()
for d in [cfg.OUT_CHECKPOINTS, cfg.OUT_PREDICTIONS, cfg.OUT_METRICS,
          cfg.OUT_FIGURES, cfg.OUT_FUSION]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# PHASE 1: Data loading and reconstruction
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 1: Loading data and kinetic-anchored reconstruction")
print("=" * 60)

runs = load_all_runs(cfg.ALL_RUNS)
summary = run_summary(runs)
summary.to_csv(cfg.OUT_METRICS / "run_summary.csv")
print(summary.to_string())

profiles, masks, sparses = {}, {}, {}
for run_id, run_data in runs.items():
    df = run_data["df"]
    y_sparse, obs_mask, y_profile = reconstruct_targets(
        df["at400_frac"].values, df["label"].values,
        kinetic=run_data["kinetic"], method="kinetic_residual"
    )
    profiles[run_id] = y_profile
    masks[run_id]    = obs_mask
    sparses[run_id]  = y_sparse

train_runs, val_runs, test_runs = split_runs(runs)

# Fit scalers on train
all_feats_tr = np.concatenate([
    _get_feature_matrix(rd["df"], use_label_onehot=True)
    for rd in train_runs.values()
])
feat_scaler = StandardScaler().fit(all_feats_tr)

all_prof_tr = np.concatenate([profiles[r] for r in train_runs])
target_scaler = StandardScaler().fit(all_prof_tr)

save_scaler(feat_scaler,   cfg.OUT_METRICS / "feature_scaler.pkl")
save_scaler(target_scaler, cfg.OUT_METRICS / "target_scaler.pkl")

# ── Phase 1 plots ─────────────────────────────────────────────
def plot_reconstruction(run_id):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    y_sp = sparses[run_id]; y_pr = profiles[run_id]
    kn   = runs[run_id]["kinetic"]
    for pt in range(6):
        ax = axes[pt]
        obs_idx = np.where(~np.isnan(y_sp[:, pt]))[0]
        ax.plot(kn[:, pt],    color="darkorange", lw=1.5, ls="--", alpha=0.7, label="Kinetic prior")
        ax.plot(y_pr[:, pt],  color="steelblue",  lw=2,            label="Reconstructed (KAR)")
        ax.scatter(obs_idx, y_sp[obs_idx, pt], s=30, color="red", zorder=5,
                   label="AT400 observed" if pt==0 else "")
        ax.set_title(f"Sampling point {pt+1}"); ax.set_ylabel("CO2 fraction")
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{run_id}: Kinetic-Anchored Reconstruction vs Linear Interpolation baseline", fontsize=11)
    plt.tight_layout()
    fig.savefig(cfg.OUT_FIGURES / f"01_reconstruction_{run_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

for r in ["140313_1", "140207_1"]:
    plot_reconstruction(r)
    print(f"  Saved reconstruction plot: {r}")

# Reconstruction quality: residual std comparison
print("\n[Reconstruction] Residual std (kinetic-anchored vs linear):")
for run_id, run_data in runs.items():
    df = run_data["df"]; kn = run_data["kinetic"]
    at400 = df["at400_frac"].values; label = df["label"].values.astype(int)
    obs_kn = kn[np.arange(len(label)), label-1]
    r_std = np.std(at400 - obs_kn)
    at_std = np.std(at400)
    print(f"  {run_id}: AT400 std={at_std:.5f}  residual std={r_std:.5f}  ratio={r_std/at_std:.3f}")

# ══════════════════════════════════════════════════════════════
# PHASE 2: SDAE pre-training
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 2: SDAE dimensionality reduction (89 -> 16)")
print("=" * 60)

X_tr_raw = feat_scaler.transform(all_feats_tr)
X_va_raw = feat_scaler.transform(np.concatenate([
    _get_feature_matrix(rd["df"], use_label_onehot=True)
    for rd in val_runs.values()
]))

n_features = X_tr_raw.shape[1]
print(f"  Input features: {n_features}, Latent dim: {cfg.SDAE_LATENT_DIM}")

sdae = SDAE(n_features)
sdae_ckpt = cfg.OUT_CHECKPOINTS / "sdae.pt"
sdae_hist = train_sdae(sdae, X_tr_raw, X_va_raw, sdae_ckpt,
                       epochs=cfg.SDAE_EPOCHS, verbose=True)

# Save SDAE history
np.save(cfg.OUT_METRICS / "history_sdae.npy", sdae_hist)

# SDAE loss curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sdae_hist["train_loss"], label="Train", color="steelblue")
ax.plot(sdae_hist["val_loss"],   label="Val",   color="coral")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_title("SDAE Pre-training: Reconstruction Loss"); ax.legend()
fig.savefig(cfg.OUT_FIGURES / "02_sdae_loss.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# SDAE latent space visualisation (PCA of encoded val set)
from sklearn.decomposition import PCA
Z_va = encode_dataset(sdae, X_va_raw)
pca  = PCA(n_components=2).fit(Z_va)
Z2   = pca.transform(Z_va)
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=np.arange(len(Z2)), cmap="viridis", s=10, alpha=0.7)
plt.colorbar(sc, ax=ax, label="Time step")
ax.set_title(f"SDAE Latent Space (PCA 2D) - Val run\n"
             f"Explained var: {pca.explained_variance_ratio_.sum():.2%}")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
fig.savefig(cfg.OUT_FIGURES / "03_sdae_latent_space.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"  SDAE trained. Best val reconstruction MSE: {min(sdae_hist['val_loss']):.6f}")
print(f"  Compression: {n_features} -> {cfg.SDAE_LATENT_DIM} features ({cfg.SDAE_LATENT_DIM/n_features:.1%} of original)")

# ══════════════════════════════════════════════════════════════
# PHASE 3: Baselines
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 3: Baselines")
print("=" * 60)

baseline_results = []
for h in cfg.HORIZONS:
    p_pers, t_pers, m_pers = persistence_predictions(test_runs, profiles, masks, h, cfg.WINDOW_LENGTH)
    p_kn,   t_kn,   m_kn   = kinetic_predictions(    test_runs, profiles, masks, h, cfg.WINDOW_LENGTH)
    baseline_results.append(compute_all_metrics(p_pers, t_pers, m_pers, "persistence",   h))
    baseline_results.append(compute_all_metrics(p_kn,   t_kn,   m_kn,   "kinetic_prior", h))
    print(f"  h={h}: persistence obs_rmse={baseline_results[-2]['obs_rmse']:.5f}  "
          f"kinetic obs_rmse={baseline_results[-1]['obs_rmse']:.5f}")

# ══════════════════════════════════════════════════════════════
# PHASE 4: Train GRU and Transformer
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 4: Training GRU and Mini-Transformer (SDAE features)")
print("=" * 60)

all_results = baseline_results.copy()
per_point_store = {}
latent_dim = cfg.SDAE_LATENT_DIM

for arch in ["gru", "transformer"]:
    for h in cfg.HORIZONS:
        print(f"\n  --- {arch.upper()} | h={h} ---")
        X_tr, y_tr, m_tr, k_tr = build_dataset(
            train_runs, profiles, masks, h,
            feature_scaler=feat_scaler, target_scaler=target_scaler,
            sdae_encoder=sdae)
        X_va, y_va, m_va, k_va = build_dataset(
            val_runs,   profiles, masks, h,
            feature_scaler=feat_scaler, target_scaler=target_scaler,
            sdae_encoder=sdae)
        X_te, y_te, m_te, k_te = build_dataset(
            test_runs,  profiles, masks, h,
            feature_scaler=feat_scaler, target_scaler=target_scaler,
            sdae_encoder=sdae)

        print(f"     train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")

        model = build_model(arch, latent_dim, cfg)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"     {arch} params: {n_params:,}")

        ckpt = cfg.OUT_CHECKPOINTS / f"{arch}_h{h:02d}.pt"
        hist = train_model(model,
                           make_loader(X_tr, y_tr, m_tr, k_tr),
                           make_loader(X_va, y_va, m_va, k_va, shuffle=False),
                           ckpt, verbose=True)
        np.save(cfg.OUT_METRICS / f"history_{arch}_h{h:02d}.npy", hist)

        # Evaluate on test
        loader_te = make_loader(X_te, y_te, m_te, k_te, shuffle=False)
        preds_sc, _, _ = predict(model, loader_te)
        preds_real = target_scaler.inverse_transform(preds_sc)
        y_te_real  = target_scaler.inverse_transform(y_te)

        np.save(cfg.OUT_PREDICTIONS / f"pred_{arch}_h{h:02d}.npy", preds_real)
        np.save(cfg.OUT_PREDICTIONS / f"target_h{h:02d}.npy",      y_te_real)
        np.save(cfg.OUT_PREDICTIONS / f"mask_h{h:02d}.npy",        m_te)
        np.save(cfg.OUT_PREDICTIONS / f"kinetic_h{h:02d}.npy",     k_te)

        metrics = compute_all_metrics(preds_real, y_te_real, m_te, arch, h)
        all_results.append(metrics)
        print(f"     full_rmse={metrics['full_rmse']:.5f}  obs_rmse={metrics['obs_rmse']:.5f}  mape={metrics['mape']:.2f}%")

        if h == 1:
            per_point_store[arch] = per_point_metrics(preds_real, y_te_real, m_te)

        # Loss curve
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(hist["train_loss"], label="Train", color="steelblue")
        ax.plot(hist["val_loss"],   label="Val",   color="coral")
        ax.set_title(f"{arch.upper()} Loss Curves | h={h}"); ax.set_xlabel("Epoch")
        ax.legend(); ax.set_ylabel("Combined Loss")
        fig.savefig(cfg.OUT_FIGURES / f"04_loss_{arch}_h{h:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Prediction plot
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        for pt in range(6):
            ax = axes[pt]
            obs = m_te[:, pt] > 0.5
            ax.plot(y_te_real[:, pt], color="gray", lw=1, alpha=0.6, label="Target")
            ax.plot(preds_real[:, pt], color="steelblue", lw=1.5, label=arch.upper())
            ax.scatter(np.where(obs)[0], y_te_real[obs, pt], s=15, c="red", zorder=5,
                       label="Observed" if pt==0 else "")
            ax.set_title(f"Pt {pt+1}")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{arch.upper()} Predictions vs Target | h={h} | Test run 140207_1", fontsize=11)
        plt.tight_layout()
        fig.savefig(cfg.OUT_FIGURES / f"05_pred_{arch}_h{h:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

# ── Horizon comparison plot ───────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(cfg.OUT_METRICS / "results_table.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = {"persistence":"gray", "kinetic_prior":"darkorange", "gru":"steelblue", "transformer":"purple"}
for ax, col, title in zip(axes, ["full_rmse","obs_rmse"], ["Full-Profile RMSE","Observed-Point RMSE"]):
    for name, grp in results_df.groupby("label"):
        ax.plot(grp["horizon"], grp[col], marker="o", label=name,
                color=colors.get(name,"black"), lw=2)
    ax.set_xlabel("Forecast horizon (steps)"); ax.set_ylabel("RMSE (CO2 fraction)")
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(cfg.OUT_FIGURES / "06_rmse_by_horizon.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Per-point RMSE heatmap ────────────────────────────────────
if per_point_store:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (name, df_pp) in zip(axes, per_point_store.items()):
        data = df_pp["rmse_observed"].values.reshape(6, 1)
        data = np.nan_to_num(data, nan=0)
        sns.heatmap(data, ax=ax, annot=True, fmt=".4f", cmap="YlOrRd",
                    yticklabels=[f"Pt{i}" for i in range(1,7)],
                    xticklabels=[name.upper()], vmin=0)
        ax.set_title(f"{name.upper()} | h=1 | Observed-point RMSE")
    plt.tight_layout()
    fig.savefig(cfg.OUT_FIGURES / "07_per_point_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# GRU vs Transformer scatter at h=1
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, arch in zip(axes, ["gru", "transformer"]):
    y_te_real = np.load(cfg.OUT_PREDICTIONS / f"target_h01.npy")
    preds_real = np.load(cfg.OUT_PREDICTIONS / f"pred_{arch}_h01.npy")
    ax.scatter(y_te_real.ravel(), preds_real.ravel(), s=6, alpha=0.3, color="steelblue")
    lims = [min(y_te_real.min(), preds_real.min()), max(y_te_real.max(), preds_real.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Target"); ax.set_ylabel("Predicted")
    ax.set_title(f"{arch.upper()} | h=1 | Scatter (all 6 points)")
plt.tight_layout()
fig.savefig(cfg.OUT_FIGURES / "08_scatter_h01.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
# PHASE 5: Fusion (Inverse-variance + Kalman-GC)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 5: Fusion (Inverse-Variance + Kalman/Gaspari-Cohn)")
print("=" * 60)

GC_mat = gaspari_cohn_matrix(n=6, L=cfg.GC_CORR_LENGTH)
fusion_results = []

for arch in ["gru", "transformer"]:
    # Kinetic variance from val set
    p_kn_va, t_kn_va, _ = kinetic_predictions(val_runs, profiles, masks, 1, cfg.WINDOW_LENGTH)
    var_kinetic = estimate_residual_variances(p_kn_va, t_kn_va)
    B = estimate_residual_covariance(p_kn_va, t_kn_va)

    for h in cfg.HORIZONS:
        print(f"\n  --- Fusion {arch.upper()} | h={h} ---")

        # Val predictions for model variance
        X_va, y_va, m_va, k_va = build_dataset(
            val_runs, profiles, masks, h,
            feature_scaler=feat_scaler, target_scaler=target_scaler, sdae_encoder=sdae)
        model = build_model(arch, latent_dim, cfg)
        ckpt  = cfg.OUT_CHECKPOINTS / f"{arch}_h{h:02d}.pt"
        model.load_state_dict(torch.load(ckpt, weights_only=True))

        preds_va_sc, _, _ = predict(model, make_loader(X_va, y_va, m_va, k_va, shuffle=False))
        preds_va   = target_scaler.inverse_transform(preds_va_sc)
        y_va_real  = target_scaler.inverse_transform(y_va)
        var_model  = estimate_residual_variances(preds_va, y_va_real)
        R = estimate_residual_covariance(preds_va, y_va_real)

        # Test fusion
        preds_te = np.load(cfg.OUT_PREDICTIONS / f"pred_{arch}_h{h:02d}.npy")
        y_te_r   = np.load(cfg.OUT_PREDICTIONS / f"target_h{h:02d}.npy")
        m_te     = np.load(cfg.OUT_PREDICTIONS / f"mask_h{h:02d}.npy")
        k_te     = np.load(cfg.OUT_PREDICTIONS / f"kinetic_h{h:02d}.npy")

        # Inverse-variance fusion
        fused_iv = inverse_variance_fusion(k_te, preds_te, var_kinetic, var_model)
        m_iv = compute_all_metrics(fused_iv, y_te_r, m_te, f"{arch}_fused_iv", h)
        fusion_results.append(m_iv)

        # Kalman + Gaspari-Cohn
        fused_kgc = kalman_fusion_gc(k_te, preds_te, B, R, L=cfg.GC_CORR_LENGTH)
        m_kgc = compute_all_metrics(fused_kgc, y_te_r, m_te, f"{arch}_fused_kgc", h)
        fusion_results.append(m_kgc)

        np.save(cfg.OUT_FUSION / f"fused_iv_{arch}_h{h:02d}.npy",  fused_iv)
        np.save(cfg.OUT_FUSION / f"fused_kgc_{arch}_h{h:02d}.npy", fused_kgc)

        wk, wm = fusion_weights(var_kinetic, var_model)
        print(f"     IV fusion   obs_rmse={m_iv['obs_rmse']:.5f}  mape={m_iv['mape']:.2f}%")
        print(f"     Kalman-GC   obs_rmse={m_kgc['obs_rmse']:.5f}  mape={m_kgc['mape']:.2f}%")
        print(f"     Fusion weights (model): {wm.round(3)}")

        # Fusion comparison plot (h=1)
        if h == 1:
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            axes = axes.flatten()
            for pt in range(6):
                ax = axes[pt]
                obs = m_te[:, pt] > 0.5
                ax.plot(y_te_r[:, pt],   color="gray",       lw=1, alpha=0.6, label="Target")
                ax.plot(k_te[:, pt],     color="darkorange",  lw=1.5, ls=":",  label="Kinetic")
                ax.plot(preds_te[:, pt], color="steelblue",   lw=1.5, ls="--", label=arch.upper())
                ax.plot(fused_kgc[:, pt],color="purple",      lw=2,            label="Fused (KGC)")
                ax.scatter(np.where(obs)[0], y_te_r[obs, pt], s=12, c="black", zorder=5)
                ax.set_title(f"Pt {pt+1}")
            axes[0].legend(fontsize=8)
            fig.suptitle(f"Fusion: {arch.upper()} + Kinetic (Gaspari-Cohn) | h=1 | Test 140207_1", fontsize=11)
            plt.tight_layout()
            fig.savefig(cfg.OUT_FIGURES / f"09_fusion_{arch}_h01.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Fusion weights bar chart
    wk, wm = fusion_weights(var_kinetic, var_model)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(1, 7)
    ax.bar(x - 0.2, wk, 0.4, label="Kinetic",    color="darkorange")
    ax.bar(x + 0.2, wm, 0.4, label=arch.upper(), color="steelblue")
    ax.set_xticks(x); ax.set_xticklabels([f"Pt{i}" for i in range(1,7)])
    ax.set_ylabel("Normalised weight"); ax.legend()
    ax.set_title(f"Inverse-Variance Fusion Weights | {arch.upper()}")
    fig.savefig(cfg.OUT_FIGURES / f"10_fusion_weights_{arch}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Gaspari-Cohn matrix
    if arch == "gru":
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(GC_mat, ax=ax, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[f"Pt{i}" for i in range(1,7)],
                    yticklabels=[f"Pt{i}" for i in range(1,7)])
        ax.set_title(f"Gaspari-Cohn Localization Matrix (L={cfg.GC_CORR_LENGTH})")
        fig.savefig(cfg.OUT_FIGURES / "11_gaspari_cohn_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

# ── Final summary table ───────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS (Test run 140207_1, Observed-point RMSE)")
print("=" * 60)

all_combined = pd.DataFrame(all_results + fusion_results)
all_combined.to_csv(cfg.OUT_METRICS / "all_results.csv", index=False)

pivot = all_combined.pivot_table(index="label", columns="horizon", values="obs_rmse").round(5)
print(pivot.to_string())

# Final comparison bar chart (h=1)
h1 = all_combined[all_combined["horizon"] == 1].copy()
h1 = h1.sort_values("obs_rmse")
fig, ax = plt.subplots(figsize=(10, 5))
colors_bar = []
for lbl in h1["label"]:
    if "fused" in lbl: colors_bar.append("purple")
    elif lbl in ["gru","transformer"]: colors_bar.append("steelblue")
    elif lbl == "kinetic_prior": colors_bar.append("darkorange")
    else: colors_bar.append("gray")
bars = ax.barh(h1["label"], h1["obs_rmse"], color=colors_bar)
ax.set_xlabel("Observed-Point RMSE (CO2 fraction)")
ax.set_title("Model Comparison | h=1 | Test run 140207_1")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, h1["obs_rmse"]):
    ax.text(val + 0.0001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
fig.savefig(cfg.OUT_FIGURES / "12_final_comparison_h1.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nAll outputs saved to outputs/")
print("Figures:", len(list(cfg.OUT_FIGURES.glob("*.png"))), "plots")
print("Done.")
