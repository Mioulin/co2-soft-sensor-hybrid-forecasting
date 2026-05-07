"""
Visualizations for CO2 soft-sensor project.
All functions save figures to outputs/figures/ and optionally return the Figure.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import N_SAMPLING_POINTS, OUT_FIGURES

POINT_COLORS = plt.cm.tab10(np.linspace(0, 0.6, N_SAMPLING_POINTS))


def _save(fig, name: str, out_dir: Path = OUT_FIGURES) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Phase 1: Data understanding ───────────────────────────────────────────────

def plot_at400_per_run(
    runs: Dict[str, dict],
    profiles: Dict[str, np.ndarray],
) -> None:
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (run_id, run_data) in zip(axes, runs.items()):
        at400 = run_data["df"]["at400_frac"].values
        ax.plot(at400, color="steelblue", lw=1.5, label="AT400 measured")
        ax.set_title(run_id, fontsize=9)
        ax.set_ylabel("CO2 fraction")
        ax.legend(fontsize=8)
    fig.suptitle("AT400 CO2 fraction per run", fontsize=11)
    plt.tight_layout()
    _save(fig, "01_at400_per_run")


def plot_label_schedule(runs: Dict[str, dict]) -> None:
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (run_id, run_data) in zip(axes, runs.items()):
        labels = run_data["df"]["label"].values
        ax.step(np.arange(len(labels)), labels, where="post", color="coral", lw=1.5)
        ax.set_yticks(range(1, 7))
        ax.set_ylabel("Active point")
        ax.set_title(run_id, fontsize=9)
    fig.suptitle("Label (active sampling point) schedule per run", fontsize=11)
    plt.tight_layout()
    _save(fig, "02_label_schedule")


def plot_sparse_vs_reconstructed(
    run_id: str,
    y_sparse: np.ndarray,
    y_profile: np.ndarray,
) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    for pt in range(6):
        ax = axes[pt]
        obs_idx = np.where(~np.isnan(y_sparse[:, pt]))[0]
        ax.plot(y_profile[:, pt], color="steelblue", lw=1.5, label="Interpolated")
        ax.scatter(obs_idx, y_sparse[obs_idx, pt], s=20, color="red",
                   zorder=5, label="Observed" if pt == 0 else "")
        ax.set_ylabel(f"Pt {pt+1}")
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{run_id}: Sparse observations vs reconstructed profile", fontsize=11)
    plt.tight_layout()
    _save(fig, f"03_sparse_vs_reconstructed_{run_id}")


def plot_kinetic_vs_reconstructed(
    run_id: str,
    y_profile: np.ndarray,
    kinetic: np.ndarray,
) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    for pt in range(6):
        ax = axes[pt]
        ax.plot(y_profile[:, pt], color="steelblue", lw=1.5, label="Reconstructed")
        ax.plot(kinetic[:, pt],   color="darkorange", lw=1.5, ls="--", label="Kinetic prior")
        ax.set_ylabel(f"Pt {pt+1}")
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{run_id}: Reconstructed profile vs kinetic prior", fontsize=11)
    plt.tight_layout()
    _save(fig, f"04_kinetic_vs_reconstructed_{run_id}")


# ── Phase 2: Baselines ────────────────────────────────────────────────────────

def plot_baseline_per_point(
    pred_pers: np.ndarray,
    pred_kn:   np.ndarray,
    target:    np.ndarray,
    mask:      np.ndarray,
    horizon:   int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False)
    axes = axes.flatten()
    for pt in range(6):
        ax = axes[pt]
        obs = mask[:, pt] > 0.5
        ax.plot(target[:, pt], color="gray", lw=1, alpha=0.7, label="Target")
        ax.plot(pred_pers[:, pt], color="coral",      lw=1.5, ls="--", label="Persistence")
        ax.plot(pred_kn[:, pt],   color="darkorange",  lw=1.5, ls=":",  label="Kinetic")
        ax.scatter(np.where(obs)[0], target[obs, pt], s=12, c="black", zorder=5)
        ax.set_title(f"Point {pt+1}")
        ax.set_ylabel("CO2 fraction")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Baselines vs target | h={horizon}", fontsize=11)
    plt.tight_layout()
    _save(fig, f"05_baselines_h{horizon:02d}")


# ── Phase 3: Training curves ──────────────────────────────────────────────────

def plot_loss_curves(
    history: Dict[str, list],
    arch: str,
    horizon: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history["train_loss"], label="Train loss", color="steelblue")
    ax.plot(history["val_loss"],   label="Val loss",   color="coral")
    ax.plot(history["train_full_mse"], label="Train full MSE", color="steelblue", ls="--", alpha=0.6)
    ax.plot(history["val_full_mse"],   label="Val full MSE",   color="coral",     ls="--", alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.set_title(f"{arch.upper()} training curves | h={horizon}")
    _save(fig, f"06_loss_{arch}_h{horizon:02d}")


# ── Phase 4: Evaluation ───────────────────────────────────────────────────────

def plot_predictions_per_point(
    pred:    np.ndarray,
    target:  np.ndarray,
    mask:    np.ndarray,
    arch:    str,
    horizon: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False)
    axes = axes.flatten()
    for pt in range(6):
        ax = axes[pt]
        obs = mask[:, pt] > 0.5
        ax.plot(target[:, pt], color="gray", lw=1, alpha=0.7, label="Target")
        ax.plot(pred[:, pt],   color="steelblue", lw=1.5, label=arch.upper())
        ax.scatter(np.where(obs)[0], target[obs, pt], s=15, c="red", zorder=5,
                   label="Observed" if pt == 0 else "")
        ax.set_title(f"Point {pt+1}")
        ax.set_ylabel("CO2 fraction")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{arch.upper()} predictions | h={horizon}", fontsize=11)
    plt.tight_layout()
    _save(fig, f"07_pred_{arch}_h{horizon:02d}")


def plot_scatter_pred_vs_target(
    pred: np.ndarray, target: np.ndarray, arch: str, horizon: int
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target.ravel(), pred.ravel(), s=8, alpha=0.4, color="steelblue")
    lims = [min(target.min(), pred.min()), max(target.max(), pred.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Target (CO2 fraction)")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{arch.upper()} scatter | h={horizon}")
    _save(fig, f"08_scatter_{arch}_h{horizon:02d}")


def plot_rmse_by_horizon(
    results: pd.DataFrame,
) -> None:
    """results must have columns: model, horizon, full_rmse, obs_rmse"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in zip(
        axes,
        ["full_rmse", "obs_rmse"],
        ["Full-profile RMSE", "Observed-point RMSE"],
    ):
        for model_name, grp in results.groupby("label"):
            ax.plot(grp["horizon"], grp[col], marker="o", label=model_name)
        ax.set_xlabel("Horizon (steps)")
        ax.set_ylabel("RMSE (CO2 fraction)")
        ax.set_title(title)
        ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "09_rmse_by_horizon")


def plot_per_point_rmse_heatmap(
    per_point_results: Dict[str, pd.DataFrame],
) -> None:
    """per_point_results: {model_name: per_point_metrics DataFrame}"""
    n_models = len(per_point_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, df) in zip(axes, per_point_results.items()):
        data = df["rmse_full"].values.reshape(6, 1)
        sns.heatmap(data, ax=ax, annot=True, fmt=".4f", cmap="YlOrRd",
                    yticklabels=[f"Pt {i}" for i in range(1, 7)],
                    xticklabels=[name], vmin=0)
        ax.set_title(name)
    fig.suptitle("Per-point RMSE", fontsize=11)
    plt.tight_layout()
    _save(fig, "10_per_point_rmse_heatmap")


# ── Phase 5: Fusion ───────────────────────────────────────────────────────────

def plot_fusion_weights(
    w_kinetic: np.ndarray,
    w_model:   np.ndarray,
    arch:      str,
    horizon:   int,
) -> None:
    x = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, w_kinetic, 0.4, label="Kinetic weight", color="darkorange")
    ax.bar(x + 0.2, w_model,   0.4, label=f"{arch.upper()} weight", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Pt {i}" for i in range(1, 7)])
    ax.set_ylabel("Normalised weight")
    ax.set_title(f"Inverse-variance fusion weights | {arch.upper()} h={horizon}")
    ax.legend()
    _save(fig, f"11_fusion_weights_{arch}_h{horizon:02d}")


def plot_fusion_comparison(
    kinetic: np.ndarray,
    model:   np.ndarray,
    fused:   np.ndarray,
    target:  np.ndarray,
    mask:    np.ndarray,
    arch:    str,
    horizon: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False)
    axes = axes.flatten()
    for pt in range(6):
        ax = axes[pt]
        obs = mask[:, pt] > 0.5
        ax.plot(target[:, pt],  color="gray",        lw=1, alpha=0.6, label="Target")
        ax.plot(kinetic[:, pt], color="darkorange",  lw=1.5, ls=":",  label="Kinetic")
        ax.plot(model[:, pt],   color="steelblue",   lw=1.5, ls="--", label=arch.upper())
        ax.plot(fused[:, pt],   color="purple",      lw=2,             label="Fused")
        ax.scatter(np.where(obs)[0], target[obs, pt], s=12, c="black", zorder=5)
        ax.set_title(f"Point {pt+1}")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Fusion comparison | {arch.upper()} h={horizon}", fontsize=11)
    plt.tight_layout()
    _save(fig, f"12_fusion_comparison_{arch}_h{horizon:02d}")


def plot_covariance_heatmaps(B: np.ndarray, R: np.ndarray, arch: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mat, title in zip(
        axes, [B, R], ["Kinetic residual cov (B)", f"{arch.upper()} residual cov (R)"]
    ):
        sns.heatmap(mat, ax=ax, annot=True, fmt=".2e", cmap="coolwarm", center=0,
                    xticklabels=[f"Pt{i}" for i in range(1, 7)],
                    yticklabels=[f"Pt{i}" for i in range(1, 7)])
        ax.set_title(title)
    plt.tight_layout()
    _save(fig, f"13_covariance_heatmaps_{arch}")
