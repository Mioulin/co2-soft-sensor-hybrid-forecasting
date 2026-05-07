"""Sliding-window dataset construction over latent (SDAE-encoded) features."""
from typing import Dict, Tuple
import numpy as np
from src.config import WINDOW_LENGTH, N_SAMPLING_POINTS, USE_LABEL_ONEHOT


def _get_feature_matrix(df, use_label_onehot=True) -> np.ndarray:
    EXCLUDE = {"run_id", "row_idx", "at400_frac", "label"}
    at400_names = {c for c in df.columns if c.startswith("AT400")}
    EXCLUDE |= at400_names
    raw_cols = [c for c in df.columns
                if c not in EXCLUDE and np.issubdtype(df[c].dtype, np.number)]
    feats = df[raw_cols].values.astype(np.float64)
    if use_label_onehot:
        labels = df["label"].values.astype(int)
        oh = np.zeros((len(labels), N_SAMPLING_POINTS))
        oh[np.arange(len(labels)), labels - 1] = 1.0
        feats = np.concatenate([feats, oh], axis=1)
    return feats


def build_windows_for_run(
    run_id, run_data, y_profile, observed_mask, horizon,
    window_length=WINDOW_LENGTH, use_label_onehot=True,
    feature_scaler=None, target_scaler=None, sdae_encoder=None,
):
    import torch
    df  = run_data["df"]
    kn  = run_data["kinetic"]
    n   = len(df)

    feats = _get_feature_matrix(df, use_label_onehot=use_label_onehot)
    if feature_scaler is not None:
        feats = feature_scaler.transform(feats)

    # Apply SDAE encoder if provided
    if sdae_encoder is not None:
        sdae_encoder.eval()
        with torch.no_grad():
            z = sdae_encoder.encode(
                torch.from_numpy(feats).float()
            ).numpy()
    else:
        z = feats

    y_sc = target_scaler.transform(y_profile) if target_scaler else y_profile

    Xs, ys, ms, ks = [], [], [], []
    start = window_length - 1
    end   = n - horizon - 1
    for t in range(start, end + 1):
        Xs.append(z[t - window_length + 1 : t + 1])
        ys.append(y_sc[t + horizon])
        ms.append(observed_mask[t + horizon])
        ks.append(kn[t + horizon])

    if not Xs:
        F = z.shape[1]
        return (np.empty((0, window_length, F)), np.empty((0, 6)),
                np.empty((0, 6), dtype=np.int8), np.empty((0, 6)))
    return (np.stack(Xs).astype(np.float32), np.stack(ys).astype(np.float32),
            np.stack(ms).astype(np.float32),  np.stack(ks).astype(np.float32))


def build_dataset(runs, profiles, masks, horizon, window_length=WINDOW_LENGTH,
                  use_label_onehot=True, feature_scaler=None,
                  target_scaler=None, sdae_encoder=None):
    Xs, ys, ms, ks = [], [], [], []
    for run_id, run_data in runs.items():
        X, y, m, k = build_windows_for_run(
            run_id, run_data, profiles[run_id], masks[run_id],
            horizon, window_length, use_label_onehot,
            feature_scaler, target_scaler, sdae_encoder)
        if len(X): Xs.append(X); ys.append(y); ms.append(m); ks.append(k)
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(ms), np.concatenate(ks))
