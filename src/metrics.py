"""Evaluation metrics: full-profile RMSE, observed-point RMSE, MAPE, per-point."""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def _rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))
def _mae(a, b):  return float(np.mean(np.abs(a - b)))

def _mape(pred, target, min_target=1e-4):
    """MAPE only where |target| > min_target to avoid division issues."""
    valid = np.abs(target) > min_target
    if valid.sum() == 0: return float("nan")
    return float(np.mean(np.abs((pred[valid] - target[valid]) / target[valid])) * 100)

def full_profile_rmse(pred, target): return _rmse(pred, target)
def observed_point_rmse(pred, target, mask):
    obs = mask > 0.5
    return _rmse(pred[obs], target[obs]) if obs.sum() > 0 else float("nan")

def per_point_metrics(pred, target, mask):
    rows = []
    for pt in range(6):
        p, t = pred[:, pt], target[:, pt]
        obs  = mask[:, pt] > 0.5
        rows.append({"point": pt+1,
            "rmse_full":     _rmse(p, t),     "mae_full":     _mae(p, t),
            "rmse_observed": _rmse(p[obs], t[obs]) if obs.sum()>0 else np.nan,
            "mape_observed": _mape(p[obs], t[obs]) if obs.sum()>0 else np.nan,
            "n_observed":    int(obs.sum())})
    return pd.DataFrame(rows).set_index("point")

def compute_all_metrics(pred, target, mask, label="model", horizon=None):
    return {"label": label, "horizon": horizon,
            "full_rmse": full_profile_rmse(pred, target),
            "obs_rmse":  observed_point_rmse(pred, target, mask),
            "full_mae":  _mae(pred, target),
            "mape":      _mape(pred[mask > 0.5], target[mask > 0.5])}

def persistence_predictions(runs, profiles, masks, horizon, window_length):
    preds, targets, mks = [], [], []
    for run_id, run_data in runs.items():
        prof = profiles[run_id]; mask = masks[run_id]; n = len(prof)
        for t in range(window_length - 1, n - horizon):
            preds.append(prof[t]); targets.append(prof[t+horizon]); mks.append(mask[t+horizon])
    return (np.stack(preds).astype(np.float32), np.stack(targets).astype(np.float32),
            np.stack(mks).astype(np.float32))

def kinetic_predictions(runs, profiles, masks, horizon, window_length):
    preds, targets, mks = [], [], []
    for run_id, run_data in runs.items():
        kn = run_data["kinetic"]; prof = profiles[run_id]; mask = masks[run_id]; n = len(prof)
        for t in range(window_length - 1, n - horizon):
            preds.append(kn[t+horizon]); targets.append(prof[t+horizon]); mks.append(mask[t+horizon])
    return (np.stack(preds).astype(np.float32), np.stack(targets).astype(np.float32),
            np.stack(mks).astype(np.float32))
