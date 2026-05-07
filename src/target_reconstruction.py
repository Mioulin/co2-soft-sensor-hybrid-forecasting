"""
Six-point CO2 profile reconstruction.

Two methods:
  1. kinetic_residual (default, recommended):
     Interpolate residuals r = AT400 - kinetic_prior, then
     y_profile = kinetic_prior + r_profile
     Justified: kinetic prior captures ~92% of CO2 variance; residual std
     is 12x smaller than raw CO2 std => interpolation error is minimal.

  2. linear (legacy, not recommended):
     Direct linear interpolation of sparse AT400 observations.
     No physical basis; produces distorted training data (Chai et al. 2026).
"""
from typing import Tuple
import numpy as np
import pandas as pd
from src.config import N_SAMPLING_POINTS, RECONSTRUCTION


def build_sparse_target(
    at400_frac: np.ndarray,
    label: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(at400_frac)
    y_sparse      = np.full((n, N_SAMPLING_POINTS), np.nan)
    observed_mask = np.zeros((n, N_SAMPLING_POINTS), dtype=np.int8)
    for t in range(n):
        pt = int(label[t]) - 1
        y_sparse[t, pt]      = at400_frac[t]
        observed_mask[t, pt] = 1
    return y_sparse, observed_mask


def _interpolate_channels(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation per channel, ffill/bfill at edges."""
    df = pd.DataFrame(arr)
    df = df.interpolate(method="linear", axis=0, limit_direction="both")
    df = df.ffill().bfill()
    return df.values.astype(np.float64)


def reconstruct_targets(
    at400_frac: np.ndarray,
    label: np.ndarray,
    kinetic: np.ndarray = None,
    method: str = RECONSTRUCTION,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full target reconstruction pipeline.

    Returns
    -------
    y_sparse      : (n, 6) raw sparse observations (NaN where missing)
    observed_mask : (n, 6) binary observation indicator
    y_profile     : (n, 6) reconstructed pseudo-label profile
    """
    y_sparse, observed_mask = build_sparse_target(at400_frac, label)

    if method == "kinetic_residual" and kinetic is not None:
        # Compute sparse residuals at observed positions
        r_sparse = np.full_like(y_sparse, np.nan)
        for t in range(len(at400_frac)):
            pt = int(label[t]) - 1
            r_sparse[t, pt] = at400_frac[t] - kinetic[t, pt]
        # Interpolate residuals (12x smaller variance than raw CO2)
        r_profile = _interpolate_channels(r_sparse)
        y_profile = kinetic + r_profile
        # Clip to physically valid range
        y_profile = np.clip(y_profile, 0.0, 1.0)
    else:
        # Fallback: direct linear interpolation
        y_profile = _interpolate_channels(y_sparse)

    if np.isnan(y_profile).any():
        raise ValueError("NaN in y_profile after reconstruction.")
    return y_sparse, observed_mask, y_profile
