"""
Data loading for CO2 soft-sensor project.

Loads raw Excel runs and matching kinetic prior CSVs.
Returns a dict keyed by run_id with a validated, unified DataFrame per run.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    COL_AT400,
    COL_LABEL,
    DATA_KINETIC,
    DATA_RAW,
    KINETIC_HEADER,
    AT400_SCALE,
    N_SAMPLING_POINTS,
)


# ── Excel loading ─────────────────────────────────────────────────────────────

def _flatten_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten two-level column MultiIndex to readable strings."""
    new_cols = []
    for top, bot in df.columns:
        top = str(top).strip()
        bot = str(bot).strip()
        if bot.startswith("Unnamed"):
            new_cols.append(top)
        else:
            new_cols.append(f"{top}_{bot}")
    df.columns = new_cols
    return df


def load_excel_run(run_id: str, data_dir: Path = DATA_RAW) -> pd.DataFrame:
    """
    Load a single labelled Excel run.

    Returns a DataFrame with flattened column names, a 'run_id' column,
    and a 'row_idx' column (0-based within run).

    The 'at400_frac' column is AT400 converted to CO2 fraction (/ 100).
    The 'label' column is the active sampling point as int in [1, 6].
    """
    path = data_dir / f"{run_id}.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Excel run not found: {path}")

    df = pd.read_excel(path, header=[0, 1])
    df = _flatten_multiindex(df)

    # ── Validate key columns ─────────────────────────────────────────────────
    assert df.shape[1] >= 5, f"{run_id}: too few columns ({df.shape[1]})"

    at400_raw = df.iloc[:, COL_AT400]
    label_raw = df.iloc[:, COL_LABEL]

    assert not at400_raw.isna().all(), f"{run_id}: AT400 column is all NaN"
    labels_unique = label_raw.dropna().astype(int).unique()
    assert set(labels_unique).issubset(set(range(1, N_SAMPLING_POINTS + 1))), \
        f"{run_id}: unexpected label values {labels_unique}"

    # ── Build clean output ────────────────────────────────────────────────────
    out = df.copy()
    out["run_id"]    = run_id
    out["row_idx"]   = np.arange(len(df))
    out["at400_frac"] = at400_raw.values / AT400_SCALE
    out["label"]     = label_raw.values.astype(int)

    return out


def load_kinetic_run(run_id: str, data_dir: Path = DATA_KINETIC) -> np.ndarray:
    """
    Load kinetic prior CSV for a run.

    Returns ndarray of shape (n_rows, 6) in fractional CO2 units.
    Columns 0-5 correspond to sampling points 1-6 (verified empirically).
    """
    path = data_dir / f"{run_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Kinetic CSV not found: {path}")

    kn = pd.read_csv(path, header=KINETIC_HEADER).values.astype(np.float64)

    assert kn.shape[1] == N_SAMPLING_POINTS, \
        f"{run_id}: kinetic CSV has {kn.shape[1]} columns, expected {N_SAMPLING_POINTS}"
    assert not np.isnan(kn).any(), f"{run_id}: kinetic CSV contains NaN"
    assert not np.isinf(kn).any(), f"{run_id}: kinetic CSV contains Inf"

    return kn


def load_all_runs(
    run_ids: List[str],
    data_raw: Path = DATA_RAW,
    data_kinetic: Path = DATA_KINETIC,
) -> Dict[str, dict]:
    """
    Load all runs. Returns dict:
      {run_id: {'df': DataFrame, 'kinetic': ndarray(n, 6)}}

    Validates that Excel and kinetic row counts match.
    """
    runs = {}
    for run_id in run_ids:
        df  = load_excel_run(run_id, data_raw)
        kn  = load_kinetic_run(run_id, data_kinetic)

        assert len(df) == len(kn), (
            f"{run_id}: Excel has {len(df)} rows, kinetic has {len(kn)} rows. "
            "Ensure kinetic CSV is read with header=None."
        )

        runs[run_id] = {"df": df, "kinetic": kn}

    return runs


def run_summary(runs: Dict[str, dict]) -> pd.DataFrame:
    """Return a summary table of run lengths and label counts."""
    rows = []
    for run_id, data in runs.items():
        df = data["df"]
        label_counts = df["label"].value_counts().sort_index()
        row = {"run_id": run_id, "n_rows": len(df)}
        for pt in range(1, N_SAMPLING_POINTS + 1):
            row[f"pt{pt}_count"] = label_counts.get(pt, 0)
        rows.append(row)
    return pd.DataFrame(rows).set_index("run_id")
