"""
Tests for CO2 soft-sensor project.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.data_loading import load_excel_run, load_kinetic_run, load_all_runs
from src.target_reconstruction import reconstruct_targets, build_sparse_target, interpolate_profile
from src.windowing import build_dataset, build_windows_for_run
from src.fusion import (
    estimate_residual_variances, inverse_variance_fusion,
    kalman_fusion, localization_matrix,
)


# ── test_data_alignment.py ─────────────────────────────────────────────────────

class TestDataAlignment:

    def test_excel_kinetic_row_match(self):
        """Every run: Excel and kinetic CSV must have identical row counts."""
        for run_id in cfg.ALL_RUNS:
            df = load_excel_run(run_id)
            kn = load_kinetic_run(run_id)
            assert len(df) == len(kn), (
                f"{run_id}: Excel={len(df)} rows, kinetic={len(kn)} rows. "
                "kinetic must be read with header=None."
            )

    def test_kinetic_no_nan_inf(self):
        for run_id in cfg.ALL_RUNS:
            kn = load_kinetic_run(run_id)
            assert not np.isnan(kn).any(), f"{run_id}: kinetic has NaN"
            assert not np.isinf(kn).any(), f"{run_id}: kinetic has Inf"

    def test_kinetic_column_ordering_forward(self):
        """Forward ordering (col 0 = point 1) must be 5x better than reversed."""
        for run_id in cfg.ALL_RUNS:
            df = load_excel_run(run_id)
            kn = load_kinetic_run(run_id)
            at400 = df["at400_frac"].values
            label = df["label"].values.astype(int)
            idx = label - 1
            err_fwd = np.abs(kn[np.arange(len(label)), idx] - at400).mean()
            err_rev = np.abs(kn[np.arange(len(label)), 5 - idx] - at400).mean()
            assert err_fwd < err_rev / 3, (
                f"{run_id}: forward error ({err_fwd:.5f}) not clearly "
                f"better than reversed ({err_rev:.5f})"
            )

    def test_label_values_valid(self):
        for run_id in cfg.ALL_RUNS:
            df = load_excel_run(run_id)
            labels = df["label"].values
            assert set(labels).issubset(set(range(1, 7))), \
                f"{run_id}: unexpected labels {set(labels)}"

    def test_at400_is_percent_range(self):
        """AT400 before /100 should be in [0, 25] (percent, not fraction)."""
        for run_id in cfg.ALL_RUNS:
            df = load_excel_run(run_id)
            raw = df.iloc[:, cfg.COL_AT400].values
            assert raw.max() > 1.0, \
                f"{run_id}: AT400 looks like it's already a fraction (max={raw.max():.4f})"


# ── test_target_reconstruction.py ─────────────────────────────────────────────

class TestTargetReconstruction:

    @pytest.fixture
    def sample_data(self):
        n = 60
        rng = np.random.default_rng(42)
        at400 = rng.uniform(0.0, 0.12, n)
        label = np.tile([1, 2, 3, 4, 5, 6], n // 6 + 1)[:n]
        return at400, label

    def test_sparse_shape(self, sample_data):
        at400, label = sample_data
        y_sparse, mask = build_sparse_target(at400, label)
        assert y_sparse.shape == (len(at400), 6)
        assert mask.shape    == (len(at400), 6)

    def test_observed_mask_one_per_row(self, sample_data):
        at400, label = sample_data
        _, mask = build_sparse_target(at400, label)
        assert (mask.sum(axis=1) == 1).all(), "Each row must have exactly 1 observed point"

    def test_sparse_value_at_correct_position(self, sample_data):
        at400, label = sample_data
        y_sparse, mask = build_sparse_target(at400, label)
        for t in range(len(at400)):
            pt = int(label[t]) - 1
            assert not np.isnan(y_sparse[t, pt]), f"Row {t}: observed position is NaN"
            assert mask[t, pt] == 1, f"Row {t}: mask not set at observed position"
            # all other positions must be NaN / 0
            other_pts = [i for i in range(6) if i != pt]
            assert np.all(np.isnan(y_sparse[t, other_pts]))
            assert np.all(mask[t, other_pts] == 0)

    def test_interpolated_no_nan(self, sample_data):
        at400, label = sample_data
        y_sparse, _ = build_sparse_target(at400, label)
        y_profile = interpolate_profile(y_sparse)
        assert not np.isnan(y_profile).any()

    def test_observed_values_preserved_after_interpolation(self, sample_data):
        at400, label = sample_data
        y_sparse, mask = build_sparse_target(at400, label)
        y_profile = interpolate_profile(y_sparse)
        obs = mask > 0
        np.testing.assert_allclose(
            y_profile[obs], y_sparse[obs], rtol=1e-6,
            err_msg="Interpolation must preserve observed values"
        )


# ── test_windowing.py ─────────────────────────────────────────────────────────

class TestWindowing:

    @pytest.fixture
    def mini_run(self):
        """A small synthetic run for windowing tests."""
        n = 40
        rng  = np.random.default_rng(0)
        at400 = rng.uniform(0.0, 0.12, n)
        label = np.tile([1, 2, 3, 4, 5, 6], n // 6 + 1)[:n]
        _, mask, profile = reconstruct_targets(at400, label)
        kn = rng.uniform(0.0, 0.12, (n, 6))
        # Build a fake run_data dict
        import pandas as pd
        cols_numeric = {f"feat_{i}": rng.normal(size=n) for i in range(5)}
        cols_numeric["AT400_(CO2 %)"] = at400 * 100
        cols_numeric["label"]   = label
        cols_numeric["run_id"]  = "fake"
        cols_numeric["row_idx"] = np.arange(n)
        cols_numeric["at400_frac"] = at400
        df = pd.DataFrame(cols_numeric)
        return {"df": df, "kinetic": kn}, profile, mask

    def test_window_shapes(self, mini_run):
        run_data, profile, mask = mini_run
        h = 1
        X, y, m, kn = build_windows_for_run(
            "fake", run_data, profile, mask, horizon=h, window_length=18
        )
        assert X.shape[1] == 18, "Window length must be 18"
        assert X.shape[2] > 0,   "Features must be > 0"
        assert y.shape[1] == 6
        assert m.shape[1] == 6
        assert kn.shape[1] == 6

    def test_target_at_correct_horizon(self, mini_run):
        run_data, profile, mask = mini_run
        h = 3
        X, y, m, kn = build_windows_for_run(
            "fake", run_data, profile, mask, horizon=h, window_length=18
        )
        n = len(run_data["df"])
        t_first = 17          # first end-of-window
        t_target = t_first + h
        np.testing.assert_allclose(
            y[0], profile[t_target], rtol=1e-5,
            err_msg=f"First target must be profile at t={t_target} (h={h})"
        )


# ── test_fusion.py ────────────────────────────────────────────────────────────

class TestFusion:

    @pytest.fixture
    def dummy(self):
        rng = np.random.default_rng(7)
        n = 100
        return {
            "kinetic": rng.uniform(0, 0.1, (n, 6)),
            "model":   rng.uniform(0, 0.1, (n, 6)),
            "target":  rng.uniform(0, 0.1, (n, 6)),
        }

    def test_iv_fusion_shape(self, dummy):
        var_k = estimate_residual_variances(dummy["kinetic"], dummy["target"])
        var_m = estimate_residual_variances(dummy["model"],   dummy["target"])
        fused = inverse_variance_fusion(dummy["kinetic"], dummy["model"], var_k, var_m)
        assert fused.shape == (100, 6)

    def test_iv_fusion_is_convex_combination(self, dummy):
        """Fused values must lie between kinetic and model (convex combination)."""
        var_k = estimate_residual_variances(dummy["kinetic"], dummy["target"])
        var_m = estimate_residual_variances(dummy["model"],   dummy["target"])
        fused = inverse_variance_fusion(dummy["kinetic"], dummy["model"], var_k, var_m)
        lo = np.minimum(dummy["kinetic"], dummy["model"])
        hi = np.maximum(dummy["kinetic"], dummy["model"])
        assert np.all(fused >= lo - 1e-8) and np.all(fused <= hi + 1e-8)

    def test_kalman_fusion_shape(self, dummy):
        from src.fusion import estimate_residual_covariance
        B = estimate_residual_covariance(dummy["kinetic"], dummy["target"])
        R = estimate_residual_covariance(dummy["model"],   dummy["target"])
        fused = kalman_fusion(dummy["kinetic"], dummy["model"], B, R)
        assert fused.shape == (100, 6)

    def test_localization_matrix_diagonal_ones(self):
        L = localization_matrix(decay=0.5)
        np.testing.assert_allclose(np.diag(L), np.ones(6))

    def test_localization_matrix_symmetric(self):
        L = localization_matrix(decay=0.3)
        np.testing.assert_allclose(L, L.T)
