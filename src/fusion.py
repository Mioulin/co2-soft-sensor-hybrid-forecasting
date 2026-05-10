"""
Fusion of kinetic prior with neural model predictions.

Inverse-variance fusion (point-wise):
  fused_i = (k_i/var_k_i + m_i/var_m_i) / (1/var_k_i + 1/var_m_i)

Kalman-like covariance fusion with Gaspari-Cohn localization:
  x_fused = x_k + B_loc (B_loc + R_loc)^{-1} (x_m - x_k)
  B_loc = B * GC(rho),  R_loc = R * GC(rho)

Gaspari-Cohn function (fifth-order, compact support at rho>=2):
  Standard in EnKF/variational assimilation. Preserves positive
  semidefiniteness when applied to a PSD covariance matrix (Schur product
  theorem), and damps spurious long-range correlations between sampling points.
  Reference: Chai et al. 2026, eq. 25.
"""
import numpy as np


def gaspari_cohn(rho: float) -> float:
    """Fifth-order piecewise polynomial, compactly supported at rho >= 2."""
    r = abs(rho)
    if r >= 2.0:
        return 0.0
    elif r >= 1.0:
        return (4 - 5*r + 5/3*r**2 + 5/8*r**3 - 1/2*r**4 + 1/12*r**5 - 2/(3*r))
    else:
        return (1 - 5/3*r**2 + 5/8*r**3 + 1/2*r**4 - 1/4*r**5)


def gaspari_cohn_matrix(n: int = 6, L: float = 2.0) -> np.ndarray:
    """
    Build n x n Gaspari-Cohn localization matrix.
    L: correlation length in index units.
    """
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = gaspari_cohn(abs(i - j) / L)
    return M


def localization_matrix(n: int = 6, decay: float = 0.5) -> np.ndarray:
    """
    Simple exponential localization matrix: L[i,j] = exp(-decay * |i-j|).
    Diagonal is 1 by construction; matrix is symmetric.
    Used in tests and as an alternative to Gaspari-Cohn for pre-localizing
    B and R before passing to kalman_fusion.
    """
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = np.exp(-decay * abs(i - j))
    return M


def estimate_residual_variances(pred, target, epsilon=1e-8):
    var = np.var(target - pred, axis=0)
    return np.maximum(var, epsilon)


def estimate_residual_covariance(pred, target, reg_eps=1e-6):
    residuals = target - pred
    cov = np.cov(residuals.T)
    return cov + reg_eps * np.eye(cov.shape[0])


def inverse_variance_fusion(kinetic, model, var_kinetic, var_model, epsilon=1e-8):
    w_k = 1.0 / np.maximum(var_kinetic, epsilon)
    w_m = 1.0 / np.maximum(var_model,   epsilon)
    return (kinetic * w_k + model * w_m) / (w_k + w_m)


def fusion_weights(var_kinetic, var_model, epsilon=1e-8):
    w_k = 1.0 / np.maximum(var_kinetic, epsilon)
    w_m = 1.0 / np.maximum(var_model,   epsilon)
    s   = w_k + w_m
    return w_k / s, w_m / s


def kalman_fusion(kinetic, model, B, R, reg=1e-6):
    """
    Kalman-like fusion with pre-localized covariance matrices.
    x_fused = x_k + B (B + R)^{-1} (x_m - x_k)

    B and R must already be localized by the caller (e.g. multiplied by
    localization_matrix() or gaspari_cohn_matrix()). This is the bare
    Kalman update step.
    """
    n = B.shape[0]
    B_reg = B + reg * np.eye(n)
    R_reg = R + reg * np.eye(n)
    K = B_reg @ np.linalg.inv(B_reg + R_reg)
    return kinetic + (model - kinetic) @ K.T


def kalman_fusion_gc(kinetic, model, B, R, L=2.0, reg=1e-6):
    """
    Kalman-like fusion with Gaspari-Cohn localization applied internally.
    x_fused = x_k + B_loc (B_loc + R_loc)^{-1} (x_m - x_k)

    Localization is applied here; pass raw B, R from estimate_residual_covariance.
    """
    GC = gaspari_cohn_matrix(n=6, L=L)
    B_loc = B * GC + reg * np.eye(6)
    R_loc = R * GC + reg * np.eye(6)
    K     = B_loc @ np.linalg.inv(B_loc + R_loc)
    return kinetic + (model - kinetic) @ K.T
