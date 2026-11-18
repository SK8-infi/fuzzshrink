"""Sample moments shared by every estimator (docs/method.md, section 1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WindowStats", "sample_covariance", "cov_to_corr", "corr_variance"]


def sample_covariance(X: np.ndarray) -> np.ndarray:
    """Sample covariance of a (T, N) return matrix, divisor T - 1."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / (X.shape[0] - 1)


def cov_to_corr(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a covariance matrix into (correlation matrix, standard deviations)."""
    sd = np.sqrt(np.diag(S))
    if np.any(sd <= 0):
        raise ValueError("every asset needs a positive sample variance")
    R = S / np.outer(sd, sd)
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.0)
    return R, sd


def corr_variance(X: np.ndarray) -> np.ndarray:
    """Estimated variance of each sample correlation (Schafer and Strimmer, 2005).

    With standardised returns z, w_tij = z_ti z_tj and wbar_ij = mean_t w_tij:
    Var(r_ij) = T / (T - 1)^3 * sum_t (w_tij - wbar_ij)^2.
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    Xc = X - X.mean(axis=0)
    Z = Xc / Xc.std(axis=0, ddof=1)
    wbar = Z.T @ Z / T
    Z2 = Z * Z
    sum_w2 = Z2.T @ Z2
    var = T / (T - 1) ** 3 * (sum_w2 - T * wbar * wbar)
    return np.maximum(var, 0.0)


@dataclass
class WindowStats:
    """Everything the estimators need from one estimation window, computed once."""

    X: np.ndarray
    S: np.ndarray
    R: np.ndarray
    sd: np.ndarray
    varR: np.ndarray

    @property
    def T(self) -> int:
        return self.X.shape[0]

    @property
    def N(self) -> int:
        return self.X.shape[1]

    @classmethod
    def from_returns(cls, X: np.ndarray) -> "WindowStats":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 3:
            raise ValueError("X must be a (T, N) array with T >= 3")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains missing or non-finite returns")
        S = sample_covariance(X)
        R, sd = cov_to_corr(S)
        return cls(X=X, S=S, R=R, sd=sd, varR=corr_variance(X))
