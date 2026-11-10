"""Minimum-variance portfolios (docs/method.md, section 8)."""

from __future__ import annotations

import numpy as np

__all__ = ["gmv_weights", "project_simplex", "gmv_long_only"]


def gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """Global minimum-variance weights: Sigma^-1 1 / (1' Sigma^-1 1)."""
    ones = np.ones(Sigma.shape[0])
    try:
        x = np.linalg.solve(Sigma, ones)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(Sigma, ones, rcond=None)[0]
    return x / x.sum()


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w : w >= 0, sum w = 1} (Duchi et al., 2008)."""
    v = np.asarray(v, dtype=float)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.arange(1, v.size + 1)
    rho = int(np.nonzero(u * k > css - 1.0)[0][-1])
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def gmv_long_only(Sigma: np.ndarray, tol: float = 1e-10, max_iter: int = 20000) -> np.ndarray:
    """Long-only minimum variance by FISTA with adaptive restart and simplex projection."""
    n = Sigma.shape[0]
    step = 1.0 / (2.0 * float(np.linalg.eigvalsh(Sigma).max()))
    x = np.full(n, 1.0 / n)
    y = x.copy()
    t = 1.0
    for _ in range(max_iter):
        x_new = project_simplex(y - step * 2.0 * (Sigma @ y))
        if np.max(np.abs(x_new - x)) < tol:
            return x_new
        if (y - x_new) @ (x_new - x) > 0:  # momentum is pointing uphill: restart
            t = 1.0
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x
