"""Synthetic market with known covariance (docs/protocol.md, section 7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SimMarket", "simulate_market"]


@dataclass
class SimMarket:
    X: np.ndarray  # (T, N) returns
    Sigma: np.ndarray  # (N, N) true covariance
    U_true: np.ndarray  # (N, C) true memberships (outliers: uniform)
    types: np.ndarray  # (N,) "crisp", "bridge" or "outlier"


def _unit_t(rng: np.random.Generator, df: float, size) -> np.ndarray:
    """Student-t draws scaled to unit variance (Gaussian if df is infinite)."""
    if not np.isfinite(df):
        return rng.standard_normal(size)
    return rng.standard_t(df, size=size) / np.sqrt(df / (df - 2.0))


def simulate_market(
    N: int = 100,
    n_groups: int = 5,
    pi_bridge: float = 0.0,
    pi_outlier: float = 0.0,
    T: int = 252,
    df: float = 5.0,
    seed: int = 0,
) -> SimMarket:
    """Draw one market and one sample of T daily returns from it.

    Asset types (protocol section 7):
    - crisp: one group; market loading a ~ U(0.4, 0.6), group loading b ~ U(0.5, 0.7)
    - bridge: Dirichlet(1, 1) memberships over two groups, same loadings as crisp
    - outlier: no group loading; a ~ U(0.15, 0.35); a private pair factor (loading 0.6)
      shared with one other outlier
    Each asset has unit variance before scaling by a daily volatility in U(0.01, 0.03).
    """
    rng = np.random.default_rng(seed)
    C = n_groups
    n_out = int(round(pi_outlier * N))
    n_out -= n_out % 2  # outliers come in pairs
    n_bridge = int(round(pi_bridge * N))
    n_crisp = N - n_out - n_bridge
    if n_crisp < C:
        raise ValueError("too few crisp assets for the number of groups")

    types = np.array(["crisp"] * n_crisp + ["bridge"] * n_bridge + ["outlier"] * n_out)
    U = np.zeros((N, C))
    U[np.arange(n_crisp), rng.permutation(np.arange(n_crisp) % C)] = 1.0
    for i in range(n_crisp, n_crisp + n_bridge):
        pair = rng.choice(C, size=2, replace=False)
        U[i, pair] = rng.dirichlet([1.0, 1.0])
    U[n_crisp + n_bridge :] = 1.0 / C

    n_pairs = n_out // 2
    B = np.zeros((N, 1 + C + n_pairs))
    a = np.where(types == "outlier", rng.uniform(0.15, 0.35, N), rng.uniform(0.4, 0.6, N))
    b = np.where(types == "outlier", 0.0, rng.uniform(0.5, 0.7, N))
    B[:, 0] = a
    B[:, 1 : 1 + C] = b[:, None] * U * (types != "outlier")[:, None]
    outliers = rng.permutation(np.arange(n_crisp + n_bridge, N))
    for k in range(n_pairs):
        B[outliers[2 * k : 2 * k + 2], 1 + C + k] = 0.6
    e2 = 1.0 - (B**2).sum(axis=1)
    if np.any(e2 <= 0):
        raise ValueError("loadings leave no idiosyncratic variance")
    P = B @ B.T + np.diag(e2)
    vol = rng.uniform(0.01, 0.03, N)
    Sigma = P * np.outer(vol, vol)

    factors = _unit_t(rng, df, (T, B.shape[1]))
    idio = _unit_t(rng, df, (T, N))
    X = (factors @ B.T + idio * np.sqrt(e2)) * vol

    order = rng.permutation(N)  # so asset type is not tied to column position
    return SimMarket(
        X=X[:, order],
        Sigma=Sigma[np.ix_(order, order)],
        U_true=U[order],
        types=types[order],
    )
