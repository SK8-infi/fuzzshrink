"""Small data generators shared by the unit tests."""

import numpy as np


def block_returns(n_blocks=3, per_block=20, T=500, a=0.5, b=0.55, seed=0):
    """Returns from a market factor plus one factor per block; returns (X, labels)."""
    rng = np.random.default_rng(seed)
    N = n_blocks * per_block
    labels = np.repeat(np.arange(n_blocks), per_block)
    f = rng.standard_normal(T)
    g = rng.standard_normal((T, n_blocks))
    e = np.sqrt(1.0 - a**2 - b**2)
    X = a * f[:, None] + b * g[:, labels] + e * rng.standard_normal((T, N))
    vol = rng.uniform(0.01, 0.03, size=N)
    return X * vol, labels


def random_memberships(N, C, seed=0, alpha=0.7):
    """Dirichlet membership rows (soft partition)."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.full(C, alpha), size=N)
