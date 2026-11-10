"""Stage 1 of FuzzShrink: fuzzy memberships of assets (docs/method.md, section 2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .moments import WindowStats

__all__ = [
    "FuzzyPartition",
    "comovement_features",
    "riskreturn_features",
    "fuzzy_cmeans",
    "xie_beni",
    "fit_memberships",
    "co_membership",
    "membership_entropy",
]


@dataclass
class FuzzyPartition:
    """Result of fuzzy c-means on one feature matrix."""

    U: np.ndarray  # (N, C) memberships, rows sum to 1
    centers: np.ndarray  # (C, k) prototypes
    d2: np.ndarray  # (N, C) squared distances to the prototypes
    m: float
    objective: float
    n_iter: int
    xb_by_c: dict = field(default_factory=dict)  # Xie-Beni index for each C tried

    @property
    def n_clusters(self) -> int:
        return self.U.shape[1]


# ----------------------------------------------------------------------------- features


def comovement_features(R: np.ndarray, n_obs: int, k_min: int = 2, k_max: int = 10) -> np.ndarray:
    """Market-neutral principal-component loadings (method section 2.1).

    The market mode (largest eigenvalue) is dropped. The next k components are kept,
    where k counts the eigenvalues above the Marchenko-Pastur edge (1 + sqrt(N/T))^2,
    clipped to [k_min, k_max]. Loadings are scaled by sqrt(eigenvalue) and rows are
    not normalised, so assets with weak group loadings sit near the origin.
    """
    N = R.shape[0]
    evals, evecs = np.linalg.eigh(R)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    edge = (1.0 + np.sqrt(N / n_obs)) ** 2
    k = int(np.sum(evals[1:] > edge))
    k = int(np.clip(k, k_min, min(k_max, N - 1)))
    lam = np.clip(evals[1 : k + 1], 0.0, None)
    return evecs[:, 1 : k + 1] * np.sqrt(lam)


def riskreturn_features(X: np.ndarray, periods_per_year: int = 252) -> np.ndarray:
    """Draft-faithful features (ablation X4): six risk-return statistics, z-scored."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    sd0 = np.maximum(X.std(axis=0, ddof=0), 1e-12)
    mean = X.mean(axis=0) * periods_per_year
    vol = X.std(axis=0, ddof=1) * np.sqrt(periods_per_year)
    skew = (Xc**3).mean(axis=0) / sd0**3
    kurt = (Xc**4).mean(axis=0) / sd0**4 - 3.0
    mkt = X.mean(axis=1)
    mkt_c = mkt - mkt.mean()
    beta = (Xc * mkt_c[:, None]).mean(axis=0) / max(mkt_c.var(), 1e-18)
    wealth = np.cumprod(1.0 + X, axis=0)
    mdd = (wealth / np.maximum.accumulate(wealth, axis=0) - 1.0).min(axis=0)
    F = np.column_stack([mean, vol, skew, kurt, beta, mdd])
    scale = F.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    return (F - F.mean(axis=0)) / scale


# ----------------------------------------------------------------------- fuzzy c-means


def _sqdist(Z: np.ndarray, V: np.ndarray) -> np.ndarray:
    d2 = (Z * Z).sum(1)[:, None] - 2.0 * Z @ V.T + (V * V).sum(1)[None, :]
    return np.maximum(d2, 0.0)


def _memberships(d2: np.ndarray, m: float) -> np.ndarray:
    """FCM membership update; an asset on a prototype gets that cluster crisply."""
    U = np.empty_like(d2)
    hits = d2 <= 1e-300
    hit_rows = hits.any(axis=1)
    if hit_rows.any():
        U[hit_rows] = hits[hit_rows] / hits[hit_rows].sum(axis=1, keepdims=True)
    rows = ~hit_rows
    if rows.any():
        d = d2[rows]
        ratio = d / d.min(axis=1, keepdims=True)  # in [1, inf): numerically stable
        inv = ratio ** (-1.0 / (m - 1.0))
        U[rows] = inv / inv.sum(axis=1, keepdims=True)
    return U


def _kmeanspp(Z: np.ndarray, C: int, rng: np.random.Generator) -> np.ndarray:
    N = Z.shape[0]
    idx = [int(rng.integers(N))]
    d2 = ((Z - Z[idx[0]]) ** 2).sum(axis=1)
    for _ in range(1, C):
        total = d2.sum()
        nxt = int(rng.integers(N)) if total <= 0 else int(rng.choice(N, p=d2 / total))
        idx.append(nxt)
        d2 = np.minimum(d2, ((Z - Z[nxt]) ** 2).sum(axis=1))
    return Z[idx].copy()


def fuzzy_cmeans(
    Z: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    seed: int = 0,
    n_init: int = 5,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> FuzzyPartition:
    """Fuzzy c-means (Bezdek, 1981) with k-means++ seeding; best of n_init restarts."""
    Z = np.asarray(Z, dtype=float)
    N = Z.shape[0]
    if not 1 <= n_clusters <= N:
        raise ValueError("need 1 <= n_clusters <= number of assets")
    if m <= 1.0:
        raise ValueError("the fuzzifier m must be > 1")
    if n_clusters == 1:
        V = Z.mean(axis=0, keepdims=True)
        d2 = _sqdist(Z, V)
        return FuzzyPartition(np.ones((N, 1)), V, d2, m, float(d2.sum()), 0)

    rng = np.random.default_rng(seed)
    best: FuzzyPartition | None = None
    for _ in range(n_init):
        V = _kmeanspp(Z, n_clusters, rng)
        d2 = _sqdist(Z, V)
        U = _memberships(d2, m)
        n_iter = 0
        for n_iter in range(1, max_iter + 1):
            Um = U**m
            V = (Um.T @ Z) / np.maximum(Um.sum(axis=0), 1e-300)[:, None]
            d2 = _sqdist(Z, V)
            U_new = _memberships(d2, m)
            change = float(np.abs(U_new - U).max())
            U = U_new
            if change < tol:
                break
        obj = float(((U**m) * d2).sum())
        if best is None or obj < best.objective:
            best = FuzzyPartition(U, V, d2, m, obj, n_iter)
    assert best is not None
    return best


def xie_beni(part: FuzzyPartition) -> float:
    """Xie-Beni validity index (lower is better); infinite for one cluster."""
    if part.n_clusters < 2:
        return float("inf")
    sep = _sqdist(part.centers, part.centers)
    np.fill_diagonal(sep, np.inf)
    min_sep = float(sep.min())
    if min_sep <= 0:
        return float("inf")
    compact = float(((part.U**part.m) * part.d2).sum())
    return compact / (part.U.shape[0] * min_sep)


def fit_memberships(
    stats: WindowStats,
    *,
    features: str = "comove",
    n_clusters: int | str = "auto",
    c_range: tuple[int, int] = (2, 8),
    m: float = 2.0,
    seed: int = 0,
    n_init: int = 5,
) -> FuzzyPartition:
    """Stage 1 end to end: features, then FCM with C chosen by Xie-Beni (method 2.1-2.3)."""
    if features == "comove":
        Z = comovement_features(stats.R, stats.T)
    elif features == "riskret":
        Z = riskreturn_features(stats.X)
    else:
        raise ValueError(f"unknown features: {features!r}")

    if n_clusters != "auto":
        return fuzzy_cmeans(Z, int(n_clusters), m=m, seed=seed, n_init=n_init)

    lo, hi = c_range
    hi = min(hi, stats.N - 1)
    best: FuzzyPartition | None = None
    best_xb = float("inf")
    xb_by_c: dict[int, float] = {}
    for C in range(lo, hi + 1):
        part = fuzzy_cmeans(Z, C, m=m, seed=seed * 1009 + C, n_init=n_init)
        xb = xie_beni(part)
        xb_by_c[C] = xb
        if best is None or xb < best_xb:
            best, best_xb = part, xb
    assert best is not None
    best.xb_by_c = xb_by_c
    return best


# ------------------------------------------------------------------- membership summaries


def co_membership(U: np.ndarray) -> np.ndarray:
    """s_ij = sum_c u_ic u_jc (Eq. 2)."""
    return U @ U.T


def membership_entropy(U: np.ndarray) -> np.ndarray:
    """Normalised entropy H_i in [0, 1] (Eq. 5); zero when C = 1."""
    U = np.asarray(U, dtype=float)
    C = U.shape[1]
    if C == 1:
        return np.zeros(U.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(U > 0, U * np.log(U), 0.0)
    return np.clip(-plogp.sum(axis=1) / np.log(C), 0.0, 1.0)
