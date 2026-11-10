"""Long-only allocators for the base-paper comparison (docs/protocol-base-comparison.md)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = ["repair_weights", "nsga2_mean_variance", "max_sharpe_exact", "min_variance_capped"]


def repair_weights(w: np.ndarray, cap: float) -> np.ndarray:
    """Project onto {w >= 0, sum w = 1, w <= cap} by clipping and redistributing."""
    w = np.clip(np.asarray(w, float), 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    for _ in range(50):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum() if w[free].sum() > 0 else excess / free.sum()
    return w / w.sum()


def _nondominated_sort(F: np.ndarray) -> list[np.ndarray]:
    n = len(F)
    dom = (F[:, None, :] <= F[None, :, :]).all(-1) & (F[:, None, :] < F[None, :, :]).any(-1)
    count = dom.sum(axis=0)  # how many dominate each point
    fronts, remaining = [], np.arange(n)
    while len(remaining):
        front = remaining[count[remaining] == 0]
        fronts.append(front)
        remaining = np.setdiff1d(remaining, front)
        count = count - dom[front].sum(axis=0)
        count[front] = n + 1
    return fronts


def _crowding(F: np.ndarray) -> np.ndarray:
    d = np.zeros(len(F))
    for k in range(F.shape[1]):
        order = np.argsort(F[:, k])
        span = F[order[-1], k] - F[order[0], k]
        d[order[[0, -1]]] = np.inf
        if span > 0 and len(F) > 2:
            d[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / span
    return d


def nsga2_mean_variance(mu: np.ndarray, Sigma: np.ndarray, cap: float = 0.3, pop: int = 100,
                        generations: int = 150, seed: int = 0) -> np.ndarray:
    """NSGA-II on (maximise mu'w, minimise w'Sigma w); returns the max mu/sigma Pareto point."""
    rng = np.random.default_rng(seed)
    n = len(mu)
    P = np.array([repair_weights(rng.dirichlet(np.ones(n)), cap) for _ in range(pop)])

    def objectives(X):
        return np.column_stack([-(X @ mu), np.einsum("ij,jk,ik->i", X, Sigma, X)])

    F = objectives(P)
    for _ in range(generations):
        # binary tournament on (rank, crowding)
        fronts = _nondominated_sort(F)
        rank = np.empty(len(P), int)
        crowd = np.empty(len(P))
        for r, fr in enumerate(fronts):
            rank[fr] = r
            crowd[fr] = _crowding(F[fr])
        a, b = rng.integers(len(P), size=(2, pop))
        better = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
        parents = P[np.where(better, a, b)]
        # blend crossover and Gaussian mutation, then repair
        mates = parents[rng.permutation(pop)]
        alpha = rng.random((pop, 1))
        kids = alpha * parents + (1 - alpha) * mates
        mutate = rng.random((pop, n)) < 1.0 / n
        kids = kids + mutate * rng.normal(0, 0.05, (pop, n))
        kids = np.array([repair_weights(k, cap) for k in kids])
        # environmental selection
        U = np.vstack([P, kids])
        FU = objectives(U)
        chosen = []
        for fr in _nondominated_sort(FU):
            if len(chosen) + len(fr) <= pop:
                chosen.extend(fr)
            else:
                c = _crowding(FU[fr])
                chosen.extend(fr[np.argsort(-c)][: pop - len(chosen)])
                break
        P, F = U[chosen], FU[chosen]
    front = _nondominated_sort(F)[0]
    ret, var = -F[front, 0], F[front, 1]
    best = front[np.argmax(ret / np.sqrt(np.maximum(var, 1e-18)))]
    return P[best]


def min_variance_capped(Sigma: np.ndarray, cap: float = 0.3) -> np.ndarray:
    n = len(Sigma)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = minimize(lambda w: w @ Sigma @ w, np.full(n, 1.0 / n), jac=lambda w: 2 * Sigma @ w,
                   bounds=[(0.0, cap)] * n, constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-14})
    return repair_weights(res.x, cap)


def max_sharpe_exact(mu: np.ndarray, Sigma: np.ndarray, cap: float = 0.3) -> np.ndarray:
    """Exact long-only, capped maximum-Sharpe portfolio via its convex reformulation.

    Minimise y' Sigma y subject to mu'y = 1, y >= 0 and y_i <= cap * sum(y); then w = y / sum(y).
    Falls back to capped minimum variance when no stock has a positive forecast.
    """
    n = len(mu)
    if np.max(mu) <= 0:
        return min_variance_capped(Sigma, cap)
    y0 = np.where(mu > 0, 1.0, 0.0)
    y0 = y0 / (mu @ y0)
    cons = [{"type": "eq", "fun": lambda y: mu @ y - 1.0, "jac": lambda y: mu},
            {"type": "ineq", "fun": lambda y: cap * y.sum() - y,
             "jac": lambda y: cap * np.ones((n, n)) - np.eye(n)}]
    res = minimize(lambda y: y @ Sigma @ y, y0, jac=lambda y: 2 * Sigma @ y, bounds=[(0.0, None)] * n,
                   constraints=cons, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-16})
    y = np.clip(res.x, 0.0, None)
    if not res.success or y.sum() <= 0:
        return min_variance_capped(Sigma, cap)
    return repair_weights(y / y.sum(), cap)
