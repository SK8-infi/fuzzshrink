"""Covariance estimators: FuzzShrink and its baselines (docs/method.md, sections 3-7)."""

from __future__ import annotations

import numpy as np

from .clustering import co_membership, membership_entropy
from .moments import WindowStats

__all__ = [
    "two_level_target",
    "block_pair_target",
    "intensity_weights",
    "fit_ratio",
    "base_intensity",
    "combine_angle",
    "combine_arithmetic",
    "nearest_correlation",
    "fuzzshrink",
    "sample_estimator",
    "constant_correlation",
    "hard_block",
    "ledoit_wolf_cc",
    "ledoit_wolf_nonlinear",
]

_OFF_TOL = 1e-12


def _offdiag(N: int) -> np.ndarray:
    return ~np.eye(N, dtype=bool)


# --------------------------------------------------------------------------- targets


def two_level_target(R: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, dict]:
    """Soft block target (Eq. 3-4): F_ij = s_ij rho_w + (1 - s_ij) rho_b, F_ii = 1."""
    N = R.shape[0]
    S = co_membership(U)
    off = _offdiag(N)
    s, r = S[off], R[off]
    w_within, w_between = s.sum(), (1.0 - s).sum()
    rho_w = float((s * r).sum() / w_within) if w_within > _OFF_TOL else float(r.mean())
    if w_between > _OFF_TOL * off.sum():
        rho_b = float(((1.0 - s) * r).sum() / w_between)
    else:  # every pair fully co-member (e.g. C = 1): rho_b never enters F
        rho_b = rho_w
    F = rho_b + (rho_w - rho_b) * S
    np.fill_diagonal(F, 1.0)
    info = {
        "rho_within": rho_w,
        "rho_between": rho_b,
        "lemma1_holds": bool(0.0 <= rho_b <= rho_w <= 1.0),
    }
    return F, info


def block_pair_target(R: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, dict]:
    """Block-pair target (extension X2, method N3): F_ij = u_i' P u_j, F_ii = 1."""
    N = R.shape[0]
    num = U.T @ (R - np.eye(N)) @ U
    col = U.sum(axis=0)
    den = np.outer(col, col) - U.T @ U
    P = np.where(den > _OFF_TOL, num / np.maximum(den, _OFF_TOL), 0.0)
    P = (P + P.T) / 2
    evals, evecs = np.linalg.eigh(P)
    clipped = bool(evals.min() < 0)
    if clipped:
        P = (evecs * np.clip(evals, 0.0, None)) @ evecs.T
    quad = np.einsum("ic,ck,ik->i", U, P, U)
    rescaled = bool(quad.max() > 1.0)
    if rescaled:  # keeps the diagonal correction non-negative, so F stays PSD
        P = P / quad.max()
    F = U @ P @ U.T
    np.fill_diagonal(F, 1.0)
    return F, {"block_means": P, "eig_clipped": clipped, "rescaled": rescaled}


# ------------------------------------------------------------------------ intensities


def fit_ratio(R: np.ndarray, F: np.ndarray, varR: np.ndarray) -> np.ndarray:
    """q_i: row-wise noise-to-misfit ratio of asset i's correlations (method N1)."""
    off = _offdiag(R.shape[0])
    noise = np.where(off, varR, 0.0).sum(axis=1)
    misfit = np.where(off, (R - F) ** 2, 0.0).sum(axis=1)
    return noise / np.maximum(misfit, 1e-300)


def intensity_weights(
    gamma: float,
    *,
    entropy: np.ndarray | None = None,
    fit: np.ndarray | None = None,
    kind: str = "entropy",
) -> np.ndarray:
    """Per-asset weights w_i (method 4.1): (1 - H_i)^gamma or q_i^gamma."""
    if kind == "entropy":
        if entropy is None:
            raise ValueError("entropy weights need the membership entropy")
        base = np.clip(1.0 - np.asarray(entropy, dtype=float), 1e-6, 1.0)
    elif kind == "fit":
        if fit is None:
            raise ValueError("fit weights need the fit ratios")
        base = np.clip(np.asarray(fit, dtype=float), 1e-6, 1e6)
    else:
        raise ValueError(f"unknown weight kind: {kind!r}")
    if gamma == 0:
        return np.ones_like(base)
    return base**gamma


def base_intensity(R: np.ndarray, F: np.ndarray, varR: np.ndarray, w: np.ndarray) -> float:
    """Closed-form base intensity delta_0 (method N5), clipped to [0, 1 / min w]."""
    off = _offdiag(R.shape[0])
    a = np.sqrt(np.outer(w, w))[off]
    num = float((a * varR[off]).sum())
    den = float((a * a * (R[off] - F[off]) ** 2).sum())
    upper = 1.0 / float(np.min(w))
    if den <= 0.0:
        return upper if num > 0 else 0.0
    return float(np.clip(num / den, 0.0, upper))


# ---------------------------------------------------------------------- combinations


def combine_angle(R: np.ndarray, F: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Angle rule (method N6): diag(c) R diag(c) + diag(s) F diag(s); PSD if R, F are."""
    s = np.sqrt(delta)
    c = np.sqrt(1.0 - delta)
    Rhat = c[:, None] * R * c[None, :] + s[:, None] * F * s[None, :]
    Rhat = (Rhat + Rhat.T) / 2
    np.fill_diagonal(Rhat, 1.0)
    return Rhat


def nearest_correlation(
    A: np.ndarray, eig_floor: float = 1e-8, tol: float = 1e-10, max_iter: int = 1000
) -> np.ndarray:
    """Nearest correlation matrix by Higham's (2002) alternating projections."""
    Y = (A + A.T) / 2
    dS = np.zeros_like(Y)
    for _ in range(max_iter):
        Rk = Y - dS
        evals, evecs = np.linalg.eigh(Rk)
        X = (evecs * np.maximum(evals, eig_floor)) @ evecs.T
        dS = X - Rk
        Y_old = Y
        Y = (X + X.T) / 2
        np.fill_diagonal(Y, 1.0)
        if np.linalg.norm(Y - Y_old) <= tol * np.linalg.norm(Y):
            break
    return Y


def combine_arithmetic(
    R: np.ndarray, F: np.ndarray, delta: np.ndarray, tol: float = 1e-10
) -> tuple[np.ndarray, dict]:
    """The draft's rule (Eq. 7-8) with its projection safeguard (draft section 5.7)."""
    d = np.sqrt(delta)
    W = np.outer(d, d)
    Rhat = (1.0 - W) * R + W * F
    Rhat = (Rhat + Rhat.T) / 2
    np.fill_diagonal(Rhat, 1.0)
    min_eig = float(np.linalg.eigvalsh(Rhat).min())
    projected = min_eig < tol
    moved = 0.0
    if projected:
        fixed = nearest_correlation(Rhat)
        moved = float(np.linalg.norm(fixed - Rhat))
        Rhat = fixed
    return Rhat, {"min_eig_before": min_eig, "projected": projected, "frobenius_moved": moved}


# ------------------------------------------------------------------------- FuzzShrink


def fuzzshrink(
    stats: WindowStats,
    U: np.ndarray,
    *,
    gamma: float = 1.0,
    weights: str = "entropy",
    target: str = "two_level",
    combine: str = "angle",
    delta0_scale: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """FuzzShrink covariance estimate (method sections 3-5).

    Returns (Sigma_hat, info). Sigma_hat keeps the sample variances; info records the
    target means, the base intensity, the per-asset intensities and the PSD diagnostics.
    delta0_scale multiplies the base intensity (1 = the pre-registered rule; other
    values are only used by the post-hoc analysis in scripts/run_exploratory.py).
    """
    U = np.asarray(U, dtype=float)
    if U.shape[0] != stats.N:
        raise ValueError("U must have one row per asset")
    R = stats.R
    if target == "two_level":
        F, tinfo = two_level_target(R, U)
    elif target == "block_pair":
        F, tinfo = block_pair_target(R, U)
    else:
        raise ValueError(f"unknown target: {target!r}")

    H = membership_entropy(U)
    q = fit_ratio(R, F, stats.varR) if weights == "fit" else None
    w = intensity_weights(gamma, entropy=H, fit=q, kind=weights)
    delta0 = base_intensity(R, F, stats.varR, w) * delta0_scale
    delta = np.minimum(1.0, delta0 * w)

    if combine == "angle":
        Rhat = combine_angle(R, F, delta)
        cinfo = {"projected": False, "frobenius_moved": 0.0}
    elif combine == "arithmetic":
        Rhat, cinfo = combine_arithmetic(R, F, delta)
    else:
        raise ValueError(f"unknown combination rule: {combine!r}")

    Sigma = Rhat * np.outer(stats.sd, stats.sd)
    info = {**tinfo, **cinfo, "delta0": delta0, "delta": delta, "entropy": H, "fit": q}
    return Sigma, info


def sample_estimator(stats: WindowStats) -> np.ndarray:
    """A0: the sample covariance."""
    return stats.S.copy()


def constant_correlation(stats: WindowStats) -> tuple[np.ndarray, dict]:
    """A1: FuzzShrink with a single cluster = constant-correlation shrinkage (Prop. 1)."""
    return fuzzshrink(stats, np.ones((stats.N, 1)), gamma=0.0)


def hard_block(stats: WindowStats, U: np.ndarray) -> tuple[np.ndarray, dict]:
    """A2: FuzzShrink with the hardened partition = hard block shrinkage (Prop. 2)."""
    U = np.asarray(U, dtype=float)
    hard = np.zeros_like(U)
    hard[np.arange(U.shape[0]), U.argmax(axis=1)] = 1.0
    return fuzzshrink(stats, hard, gamma=0.0)


# ------------------------------------------------------------------- external references


def ledoit_wolf_cc(X: np.ndarray) -> tuple[np.ndarray, float]:
    """R1: Ledoit and Wolf (2004) constant-correlation shrinkage, as in their covCor code.

    Returns (Sigma_hat, shrinkage intensity). Uses the divisor T, as published.
    """
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T
    var = np.diag(S).copy()
    sd = np.sqrt(var)
    rbar = (float((S / np.outer(sd, sd)).sum()) - N) / (N * (N - 1))
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, var)

    Y = Xc**2
    phi_mat = Y.T @ Y / T - S**2
    phi = float(phi_mat.sum())
    theta = (Xc**3).T @ Xc / T - var[:, None] * S
    np.fill_diagonal(theta, 0.0)
    rho = float(np.trace(phi_mat) + rbar * (np.outer(1.0 / sd, sd) * theta).sum())
    gamma = float(((S - F) ** 2).sum())
    kappa = (phi - rho) / gamma if gamma > 0 else 0.0
    delta = float(max(0.0, min(1.0, kappa / T)))
    return delta * F + (1.0 - delta) * S, delta


def ledoit_wolf_nonlinear(X: np.ndarray) -> np.ndarray:
    """R2: Ledoit and Wolf (2020) analytical nonlinear shrinkage.

    Follows the published kernel formula (Epanechnikov kernel, h = n^(-1/3)) after
    demeaning, with effective sample size n = T - 1. Handles both p <= n and p > n.
    """
    X = np.asarray(X, dtype=float)
    T, p = X.shape
    n = T - 1
    if n < 12:
        raise ValueError("analytical nonlinear shrinkage needs at least 13 observations")
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / n
    S = (S + S.T) / 2
    lam, u = np.linalg.eigh(S)  # ascending
    lam = np.clip(lam, 0.0, None)
    m = min(p, n)
    lam_nz = np.maximum(lam[p - m :], 1e-300)

    L = np.repeat(lam_nz[:, None], m, axis=1)  # L_ij = lambda_i
    h = n ** (-1.0 / 3.0)
    H = h * L.T  # H_ij = h * lambda_j
    x = (L - L.T) / H
    r5 = np.sqrt(5.0)
    f_tilde = (3.0 / 4.0 / r5) * np.mean(np.maximum(1.0 - x**2 / 5.0, 0.0) / H, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log(np.abs((r5 - x) / (r5 + x)))
    hf = (-3.0 / 10.0 / np.pi) * x + (3.0 / 4.0 / r5 / np.pi) * (1.0 - x**2 / 5.0) * log_term
    edge = np.isclose(np.abs(x), r5) | ~np.isfinite(hf)
    hf[edge] = (-3.0 / 10.0 / np.pi) * x[edge]
    hf_tilde = np.mean(hf / H, axis=1)

    if p <= n:
        c = p / n
        d = lam_nz / ((np.pi * c * lam_nz * f_tilde) ** 2 + (1.0 - c - np.pi * c * lam_nz * hf_tilde) ** 2)
    else:
        hf0 = (
            (1.0 / np.pi)
            * (3.0 / 10.0 / h**2 + 3.0 / 4.0 / r5 / h * (1.0 - 1.0 / 5.0 / h**2) * np.log((1.0 + r5 * h) / (1.0 - r5 * h)))
            * np.mean(1.0 / lam_nz)
        )
        d0 = 1.0 / (np.pi * (p - n) / n * hf0)
        d1 = lam_nz / (np.pi**2 * lam_nz**2 * (f_tilde**2 + hf_tilde**2))
        d = np.concatenate([np.full(p - n, d0), d1])
    Sigma = (u * d) @ u.T
    return (Sigma + Sigma.T) / 2
