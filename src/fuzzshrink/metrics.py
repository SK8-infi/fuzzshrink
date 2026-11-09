"""Performance measures and the tests named in docs/protocol.md (sections 5-6)."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

__all__ = [
    "performance",
    "newey_west_lag",
    "hac_variance",
    "diebold_mariano",
    "stationary_bootstrap_indices",
    "variance_ratio_bootstrap",
    "sharpe_difference_test",
    "deflated_sharpe_ratio",
    "holm",
    "cer_difference_bps",
    "permutation_pvalue",
]

DAYS = 252


def performance(gross: np.ndarray, net: np.ndarray, turnover: np.ndarray | None = None) -> dict:
    """Annualised risk (gross) and performance (net) of a daily return series."""
    gross, net = np.asarray(gross, float), np.asarray(net, float)
    sd = net.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(net, 0.0) ** 2))
    wealth = np.cumprod(1.0 + net)
    mdd = float((wealth / np.maximum.accumulate(wealth) - 1.0).min())
    ann_mean = net.mean() * DAYS
    out = {
        "sd": gross.std(ddof=1) * np.sqrt(DAYS),
        "mean": ann_mean,
        "sharpe": net.mean() / sd * np.sqrt(DAYS) if sd > 0 else np.nan,
        "sortino": net.mean() / downside * np.sqrt(DAYS) if downside > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": ann_mean / abs(mdd) if mdd < 0 else np.nan,
    }
    if turnover is not None and len(turnover) > 1:
        out["turnover"] = float(np.mean(turnover[1:]))  # the first rebalance sets up the book
    return out


def newey_west_lag(n: int) -> int:
    """Newey-West (1994) rule of thumb: floor(4 (n / 100)^(2/9))."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_variance(V: np.ndarray, lag: int | None = None) -> np.ndarray:
    """Bartlett-kernel long-run covariance of the columns of V (already centred)."""
    V = np.atleast_2d(np.asarray(V, float).T).T
    n = V.shape[0]
    lag = newey_west_lag(n) if lag is None else lag
    S = V.T @ V / n
    for j in range(1, lag + 1):
        G = V[j:].T @ V[:-j] / n
        S += (1.0 - j / (lag + 1.0)) * (G + G.T)
    return S


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> dict:
    """DM test of E[loss_a - loss_b] < 0 with HAC variance and the HLN correction (h = 1)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    lrv = float(hac_variance((d - d.mean())[:, None])[0, 0])
    stat = d.mean() / np.sqrt(lrv / n) if lrv > 0 else 0.0
    stat *= np.sqrt((n - 1.0) / n)  # Harvey, Leybourne and Newbold (1997), h = 1
    return {"mean_diff": float(d.mean()), "stat": float(stat), "p_less": float(sps.t.cdf(stat, df=n - 1))}


def stationary_bootstrap_indices(n: int, n_boot: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary-bootstrap index matrix (n_boot x n), vectorised."""
    starts = rng.integers(n, size=(n_boot, n))
    new = rng.random((n_boot, n)) < 1.0 / block
    new[:, 0] = True
    pos = np.arange(n)[None, :]
    last_start = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    origin = np.take_along_axis(starts, last_start, axis=1)
    return (origin + pos - last_start) % n


def variance_ratio_bootstrap(ra: np.ndarray, rb: np.ndarray, n_boot: int = 2000, block: float = 21.0,
                             seed: int = 0) -> dict:
    """Paired stationary bootstrap for var(ra) / var(rb): estimate, 95% CI, P*(ratio >= 1)."""
    ra, rb = np.asarray(ra, float), np.asarray(rb, float)
    idx = stationary_bootstrap_indices(len(ra), n_boot, block, np.random.default_rng(seed))
    ratios = ra[idx].var(axis=1, ddof=1) / rb[idx].var(axis=1, ddof=1)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return {"ratio": float(ra.var(ddof=1) / rb.var(ddof=1)), "ci_low": float(lo), "ci_high": float(hi),
            "p_boot": float(np.mean(ratios >= 1.0))}


def sharpe_difference_test(ra: np.ndarray, rb: np.ndarray) -> dict:
    """Ledoit and Wolf (2008) HAC test of equal Sharpe ratios (two-sided)."""
    ra, rb = np.asarray(ra, float), np.asarray(rb, float)
    n = len(ra)
    mu_a, mu_b = ra.mean(), rb.mean()
    g_a, g_b = (ra * ra).mean(), (rb * rb).mean()
    va, vb = g_a - mu_a**2, g_b - mu_b**2
    diff = mu_a / np.sqrt(va) - mu_b / np.sqrt(vb)
    V = np.column_stack([ra - mu_a, rb - mu_b, ra * ra - g_a, rb * rb - g_b])
    psi = hac_variance(V)
    grad = np.array([g_a / va**1.5, -g_b / vb**1.5, -mu_a / (2 * va**1.5), mu_b / (2 * vb**1.5)])
    se = float(np.sqrt(grad @ psi @ grad / n))
    z = diff / se if se > 0 else 0.0
    return {"diff_annual": float(diff * np.sqrt(DAYS)), "z": float(z), "p_two_sided": float(2 * sps.norm.sf(abs(z)))}


def deflated_sharpe_ratio(returns: np.ndarray, sr_trials_daily: np.ndarray) -> dict:
    """Bailey and Lopez de Prado (2014): probability that the true Sharpe ratio exceeds
    the maximum expected under N independent trials with zero true Sharpe."""
    r = np.asarray(returns, float)
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=False))
    trials = np.asarray(sr_trials_daily, float)
    N = len(trials)
    euler = 0.5772156649
    sr0 = np.sqrt(trials.var(ddof=1)) * ((1 - euler) * sps.norm.ppf(1 - 1 / N)
                                          + euler * sps.norm.ppf(1 - 1 / (N * np.e)))
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    dsr = float(sps.norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom))
    return {"sharpe_annual": float(sr * np.sqrt(DAYS)), "sr0_annual": float(sr0 * np.sqrt(DAYS)),
            "n_trials": N, "dsr": dsr}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values for one family of tests."""
    names = sorted(pvalues, key=pvalues.get)
    m = len(names)
    adjusted, running = {}, 0.0
    for k, name in enumerate(names):
        running = max(running, min(1.0, (m - k) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def cer_difference_bps(ra: np.ndarray, rb: np.ndarray, risk_aversion: float) -> float:
    """Annual change in certainty-equivalent return of a over b, in basis points."""
    ra, rb = np.asarray(ra, float), np.asarray(rb, float)
    d = (ra.mean() - rb.mean()) - risk_aversion / 2 * (ra.var(ddof=1) - rb.var(ddof=1))
    return float(DAYS * d * 1e4)


def permutation_pvalue(observed: float, null: np.ndarray) -> float:
    """One-sided p-value that the observed value is this low by chance: (1 + #{null <= obs}) / (1 + n)."""
    null = np.asarray(null, float)
    return float((1 + np.sum(null <= observed)) / (1 + len(null)))
