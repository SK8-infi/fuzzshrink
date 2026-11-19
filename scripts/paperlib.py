"""Shared pieces for the stage 9 replication study (docs/protocol-replication-paper.md)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_base import forecast, train_lstm  # noqa: E402,F401

BUY, SELL = 0.001, 0.002
CRITERIA = ("max_return", "max_sharpe", "min_risk")
C_RANGE = range(2, 9)


def load_returns() -> pd.DataFrame:
    prices = pd.read_csv(ROOT / "data/raw/base_universe_prices.csv.gz", index_col=0, parse_dates=True)
    prices = prices.loc["2017-01-01":"2024-12-31"].dropna(axis=1, how="any")
    return prices.pct_change().iloc[1:]


def net_of_costs(gross):
    """The paper's cost rule: buy 0.1% and sell 0.2% of the whole portfolio every month."""
    return (1 - BUY) * (1 + np.asarray(gross)) * (1 - SELL) - 1


def lstm_seed(seed: int, date: pd.Timestamp) -> int:
    """Seed for the LSTM trained at a rebalance date, shared by every stage 9 script."""
    return seed * 10_000 + (date.year - 2000) * 12 + date.month


def whitened_mean_sd(hist: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, sd) per stock, whitened by their 2x2 covariance: Euclidean FCM on it = Mahalanobis FCM."""
    mean, sd = hist.mean(axis=0), hist.std(axis=0, ddof=1)
    F = np.column_stack([mean, sd])
    F = F - F.mean(axis=0)
    L = np.linalg.cholesky(np.linalg.inv(np.cov(F, rowvar=False)))
    return F @ L, mean, sd


def score(mean: np.ndarray, sd: np.ndarray, criterion: str) -> np.ndarray:
    return {"max_return": mean, "max_sharpe": mean / sd, "min_risk": -sd}[criterion]


def representatives(U: np.ndarray, s: np.ndarray, per_cluster: int = 1) -> np.ndarray:
    """The top `per_cluster` stocks of each cluster (hard labels) by score s."""
    labels = U.argmax(axis=1)
    out = []
    for c in range(U.shape[1]):
        members = np.where(labels == c)[0]
        if len(members):
            out.extend(members[np.argsort(-s[members])][:per_cluster])
    return np.array(sorted(set(out)))


def hac_t_greater(diff: np.ndarray) -> tuple[float, float]:
    """One-sided HAC t-test of mean(diff) > 0: (t statistic, p-value)."""
    from scipy import stats as sps

    from fuzzshrink.metrics import hac_variance

    d = np.asarray(diff, float)
    n = len(d)
    lrv = float(hac_variance((d - d.mean())[:, None])[0, 0])
    t = d.mean() / np.sqrt(lrv / n) if lrv > 0 else 0.0
    return float(t), float(sps.t.sf(t, df=n - 1))
