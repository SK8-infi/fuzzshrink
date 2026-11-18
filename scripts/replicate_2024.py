"""Stage 8: rerun the base paper's own 2024 test for B24 (its pipeline) and P24 (ours).

Design: docs/protocol-2024-replication.md. Runs 2023 (for selection rule V) and 2024.
Usage:
    python scripts/replicate_2024.py [--seeds 5]
Writes results/runs/replication/seed{k}.csv (monthly net returns of every cell).
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_base import forecast, train_lstm  # noqa: E402
from fuzzshrink.allocation import ga_sharpe_paper, max_sharpe_exact  # noqa: E402
from fuzzshrink.clustering import comovement_features, fuzzy_cmeans  # noqa: E402
from fuzzshrink.data import rebalance_dates  # noqa: E402
from fuzzshrink.estimators import ledoit_wolf_nonlinear  # noqa: E402

BUY, SELL = 0.001, 0.002
CRITERIA = ("max_return", "max_sharpe", "min_risk")
C_RANGE = range(2, 9)


def representatives(U: np.ndarray, mean: np.ndarray, sd: np.ndarray, criterion: str) -> np.ndarray:
    labels = U.argmax(axis=1)
    score = {"max_return": mean, "max_sharpe": mean / sd, "min_risk": -sd}[criterion]
    reps = [np.where(labels == c)[0][np.argmax(score[labels == c])]
            for c in range(U.shape[1]) if np.any(labels == c)]
    return np.array(sorted(set(reps)))


def run_seed(seed: int) -> int:
    prices = pd.read_csv(ROOT / "data/raw/base_universe_prices.csv.gz", index_col=0, parse_dates=True)
    prices = prices.loc["2017-01-01":"2024-12-31"].dropna(axis=1, how="any")
    rets = prices.pct_change().iloc[1:]
    R = rets.to_numpy()
    dates = rebalance_dates(rets.index, "2023-01-01", "2024-12-31")
    rows = []
    for m, d in enumerate(dates):
        pos = rets.index.get_loc(d)
        end = rets.index.get_loc(dates[m + 1]) if m + 1 < len(dates) else len(rets)
        hold = R[pos:end]
        month_ret = np.prod(1 + hold, axis=0) - 1  # buy-and-hold return of each stock over the month
        hist = R[:pos]  # everything since 2017-01, as in the paper
        mean, sd = hist.mean(axis=0), hist.std(axis=0, ddof=1)
        net = train_lstm(R[max(0, pos - 504):pos], seed=seed * 100 + m)
        mu_daily = forecast(net, hist, R[max(0, pos - 252):pos].std(axis=0, ddof=1))
        mu_month = 21 * mu_daily

        # B24 features: (mean, sd) whitened by their covariance -> Euclidean FCM == Mahalanobis FCM
        F = np.column_stack([mean, sd])
        F = F - F.mean(axis=0)
        L = np.linalg.cholesky(np.linalg.inv(np.cov(F, rowvar=False)))
        Z_b = F @ L
        win = R[max(0, pos - 252):pos]
        Z_p = comovement_features(np.corrcoef(win, rowvar=False), len(win))

        for pipe, Z in (("B24", Z_b), ("P24", Z_p)):
            for C in C_RANGE:
                U = fuzzy_cmeans(Z, C, seed=seed * 1009 + C + 31 * m, n_init=3).U
                for crit in CRITERIA:
                    idx = representatives(U, mean, sd, crit)
                    if pipe == "B24":
                        Sigma = 21 * np.cov(hist[:, idx], rowvar=False)
                        w = (ga_sharpe_paper(mu_month[idx], np.atleast_2d(Sigma), seed=seed * 7919 + m * 37 + C)
                             if len(idx) > 1 else np.ones(1))
                    else:
                        mu = mu_month[idx]
                        mu = mu.mean() + 0.5 * (mu - mu.mean())
                        Sigma = 21 * ledoit_wolf_nonlinear(win[:, idx]) if len(idx) > 1 else np.ones((1, 1))
                        w = max_sharpe_exact(mu, Sigma, cap=1.0) if len(idx) > 1 else np.ones(1)
                    gross = float(w @ month_ret[idx])
                    rows.append({"seed": seed, "pipeline": pipe, "C": C, "criterion": crit,
                                 "month": d.strftime("%Y-%m"), "gross": gross,
                                 "net": (1 - BUY) * (1 + gross) * (1 - SELL) - 1,
                                 "holdings": " ".join(rets.columns[idx[w > 1e-6]])})
    out = ROOT / "results/runs/replication"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"seed{seed}.csv", index=False)
    return seed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    with Pool(args.seeds) as pool:
        for s in pool.imap_unordered(run_seed, range(args.seeds)):
            print(f"seed {s} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
