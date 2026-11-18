"""Stage 9, Q3 (S1): random-selection null for the paper's best-of-21 2024 result.

Design: docs/protocol-replication-paper.md, section 4. For each seed: 2024, C in 2..8, 100 random
C-stock paths (new random stocks every month), same LSTM forecasts, covariance and GA as FULL.
Usage:
    python scripts/random_null_2024.py [--seeds 5]
Writes results/runs/random_null/seed{k}.csv (2024 final wealth of every random path).
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from paperlib import C_RANGE, ROOT, forecast, load_returns, lstm_seed, net_of_costs, train_lstm
from fuzzshrink.allocation import ga_sharpe_paper
from fuzzshrink.data import rebalance_dates

N_DRAWS = 100


def run_seed(seed: int) -> int:
    rets = load_returns()
    R = rets.to_numpy()
    N = R.shape[1]
    dates = rebalance_dates(rets.index, "2024-01-01", "2024-12-31")
    wealth = {(C, k): 1000.0 for C in C_RANGE for k in range(N_DRAWS)}
    for m, d in enumerate(dates):
        pos = rets.index.get_loc(d)
        end = rets.index.get_loc(dates[m + 1]) if m + 1 < len(dates) else len(rets)
        month_ret = np.prod(1 + R[pos:end], axis=0) - 1
        hist = R[:pos]
        win = R[max(0, pos - 252):pos]
        net = train_lstm(R[max(0, pos - 504):pos], seed=lstm_seed(seed, d))
        mu = 21 * forecast(net, hist, win.std(axis=0, ddof=1))
        cov = 21 * np.cov(hist, rowvar=False)
        for C in C_RANGE:
            rng = np.random.default_rng([seed, 7, m, C])
            for k in range(N_DRAWS):
                idx = np.sort(rng.choice(N, size=C, replace=False))
                w = ga_sharpe_paper(mu[idx], cov[np.ix_(idx, idx)], seed=seed * 999_983 + m * 1_009 + C * 131 + k)
                wealth[(C, k)] *= 1 + float(net_of_costs(w @ month_ret[idx]))
    out = ROOT / "results/runs/random_null"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"seed": seed, "C": C, "draw": k, "w2024": v} for (C, k), v in wealth.items()]).to_csv(
        out / f"seed{seed}.csv", index=False)
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
