"""Stage 9, Q2 (F2) and Q4-Q5: the paper's pipeline, its alternative models and IMP, 2018-2024.

Design: docs/protocol-replication-paper.md, sections 3, 5 and 6.
Usage:
    python scripts/walkforward_paper.py [--seeds 5]
Writes results/runs/walkforward/cells_seed{k}.csv and ic_seed{k}.csv.
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
from scipy import stats as sps

from paperlib import (C_RANGE, CRITERIA, ROOT, forecast, load_returns, lstm_seed, net_of_costs,
                      representatives, score, train_lstm, whitened_mean_sd)
from fuzzshrink.allocation import ga_sharpe_paper, max_sharpe_exact
from fuzzshrink.clustering import fuzzy_cmeans
from fuzzshrink.data import rebalance_dates
from fuzzshrink.estimators import ledoit_wolf_nonlinear


def ga_or_single(mu, Sigma, seed):
    if len(mu) == 1:
        return np.ones(1)
    return ga_sharpe_paper(mu, np.atleast_2d(Sigma), seed=seed)


def run_seed(seed: int) -> int:
    rets = load_returns()
    R = rets.to_numpy()
    N = R.shape[1]
    dates = rebalance_dates(rets.index, "2018-01-01", "2024-12-31")
    rows, ics = [], []
    for m, d in enumerate(dates):
        pos = rets.index.get_loc(d)
        end = rets.index.get_loc(dates[m + 1]) if m + 1 < len(dates) else len(rets)
        month_ret = np.prod(1 + R[pos:end], axis=0) - 1
        hist = R[:pos]
        win = R[max(0, pos - 252):pos]
        net = train_lstm(R[max(0, pos - 504):pos], seed=lstm_seed(seed, d))
        mu_month = 21 * forecast(net, hist, win.std(axis=0, ddof=1))
        ics.append({"seed": seed, "month": d.strftime("%Y-%m"),
                    "ic": float(sps.spearmanr(mu_month, month_ret).statistic)})
        Z, mean, sd = whitened_mean_sd(hist)
        hist_mu = 21 * mean
        cov_hist = 21 * np.cov(hist, rowvar=False)
        gseed = seed * 1_000_003 + m * 101

        def record(pipe, C, crit, idx, w):
            g = float(w @ month_ret[idx])
            rows.append({"seed": seed, "pipeline": pipe, "C": C, "criterion": crit,
                         "month": d.strftime("%Y-%m"), "gross": g, "net": float(net_of_costs(g)),
                         "n_held": int((w > 1e-6).sum())})

        for C in C_RANGE:
            U = fuzzy_cmeans(Z, C, seed=seed * 1009 + C + 31 * m, n_init=3).U
            for k, crit in enumerate(CRITERIA):
                s = score(mean, sd, crit)
                idx = representatives(U, s, 1)
                cov = cov_hist[np.ix_(idx, idx)]
                record("FULL", C, crit, idx, ga_or_single(mu_month[idx], cov, gseed + C * 7 + k))
                record("ALT2", C, crit, idx, ga_or_single(hist_mu[idx], cov, gseed + C * 7 + k + 3))
                record("ALT3", C, crit, idx, np.full(len(idx), 1.0 / len(idx)))
                idx3 = representatives(U, s, 3)
                mu = mu_month[idx3]
                mu = mu.mean() + 0.5 * (mu - mu.mean())
                cap = max(0.3, 1.0 / len(idx3))
                record("IMP", C, crit, idx3, max_sharpe_exact(mu, 21 * ledoit_wolf_nonlinear(win[:, idx3]), cap=cap))
            rng = np.random.default_rng([seed, m, C])
            for r in range(3):
                idx = np.sort(rng.choice(N, size=C, replace=False))
                record("ALT1", C, f"random{r + 1}", idx,
                       ga_or_single(mu_month[idx], cov_hist[np.ix_(idx, idx)], gseed + C * 7 + r + 50))
    out = ROOT / "results/runs/walkforward"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"cells_seed{seed}.csv", index=False)
    pd.DataFrame(ics).to_csv(out / f"ic_seed{seed}.csv", index=False)
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
