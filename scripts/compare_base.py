"""Head-to-head against a reconstruction of the base paper (docs/protocol-base-comparison.md).

Usage:
    python scripts/compare_base.py            # 5 seeds, writes results/runs/base/seed{k}.pkl
    python scripts/compare_base.py --smoke    # three rebalances, plumbing check only
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzshrink.allocation import max_sharpe_exact, nsga2_mean_variance  # noqa: E402
from fuzzshrink.clustering import comovement_features, fuzzy_cmeans, xie_beni  # noqa: E402
from fuzzshrink.data import rebalance_dates  # noqa: E402
from fuzzshrink.estimators import ledoit_wolf_nonlinear  # noqa: E402

N_SELECT, CAP, TC = 20, 0.3, 0.001
LOOKBACK, SEQ, HORIZON, TRAIN_DAYS, RETRAIN_EVERY = 252, 60, 21, 504, 3

# configuration -> (selection features, covariance, optimiser, forecast)
LADDER = {
    "B": ("riskret", "sample", "ga", "raw"),
    "L1": ("riskret", "nonlinear", "ga", "raw"),
    "L2": ("riskret", "nonlinear", "exact", "raw"),
    "L3": ("riskret", "nonlinear", "exact", "shrunk"),
    "P": ("comove", "nonlinear", "exact", "shrunk"),
}


# ------------------------------------------------------------------------------ LSTM


def train_lstm(R: np.ndarray, seed: int, epochs: int = 20):
    """Pooled LSTM: last SEQ standardised daily returns -> next HORIZON-day standardised return."""
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    sd = R.std(axis=0, ddof=1)
    Z = R / sd
    xs, ys = [], []
    for t in range(SEQ, len(Z) - HORIZON + 1, 5):
        xs.append(Z[t - SEQ : t].T)
        ys.append(Z[t : t + HORIZON].sum(axis=0) / np.sqrt(HORIZON))
    X = torch.tensor(np.concatenate(xs)[:, :, None], dtype=torch.float32)
    y = torch.tensor(np.concatenate(ys)[:, None], dtype=torch.float32)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(1, 16, batch_first=True)
            self.head = torch.nn.Linear(16, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1])

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=gen).split(256):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(X[idx]), y[idx])
            loss.backward()
            opt.step()
    return net


def forecast(net, R_recent: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Expected mean daily return over the next HORIZON days, per stock."""
    import torch

    x = torch.tensor((R_recent[-SEQ:] / sd).T[:, :, None], dtype=torch.float32)
    with torch.no_grad():
        z = net(x).numpy().ravel()
    return z * np.sqrt(HORIZON) * sd / HORIZON


# ------------------------------------------------------------------------- selection


def select(R: np.ndarray, features: str, seed: int) -> np.ndarray:
    """Outlier-aware fuzzy clustering, then top trailing-Sharpe stocks per cluster."""
    mean, sd = R.mean(axis=0), R.std(axis=0, ddof=1)
    C_mat = np.corrcoef(R, rowvar=False)
    if features == "riskret":
        avg_corr = (C_mat.sum(axis=1) - 1) / (len(C_mat) - 1)
        Z = np.column_stack([mean, sd, avg_corr])
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    else:
        Z = comovement_features(C_mat, len(R))
    best, best_xb = None, np.inf
    for C in range(2, 9):
        part = fuzzy_cmeans(Z, C, seed=seed * 1009 + C, n_init=3)
        xb = xie_beni(part)
        if xb < best_xb:
            best, best_xb = part, xb
    U = best.U
    keep = U.max(axis=1) >= 0.5  # outliers belong to no cluster clearly
    labels = U.argmax(axis=1)
    sharpe = mean / sd
    chosen = []
    clusters = [c for c in range(U.shape[1]) if np.any(keep & (labels == c))]
    sizes = np.array([np.sum(keep & (labels == c)) for c in clusters])
    quota = np.maximum(1, np.round(N_SELECT * sizes / sizes.sum())).astype(int)
    for c, q in zip(clusters, quota):
        members = np.where(keep & (labels == c))[0]
        chosen.extend(members[np.argsort(-sharpe[members])][:q])
    chosen = np.array(sorted(set(chosen)))
    if len(chosen) > N_SELECT:
        chosen = chosen[np.argsort(-sharpe[chosen])][:N_SELECT]
    return np.sort(chosen)


# -------------------------------------------------------------------------- backtest


def allocate(cfg, R_win, mu_all, idx, seed):
    _, cov, opt, fc = cfg
    mu = mu_all[idx]
    if fc == "shrunk":
        mu = mu.mean() + 0.5 * (mu - mu.mean())
    X = R_win[:, idx]
    Sigma = np.cov(X, rowvar=False) if cov == "sample" else ledoit_wolf_nonlinear(X)
    if opt == "ga":
        return nsga2_mean_variance(mu, Sigma, cap=CAP, seed=seed)
    return max_sharpe_exact(mu, Sigma, cap=CAP)


def run_seed(task):
    seed, smoke = task
    prices = pd.read_csv("data/raw/base_universe_prices.csv.gz", index_col=0, parse_dates=True)
    prices = prices.loc["2017-01-01":"2024-12-31"].dropna(axis=1, how="any")
    rets = prices.pct_change().iloc[1:]
    tickers = rets.columns
    R_all = rets.to_numpy()
    dates = rebalance_dates(rets.index, "2018-01-01", "2024-12-31")
    if smoke:
        dates = dates[:3]
    idx_of = {d: rets.index.get_loc(d) for d in dates}
    out = {k: [] for k in LADDER}
    weights = {k: [] for k in LADDER}
    turnover = {k: [] for k in LADDER}
    prev = {k: None for k in LADDER}
    net = None
    for m, d in enumerate(dates):
        pos = idx_of[d]
        end = idx_of[dates[m + 1]] if m + 1 < len(dates) else len(rets)
        if m % RETRAIN_EVERY == 0 or net is None:
            net = train_lstm(R_all[max(0, pos - TRAIN_DAYS) : pos], seed=seed * 100 + m)
        R_win = R_all[max(0, pos - LOOKBACK) : pos]
        mu_all = forecast(net, R_all[:pos], R_win.std(axis=0, ddof=1))
        sel = {f: select(R_win, f, seed) for f in ("riskret", "comove")}
        hold = R_all[pos:end]
        for k, cfg in LADDER.items():
            idx = sel[cfg[0]]
            w = np.zeros(len(tickers))
            w[idx] = allocate(cfg, R_win, mu_all, idx, seed * 7919 + m)
            if prev[k] is None:
                to = 0.0
            else:
                drift = prev[k][0] * np.prod(1 + prev[k][1], axis=0)
                drift = drift / drift.sum()
                to = float(np.abs(w - drift).sum())
            r = hold @ w
            r[0] -= TC * to
            out[k].append(pd.Series(r, index=rets.index[pos:end]))
            weights[k].append(w)
            turnover[k].append(to)
            prev[k] = (w, hold)
    res = {
        "net": pd.DataFrame({k: pd.concat(v) for k, v in out.items()}),
        "turnover": pd.DataFrame(turnover, index=dates),
        "weights": {k: np.array(v) for k, v in weights.items()},
        "tickers": list(tickers),
    }
    path = Path("results/runs/base-smoke" if smoke else "results/runs/base")
    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"seed{seed}.pkl", "wb") as fh:
        pickle.dump(res, fh)
    return seed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    with Pool(min(args.seeds, (os.cpu_count() or 2) - 1)) as pool:
        for s in pool.imap_unordered(run_seed, [(s, args.smoke) for s in range(args.seeds)]):
            print(f"seed {s} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
