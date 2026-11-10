"""Run the pre-registered NASDAQ-100 backtests (docs/protocol.md, sections 3-4).

Usage:
    python scripts/run_backtest.py                 # every experiment, 10 seeds
    python scripts/run_backtest.py --only primary  # just the T = 252 GMV study
    python scripts/run_backtest.py --smoke         # two rebalances, to check the plumbing

Each (experiment, seed) result is pickled to results/runs/backtest/ (not committed);
scripts/report_backtest.py turns them into results/backtest/.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import pickle
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzshrink.backtest import BacktestSettings, run_backtest  # noqa: E402
from fuzzshrink.data import load_prices, prices_to_returns  # noqa: E402
from fuzzshrink.registry import CORE, EXTENSIONS, REFERENCES, Config, fuzz_variant  # noqa: E402

ALL = CORE + EXTENSIONS + REFERENCES
GAMMA_GRID = {f"G{g:+.1f}": Config(f"FuzzShrink, gamma = {g:+.1f}", fuzz_variant(g), True)
              for g in (-2.0, -0.5, 0.5, 2.0)}  # gamma = -1, 0, +1 are A6, A3 and A4

EXPERIMENTS = {
    # name: (settings changes, configurations, extra configurations)
    "primary": ({"T": 252, "n_permutations": 20}, ALL + list(GAMMA_GRID), GAMMA_GRID),
    "T126": ({"T": 126}, ALL, None),
    "T504": ({"T": 504}, ALL, None),
    "m1.5": ({"T": 252, "m": 1.5}, ["A1", "A3", "A4", "X1"], None),
    "m2.5": ({"T": 252, "m": 2.5}, ["A1", "A3", "A4", "X1"], None),
    "longonly": ({"T": 252, "long_only": True}, ALL, None),
}

_RETURNS = None


def _returns():
    global _RETURNS
    if _RETURNS is None:
        prices = load_prices("data/raw/prices.csv.gz", overrides="data/universe/listing_overrides.csv")
        _RETURNS = prices_to_returns(prices)
    return _RETURNS


def run_task(task):
    exp, seed, out_dir, smoke = task
    changes, names, extra = EXPERIMENTS[exp]
    settings = replace(BacktestSettings(), **changes)
    if smoke:
        settings = replace(settings, start="2017-01-01", end="2017-02-28", n_permutations=min(2, settings.n_permutations))
    t0 = time.time()
    result = run_backtest(_returns(), names, seed=seed, settings=settings, extra_configs=extra)
    with open(Path(out_dir) / f"{exp}_seed{seed}.pkl", "wb") as fh:
        pickle.dump(result, fh)
    return exp, seed, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = Path("results/runs/backtest-smoke" if args.smoke else "results/runs/backtest")
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(e, s, str(out), args.smoke) for e in args.only for s in range(args.seeds)]
    tasks.sort(key=lambda t: t[0] != "longonly")  # slowest first
    start = time.time()
    with Pool(args.workers) as pool:
        for i, (exp, seed, secs) in enumerate(pool.imap_unordered(run_task, tasks), 1):
            print(f"[{i}/{len(tasks)}] {exp} seed {seed}: {secs:.0f}s (elapsed {time.time() - start:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
