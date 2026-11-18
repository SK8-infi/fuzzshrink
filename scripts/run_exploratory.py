"""Post-hoc analysis (NOT pre-registered): is the null result an artefact of the base intensity?

On real data, the pre-registered base-intensity rule shrinks harder than Ledoit and Wolf's
(2004) intensity for the same constant-correlation target (mean 0.29 against 0.19). In
each window this script rescales every FuzzShrink variant's base intensity by
kappa = delta_LW / delta_SS from the constant-correlation case, reruns the T = 252 GMV
backtest, and repeats the permutation test.

Usage:
    python scripts/run_exploratory.py
Writes results/backtest/EXPLORATORY.md.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzshrink.backtest import BacktestSettings, run_backtest  # noqa: E402
from fuzzshrink.data import load_prices, prices_to_returns  # noqa: E402
from fuzzshrink.estimators import constant_correlation, fuzzshrink, ledoit_wolf_cc  # noqa: E402
from fuzzshrink.metrics import diebold_mariano, performance, permutation_pvalue  # noqa: E402
from fuzzshrink.registry import Config  # noqa: E402

N_PERM = 20


def _kappa(stats):
    _, d_lw = ledoit_wolf_cc(stats.X)
    _, info = constant_correlation(stats)
    return d_lw / info["delta0"] if info["delta0"] > 0 else 1.0


def calibrated(gamma, target="two_level", permute=False, single=False):
    def fn(ctx):
        k = _kappa(ctx.stats)
        U = np.ones((ctx.stats.N, 1)) if single else ctx.U
        if permute:
            U = U[ctx.rng.permutation(U.shape[0])]
        return fuzzshrink(ctx.stats, U, gamma=gamma, target=target, delta0_scale=k)
    return fn


EXTRA = {
    "cA1": Config("Calibrated constant correlation (= R1)", calibrated(0.0, single=True)),
    "cA3": Config("Calibrated soft target, gamma = 0", calibrated(0.0), True),
    "cA4": Config("Calibrated soft target, gamma = +1", calibrated(1.0), True),
    "cX2": Config("Calibrated block-pair target, gamma = +1", calibrated(1.0, "block_pair"), True),
    **{f"cA5_{p}": Config("Calibrated A4, permuted", calibrated(1.0, permute=True), True) for p in range(N_PERM)},
}
NAMES = ["A1", "R1", "R2"] + list(EXTRA)


def task(seed):
    prices = load_prices("data/raw/prices.csv.gz", overrides="data/universe/listing_overrides.csv")
    settings = replace(BacktestSettings(), T=252, mechanism=())
    return seed, run_backtest(prices_to_returns(prices), NAMES, seed=seed, settings=settings, extra_configs=EXTRA)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        results = dict(pool.map(task, range(10)))

    sd = {s: {n: performance(r.gross[n], r.net[n])["sd"] for n in r.gross} for s, r in results.items()}
    sd = pd.DataFrame(sd).T  # seeds x configs
    rows = []
    for n, label in [("A1", "A1 (pre-registered intensity)"), ("R1", "R1 Ledoit–Wolf 2004"),
                     ("cA1", "cA1 calibrated constant correlation"), ("cA3", "cA3 calibrated soft, γ = 0"),
                     ("cA4", "cA4 calibrated soft, γ = +1"), ("cX2", "cX2 calibrated block-pair, γ = +1"),
                     ("R2", "R2 Ledoit–Wolf 2020 nonlinear")]:
        dm = [diebold_mariano(results[s].gross[n] ** 2, results[s].gross["R1"] ** 2)["p_less"] for s in results] \
            if n != "R1" else [np.nan]
        rows.append({"Configuration": label, "Ann. SD (%)": f"{100 * sd[n].mean():.2f}",
                     "Var. ratio vs R1": f"{(sd[n].mean() / sd['R1'].mean()) ** 2:.3f}",
                     "DM p (lower than R1)": "" if n == "R1" else f"{np.median(dm):.3f}"})
    perm = sd[[c for c in sd.columns if c.startswith("cA5_")]].to_numpy().ravel()
    p_perm = permutation_pvalue(sd["cA4"].mean(), perm)

    def md(df):
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
        return "\n".join(out)

    text = f"""# Post-hoc analysis: the base-intensity rule (not pre-registered)

**Status: exploratory.** This analysis was designed after seeing the pre-registered results.
It cannot change any verdict in `README.md`; it asks whether those verdicts are an artefact
of the pre-registered base-intensity rule.

**Why we ran it.** On the NASDAQ-100 the pre-registered rule (Schäfer–Strimmer type, which treats
the target as fixed) shrinks harder than Ledoit and Wolf's (2004) intensity for the same
constant-correlation target: mean δ₀ 0.29 against 0.19. With an identical target, R1 beats A1,
so the intensity choice matters.

**What we did.** In each window we computed κ = δ_LW / δ_SS for the constant-correlation case and
multiplied every FuzzShrink variant's base intensity by κ. With one cluster this reproduces R1
exactly, which serves as a check. Everything else is unchanged: T = 252, unconstrained GMV,
10 seeds, {N_PERM} permutation runs per seed.

{md(pd.DataFrame(rows))}

Permutation test with the calibrated intensity: calibrated A4 SD {100 * sd['cA4'].mean():.2f}% against
a permuted median of {100 * np.median(perm):.2f}% (5th percentile {100 * np.percentile(perm, 5):.2f}%,
n = {len(perm)}); p = {p_perm:.3f}.

Check: calibrated constant correlation (cA1) and R1 differ by at most
{100 * (sd['cA1'] - sd['R1']).abs().max():.4f} percentage points of annualised SD.
"""
    out = Path("results/backtest/EXPLORATORY.md")
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
