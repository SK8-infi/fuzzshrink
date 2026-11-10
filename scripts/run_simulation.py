"""Simulation study (docs/protocol.md, section 7).

Usage:
    python scripts/run_simulation.py --reps 100 --out results/simulation
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one BLAS thread per worker process

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzshrink.clustering import fit_memberships, membership_entropy  # noqa: E402
from fuzzshrink.moments import WindowStats  # noqa: E402
from fuzzshrink.portfolio import gmv_weights  # noqa: E402
from fuzzshrink.registry import CONFIGS, U_BASED, Context, portfolio_weights  # noqa: E402
from fuzzshrink.simulate import simulate_market  # noqa: E402

SCENARIOS = {
    "crisp": (0.0, 0.0),
    "bridges": (0.3, 0.0),
    "outliers": (0.0, 0.15),
    "mixed": (0.3, 0.15),
}
T_VALUES = (126, 252)
ESTIMATED = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "X1", "X2", "X3", "X4", "R1", "R2", "R3"]
ORACLE = ["A2", "A3", "A4", "A6", "X1", "X2"]


def gmv_loss(Sigma_hat, Sigma_true) -> float:
    """Excess out-of-sample GMV variance relative to the true-covariance GMV."""
    w = portfolio_weights(Sigma_hat, Sigma_true.shape[0])
    w_star = gmv_weights(Sigma_true)
    return float(w @ Sigma_true @ w / (w_star @ Sigma_true @ w_star) - 1.0)


def one_rep(task):
    scenario, T, rep = task
    pb, po = SCENARIOS[scenario]
    seed = int(np.random.SeedSequence([list(SCENARIOS).index(scenario), T, rep]).generate_state(1)[0])
    market = simulate_market(pi_bridge=pb, pi_outlier=po, T=T, seed=seed)
    stats = WindowStats.from_returns(market.X)
    part = fit_memberships(stats, features="comove", seed=rep)
    part_rr = fit_memberships(stats, features="riskret", seed=rep)
    rng = np.random.default_rng(seed + 1)

    rows, diag = [], []
    base = {"scenario": scenario, "T": T, "rep": rep}
    for oracle, names, U, U_rr in ((False, ESTIMATED, part.U, part_rr.U), (True, ORACLE, market.U_true, None)):
        ctx = Context(stats, U, U_rr, rng)
        for name in names:
            Sigma_hat, info = CONFIGS[name].fn(ctx)
            rows.append({**base, "config": name, "oracle": oracle, "loss": gmv_loss(Sigma_hat, market.Sigma)})
            if name == "X3" and not oracle:
                diag.append({**base, "config": "X3", "type": "all", "projected": bool(info["projected"]),
                             "min_eig": float(np.linalg.eigvalsh(Sigma_hat / np.outer(stats.sd, stats.sd)).min())})
            if name in ("A4", "X1") and not oracle:
                for typ in ("crisp", "bridge", "outlier"):
                    mask = market.types == typ
                    if mask.any():
                        diag.append({**base, "config": name, "type": typ,
                                     "mean_delta": float(info["delta"][mask].mean())})

    H = membership_entropy(part.U)
    for typ in ("crisp", "bridge", "outlier"):
        mask = market.types == typ
        if mask.any():
            diag.append({**base, "config": "U", "type": typ, "mean_entropy": float(H[mask].mean()),
                         "n_clusters": part.n_clusters})
    return rows, diag


def paired_test(df, a, b, scenario, T, oracle=False):
    """One-sided paired t-test of mean(loss_a - loss_b) < 0 across replications."""
    sub = df[(df.scenario == scenario) & (df["T"] == T) & (df.oracle == oracle)]
    wide = sub.pivot(index="rep", columns="config", values="loss")
    d = (wide[a] - wide[b]).to_numpy()
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else 0.0
    return {"diff": d.mean(), "se": se, "t": t, "p_less": float(sps.t.cdf(t, df=len(d) - 1))}


def evaluate_predictions(df):
    """Pre-registered predictions S1-S4, evaluated separately at each T."""
    out = []
    means = df[~df.oracle].groupby(["scenario", "T", "config"]).loss.mean()
    for T in T_VALUES:
        m = means.xs(("crisp", T), level=("scenario", "T"))
        trio = m[["A2", "A3", "A4"]]
        out.append(("S1", T, "A2, A3, A4 within 5% (crisp)",
                    f"spread {(trio.max() - trio.min()) / trio.min():.1%}",
                    bool((trio.max() - trio.min()) / trio.min() <= 0.05)))

        s2a = paired_test(df, "A3", "A2", "bridges", T)
        s2b = paired_test(df, "A4", "A3", "bridges", T)
        out.append(("S2", T, "A3 beats A2, and A4 does not beat A3 (bridges)",
                    f"A3-A2 {s2a['diff']:+.4f} (p={s2a['p_less']:.3f}); "
                    f"A4-A3 {s2b['diff']:+.4f} (p={s2b['p_less']:.3f})",
                    bool(s2a["p_less"] < 0.05 and not s2b["p_less"] < 0.05)))

        s3a = paired_test(df, "A4", "A3", "outliers", T)
        m3 = means.xs(("outliers", T), level=("scenario", "T"))
        out.append(("S3", T, "A4 beats A3, and X1 <= A4 (outliers)",
                    f"A4-A3 {s3a['diff']:+.4f} (p={s3a['p_less']:.3f}); "
                    f"X1 {m3['X1']:.4f} vs A4 {m3['A4']:.4f}",
                    bool(s3a["p_less"] < 0.05 and m3["X1"] <= m3["A4"])))

        m4 = means.xs(("mixed", T), level=("scenario", "T"))[U_BASED]
        out.append(("S4", T, "X1 lowest among U-based variants (mixed)",
                    f"best {m4.idxmin()} {m4.min():.4f}; X1 {m4['X1']:.4f}",
                    bool(m4.idxmin() == "X1")))
    return pd.DataFrame(out, columns=["prediction", "T", "statement", "evidence", "holds"])


def write_report(df, diag, preds, out: Path, reps: int, seconds: float):
    summary = (df.groupby(["oracle", "config", "scenario", "T"]).loss
                 .agg(["mean", "sem"]).reset_index())
    summary.to_csv(out / "summary.csv", index=False)
    preds.to_csv(out / "predictions.csv", index=False)

    table = summary[~summary.oracle].pivot_table(index="config", columns=["scenario", "T"], values="mean")
    table = table.reindex(columns=[(s, T) for s in SCENARIOS for T in T_VALUES])
    oracle = summary[summary.oracle].pivot_table(index="config", columns=["scenario", "T"], values="mean")
    oracle = oracle.reindex(columns=table.columns)
    oracle.index = [f"{c} (true U)" for c in oracle.index]

    ent = diag[diag.config == "U"].groupby(["scenario", "type"]).mean_entropy.mean().unstack()
    dlt = (diag[diag.config.isin(["A4", "X1"])]
           .groupby(["scenario", "config", "type"]).mean_delta.mean().unstack())
    n_c = diag[diag.config == "U"].groupby("scenario").n_clusters.mean()
    x3 = diag[diag.config == "X3"].groupby(["scenario", "T"]).agg(
        projection_rate=("projected", "mean"), median_min_eig=("min_eig", "median"))

    def md(frame, fmt="{:.4f}"):
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in frame.columns]
        lines = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for idx, row in frame.iterrows():
            idx = " ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
            lines.append(f"| {idx} | " + " | ".join("" if pd.isna(v) else fmt.format(v) for v in row) + " |")
        return "\n".join(lines)

    labels = "\n".join(f"- **{k}**: {CONFIGS[k].label}" for k in ESTIMATED)
    verdicts = "\n".join(f"| {r.prediction} | {r.T} | {r.statement} | {r.evidence} | "
                         f"{'holds' if r.holds else 'does not hold'} |" for r in preds.itertuples())
    text = f"""# Simulation study results

Generated by `scripts/run_simulation.py` ({reps} replications per cell, {seconds/60:.1f} minutes).
The design is in `docs/protocol.md` §7.

**Loss.** Excess out-of-sample variance of the GMV portfolio under the true covariance:
L = w'Σw / w*'Σw* − 1, where 0 means as good as knowing Σ. Lower is better.

## Pre-registered predictions

| Prediction | T | Statement | Evidence | Verdict |
|---|---|---|---|---|
{verdicts}

## Mean loss with estimated memberships

{md(table)}

## Mean loss with the true memberships (oracle U)

{md(oracle)}

## Diagnostics

Mean membership entropy by asset type:

{md(ent, "{:.3f}")}

Mean intensity δ by asset type, for entropy weights (A4) and fit weights (X1):

{md(dlt, "{:.3f}")}

Average number of clusters chosen by Xie–Beni (5 true groups):

{md(n_c.to_frame("C"), "{:.2f}")}

The draft's arithmetic rule (X3): how often the projection safeguard fired, and the
median smallest eigenvalue of the resulting correlation matrix:

{md(x3, "{:.3g}")}

## Configurations

{labels}
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, default=Path("results/simulation"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tasks = [(s, T, r) for s in SCENARIOS for T in T_VALUES for r in range(args.reps)]
    start = time.time()
    rows, diag = [], []
    with Pool(args.workers) as pool:
        for i, (r, d) in enumerate(pool.imap_unordered(one_rep, tasks, chunksize=4), 1):
            rows += r
            diag += d
            if i % 50 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} replications, {time.time() - start:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    diag = pd.DataFrame(diag)
    df.to_csv(args.out / "losses.csv.gz", index=False)
    preds = evaluate_predictions(df)
    write_report(df, diag, preds, args.out, args.reps, time.time() - start)
    print(preds.to_string(index=False))


if __name__ == "__main__":
    main()
