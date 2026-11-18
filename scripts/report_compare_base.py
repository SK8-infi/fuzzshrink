"""Report for the base-paper comparison (writes results/base_comparison/README.md).

Applies the decision rules of docs/protocol-base-comparison.md, section 5.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzshrink.metrics import diebold_mariano, sharpe_difference_test  # noqa: E402

LABELS = {
    "B": "B: base-paper reconstruction (risk-return fuzzy selection, LSTM, NSGA-II, sample covariance)",
    "L1": "L1: B + Ledoit–Wolf 2020 covariance",
    "L2": "L2: L1 + exact max-Sharpe optimiser",
    "L3": "L3: L2 + shrunk forecasts",
    "P": "P: ours (L3 + co-movement fuzzy selection)",
    "EW": "Equal weight, all stocks (monthly rebalanced)",
    "QQQ": "QQQ buy and hold",
}


def stats(r: pd.Series, turnover=None) -> dict:
    r = r.dropna()
    wealth = (1 + r).cumprod()
    mdd = float((wealth / wealth.cummax() - 1).min())
    ann = r.mean() * 252
    vol = r.std(ddof=1) * np.sqrt(252)
    down = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(252)
    return {"ann_return": ann, "vol": vol, "sharpe": ann / vol, "sortino": ann / down, "mdd": mdd,
            "calmar": ann / abs(mdd), "cumulative": float(wealth.iloc[-1] - 1),
            "turnover": float(np.mean(turnover[1:])) if turnover is not None else np.nan}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    runs = {}
    for p in sorted(Path("results/runs/base").glob("seed*.pkl")):
        with open(p, "rb") as fh:
            runs[int(p.stem[4:])] = pickle.load(fh)
    seeds = sorted(runs)
    first = runs[seeds[0]]
    idx = first["net"].index

    rows = []
    for s in seeds:
        for k in first["net"].columns:
            rows.append({"seed": s, "config": k, **stats(runs[s]["net"][k], runs[s]["turnover"][k].to_numpy())})
    per_seed = pd.DataFrame(rows)

    prices = pd.read_csv("data/raw/base_universe_prices.csv.gz", index_col=0, parse_dates=True)
    prices = prices.loc["2017-01-01":"2024-12-31"].dropna(axis=1, how="any")
    rets = prices.pct_change().iloc[1:].loc[idx]
    reb = runs[seeds[0]]["turnover"].index
    ew = rets.mean(axis=1).copy()
    ew_to = []
    w_prev = None
    for k, d in enumerate(reb):
        end = reb[k + 1] if k + 1 < len(reb) else None
        hold = rets.loc[d:end].iloc[:-1] if end is not None else rets.loc[d:]
        w = np.full(rets.shape[1], 1 / rets.shape[1])
        to = 0.0 if w_prev is None else float(np.abs(w - w_prev).sum())
        ew.loc[d] -= 0.001 * to
        ew_to.append(to)
        g = w * np.prod(1 + hold.to_numpy(), axis=0)
        w_prev = g / g.sum()
    qqq = pd.read_csv("data/raw/qqq.csv.gz", index_col=0, parse_dates=True).iloc[:, 0].pct_change().loc[idx]
    bench = {"EW": stats(ew, np.array(ew_to)), "QQQ": stats(qqq)}

    agg = per_seed.groupby("config").agg(["mean", "std"])
    order = ["B", "L1", "L2", "L3", "P"]
    lines = []
    for k in order + ["EW", "QQQ"]:
        if k in bench:
            v = {m: (bench[k][m], 0.0) for m in bench[k]}
        else:
            v = {m: (agg.loc[k, (m, "mean")], agg.loc[k, (m, "std")]) for m in ["ann_return", "vol", "sharpe",
                                                                                  "sortino", "mdd", "calmar",
                                                                                  "cumulative", "turnover"]}
        sh = f"{v['sharpe'][0]:.3f}" + (f" ± {v['sharpe'][1]:.3f}" if v["sharpe"][1] > 0 else "")
        lines.append(f"| {LABELS[k]} | {100 * v['ann_return'][0]:.1f}% | {100 * v['vol'][0]:.1f}% | {sh} | "
                     f"{v['sortino'][0]:.2f} | {100 * v['mdd'][0]:.1f}% | {v['calmar'][0]:.2f} | "
                     f"{100 * v['cumulative'][0]:.0f}% | "
                     + ("" if np.isnan(v["turnover"][0]) else f"{v['turnover'][0]:.2f}") + " |")

    # hypotheses
    sh = per_seed.pivot(index="seed", columns="config", values="sharpe")
    wins = int((sh["P"] > sh["B"]).sum())
    sr_p = [sharpe_difference_test(runs[s]["net"]["P"], runs[s]["net"]["B"])["p_two_sided"] for s in seeds]
    dm_p = [diebold_mariano(runs[s]["net"]["P"] ** 2, runs[s]["net"]["B"] ** 2)["p_less"] for s in seeds]
    mdd = per_seed.pivot(index="seed", columns="config", values="mdd")
    b1 = bool(sh["P"].mean() > sh["B"].mean() and wins >= 4)
    b2 = bool(np.median(dm_p) < 0.05)
    b3 = bool(mdd["P"].mean() > mdd["B"].mean())  # drawdowns are negative: larger = smaller loss
    hyp = [
        ("B1 (primary): P has a higher Sharpe ratio than B",
         f"P {sh['P'].mean():.3f} vs B {sh['B'].mean():.3f}; higher in {wins}/{len(seeds)} seeds; "
         f"HAC test median p = {np.median(sr_p):.3f}", b1),
        ("B2: P has lower volatility than B",
         f"DM median one-sided p = {np.median(dm_p):.3f}", b2),
        ("B3: P has a smaller maximum drawdown than B",
         f"P {100 * mdd['P'].mean():.1f}% vs B {100 * mdd['B'].mean():.1f}%", b3),
    ]
    hyp_md = "\n".join(f"| {h} | {e} | {'supported' if ok else 'not supported'} |" for h, e, ok in hyp)

    text = f"""# Head-to-head against the base paper

Generated by `scripts/report_compare_base.py`. Design fixed in advance: `docs/protocol-base-comparison.md`.
85 of the base paper's 88 stocks, monthly rebalancing, long-only with a 30% cap, 10 bps costs,
out of sample {idx.min().date()} to {idx.max().date()}, {len(seeds)} seeds. ± is the standard deviation over seeds.

## Hypotheses

| Hypothesis | Evidence | Verdict |
|---|---|---|
{hyp_md}

## Performance (after costs)

| Pipeline | Ann. return | Volatility | Sharpe | Sortino | Max drawdown | Calmar | Cumulative | Turnover |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(lines)}

Sharpe ratio by seed:

| Seed | {' | '.join(order)} |
|---|{'---|' * len(order)}
{chr(10).join(f"| {s} | " + " | ".join(f"{sh.loc[s, k]:.3f}" for k in order) + " |" for s in seeds)}
"""
    out = Path("results/base_comparison")
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "by_seed.csv", index=False)
    (out / "README.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
