"""Stage 9 report: Q2 (F2), Q3 (S1, S2), Q4 (D1-D4), Q5 (I1-I2).

Applies docs/protocol-replication-paper.md. Writes results/replication_paper/README.md, tables and figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from paperlib import ROOT, hac_t_greater, load_returns, net_of_costs
from fuzzshrink.data import rebalance_dates
from fuzzshrink.metrics import holm, probability_of_backtest_overfitting

OUT = ROOT / "results/replication_paper"
PIPES = ["FULL", "ALT1", "ALT2", "ALT3", "IMP"]
LABEL = {"FULL": "The paper's pipeline (FULL)", "ALT1": "Alt. model 1: random selection",
         "ALT2": "Alt. model 2: historical means", "ALT3": "Alt. model 3: equal weights",
         "IMP": "Our improvement (IMP)", "QQQ": "QQQ", "EW": "Equal weight, 85 stocks"}


def wealth(r):
    return 1000 * float(np.prod(1 + np.asarray(r)))


def sharpe(r):
    r = np.asarray(r)
    return float(r.mean() / r.std(ddof=1) * np.sqrt(12))


def mdd(r):
    w = np.cumprod(1 + np.asarray(r))
    return float((w / np.maximum.accumulate(w) - 1).min())


def benchmarks(months):
    rets = load_returns()
    qqq = pd.read_csv(ROOT / "data/raw/qqq.csv.gz", index_col=0, parse_dates=True).iloc[:, 0].pct_change()
    dates = rebalance_dates(rets.index, "2018-01-01", "2024-12-31")
    out = {"QQQ": [], "EW": []}
    for m, d in enumerate(dates):
        end = dates[m + 1] if m + 1 < len(dates) else None
        last = rets.index[rets.index.get_loc(end) - 1] if end is not None else rets.index[-1]
        out["QQQ"].append(float(np.prod(1 + qqq.loc[d:last]) - 1))
        out["EW"].append(float(np.mean(np.prod(1 + rets.loc[d:last].to_numpy(), axis=0) - 1)))
    idx = [d.strftime("%Y-%m") for d in dates]
    return pd.DataFrame({k: net_of_costs(v) for k, v in out.items()}, index=idx).loc[months]


def rule_v_series(cells: pd.DataFrame) -> pd.Series:
    """Annual honest selection: in year Y use the cell with the best wealth over year Y-1."""
    cells = cells.assign(year=cells.month.str[:4].astype(int))
    pieces = []
    for y in range(2019, 2025):
        prev = cells[cells.year == y - 1].groupby(["C", "criterion"]).net.apply(lambda r: np.prod(1 + r))
        C, crit = prev.idxmax()
        cur = cells[(cells.year == y) & (cells.C == C) & (cells.criterion == crit)].sort_values("month")
        pieces.append(cur.set_index("month").net)
    return pd.concat(pieces)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.concat(pd.read_csv(p) for p in sorted((ROOT / "results/runs/walkforward").glob("cells_seed*.csv")))
    ics = pd.concat(pd.read_csv(p) for p in sorted((ROOT / "results/runs/walkforward").glob("ic_seed*.csv")))
    null = pd.concat(pd.read_csv(p) for p in sorted((ROOT / "results/runs/random_null").glob("seed*.csv")))
    seeds = sorted(cells.seed.unique())

    # ---- Q4 and Q5: honest annual selection, 2019-2024
    series = {(s, p): rule_v_series(cells[(cells.seed == s) & (cells.pipeline == p)]) for s in seeds for p in PIPES}
    months = list(series[(seeds[0], "FULL")].index)
    bench = benchmarks(months)
    perf_rows = []
    for p in PIPES:
        vals = [(wealth(series[(s, p)]), sharpe(series[(s, p)]), mdd(series[(s, p)])) for s in seeds]
        w, sh, dd = np.array(vals).T
        perf_rows.append({"Pipeline": LABEL[p], "Final wealth ($1,000 in 2019)": f"${w.mean():,.0f} ± {w.std(ddof=1):,.0f}",
                          "Sharpe": f"{sh.mean():.2f} ± {sh.std(ddof=1):.2f}", "Max drawdown": f"{100 * dd.mean():.1f}%"})
    for b in ("QQQ", "EW"):
        r = bench[b].to_numpy()
        perf_rows.append({"Pipeline": LABEL[b], "Final wealth ($1,000 in 2019)": f"${wealth(r):,.0f}",
                          "Sharpe": f"{sharpe(r):.2f}", "Max drawdown": f"{100 * mdd(r):.1f}%"})
    perf = pd.DataFrame(perf_rows)

    tests = {"D1": ("FULL", "ALT1", "clustering adds value"), "D2": ("FULL", "ALT2", "the LSTM adds value"),
             "D3": ("FULL", "ALT3", "the optimiser adds value"), "D4": ("FULL", "QQQ", "beats the Nasdaq-100"),
             "I1": ("IMP", "FULL", "our improvement beats the paper's pipeline"),
             "I2": ("IMP", "QQQ", "our improvement beats the Nasdaq-100")}
    res = {}
    for h, (a, b, what) in tests.items():
        ps, ts, pos, diffs = [], [], 0, []
        for s in seeds:
            ra = series[(s, a)].to_numpy()
            rb = bench[b].to_numpy() if b in bench else series[(s, b)].to_numpy()
            t, p_ = hac_t_greater(ra - rb)
            ps.append(p_); ts.append(t); diffs.append(12 * np.mean(ra - rb)); pos += np.mean(ra - rb) > 0
        res[h] = {"a": a, "b": b, "what": what, "p": float(np.median(ps)), "t": float(np.median(ts)),
                  "pos": int(pos), "diff": float(np.mean(diffs))}
    adj = holm({h: v["p"] for h, v in res.items()})
    hyp_rows = []
    for h, v in res.items():
        ok = v["p"] < 0.05 and v["pos"] >= 3
        hyp_rows.append(f"| {h} | {LABEL.get(v['a'], v['a'])} > {LABEL.get(v['b'], v['b'])} ({v['what']}) | "
                        f"{100 * v['diff']:+.1f} pp/yr | t = {v['t']:+.2f} | {v['p']:.3f} | {adj[h]:.3f} | "
                        f"{v['pos']}/{len(seeds)} | {'supported' if ok else 'not supported'} |")

    # ---- Q2 F2: information coefficient
    ic_rows, ic_ok = [], 0
    for s in seeds:
        x = ics[ics.seed == s].ic.to_numpy()
        t, p_ = hac_t_greater(x)
        ic_ok += p_ < 0.05
        ic_rows.append((s, x.mean(), t, p_))
    f2 = ic_ok >= 3

    # ---- Q3 S1: random-selection null for the best-of-21 2024 result
    s1_rows, s1_p = [], []
    for s in seeds:
        c24 = cells[(cells.seed == s) & (cells.pipeline == "FULL") & (cells.month.str.startswith("2024"))]
        observed = c24.groupby(["C", "criterion"]).net.apply(wealth).max()
        rng = np.random.default_rng(s)
        draws = {C: null[(null.seed == s) & (null.C == C)].w2024.to_numpy() for C in range(2, 9)}
        sims = np.array([max(rng.choice(draws[C], 3, replace=False).max() for C in range(2, 9))
                         for _ in range(10_000)])
        p_ = float(np.mean(sims >= observed))
        s1_p.append(p_)
        s1_rows.append((s, observed, np.median(sims), np.percentile(sims, 95), p_))
    s1 = float(np.median(s1_p)) < 0.05

    # ---- Q3 S2: probability of backtest overfitting over the 21 FULL settings, 2018-2024
    pbo_rows = []
    for s in seeds:
        M = (cells[(cells.seed == s) & (cells.pipeline == "FULL")]
             .pivot_table(index="month", columns=["C", "criterion"], values="net").to_numpy())
        pbo_rows.append((s, probability_of_backtest_overfitting(M, n_blocks=12)))

    # ---- figure: wealth paths (seed mean)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"FULL": "#2a78d6", "IMP": "#eb6834", "QQQ": "#1baf7a", "ALT1": "#b4b2a9"}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for key, lab in (("FULL", LABEL["FULL"]), ("ALT1", LABEL["ALT1"]), ("IMP", LABEL["IMP"])):
        avg = pd.concat([series[(s, key)] for s in seeds], axis=1).mean(axis=1)
        ax.plot(pd.to_datetime(avg.index), 1000 * np.cumprod(1 + avg), color=colors[key], label=lab, linewidth=1.8)
    ax.plot(pd.to_datetime(bench.index), 1000 * np.cumprod(1 + bench["QQQ"]), color=colors["QQQ"], label="QQQ",
            linewidth=1.8)
    ax.set_ylabel("Wealth ($, after the paper's costs)")
    ax.set_title("2019–2024, setting re-chosen yearly on the prior year (5-seed mean)", loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(color="#e1e0d9", linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_walkforward_wealth.png", dpi=160)
    ax.set_title("", loc="left")  # the paper's caption carries the title
    fig.tight_layout()
    fig.savefig(ROOT / "docs/paper2/latex/figures/fig_walkforward_wealth.png", dpi=300)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    s0 = seeds[0]
    rng = np.random.default_rng(s0)
    draws = {C: null[(null.seed == s0) & (null.C == C)].w2024.to_numpy() for C in range(2, 9)}
    sims = [max(rng.choice(draws[C], 3, replace=False).max() for C in range(2, 9)) for _ in range(10_000)]
    ax.hist(sims, bins=60, color="#b4b2a9", edgecolor="#fcfcfb")
    ax.axvline(s1_rows[0][1], color="#2a78d6", linewidth=2, label=f"Paper's pipeline, best of 21: ${s1_rows[0][1]:,.0f}")
    ax.axvline(5343.96, color="#eb6834", linewidth=2, label="Paper's published $5,344")
    ax.set_xlabel("2024 final wealth of the best of 21 random-selection settings ($)")
    ax.set_title("Best-of-21 under random stock selection (seed 0)", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="center right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_random_null.png", dpi=160)
    ax.set_title("", loc="left")  # the paper's caption carries the title
    fig.tight_layout()
    fig.savefig(ROOT / "docs/paper2/latex/figures/fig_random_null.png", dpi=300)

    def md(df):
        cols = list(df.columns)
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                         + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])

    text = f"""# Stage 9 results: the replication study (paper 2)

Generated by `scripts/report_replication_paper.py`. Design fixed in advance: `docs/protocol-replication-paper.md`.
85 of the paper's 88 stocks, the paper's costs, {len(seeds)} seeds. Q1 and F1 are in [audit.md](audit.md).

## Q4–Q5: the paper's decomposition over six years, with honest annual selection (2019–2024, 72 months)

{md(perf)}

± is the standard deviation across seeds.

| ID | Hypothesis | Mean gap | Median HAC t | Median p | Holm p | Seeds with a positive gap | Verdict |
|---|---|---|---|---|---|---|---|
{chr(10).join(hyp_rows)}

![Wealth](fig_walkforward_wealth.png)

## Q2, F2: does the LSTM forecast next-month returns? (2018–2024, 84 months)

| Seed | Mean information coefficient | HAC t | One-sided p |
|---|---|---|---|
{chr(10).join(f"| {s} | {m:+.4f} | {t:+.2f} | {p:.3f} |" for s, m, t, p in ic_rows)}

Verdict F2: {'supported' if f2 else 'not supported'} ({ic_ok}/{len(seeds)} seeds with p < 0.05).

## Q3, S1: is the best-of-21 2024 result better than best-of-21 random selection?

| Seed | Paper's pipeline, best of 21 | Random selection, best of 21: median | 95th percentile | p = P(random ≥ observed) |
|---|---|---|---|---|
{chr(10).join(f"| {s} | ${o:,.0f} | ${m:,.0f} | ${q:,.0f} | {p:.3f} |" for s, o, m, q, p in s1_rows)}

Verdict S1: {'supported' if s1 else 'not supported'} (median p = {np.median(s1_p):.3f}). The paper's published $5,344 is shown in the figure for reference.

![Null](fig_random_null.png)

## Q3, S2: probability of backtest overfitting (21 settings, 2018–2024, 924 splits)

| Seed | PBO | Mean in-sample Sharpe of the chosen setting | Its mean out-of-sample Sharpe |
|---|---|---|---|
{chr(10).join(f"| {s} | {d['pbo']:.2f} | {d['mean_is_sharpe_of_best']:.3f} | {d['mean_oos_sharpe_of_best']:.3f} |" for s, d in pbo_rows)}

Sharpe ratios here are monthly (not annualised). PBO above 0.5 means choosing the best setting
in-sample is more likely than not to give below-median performance out of sample.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")
    perf.to_csv(OUT / "walkforward_performance.csv", index=False)
    print(text)


if __name__ == "__main__":
    main()
