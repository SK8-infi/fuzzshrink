"""Turn the backtest results into tables and hypothesis verdicts (results/backtest/).

Usage:
    python scripts/report_backtest.py [--runs results/runs/backtest] [--out results/backtest]

Applies the decision rules of docs/protocol.md, section 6, and the clarification in section 9.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzshrink.metrics import (  # noqa: E402
    cer_difference_bps,
    deflated_sharpe_ratio,
    diebold_mariano,
    holm,
    performance,
    permutation_pvalue,
    sharpe_difference_test,
    variance_ratio_bootstrap,
)
from fuzzshrink.registry import CONFIGS, CORE, EXTENSIONS, REFERENCES  # noqa: E402

GAMMA_ORDER = [("G-2.0", -2.0), ("A6", -1.0), ("G-0.5", -0.5), ("A3", 0.0), ("G+0.5", 0.5),
               ("A4", 1.0), ("G+2.0", 2.0)]
LABELS = {**{k: v.label for k, v in CONFIGS.items()}, **{g: f"FuzzShrink, gamma = {v:+.1f}" for g, v in GAMMA_ORDER}}


def load(runs: Path) -> dict:
    results = defaultdict(dict)
    for path in sorted(runs.glob("*_seed*.pkl")):
        exp, seed = path.stem.rsplit("_seed", 1)
        with open(path, "rb") as fh:
            results[exp][int(seed)] = pickle.load(fh)
    return results


def per_seed_performance(res_by_seed: dict) -> pd.DataFrame:
    rows = []
    for seed, res in res_by_seed.items():
        diag = res.diagnostics.groupby("config").mean(numeric_only=True)
        for name in res.gross.columns:
            perf = performance(res.gross[name], res.net[name], res.turnover[name].to_numpy())
            d = diag.loc[name]
            rows.append({"seed": seed, "config": name, **perf, "cond": d["cond"], "delta0": d["delta0"],
                         "gross_exposure": d["gross_exposure"], "C": d["C"], "mean_entropy": d["mean_entropy"],
                         "lemma1_rate": d["lemma1_holds"], "projection_rate": d["projected"]})
    return pd.DataFrame(rows)


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def md_table(df: pd.DataFrame, index_name: str = "") -> str:
    cols = list(df.columns)
    lines = [f"| {index_name} | " + " | ".join(map(str, cols)) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("results/runs/backtest"))
    ap.add_argument("--out", type=Path, default=Path("results/backtest"))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args.out.mkdir(parents=True, exist_ok=True)
    R = load(args.runs)
    prim = R["primary"]
    seeds = sorted(prim)

    # ---------------------------------------------------------------- main table
    perf = per_seed_performance(prim)
    perf.to_csv(args.out / "primary_by_seed.csv", index=False)
    agg = perf.groupby("config").agg(sd_mean=("sd", "mean"), sd_std=("sd", "std"), mean=("mean", "mean"),
                                     sharpe=("sharpe", "mean"), sortino=("sortino", "mean"),
                                     mdd=("max_drawdown", "mean"), calmar=("calmar", "mean"),
                                     turnover=("turnover", "mean"), cond=("cond", "mean"),
                                     delta0=("delta0", "mean"), exposure=("gross_exposure", "mean"))
    vr_rows, sr_rows, cer_rows = [], [], []
    for seed in seeds:
        res = prim[seed]
        base_g, base_n = res.gross["A1"].to_numpy(), res.net["A1"].to_numpy()
        for name in res.gross.columns:
            g, n = res.gross[name].to_numpy(), res.net[name].to_numpy()
            if name != "A1":
                vr = variance_ratio_bootstrap(g, base_g, seed=seed)
                dm = diebold_mariano(g**2, base_g**2)
                sr = sharpe_difference_test(n, base_n)
                vr_rows.append({"seed": seed, "config": name, **vr, "dm_p": dm["p_less"]})
                sr_rows.append({"seed": seed, "config": name, **sr})
            cer_rows.append({"seed": seed, "config": name,
                             **{f"cer_{lam}": cer_difference_bps(n, base_n, lam) for lam in (1, 5, 10)}})
    vr_df = pd.DataFrame(vr_rows).groupby("config").median(numeric_only=True)
    sr_df = pd.DataFrame(sr_rows).groupby("config").median(numeric_only=True)
    cer_df = pd.DataFrame(cer_rows).groupby("config").mean(numeric_only=True)

    order = CORE + EXTENSIONS + REFERENCES
    main_rows = []
    for name in order:
        a = agg.loc[name]
        vr = vr_df.loc[name] if name in vr_df.index else None
        main_rows.append({
            "Configuration": f"{name}: {LABELS[name]}",
            "Ann. SD (%)": f"{100 * a.sd_mean:.2f} ± {100 * a.sd_std:.2f}" if a.sd_std > 1e-12 else f"{100 * a.sd_mean:.2f}",
            "Var. ratio vs A1 [95% CI]": "1" if vr is None else f"{vr.ratio:.3f} [{vr.ci_low:.3f}, {vr.ci_high:.3f}]",
            "DM p (lower than A1)": "" if vr is None else fmt_p(vr.dm_p),
            "Mean (%)": f"{100 * a['mean']:.1f}",
            "Sharpe": f"{a.sharpe:.2f}",
            "Sortino": f"{a.sortino:.2f}",
            "Max DD (%)": f"{100 * a.mdd:.1f}",
            "Turnover": "" if np.isnan(a.turnover) else f"{a.turnover:.2f}",
            "ΔCER λ=5 (bps/yr)": f"{cer_df.loc[name, 'cer_5']:+.0f}",
        })
    main = pd.DataFrame(main_rows).set_index("Configuration")
    main.to_csv(args.out / "primary_table.csv")

    # ---------------------------------------------------------------- hypotheses
    def dm_family(a, b):
        ps, stats, rejects = [], [], 0
        for seed in seeds:
            res = prim[seed]
            dm = diebold_mariano(res.gross[a].to_numpy() ** 2, res.gross[b].to_numpy() ** 2)
            ps.append(dm["p_less"])
            stats.append(dm["stat"])
            rejects += dm["p_less"] < 0.05
        return float(np.median(ps)), float(np.median(stats)), rejects

    tests = {"H1": ("A4", "A1"), "H3": ("A4", "A3"), "H5": ("A4", "A6"), "H6": ("A3", "A2")}
    hyp = {}
    for h, (a, b) in tests.items():
        p, stat, rej = dm_family(a, b)
        hyp[h] = {"comparison": f"{a} < {b}", "statistic": f"DM = {stat:+.2f}", "p": p, "seeds_rejecting": rej}
    p0, stat0, rej0 = dm_family("A4", "A0")

    a4_sd = np.mean([performance(prim[s].gross["A4"], prim[s].net["A4"])["sd"] for s in seeds])
    perm_sd = np.concatenate([[c.std(ddof=1) * np.sqrt(252) for _, c in prim[s].permutations.items()] for s in seeds])
    p_perm = permutation_pvalue(a4_sd, perm_sd)
    hyp["H4"] = {"comparison": "A4 < A5 (permuted U)",
                 "statistic": f"A4 SD {100 * a4_sd:.2f}% vs permuted median {100 * np.median(perm_sd):.2f}% "
                              f"(5th pct {100 * np.percentile(perm_sd, 5):.2f}%, n = {len(perm_sd)})",
                 "p": p_perm, "seeds_rejecting": None}
    adj = holm({h: v["p"] for h, v in hyp.items()})

    # H2: where the A3 -> A4 change in realised variance sits, by entropy quartile
    mech_rows = []
    for seed in seeds:
        m = prim[seed].mechanism
        wide = m.pivot_table(index=["date", "quartile"], columns="config", values="risk_contribution")
        delta = (wide["A4"] - wide["A3"]).unstack("quartile")  # dates x quartiles
        tot = delta.sum(axis=1)
        share = delta[4].mean() / tot.mean() if tot.mean() != 0 else np.nan
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(2000):
            idx = rng.integers(len(delta), size=len(delta))
            d = delta.iloc[idx]
            boots.append(d[4].mean() / d.sum(axis=1).mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        row = {"seed": seed, "share_q4": share, "ci_low": lo, "ci_high": hi,
               "total_change": tot.mean() * 252}
        row.update({f"q{q}": delta[q].mean() * 252 for q in range(1, 5)})
        mech_rows.append(row)
    mech = pd.DataFrame(mech_rows)
    mech.to_csv(args.out / "mechanism_by_seed.csv", index=False)
    mm = mech.median(numeric_only=True)
    h2_holds = bool(mm.share_q4 > 0.5 and mm.ci_low > 0.25)

    verdict_rows = []
    for h in ["H1", "H2", "H3", "H4", "H5", "H6"]:
        if h == "H2":
            verdict_rows.append({
                "Hypothesis": "H2: A3→A4 change concentrates in the top-entropy quartile",
                "Evidence": (f"top-quartile share {mm.share_q4:.2f} [{mm.ci_low:.2f}, {mm.ci_high:.2f}]; "
                             f"total change {1e4 * mm.total_change:+.2f} bps of annual variance"),
                "p (raw)": "", "p (Holm)": "", "Seeds rejecting": "",
                "Verdict": "supported" if h2_holds else "not supported"})
            continue
        v = hyp[h]
        verdict_rows.append({
            "Hypothesis": f"{h}: {v['comparison']}", "Evidence": v["statistic"], "p (raw)": fmt_p(v["p"]),
            "p (Holm)": fmt_p(adj[h]),
            "Seeds rejecting": "" if v["seeds_rejecting"] is None else f"{v['seeds_rejecting']}/{len(seeds)}",
            "Verdict": "supported" if adj[h] < 0.05 else "not supported"})
    verdicts = pd.DataFrame(verdict_rows).set_index("Hypothesis")
    verdicts.to_csv(args.out / "hypotheses.csv")

    # ---------------------------------------------------------------- sensitivity
    def sd_by(exp, names):
        if exp not in R:
            return None
        p = per_seed_performance(R[exp])
        return p.groupby("config").sd.mean().reindex(names)

    sens_names = ["A0", "A1", "A2", "A3", "A4", "A6", "X1", "X2", "R1", "R2", "R3"]
    t_tab = pd.DataFrame({f"T = {t}": sd_by(e, sens_names) for t, e in ((126, "T126"), (252, "primary"), (504, "T504"))
                          if e in R})
    m_tab = pd.DataFrame({f"m = {m}": sd_by(e, ["A1", "A3", "A4", "X1"])
                          for m, e in ((1.5, "m1.5"), (2.0, "primary"), (2.5, "m2.5")) if e in R})
    g_sd = perf.groupby("config").sd.agg(["mean", "std"])
    g_tab = pd.DataFrame({"gamma": [g for _, g in GAMMA_ORDER],
                          "Ann. SD (%)": [f"{100 * g_sd.loc[n, 'mean']:.2f} ± {100 * g_sd.loc[n, 'std']:.2f}"
                                          for n, _ in GAMMA_ORDER]},
                         index=[n for n, _ in GAMMA_ORDER])
    lo_tab = None
    if "longonly" in R:
        lp = per_seed_performance(R["longonly"]).groupby("config").agg(
            sd=("sd", "mean"), sharpe=("sharpe", "mean"), turnover=("turnover", "mean"))
        lo_tab = lp.reindex(order).apply(lambda r: pd.Series({
            "Ann. SD (%)": f"{100 * r.sd:.2f}", "Sharpe": f"{r.sharpe:.2f}",
            "Turnover": "" if np.isnan(r.turnover) else f"{r.turnover:.2f}"}), axis=1)

    # ---------------------------------------------------------------- deflated Sharpe
    trial_sr = []
    for exp, by_seed in R.items():
        p = per_seed_performance(by_seed)
        trial_sr += list(p.groupby("config").sharpe.mean() / np.sqrt(252))
    best = agg.sharpe.idxmax()
    dsr = deflated_sharpe_ratio(prim[0].net[best].to_numpy(), np.array(trial_sr))

    # ---------------------------------------------------------------- diagnostics
    dg = perf.groupby("config")[["C", "mean_entropy", "delta0", "lemma1_rate", "projection_rate", "cond",
                                 "gross_exposure"]].mean().reindex(order)

    def f3(df, cols, fmt):
        out = df.copy()
        for c in cols:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
        return out

    dg_fmt = f3(dg, ["C", "mean_entropy", "delta0", "lemma1_rate", "projection_rate"], "{:.3f}")
    dg_fmt = f3(dg_fmt, ["cond", "gross_exposure"], "{:,.1f}")
    pct = lambda df: df.map(lambda v: "" if pd.isna(v) else f"{100 * v:.2f}")  # noqa: E731

    n_days = len(prim[seeds[0]].gross)
    text = f"""# NASDAQ-100 backtest results

Generated by `scripts/report_backtest.py`. Design: `docs/protocol.md`. Monthly rebalancing,
{prim[seeds[0]].gross.index.min().date()} to {prim[seeds[0]].gross.index.max().date()} ({n_days} trading days),
{len(seeds)} clustering seeds; tests follow the protocol's section 6 and the section 9 clarification.

## Hypotheses (T = 252, unconstrained GMV)

{md_table(verdicts)}

A4 vs A0 (H1, second part): DM = {stat0:+.2f}, median p = {fmt_p(p0)}, {rej0}/{len(seeds)} seeds rejecting.

## Main table (T = 252, unconstrained GMV)

Risk is annualised SD of gross daily returns. Mean, Sharpe, Sortino and drawdown are
after 10 bps transaction costs. Variance ratios and DM p-values are medians over seeds.
ΔCER is the annual certainty-equivalent gain over A1 for risk aversion 5.

{md_table(main)}

Sharpe-ratio differences vs A1 (Ledoit–Wolf 2008 HAC test, median over seeds):

{md_table(sr_df[["diff_annual", "p_two_sided"]].reindex([c for c in order if c != "A1"]).map(lambda v: f"{v:+.3f}"))}

Deflated Sharpe ratio of the best configuration ({best}, seed 0): Sharpe {dsr['sharpe_annual']:.2f},
expected maximum under the null {dsr['sr0_annual']:.2f} across {dsr['n_trials']} trials, DSR = {dsr['dsr']:.3f}.

## Mechanism (H2)

Change from A3 to A4 in realised risk contribution, by membership-entropy quartile
(annualised variance, bps; quartile 4 = most ambiguous), median over seeds:

{md_table(pd.DataFrame({f"Q{q}": [f"{1e4 * mm[f'q{q}']:+.2f}"] for q in range(1, 5)}, index=["A4 − A3"]))}

## Sensitivity (exploratory)

Window length, annualised SD (%):

{md_table(pct(t_tab))}

Fuzzifier m, annualised SD (%):

{md_table(pct(m_tab))}

Direction and strength of entropy conditioning (γ), annualised SD:

{md_table(g_tab.drop(columns="gamma"))}

Long-only GMV:

{md_table(lo_tab) if lo_tab is not None else "not run"}

## Diagnostics (T = 252)

{md_table(dg_fmt.rename(columns={"mean_entropy": "mean H", "lemma1_rate": "Lemma 1 holds", "projection_rate": "projection rate", "gross_exposure": "gross exposure"}))}
"""
    (args.out / "README.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
