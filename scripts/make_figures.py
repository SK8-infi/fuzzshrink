"""Figures for the README and the paper (results/figures/).

Usage:
    python scripts/make_figures.py

Colours follow the reference data-viz palette: categorical slots in fixed order
(blue, orange, aqua, yellow), with grey for de-emphasised marks. Every figure has a
table twin in results/simulation/README.md or results/backtest/README.md.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GREY = "#b4b2a9"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OUT = Path("results/figures")

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "normal", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.linewidth": 2.0,
})


def _finish(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("wrote", OUT / name)


def _grouped_bars(ax, groups, series, values, colors, labels=True, fmt="{:.2f}"):
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for k, (s, c) in enumerate(zip(series, colors)):
        pos = x - 0.4 + width * (k + 0.5)
        bars = ax.bar(pos, values[k], width=width, color=c, edgecolor=SURFACE, linewidth=1.5, label=s, zorder=3)
        if labels:
            for b, v in zip(bars, values[k]):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center",
                        va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def fig_sim_intensity():
    diag = pd.read_csv("results/simulation/diagnostics.csv.gz")
    d = diag[(diag.scenario == "mixed") & (diag["T"] == 252) & diag.config.isin(["A4", "X1"])]
    m = d.groupby(["config", "type"]).mean_delta.mean().unstack()
    types = ["crisp", "bridge", "outlier"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    _grouped_bars(ax, ["Crisp stocks", "Bridge stocks", "Outliers"],
                  ["Entropy weights (draft, A4)", "Fit weights (X1)"],
                  [m.loc["A4", types].to_numpy(), m.loc["X1", types].to_numpy()], SLOT[:2])
    ax.set_ylabel("Mean shrinkage intensity δ")
    ax.set_title("Entropy holds back on bridges as well as outliers; fit weights only on outliers")
    ax.set_ylim(0, m.to_numpy().max() * 1.45)
    ax.legend(loc="upper right")
    _finish(fig, "fig1_sim_intensity_by_type.png")


def fig_sim_loss():
    s = pd.read_csv("results/simulation/summary.csv")
    s = s[(~s.oracle) & (s["T"] == 252)]
    scen = ["crisp", "bridges", "outliers", "mixed"]
    names = {"A2": "Hard clusters (A2)", "A3": "Soft target, uniform (A3)",
             "A4": "Soft + entropy (draft, A4)", "X1": "Soft + fit weights (X1)"}
    vals = [s[s.config == c].set_index("scenario").loc[scen, "mean"].to_numpy() for c in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    _grouped_bars(ax, ["Crisp groups", "With bridges", "With outliers", "Mixed"], list(names.values()),
                  vals, SLOT)
    ax.set_ylabel("Excess GMV variance vs. true Σ")
    ax.set_title("Simulation, T = 252: out-of-sample cost of each estimator (lower is better)")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_ylim(0, max(v.max() for v in vals) * 1.35)
    _finish(fig, "fig2_sim_loss_by_scenario.png")


def _load(exp):
    runs = {}
    for p in sorted(Path("results/runs/backtest").glob(f"{exp}_seed*.pkl")):
        with open(p, "rb") as fh:
            runs[int(p.stem.rsplit("_seed", 1)[1])] = pickle.load(fh)
    return runs


def fig_backtest_sd(labels: dict[str, str], highlight: dict[str, tuple[str, str]], lo: float, hi: float):
    """Dot plot: values are close together, so position on a zoomed axis, not bar length."""
    perf = pd.read_csv("results/backtest/primary_by_seed.csv")
    g = perf[perf.config.isin(labels)].groupby("config").sd.agg(["mean", "std"]) * 100
    g = g.sort_values("mean")
    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(g) + 1.3))
    y = np.arange(len(g))
    for yi, (c, row) in zip(y, g.iterrows()):
        color = highlight.get(c, (GREY, ""))[0]
        if row["mean"] > hi:
            ax.text(hi, yi, f"{row['mean']:.1f}%  →", va="center", ha="right", fontsize=8, color=INK2)
            continue
        ax.errorbar(row["mean"], yi, xerr=row["std"], fmt="o", color=color, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, ecolor=color, capsize=2, zorder=3)
        ax.text(row["mean"] + 0.04, yi - 0.28, f"{row['mean']:.2f}", fontsize=8, color=INK2)
    ax.set_yticks(y, [labels[c] for c in g.index])
    ax.invert_yaxis()
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Annualised out-of-sample volatility of the GMV portfolio, % (axis zoomed; lower is better)")
    ax.set_title("NASDAQ-100, 2017–2024, T = 252: realised risk by estimator")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=7, label=lab)
               for c, lab in highlight.values()]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", color=GREY, markersize=7, label="Other estimators"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    _finish(fig, "fig3_backtest_sd.png")


def fig_gamma():
    perf = pd.read_csv("results/backtest/primary_by_seed.csv")
    order = [("G-2.0", -2), ("A6", -1), ("G-0.5", -0.5), ("A3", 0), ("G+0.5", 0.5), ("A4", 1), ("G+2.0", 2)]
    g = perf.groupby("config").sd.agg(["mean", "std"])
    gam = [v for _, v in order]
    mean = np.array([g.loc[c, "mean"] for c, _ in order]) * 100
    std = np.array([g.loc[c, "std"] for c, _ in order]) * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.errorbar(gam, mean, yerr=std, color=SLOT[0], marker="o", markersize=6, capsize=3, zorder=3)
    for c, v in (("A3", 0), ("A4", 1)):
        i = gam.index(v)
        ax.annotate({"A3": "uniform (A3)", "A4": "draft (A4)"}[c], (v, mean[i]), textcoords="offset points",
                    xytext=(9, 9), ha="left", fontsize=8, color=INK2)
    ax.set_xlabel("γ   (negative: shrink ambiguous stocks more · positive: shrink them less)")
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_title("Entropy conditioning on real data: risk as a function of γ (T = 252)")
    _finish(fig, "fig4_gamma_sweep.png")


def fig_rolling(configs: dict[str, str], base: str = "A1"):
    """Trailing one-year variance of each estimator's portfolio relative to the baseline's."""
    runs = _load("primary")
    var = {}
    for s, r in runs.items():
        v = r.gross[list(configs) + [base]].rolling(252).var().dropna()
        var[s] = v[list(configs)].div(v[base], axis=0)
    ratio = pd.concat(var).groupby(level=1).mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.axhline(1.0, color=AXIS, linewidth=1.2, zorder=2)
    for (c, lab), col in zip(configs.items(), SLOT):
        ax.plot(ratio.index, ratio[c], color=col, label=lab, linewidth=1.8, zorder=3)
    ax.set_ylabel("Trailing 1-year variance ÷ A1's")
    ax.set_title("Below 1 = less realised risk than constant-correlation shrinkage (A1)")
    ax.legend(loc="lower right", ncol=1, fontsize=8)
    _finish(fig, "fig5_rolling_variance_ratio.png")


if __name__ == "__main__":
    fig_sim_intensity()
    fig_sim_loss()
    fig_backtest_sd(
        labels={"A0": "A0 Sample covariance", "A1": "A1 Constant correlation", "A2": "A2 Hard block",
                "A3": "A3 Soft target, γ = 0", "A4": "A4 Soft + entropy (draft)", "A5": "A5 Permuted memberships",
                "A6": "A6 Mirror, γ = −1", "X1": "X1 Fit weights", "X2": "X2 Block-pair target",
                "X3": "X3 Draft's arithmetic rule", "X4": "X4 Risk-return features",
                "R1": "R1 Ledoit–Wolf 2004", "R2": "R2 Ledoit–Wolf 2020 nonlinear", "R3": "R3 Equal weight"},
        highlight={"A4": (SLOT[0], "Draft's estimator (A4)"), "R2": (SLOT[1], "Best: Ledoit–Wolf 2020 (R2)")},
        lo=13.9, hi=15.6)
    fig_gamma()
    fig_rolling({"A4": "A4 draft's estimator", "R1": "R1 Ledoit–Wolf 2004", "R2": "R2 Ledoit–Wolf 2020"})
