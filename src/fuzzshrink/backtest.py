"""Rolling out-of-sample backtest (docs/protocol.md, section 3)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .clustering import fit_memberships, membership_entropy
from .data import rebalance_dates, window_slices
from .estimators import fuzzshrink
from .moments import WindowStats
from .registry import CONFIGS, Config, Context, portfolio_weights

__all__ = ["BacktestSettings", "BacktestResult", "run_backtest"]


@dataclass
class BacktestSettings:
    T: int = 252
    start: str = "2017-01-01"
    end: str = "2024-12-31"
    tc_bps: float = 10.0
    long_only: bool = False
    m: float = 2.0
    c_range: tuple[int, int] = (2, 8)
    n_init: int = 5
    n_permutations: int = 0
    mechanism: tuple[str, ...] = ("A1", "A3", "A4", "A6", "X1")


@dataclass
class BacktestResult:
    settings: BacktestSettings
    seed: int
    gross: pd.DataFrame  # daily gross returns, dates x configs
    net: pd.DataFrame  # daily returns after transaction costs
    turnover: pd.DataFrame  # one-way turnover at each rebalance, dates x configs
    diagnostics: pd.DataFrame  # one row per (rebalance, config)
    mechanism: pd.DataFrame  # realised risk contribution by entropy quartile
    permutations: pd.DataFrame = field(default_factory=pd.DataFrame)  # A5 runs, gross


def _drift(w: pd.Series, hold: pd.DataFrame) -> pd.Series:
    """Weights at the end of a holding period after prices moved (buy and hold)."""
    growth = (1.0 + hold[w.index]).prod()
    value = w * growth
    return value / value.sum()


def _entropy_quartile(H: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(H, kind="stable"), kind="stable")
    return np.minimum(4 * ranks // len(H), 3)


def run_backtest(returns: pd.DataFrame, names: list[str], seed: int = 0,
                 settings: BacktestSettings | None = None,
                 extra_configs: dict[str, Config] | None = None) -> BacktestResult:
    """Run every named configuration through the same monthly rolling backtest."""
    s = settings or BacktestSettings()
    registry = {**CONFIGS, **(extra_configs or {})}
    configs = {n: registry[n] for n in names}
    need_U = s.n_permutations > 0 or any(c.uses_memberships for c in configs.values())
    need_rr = "X4" in names
    tc = s.tc_bps / 1e4

    dates = rebalance_dates(returns.index, s.start, s.end)
    gross = {n: [] for n in names}
    turnover = {n: [] for n in names}
    perm = {p: [] for p in range(s.n_permutations)}
    diag_rows, mech_rows, index, reb_dates = [], [], [], []
    prev_w: dict[str, pd.Series | None] = {n: None for n in names}
    prev_hold = None

    for k, (date, est, hold) in enumerate(window_slices(returns, dates, s.T)):
        tickers = est.columns
        stats = WindowStats.from_returns(est.to_numpy())
        part = (fit_memberships(stats, features="comove", seed=seed, m=s.m, c_range=s.c_range,
                                n_init=s.n_init) if need_U else None)
        part_rr = (fit_memberships(stats, features="riskret", seed=seed, m=s.m, c_range=s.c_range,
                                   n_init=s.n_init) if need_rr else None)
        rng = np.random.default_rng([seed, k])
        ctx = Context(stats, part.U if part else None, part_rr.U if part_rr else None, rng)
        H = membership_entropy(part.U) if part else None
        quart = _entropy_quartile(H) if part else None
        R_hold = hold.to_numpy()
        hold_c = R_hold - R_hold.mean(axis=0)
        cov_real = hold_c.T @ hold_c / len(R_hold)
        index.append(hold.index)
        reb_dates.append(date)

        for name, cfg in configs.items():
            Sigma, info = cfg.fn(ctx)
            w = portfolio_weights(Sigma, stats.N, s.long_only)
            gross[name].append(R_hold @ w)
            w_s = pd.Series(w, index=tickers)
            if prev_w[name] is None:
                to = 0.0
            else:
                drifted = _drift(prev_w[name], prev_hold)
                allt = w_s.index.union(drifted.index)
                to = float((w_s.reindex(allt, fill_value=0) - drifted.reindex(allt, fill_value=0)).abs().sum())
            turnover[name].append(to)
            prev_w[name] = w_s

            diag_rows.append({
                "date": date, "config": name, "N": stats.N,
                "C": part.n_clusters if part else np.nan,
                "mean_entropy": float(H.mean()) if H is not None else np.nan,
                "delta0": info.get("delta0", np.nan),
                "mean_delta": float(np.mean(info["delta"])) if "delta" in info else np.nan,
                "lemma1_holds": float(info["lemma1_holds"]) if "lemma1_holds" in info else np.nan,
                "projected": float(info["projected"]) if "projected" in info else np.nan,
                "frobenius_moved": info.get("frobenius_moved", np.nan),
                "cond": float(np.linalg.cond(Sigma)) if Sigma is not None else np.nan,
                "gross_exposure": float(np.abs(w).sum()),
            })
            if name in s.mechanism and quart is not None:
                rc = w * (cov_real @ w)
                for q in range(4):
                    mech_rows.append({"date": date, "config": name, "quartile": q + 1,
                                      "risk_contribution": float(rc[quart == q].sum())})

        for p in range(s.n_permutations):
            Up = part.U[rng.permutation(stats.N)]
            Sigma_p, _ = fuzzshrink(stats, Up, gamma=1.0)
            perm[p].append(R_hold @ portfolio_weights(Sigma_p, stats.N, s.long_only))
        prev_hold = hold

    all_dates = index[0].append(index[1:]) if len(index) > 1 else index[0]
    gross_df = pd.DataFrame({n: np.concatenate(v) for n, v in gross.items()}, index=all_dates)
    to_df = pd.DataFrame(turnover, index=pd.DatetimeIndex(reb_dates))
    cost = pd.DataFrame(0.0, index=all_dates, columns=names)
    first_days = [ix[0] for ix in index]
    cost.loc[first_days, :] = to_df.to_numpy() * tc
    perm_df = pd.DataFrame({f"perm{p}": np.concatenate(v) for p, v in perm.items()}, index=all_dates) \
        if s.n_permutations else pd.DataFrame(index=all_dates)
    return BacktestResult(
        settings=s, seed=seed, gross=gross_df, net=gross_df - cost, turnover=to_df,
        diagnostics=pd.DataFrame(diag_rows), mechanism=pd.DataFrame(mech_rows), permutations=perm_df,
    )
