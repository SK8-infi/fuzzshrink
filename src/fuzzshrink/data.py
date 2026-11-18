"""Price data, returns and the per-window universe (docs/protocol.md, sections 2-3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["load_prices", "apply_listing_overrides", "prices_to_returns", "rebalance_dates",
           "window_slices"]


def load_prices(path: str | Path, overrides: str | Path | None = None) -> pd.DataFrame:
    """Adjusted closes saved by scripts/download_data.py: dates x tickers.

    If an overrides file is given (ticker, valid_from, reason), each listed ticker's
    prices before its valid_from date are dropped; see data/universe/listing_overrides.csv.
    """
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices.index.name = "date"
    prices = prices.sort_index()
    if overrides is not None:
        prices = apply_listing_overrides(prices, pd.read_csv(overrides))
    return prices


def apply_listing_overrides(prices: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """Set each listed ticker's prices before its valid_from date to NaN."""
    out = prices.copy()
    for row in overrides.itertuples():
        if row.ticker in out.columns:
            out.loc[out.index < pd.Timestamp(row.valid_from), row.ticker] = np.nan
    return out


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns; NaN wherever either price is missing (no forward fill)."""
    rets = prices / prices.shift(1) - 1.0
    return rets.iloc[1:].replace([np.inf, -np.inf], np.nan)


def rebalance_dates(index: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    """First trading day of each calendar month within [start, end]."""
    idx = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    months = idx.to_period("M")
    first = ~pd.Series(months).duplicated().to_numpy()
    return idx[first]


def window_slices(returns: pd.DataFrame, rebalances: pd.DatetimeIndex, T: int):
    """Yield (date, estimation window, holding period) for each rebalance date.

    The estimation window is the T trading days strictly before the rebalance date. The
    holding period runs from the rebalance date to the day before the next rebalance;
    the last one ends with the data. Only assets with complete returns in both the
    window and the holding period are kept (protocol section 2).
    """
    index = returns.index
    for k, date in enumerate(rebalances):
        pos = index.get_loc(date)
        if pos < T:
            raise ValueError(f"not enough history before {date.date()} for T = {T}")
        end = index.get_loc(rebalances[k + 1]) if k + 1 < len(rebalances) else len(index)
        est = returns.iloc[pos - T : pos]
        hold = returns.iloc[pos:end]
        keep = est.notna().all() & hold.notna().all()
        yield date, est.loc[:, keep], hold.loc[:, keep]
