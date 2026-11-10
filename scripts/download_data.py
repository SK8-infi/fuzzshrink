"""Download the NASDAQ-100 universe and daily adjusted prices (docs/protocol.md, section 2).

Usage:
    python scripts/download_data.py

Writes:
    data/universe/nasdaq100_constituents.csv   (committed: tickers, names, sectors, date)
    data/raw/prices.csv.gz                     (not committed: adjusted closes)
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
HEADERS = {"User-Agent": "fuzzshrink-course-project/0.1 (research; python-requests)"}


def fetch_constituents() -> pd.DataFrame:
    html = requests.get(WIKI_URL, headers=HEADERS, timeout=30).text
    for table in pd.read_html(io.StringIO(html)):
        cols = {str(c).strip().lower(): c for c in table.columns}
        if "ticker" in cols and ("company" in cols or "security" in cols):
            out = pd.DataFrame({
                "ticker": table[cols["ticker"]].astype(str).str.strip().str.replace(".", "-", regex=False),
                "company": table[cols.get("company", cols.get("security"))].astype(str).str.strip(),
            })
            sector = next((cols[c] for c in cols if "industry" in c), None)
            if sector is None:
                sector = next((cols[c] for c in cols if "sector" in c), None)
            out["sector"] = table[sector].astype(str).str.strip() if sector is not None else ""
            if 90 <= len(out) <= 110:
                out["fetched_on"] = dt.date.today().isoformat()
                return out.sort_values("ticker").reset_index(drop=True)
    raise RuntimeError("could not find the constituents table on the Wikipedia page")


def download_prices(tickers: list[str], start: str, end: str, retries: int = 3) -> pd.DataFrame:
    frames, missing = [], list(tickers)
    for attempt in range(1, retries + 1):
        if not missing:
            break
        data = yf.download(missing, start=start, end=end, auto_adjust=True, progress=False,
                           threads=True, group_by="column")
        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
        close = close.dropna(axis=1, how="all")
        frames.append(close)
        missing = [t for t in missing if t not in close.columns]
        if missing:
            print(f"attempt {attempt}: {len(missing)} tickers missing, retrying: {missing}")
            time.sleep(5 * attempt)
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    if missing:
        print(f"gave up on {len(missing)} tickers: {missing}")
    return prices


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2014-07-01")
    ap.add_argument("--end", default="2025-01-01", help="exclusive end date")
    ap.add_argument("--root", type=Path, default=Path("data"))
    args = ap.parse_args()

    uni_path = args.root / "universe" / "nasdaq100_constituents.csv"
    if uni_path.exists():
        universe = pd.read_csv(uni_path)
        print(f"using saved universe from {universe.fetched_on.iloc[0]} ({len(universe)} tickers)")
    else:
        universe = fetch_constituents()
        uni_path.parent.mkdir(parents=True, exist_ok=True)
        universe.to_csv(uni_path, index=False)
        print(f"saved {len(universe)} constituents to {uni_path}")

    prices = download_prices(universe.ticker.tolist(), args.start, args.end)
    out = args.root / "raw" / "prices.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out)
    first = prices.apply(lambda s: s.first_valid_index())
    print(f"saved {prices.shape[1]} tickers x {prices.shape[0]} days to {out}")
    print(f"{(first <= pd.Timestamp('2015-01-02')).sum()} tickers have prices from the start of 2015")


if __name__ == "__main__":
    main()
