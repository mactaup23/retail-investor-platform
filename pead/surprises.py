"""
EPS surprise data pull for the PEAD signal.

Source: yfinance Ticker.get_earnings_dates() — actual vs. consensus-estimate
EPS per historical announcement, with an hour-level announcement timestamp.

Depth
-----
yfinance's default limit=12 is shallow (~3 years). Empirically, passing a
higher limit (_FETCH_LIMIT=50) returns 40-50 quarters (10+ years) of history
for names with a long public trading history (e.g. CVI, HLX, HELE all show
data back to 2014), shrinking only for recent IPOs (e.g. HAYW, listed 2021,
tops out at 22 quarters). This is deeper than the "shallow ~4-8 quarter"
assumption this signal was originally scoped under — the limitation turned
out to be the default parameter, not a hard yfinance ceiling. Accepted as-is
for this first pass regardless: no cross-referencing against a second
history source is being done here (see signal.py and CLAUDE.md for the
EDGAR-cross-check-if-triggered plan).

Session classification (BMO / AMC)
-----------------------------------
Verified empirically: yfinance's announcement timestamps carry a real
America/New_York offset and are internally consistent with known real-world
release patterns — KR (reports before the open) consistently shows
06:00-08:00 ET; AAPL/NVDA (report after the close) consistently show
16:00-18:00 ET. Classified per row, not per ticker, since a company's
disclosure timing can change over its history:

    hour <  16  -> "bmo"      that day's close already reflects the news
    hour >= 16  -> "amc"      entry anchors to the next trading day instead
    no timestamp -> "unknown"  treated as amc (conservative — never assume
                                same-day reaction without evidence)

The exact minute is not trusted, only the before/at-or-after-close bucket.

revenue_surprise_pct is a placeholder column for a future second surprise
dimension — always null in this first pass; no revenue data is pulled yet.

Cache
-----
data/pead/surprises/{ticker}.csv, one row per historical announcement.
Refresh a ticker by deleting its cache file, or pass refresh=True to
force a re-pull for every ticker requested.
"""

from __future__ import annotations

import logging
import os
import time

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pead", "surprises")
_FETCH_LIMIT = 50       # quarters of history requested per ticker (see depth note above)
_REQUEST_DELAY = 0.3    # seconds between per-ticker calls; yfinance has no documented
                         # rate limit but get_earnings_dates() has no batch endpoint
                         # (unlike yf.download used in smart_money/prices.py), so pacing
                         # is the only lever available to avoid throttling at ~1,500 calls

_AMC_HOUR_CUTOFF = 16   # market close, ET

_CSV_COLUMNS = [
    "ticker", "announcement_date", "session",
    "eps_estimate", "eps_actual", "eps_surprise_pct", "revenue_surprise_pct",
]


def _cache_path(ticker: str) -> str:
    safe = ticker.replace("/", "-")
    return os.path.join(_CACHE_DIR, f"{safe}.csv")


def _classify_session(ts) -> str:
    if ts is None or pd.isna(ts):
        return "unknown"
    return "bmo" if ts.hour < _AMC_HOUR_CUTOFF else "amc"


def _fetch_one(ticker: str) -> pd.DataFrame | None:
    """Fetch and normalize one ticker's EPS surprise history. Returns None on failure or no data."""
    try:
        raw = yf.Ticker(ticker).get_earnings_dates(limit=_FETCH_LIMIT)
    except Exception as e:
        log.warning("PEAD surprise fetch failed for %s: %s", ticker, e)
        return None

    if raw is None or raw.empty:
        return None

    raw = raw.dropna(subset=["Reported EPS"])   # drops the single unreported future-earnings row
    if raw.empty:
        return None

    df = pd.DataFrame({
        "ticker": ticker,
        "announcement_date": [idx.date() for idx in raw.index],
        "session": [_classify_session(idx) for idx in raw.index],
        "eps_estimate": raw["EPS Estimate"].to_numpy(),
        "eps_actual": raw["Reported EPS"].to_numpy(),
        "eps_surprise_pct": raw["Surprise(%)"].to_numpy(),
    })
    df["revenue_surprise_pct"] = pd.NA
    return df.sort_values("announcement_date").reset_index(drop=True)


def _load_cached(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(_cache_path(ticker), parse_dates=["announcement_date"])
    df["announcement_date"] = df["announcement_date"].dt.date
    return df


def fetch_surprises(
    tickers: list[str],
    *,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetch (or load from cache) EPS surprise history for each ticker.

    Parameters
    ----------
    tickers : list[str]
    refresh : bool
        Force a re-pull even if a cache file exists.

    Returns
    -------
    dict[ticker, DataFrame]
        Columns: ticker, announcement_date, session, eps_estimate,
        eps_actual, eps_surprise_pct, revenue_surprise_pct.
        Tickers with no usable data (fetch failure, no earnings history)
        are absent from the result.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    n_fetched = 0

    for i, ticker in enumerate(tickers):
        path = _cache_path(ticker)
        if not refresh and os.path.exists(path):
            out[ticker] = _load_cached(ticker)
            continue

        df = _fetch_one(ticker)
        n_fetched += 1
        if df is not None:
            df.to_csv(path, index=False, columns=_CSV_COLUMNS)
            out[ticker] = df
        time.sleep(_REQUEST_DELAY)

        if n_fetched and n_fetched % 100 == 0:
            log.info("PEAD surprise pull: %d/%d tickers processed (%d fetched live)", i + 1, len(tickers), n_fetched)

    return out
