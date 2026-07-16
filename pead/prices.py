"""
Ticker-keyed daily price cache for the PEAD backtest.

Deliberately separate from smart_money.models.PriceCache, which is scoped
to securities that appear in at least one 13F holding (a much smaller, and
mostly disjoint, set from the ~1,500-name PEAD universe) and is keyed by
CUSIP via the Security table — forcing PEAD's ticker-only universe through
that CUSIP-keyed schema would mean fabricating Security rows for names
with no 13F/CUSIP relationship at all. This module reuses the same
yf.download batching approach smart_money/prices.py uses, cached instead
to CSV under data/pead/prices/, consistent with the CSV-over-DB pattern
the rest of this package follows (see pead/__init__.py).

Cache
-----
data/pead/prices/{ticker}.csv — date, close, adj_close. A narrower refetch
is merged with any existing cached rows rather than overwriting them, so
repeated calls with different date windows accumulate history instead of
losing it. Refresh a ticker by deleting its cache file, or pass
refresh=True to force a re-fetch for every ticker requested.
"""

from __future__ import annotations

import datetime
import os

import pandas as pd
import yfinance as yf

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pead", "prices")
_BATCH_SIZE = 50         # tickers per yf.download() call, matches smart_money/prices.py
_CACHE_TOLERANCE = 5     # calendar days tolerance on each boundary for a cache hit

DateLike = datetime.date | str


def _to_date(d: DateLike) -> datetime.date:
    return d if isinstance(d, datetime.date) else datetime.date.fromisoformat(str(d))


def _to_yfinance_symbol(ticker: str) -> str:
    """Dual-class tickers use slash notation (e.g. BRK/B); yfinance requires hyphens."""
    return ticker.replace("/", "-")


def _cache_path(ticker: str) -> str:
    safe = ticker.replace("/", "-")
    return os.path.join(_CACHE_DIR, f"{safe}.csv")


def _load_cached(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(_cache_path(ticker), parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df.set_index("date")[["close", "adj_close"]].sort_index()


def _is_cached(ticker: str, start: datetime.date, end: datetime.date) -> bool:
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return False
    df = _load_cached(ticker)
    if df.empty:
        return False
    tolerance = datetime.timedelta(days=_CACHE_TOLERANCE)
    return df.index.min() <= start + tolerance and df.index.max() >= end - tolerance


def _fetch_batch(tickers: list[str], start: datetime.date, end: datetime.date) -> dict[str, pd.DataFrame]:
    """yf.download end date is exclusive — add 1 day to include the requested end."""
    end_excl = end + datetime.timedelta(days=1)
    symbol_map = {t: _to_yfinance_symbol(t) for t in tickers}

    raw = yf.download(
        list(symbol_map.values()),
        start=start.isoformat(),
        end=end_excl.isoformat(),
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return {}

    out: dict[str, pd.DataFrame] = {}
    for ticker, symbol in symbol_map.items():
        try:
            close_s = raw["Close"][symbol]
            adj_close_s = raw["Adj Close"][symbol]
        except (KeyError, TypeError):
            continue
        df = pd.DataFrame({"close": close_s, "adj_close": adj_close_s})
        df.index = pd.Index(
            [ts.date() if hasattr(ts, "date") else ts for ts in df.index],
            name="date",
        )
        df = df.dropna()
        if not df.empty:
            out[ticker] = df
    return out


def fetch_prices(
    tickers: list[str],
    start_date: DateLike,
    end_date: DateLike,
    *,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetch (or load from cache) daily close/adj_close prices for each ticker.

    Returns dict[ticker, DataFrame] indexed by date with columns
    [close, adj_close]. Tickers with no available data are absent.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    start = _to_date(start_date)
    end = _to_date(end_date)

    out: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for ticker in tickers:
        if not refresh and _is_cached(ticker, start, end):
            out[ticker] = _load_cached(ticker)
        else:
            to_fetch.append(ticker)

    for i in range(0, len(to_fetch), _BATCH_SIZE):
        batch = to_fetch[i:i + _BATCH_SIZE]
        fetched = _fetch_batch(batch, start, end)
        for ticker, df in fetched.items():
            path = _cache_path(ticker)
            if os.path.exists(path) and not refresh:
                existing = _load_cached(ticker)
                df = pd.concat([existing, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_csv(path, index_label="date")
            out[ticker] = df

    return out
