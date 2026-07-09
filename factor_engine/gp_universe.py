"""
Stock universe for the Gross Profitability (GP) factor.

Construction
------------
Target universe: ~1500 US equities spanning large/mid/small cap, matching the
"1500 stocks" scope for the proprietary GP factor (see factor_engine/factors/gp.py).

Russell 1000 + Russell 2000 (the literal index pairing that motivated "1500")
is not freely fetchable: iShares' holdings CSV endpoint
(ishares.com/.../1467271812596.ajax?fileType=csv&...) sits behind a JS
investor-type interstitial that returns HTML regardless of headers/cookies —
not scriptable without a real browser session.

S&P Composite 1500 (S&P 500 + S&P 400 MidCap + S&P 600 SmallCap) is used
instead: Wikipedia maintains clean, structured tables for all three indices
with no anti-bot gating, and the combined count (503 + 400 + 603 = 1506 as of
this writing) lands almost exactly on the "1500 stocks" target — a close
economic match (large/mid/small cap breadth, no OTC/micro-cap noise).

Cache: data/gp/universe.csv (ticker, name, index_source columns).
Refresh manually by deleting the cache file — constituent lists change slowly
(quarterly-ish index reconstitutions) and re-scraping on every call is wasteful.
"""

import io
import os

import pandas as pd
import requests

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gp")
_UNIVERSE_CACHE = os.path.join(_CACHE_DIR, "universe.csv")

_SOURCES: dict[str, str] = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _clean_ticker(raw: str) -> str:
    """Normalize a Wikipedia ticker string to yfinance convention (e.g. BRK.B -> BRK-B)."""
    return raw.strip().upper().replace(".", "-")


def _fetch_index_table(url: str) -> pd.DataFrame:
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    # The constituent table is always the first table on these pages and always
    # has a "Symbol" column; other tables on the page (recent changes, etc.)
    # don't share that column name.
    table = next(t for t in tables if "Symbol" in t.columns)
    return table[["Symbol", "Security"]].rename(columns={"Symbol": "ticker", "Security": "name"})


def _build_universe() -> pd.DataFrame:
    frames = []
    for source, url in _SOURCES.items():
        table = _fetch_index_table(url)
        table["ticker"] = table["ticker"].map(_clean_ticker)
        table["index_source"] = source
        frames.append(table)

    combined = pd.concat(frames, ignore_index=True)
    # A stock can appear in more than one index table around reconstitution
    # dates; keep the first (largest-cap) occurrence.
    combined = combined.drop_duplicates(subset="ticker", keep="first").reset_index(drop=True)
    return combined


def get_universe(refresh: bool = False) -> pd.DataFrame:
    """
    Return the GP factor's stock universe as a DataFrame with columns:
        ticker, name, index_source

    Cached to disk after first fetch. Pass refresh=True to re-scrape
    Wikipedia and overwrite the cache (e.g. after an index reconstitution).
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    if not refresh and os.path.exists(_UNIVERSE_CACHE):
        return pd.read_csv(_UNIVERSE_CACHE)

    universe = _build_universe()
    universe.to_csv(_UNIVERSE_CACHE, index=False)
    return universe


def get_universe_tickers(refresh: bool = False) -> list[str]:
    """Convenience accessor: just the ticker list."""
    return get_universe(refresh=refresh)["ticker"].tolist()
