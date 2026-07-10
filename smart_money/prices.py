"""
Price fetcher and cache layer for Module 3 — 13F Smart-Money Positioning & Skill Tracker.

Public interface
----------------
    fetch_prices(tickers, start_date, end_date, *, skip_cached=True)
        → dict[str, pd.DataFrame]

    fetch_prices_for_cusips(cusips, start_date, end_date, *, skip_cached=True)
        → dict[str, pd.DataFrame]   (keyed by CUSIP)

Cache strategy
--------------
Prices are stored in PriceCache (models.py) keyed by Security (CUSIP).  For
each ticker, if the DB already has rows that span [start_date, end_date] within
±_CACHE_TOLERANCE calendar days (handles weekend/holiday boundaries), the cache
is served without touching yfinance.  Pass skip_cached=False to force a fresh
fetch for all inputs.

yfinance notes
--------------
yf.download() is called with a list of tickers and auto_adjust=False so that
both raw Close and split/dividend-adjusted Adj Close are captured.  yfinance
1.4.x always returns MultiIndex columns (price_type, ticker) when given a list,
even for a single ticker — the fetch helper handles this uniformly.

adj_close is what downstream return calculations use; close is retained for
display and verification.

Batch / upsert limits
---------------------
_BATCH_SIZE tickers per yf.download() call.
DB upserts are chunked at 500 rows to stay within SQLite parameter limits.
"""

import datetime
from typing import Union

import pandas as pd
import yfinance as yf

from smart_money.models import PriceCache, Security, init_db

DateLike = Union[str, datetime.date]

_BATCH_SIZE      = 50   # tickers per yf.download() call
_CACHE_TOLERANCE = 5    # calendar days tolerance on each boundary for cache hit
_UPSERT_CHUNK    = 500  # rows per INSERT batch


def _to_yfinance_symbol(ticker: str) -> str:
    """
    Translate a dual-class ticker from OpenFIGI/Bloomberg slash notation
    (e.g. "BRK/A", "BRK/B", "BF/B") to yfinance's hyphen notation
    ("BRK-A", "BRK-B", "BF-B"). Security.ticker keeps the slash form since
    that's the conventional display format used elsewhere (display_name).
    """
    return ticker.replace("/", "-")


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _to_date(d: DateLike) -> datetime.date:
    if isinstance(d, datetime.date):
        return d
    return datetime.date.fromisoformat(str(d))


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _is_cached(security: Security, start: datetime.date, end: datetime.date) -> bool:
    """
    Return True if PriceCache has adequate coverage for [start, end].

    We compare the earliest and latest cached dates against the requested
    boundaries with a ±_CACHE_TOLERANCE day window.  This avoids false cache
    misses when start/end fall on weekends or holidays.
    """
    rows = list(
        PriceCache
        .select(PriceCache.date)
        .where(
            PriceCache.security == security,
            PriceCache.date.between(start, end),
        )
        .order_by(PriceCache.date)
    )
    if not rows:
        return False
    tolerance = datetime.timedelta(days=_CACHE_TOLERANCE)
    return rows[0].date <= start + tolerance and rows[-1].date >= end - tolerance


def _load_from_cache(
    security: Security,
    start: datetime.date,
    end: datetime.date,
) -> pd.DataFrame:
    rows = list(
        PriceCache
        .select(PriceCache.date, PriceCache.close, PriceCache.adj_close)
        .where(
            PriceCache.security == security,
            PriceCache.date.between(start, end),
        )
        .order_by(PriceCache.date)
    )
    return pd.DataFrame(
        {
            "close":     [r.close     for r in rows],
            "adj_close": [r.adj_close for r in rows],
        },
        index=pd.Index([r.date for r in rows], name="date"),
    )


_UPSERT_SQL = (
    "INSERT OR REPLACE INTO price_cache "
    "(security_id, date, close, adj_close, source, fetched_at) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)


def _date_str(d) -> str:
    """Normalise a date-like value to ISO 'YYYY-MM-DD' string."""
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)[:10]


def _upsert_prices(security: Security, df: pd.DataFrame) -> None:
    """Bulk upsert a ticker's price DataFrame into PriceCache."""
    df = df[~df.index.duplicated(keep="last")]
    sid = security.get_id()
    now_str = datetime.datetime.utcnow().isoformat()
    rows = [
        (sid, _date_str(row_date), float(row["close"]), float(row["adj_close"]), "yfinance", now_str)
        for row_date, row in df.iterrows()
        if pd.notna(row["close"]) and pd.notna(row["adj_close"])
    ]
    if not rows:
        return
    db = PriceCache._meta.database
    conn = db.connection()
    with db.atomic():
        for i in range(0, len(rows), _UPSERT_CHUNK):
            conn.executemany(_UPSERT_SQL, rows[i : i + _UPSERT_CHUNK])


# ---------------------------------------------------------------------------
# yfinance fetch
# ---------------------------------------------------------------------------

def _fetch_from_yfinance(
    tickers: list[str],
    start: datetime.date,
    end: datetime.date,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV from yfinance for a batch of tickers.

    Returns dict[ticker, DataFrame] with columns [close, adj_close] indexed
    by datetime.date.  Tickers that return no data are absent from the result.

    yfinance end date is exclusive — we add 1 day to include the requested end.
    yfinance 1.4.x always returns MultiIndex columns (price_type, ticker) when
    given a list; we access raw["Close"][symbol] and raw["Adj Close"][symbol].

    Dual-class tickers (e.g. "BRK/A") use OpenFIGI/Bloomberg slash notation
    in `tickers`; yfinance requires hyphens, so we query the translated
    symbol but key the returned dict by the original ticker.
    """
    end_excl = end + datetime.timedelta(days=1)
    symbol_map = {ticker: _to_yfinance_symbol(ticker) for ticker in tickers}

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
            close_s     = raw["Close"][symbol]
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


# ---------------------------------------------------------------------------
# Security lookup
# ---------------------------------------------------------------------------

def _get_securities_by_ticker(tickers: list[str]) -> dict[str, Security]:
    """Return a dict[ticker, Security] for all resolved tickers in the input."""
    rows = Security.select().where(
        Security.ticker.in_(tickers),
        Security.resolution_status == "resolved",
    )
    return {row.ticker: row for row in rows}


def _get_securities_by_cusip(cusips: list[str]) -> dict[str, Security]:
    """Return a dict[cusip, Security] for all resolved CUSIPs in the input."""
    rows = Security.select().where(
        Security.cusip.in_(cusips),
        Security.resolution_status == "resolved",
        Security.ticker.is_null(False),
    )
    return {row.cusip: row for row in rows}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_prices(
    tickers: list[str],
    start_date: DateLike,
    end_date: DateLike,
    *,
    skip_cached: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily close and adj_close for a list of tickers over [start_date, end_date].

    Tickers must exist in the Security table with resolution_status="resolved".
    Unknown tickers are skipped with a warning printed to stdout.

    Parameters
    ----------
    tickers : list[str]
        yfinance-compatible ticker symbols (e.g. ["V", "TSM", "SCHW"]).
    start_date, end_date : str or datetime.date
        Inclusive date range in ISO-8601 format or as datetime.date objects.
    skip_cached : bool
        When True (default), tickers already in PriceCache with full coverage
        are returned from the DB.  Set False to force re-fetch from yfinance.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are tickers.  Each DataFrame has columns [close, adj_close]
        indexed by datetime.date.  Tickers with no data are absent.
    """
    if not tickers:
        return {}

    init_db()
    start = _to_date(start_date)
    end   = _to_date(end_date)

    unique    = list(dict.fromkeys(tickers))
    sec_map   = _get_securities_by_ticker(unique)  # ticker → Security
    results: dict[str, pd.DataFrame] = {}
    to_fetch: list[tuple[str, Security]] = []

    for ticker in unique:
        sec = sec_map.get(ticker)
        if sec is None:
            print(f"  [prices] WARNING: '{ticker}' not in Security table (resolved) — skipping")
            continue
        if skip_cached and _is_cached(sec, start, end):
            results[ticker] = _load_from_cache(sec, start, end)
        else:
            to_fetch.append((ticker, sec))

    # Batch yfinance fetches
    for i in range(0, len(to_fetch), _BATCH_SIZE):
        batch         = to_fetch[i : i + _BATCH_SIZE]
        batch_tickers = [t for t, _ in batch]
        sec_by_ticker = {t: s for t, s in batch}

        fetched = _fetch_from_yfinance(batch_tickers, start, end)

        for ticker, df in fetched.items():
            sec = sec_by_ticker[ticker]
            _upsert_prices(sec, df)
            results[ticker] = df

        missing = set(batch_tickers) - set(fetched)
        if missing:
            print(f"  [prices] WARNING: yfinance returned no data for: {sorted(missing)}")

    return results


def fetch_prices_for_cusips(
    cusips: list[str],
    start_date: DateLike,
    end_date: DateLike,
    *,
    skip_cached: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Convenience: look up tickers for resolved CUSIPs and return price DataFrames.

    CUSIPs with no resolved ticker in the Security table are silently skipped.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are CUSIPs.  Each DataFrame has columns [close, adj_close]
        indexed by datetime.date.
    """
    if not cusips:
        return {}

    init_db()
    unique  = list(dict.fromkeys(cusips))
    sec_map = _get_securities_by_cusip(unique)  # cusip → Security

    # ticker → Security (for fetch_prices call) and cusip ← ticker (for remapping results)
    ticker_to_cusip: dict[str, str] = {sec.ticker: cusip for cusip, sec in sec_map.items()}
    tickers = list(ticker_to_cusip.keys())

    prices_by_ticker = fetch_prices(tickers, start_date, end_date, skip_cached=skip_cached)

    return {
        ticker_to_cusip[ticker]: df
        for ticker, df in prices_by_ticker.items()
    }
