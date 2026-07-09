"""
GP (Gross Profitability) factor — proprietary quintile long/short construction.

Novy-Marx (2013) specification
-------------------------------
    GP_ratio = (Revenue - COGS) / Total Assets

Novy-Marx's central finding: gross profitability is at least as powerful a
predictor of the cross-section of returns as traditional value metrics, and
— critically for this platform — captures genuine business-economics quality
without the growth-stage penalty that cash-flow-based quality metrics carry.

Why GP instead of FCF yield
----------------------------
FCF yield (TTM free cash flow / enterprise value, as displayed per-security
in app_pages/signals.py) was considered and rejected as a factor input.  FCF
yield systematically penalizes companies reinvesting heavily for growth —
high capex and working-capital investment suppress free cash flow even when
the underlying unit economics are excellent, so a growth-stage compounder
screens as "low quality" purely because it's investing.  Gross profitability
sits above the investment decision on the income statement (revenue minus
cost of goods sold, before capex/opex/reinvestment), so it captures
production-level economic quality without conflating it with a company's
capital allocation stage.  This matters directly for the skill-scoring
thesis: a fund holding aggressive reinvesting growth names (e.g. AMZN, NVDA)
should not be penalized on a "quality" dimension for capital intensity that
FCF yield would flag as weak.

Universe and construction
--------------------------
Universe : ~1500 US equities (S&P Composite 1500 proxy) — see
           factor_engine/gp_universe.py for sourcing rationale.
Data     : per-ticker annual + quarterly fundamentals — see
           factor_engine/gp_fundamentals.py for fetch/cache/retry design.
Rebalance: quarterly (calendar quarter boundaries).  At each rebalance date,
           each ticker's most recent GP_ratio observation is used, subject to
           REPORTING_LAG_DAYS — the observation's period_end must be at least
           that many days before the rebalance date, so the factor never uses
           data that wouldn't actually have been public yet (this mirrors the
           EDGAR 45-day 13F lag already used elsewhere in this platform,
           though 10-K/10-Q filing deadlines are a different, generally
           longer, window — 90 days is a conservative flat lag covering both
           accelerated and non-accelerated filers without needing filer-size
           classification data).
Portfolio: equal-weighted, long top quintile / short bottom quintile by
           GP_ratio, held until the next rebalance.

Historical coverage — hard limitation, not a bug
--------------------------------------------------
yfinance's free fundamentals endpoint exposes at most ~5 years of annual
statements and ~5 quarters of quarterly statements per ticker (verified
empirically; see factor_engine/gp_fundamentals.py docstring).  This bounds
GP_FACTOR coverage to roughly 2021-present — meaningfully shorter than the
Ken French RMW/CMA series (full history to 1963) or the ETF-proxy MOM series
(MTUM inception 2013).  GP factor loadings are inherently less statistically
reliable than RMW/CMA/MOM loadings estimated over their much longer windows.
Label this factor "Gross Profitability (2021-present)" wherever displayed —
never present it alongside RMW/CMA/MOM without that caveat.  Consumers that
need a coverage boundary should call get_gp_coverage_start().

Regression usage
-----------------
compute_factor_loadings() in factor_engine/factors/hml.py joins this into the
ETF-proxy 5-factor OLS (mkt+smb+hml+mom+gp) for individual holdings.
smart_money/factor_apply.py uses a two-tier regression: the primary
mkt+smb+hml+mom+rmw+cma fit runs over each fund's full available quarterly
history (unchanged sample depth vs. the prior FF6 spec), while beta_gp is
estimated from a secondary fit restricted to the GP-covered subset of those
quarters — see that module's docstring for why a single joint fit isn't
possible when GP data doesn't span the same window as the other six factors.
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from factor_engine.gp_fundamentals import fetch_universe_fundamentals
from factor_engine.gp_universe import get_universe_tickers

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "gp")
_FACTOR_CACHE = os.path.join(_CACHE_DIR, "gp_factor_daily.csv")
_PRICE_CACHE_TEMPLATE = os.path.join(_CACHE_DIR, "prices_bulk_{start}_{end}.csv")

REPORTING_LAG_DAYS = 90
MIN_UNIVERSE_FOR_REBALANCE = 100   # skip a rebalance date if fewer valid GP_ratio observations exist
QUINTILE = 0.20
_PRICE_CHUNK_SIZE = 200


# ---------------------------------------------------------------------------
# Rebalance scheduling and GP_ratio selection
# ---------------------------------------------------------------------------

def _quarterly_rebalance_dates(floor: date, ceiling: date) -> list[pd.Timestamp]:
    """Calendar-quarter-start rebalance dates (Jan/Apr/Jul/Oct 1) within [floor, ceiling]."""
    dates = []
    year, month = floor.year, ((floor.month - 1) // 3) * 3 + 1
    cursor = pd.Timestamp(year=year, month=month, day=1)
    while cursor.date() <= ceiling:
        if cursor.date() >= floor:
            dates.append(cursor)
        cursor = cursor + pd.DateOffset(months=3)
    return dates


def _select_gp_ratio(obs: pd.DataFrame, asof: pd.Timestamp) -> "float | None":
    """Most recent GP_ratio observation whose reporting lag has elapsed by `asof`."""
    if obs is None or obs.empty or "period_end" not in obs.columns:
        return None
    eligible = obs[pd.to_datetime(obs["period_end"]) + pd.Timedelta(days=REPORTING_LAG_DAYS) <= asof]
    if eligible.empty:
        return None
    return float(eligible.sort_values("period_end").iloc[-1]["gp_ratio"])


def _build_rebalance_assignments(
    fundamentals: dict[str, pd.DataFrame],
    rebalance_dates: list[pd.Timestamp],
) -> list[dict]:
    """
    For each rebalance date, quintile-sort the universe by GP_ratio and assign
    long/short baskets.  Skips any rebalance date with too few valid
    observations (early in the sample, before enough tickers have a filing).
    """
    assignments = []
    for rb_date in rebalance_dates:
        ratios = {}
        for ticker, obs in fundamentals.items():
            r = _select_gp_ratio(obs, rb_date)
            if r is not None:
                ratios[ticker] = r

        if len(ratios) < MIN_UNIVERSE_FOR_REBALANCE:
            continue

        ranked = sorted(ratios.items(), key=lambda kv: kv[1])
        n = len(ranked)
        q = max(1, int(n * QUINTILE))
        short_tickers = [t for t, _ in ranked[:q]]
        long_tickers = [t for t, _ in ranked[-q:]]
        assignments.append({"rebalance_date": rb_date, "long": long_tickers, "short": short_tickers})

    return assignments


# ---------------------------------------------------------------------------
# Bulk price fetch (batched — yfinance supports multi-ticker downloads natively,
# unlike financial statements which require one request per ticker)
# ---------------------------------------------------------------------------

def _fetch_universe_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache = _PRICE_CACHE_TEMPLATE.format(start=start, end=end)
    if os.path.exists(cache):
        print(f"  [gp] Price cache hit: {cache}")
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    n_chunks = (len(tickers) + _PRICE_CHUNK_SIZE - 1) // _PRICE_CHUNK_SIZE
    frames = []
    for i in range(0, len(tickers), _PRICE_CHUNK_SIZE):
        chunk_num = i // _PRICE_CHUNK_SIZE + 1
        chunk = tickers[i:i + _PRICE_CHUNK_SIZE]
        print(f"  [gp]   price chunk {chunk_num}/{n_chunks} ({len(chunk)} tickers)...", flush=True)
        raw = yf.download(chunk, start=start, end=end, auto_adjust=True, progress=False, group_by="column")
        if raw.empty:
            continue
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": chunk[0]})
        frames.append(close)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.index.name = "date"
    prices.to_csv(cache)
    return prices


# ---------------------------------------------------------------------------
# Full-history construction (build once, cache, slice on read — same
# philosophy as factor_engine/french_data.py's immutable-history cache)
# ---------------------------------------------------------------------------

def _build_full_history() -> pd.DataFrame:
    print("  [gp] Stage 1/4: loading stock universe...")
    universe = get_universe_tickers()
    print(f"  [gp] Universe loaded: {len(universe)} tickers")

    print(f"  [gp] Stage 2/4: fetching fundamentals for {len(universe)} tickers "
          f"(first run: ~45-90 min; resumed runs: fast, cached per-ticker)...")
    fundamentals = fetch_universe_fundamentals(universe)

    earliest_period_end = min(
        (pd.to_datetime(df["period_end"]).min() for df in fundamentals.values() if not df.empty),
        default=None,
    )
    if earliest_period_end is None:
        print("  [gp] No usable fundamentals found across the universe — aborting build.")
        return pd.DataFrame(columns=["long_return", "short_return", "gp"])

    floor = (earliest_period_end + pd.Timedelta(days=REPORTING_LAG_DAYS)).date()
    ceiling = date.today()
    rebalance_dates = _quarterly_rebalance_dates(floor, ceiling)
    print(f"  [gp] Stage 3/4: building quarterly quintile rebalance assignments "
          f"({len(rebalance_dates)} candidate rebalance dates from {floor} to {ceiling})...")
    assignments = _build_rebalance_assignments(fundamentals, rebalance_dates)
    print(f"  [gp] {len(assignments)} rebalance date(s) had enough universe coverage "
          f"(>= {MIN_UNIVERSE_FOR_REBALANCE} names) to form long/short baskets")

    if len(assignments) < 2:
        print("  [gp] Fewer than 2 valid rebalance dates — not enough coverage yet to "
              "construct a factor series. Aborting build.")
        return pd.DataFrame(columns=["long_return", "short_return", "gp"])

    needed_tickers = sorted({t for a in assignments for t in (a["long"] + a["short"])})
    price_start = assignments[0]["rebalance_date"].date().isoformat()
    price_end = (ceiling + timedelta(days=1)).isoformat()
    print(f"  [gp] Stage 4/4: fetching price history for {len(needed_tickers)} long/short "
          f"basket tickers ({price_start} to {price_end}, batched)...")
    prices = _fetch_universe_prices(needed_tickers, price_start, price_end)
    if prices.empty:
        print("  [gp] No price data returned — aborting build.")
        return pd.DataFrame(columns=["long_return", "short_return", "gp"])
    print(f"  [gp] Price data loaded for {len(prices.columns)} tickers, "
          f"{len(prices)} trading days. Assembling daily GP series...")

    log_returns = np.log(prices / prices.shift(1))

    period_frames = []
    for i, a in enumerate(assignments):
        window_start = a["rebalance_date"]
        window_end = assignments[i + 1]["rebalance_date"] if i + 1 < len(assignments) else pd.Timestamp(ceiling)
        window = log_returns.loc[(log_returns.index >= window_start) & (log_returns.index < window_end)]
        if window.empty:
            continue

        long_cols = [t for t in a["long"] if t in window.columns]
        short_cols = [t for t in a["short"] if t in window.columns]
        if not long_cols or not short_cols:
            continue

        long_return = window[long_cols].mean(axis=1)
        short_return = window[short_cols].mean(axis=1)
        period_frames.append(pd.DataFrame({
            "long_return":  long_return,
            "short_return": short_return,
            "gp":           long_return - short_return,
        }))

    if not period_frames:
        return pd.DataFrame(columns=["long_return", "short_return", "gp"])

    full = pd.concat(period_frames).sort_index()
    full = full[~full.index.duplicated(keep="first")].dropna()
    full.index.name = "date"

    os.makedirs(_CACHE_DIR, exist_ok=True)
    full.to_csv(_FACTOR_CACHE)
    return full


def build_gp_factor(start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """
    Return the daily GP factor series for [start, end].

    Returns a DataFrame with columns:
        long_return  — equal-weighted daily log return of the top-quintile basket
        short_return — equal-weighted daily log return of the bottom-quintile basket
        gp           — long_return - short_return (the GP factor)

    Coverage is bounded to roughly 2021-present (see module docstring) —
    requesting an earlier start simply yields fewer rows, not an error.
    Full history is built once and cached to data/gp/gp_factor_daily.csv;
    pass refresh=True to force a full rebuild (re-fetches fundamentals for
    any ticker not already cached, then reconstructs the factor).
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    if not refresh and os.path.exists(_FACTOR_CACHE):
        full = pd.read_csv(_FACTOR_CACHE, index_col=0, parse_dates=True)
        full.index.name = "date"
    else:
        full = _build_full_history()

    if full.empty:
        return full

    mask = (full.index >= pd.Timestamp(start)) & (full.index <= pd.Timestamp(end))
    return full.loc[mask].copy()


def get_gp_coverage_start() -> "pd.Timestamp | None":
    """
    Earliest date the constructed GP factor actually covers, or None if the
    factor hasn't been built yet.  Used by callers (stress_test.py,
    factor_apply.py) that need to know whether a given date range/quarter
    falls within GP's coverage window before including it in a regression.
    """
    if not os.path.exists(_FACTOR_CACHE):
        return None
    full = pd.read_csv(_FACTOR_CACHE, index_col=0, parse_dates=True)
    return full.index.min() if not full.empty else None
