"""
Per-ticker fundamental data fetcher for the Gross Profitability (GP) factor.

Novy-Marx (2013) gross profitability: GP_ratio = (Revenue - COGS) / Total Assets

Data source and its hard limitation
------------------------------------
yfinance's free endpoint exposes at most ~5 years of annual financial
statements and ~5 quarters of quarterly statements per ticker — this is a
hard wall of the free data source, not a rate-limit or caching artifact.
Verified empirically (2026-07-09): AAPL's annual income_stmt/balance_sheet
columns start at FY2021; quarterly statements start ~5 quarters back.  This
bounds the GP factor to roughly 2021-present (see factor_engine/factors/gp.py
GP_FACTOR_START) — meaningfully shorter than the Ken French RMW/CMA series
(full history to 1963) or the fund-skill regression's typical sample window
(13F data from 2013 onward).  GP factor loadings are therefore inherently
less statistically reliable than RMW/CMA/MOM loadings, which is reflected in
the two-tier regression design in smart_money/factor_apply.py and should be
labeled "Gross Profitability (2021-present)" wherever displayed.

Fetch strategy
--------------
Both annual and quarterly statements are pulled per ticker (yfinance doesn't
support batch-fetching financials the way it does price history — one ticker
at a time is unavoidable).  Observations from both are merged into a single
per-ticker timeline; factor_engine/factors/gp.py picks whichever observation
is freshest-as-of each quarterly rebalance date, respecting a reporting lag.

Resumability: each ticker's result (or None-marker for tickers with no usable
data) is written to data/gp/fundamentals/{ticker}.csv immediately on fetch.
fetch_universe_fundamentals() skips any ticker that already has a cache file,
so a full ~1500-ticker run (45-90 minutes) only ever pays its cost once —
interrupting and re-running picks up where it left off.
"""

import os
import time

import pandas as pd
import yfinance as yf

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gp", "fundamentals")

THROTTLE_SECONDS = 0.4          # delay between per-ticker fetches
MAX_RETRIES = 3
BACKOFF_SECONDS = (2, 8, 20)    # exponential-ish backoff per retry attempt

_REVENUE_ROW = "Total Revenue"
_COGS_ROW = "Cost Of Revenue"
_ASSETS_ROW = "Total Assets"

# A ticker with this exact single-row sentinel cached means "fetched, but no
# usable data" — distinguishes a permanently-empty result (don't retry it on
# every resumed run) from "never attempted" (no cache file at all).
_EMPTY_SENTINEL = pd.DataFrame({"period_end": [], "revenue": [], "cogs": [], "total_assets": [], "gp_ratio": [], "freq": []})


def _cache_path(ticker: str) -> str:
    return os.path.join(_CACHE_DIR, f"{ticker}.csv")


def _extract_observations(income: pd.DataFrame, balance: pd.DataFrame, freq: str) -> list[dict]:
    """
    Build raw per-period (revenue, cogs, total_assets) observations from one
    income_stmt/balance_sheet pair.

    For freq="A" these are already valid annual GP_ratio observations
    (Revenue and COGS are 12-month flows, matching the Novy-Marx spec).  For
    freq="Q" these are single-quarter flows and must NOT be used directly —
    see _ttm_from_quarterly_rows() below, which converts them to trailing-
    twelve-month observations before they're usable.
    """
    if income is None or income.empty or balance is None or balance.empty:
        return []
    if _REVENUE_ROW not in income.index or _COGS_ROW not in income.index:
        return []
    if _ASSETS_ROW not in balance.index:
        return []

    obs = []
    shared_periods = [c for c in income.columns if c in balance.columns]
    for period_end in shared_periods:
        revenue = income.at[_REVENUE_ROW, period_end]
        cogs = income.at[_COGS_ROW, period_end]
        assets = balance.at[_ASSETS_ROW, period_end]
        if pd.isna(revenue) or pd.isna(cogs) or pd.isna(assets) or assets == 0:
            continue
        obs.append({
            "period_end":   pd.Timestamp(period_end).date().isoformat(),
            "revenue":      float(revenue),
            "cogs":         float(cogs),
            "total_assets": float(assets),
            "gp_ratio":     float((revenue - cogs) / assets),
            "freq":         freq,
        })
    return obs


# A single quarter's Revenue/COGS is a ~3-month flow; total_assets is a
# point-in-time balance. Dividing a 1-quarter flow by an annual-scale balance
# understates gp_ratio by roughly 4x relative to a company reporting on an
# annual observation — a units mismatch, not a real profitability
# difference, that silently corrupted quintile ranking whenever a
# quarterly-frequency observation was selected as "most recent" alongside
# peers using annual-frequency observations. Fixed by summing 4 consecutive
# quarters' Revenue/COGS into a trailing-twelve-month flow before dividing by
# total_assets — the Novy-Marx (2013) annual specification applied at
# quarterly refresh cadence, scale-consistent with the annual observations.
_TTM_MIN_SPAN_DAYS = 250
_TTM_MAX_SPAN_DAYS = 320


def _ttm_from_quarterly_rows(quarterly_obs: pd.DataFrame) -> list[dict]:
    """
    Convert single-quarter (revenue, cogs, total_assets) rows into trailing-
    twelve-month gp_ratio observations.

    For each run of 4 consecutive quarters (by period_end), sums revenue and
    cogs across the window and divides by total_assets as of the most recent
    quarter in that window. Windows must span 250-320 days end-to-end (3
    quarter-gaps of ~90 days each ≈ 273 days) — a wider or narrower span
    means the "4 rows" aren't actually 4 consecutive fiscal quarters (a data
    gap), and that window is skipped rather than silently mixing periods.
    """
    if quarterly_obs is None or quarterly_obs.empty:
        return []

    q = quarterly_obs.sort_values("period_end").reset_index(drop=True)
    dates = pd.to_datetime(q["period_end"])

    obs = []
    for i in range(3, len(q)):
        window = q.iloc[i - 3:i + 1]
        span_days = (dates.iloc[i] - dates.iloc[i - 3]).days
        if not (_TTM_MIN_SPAN_DAYS <= span_days <= _TTM_MAX_SPAN_DAYS):
            continue

        rev_ttm = float(window["revenue"].sum())
        cogs_ttm = float(window["cogs"].sum())
        assets = float(window["total_assets"].iloc[-1])
        if assets == 0:
            continue

        obs.append({
            "period_end":   q.at[i, "period_end"],
            "revenue":      rev_ttm,
            "cogs":         cogs_ttm,
            "total_assets": assets,
            "gp_ratio":     (rev_ttm - cogs_ttm) / assets,
            "freq":         "Q",
        })
    return obs


def _fetch_one(ticker: str) -> pd.DataFrame:
    """Fetch and combine annual + TTM-quarterly observations for a single ticker."""
    t = yf.Ticker(ticker)
    annual_obs = _extract_observations(t.income_stmt, t.balance_sheet, "A")
    quarterly_raw = _extract_observations(t.quarterly_income_stmt, t.quarterly_balance_sheet, "Q")
    quarterly_ttm_obs = _ttm_from_quarterly_rows(pd.DataFrame(quarterly_raw)) if quarterly_raw else []
    observations = annual_obs + quarterly_ttm_obs

    if not observations:
        return _EMPTY_SENTINEL.copy()

    df = pd.DataFrame(observations).drop_duplicates(subset="period_end").sort_values("period_end")
    return df.reset_index(drop=True)


def fetch_ticker_fundamentals(ticker: str, force: bool = False) -> "pd.DataFrame | None":
    """
    Fetch (or load cached) fundamental observations for one ticker.

    Returns a DataFrame with columns [period_end, revenue, cogs, total_assets,
    gp_ratio, freq], or an empty DataFrame if the ticker has no usable data
    (delisted, missing statements, etc.) — never raises for a single bad
    ticker; retries transient failures with backoff before giving up.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache = _cache_path(ticker)
    if not force and os.path.exists(cache):
        return pd.read_csv(cache)

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            df = _fetch_one(ticker)
            df.to_csv(cache, index=False)
            return df
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_SECONDS[attempt])

    # Persistent failure after retries: cache the empty sentinel so a resumed
    # run doesn't keep re-attempting a ticker that's going to keep failing
    # (e.g. delisted/renamed), but log so it's visible this wasn't "no data"
    # so much as "gave up".
    print(f"  [gp_fundamentals] {ticker}: failed after {MAX_RETRIES} attempts ({last_error!r}); caching as empty")
    _EMPTY_SENTINEL.to_csv(cache, index=False)
    return _EMPTY_SENTINEL.copy()


def fetch_universe_fundamentals(
    tickers: list[str],
    force: bool = False,
    throttle_seconds: float = THROTTLE_SECONDS,
) -> dict[str, pd.DataFrame]:
    """
    Fetch fundamentals for a full ticker universe, resumable and throttled.

    Skips tickers that already have a cache file unless force=True.  A full
    ~1500-ticker first run takes 45-90 minutes (one HTTP round trip per
    ticker — yfinance has no batch financials API), so this prints verbosely:
    every ticker that's actually being fetched over the network (not loaded
    from cache) gets its own line with a running counter, remaining count,
    and a rolling ETA once enough samples exist to estimate a rate — so a
    long run is visibly progressing rather than looking stalled. Tickers
    already cached (a resumed run) are counted but not logged individually,
    to avoid flooding output on a mostly-complete resume.

    Returns dict[ticker -> observations DataFrame] (empty DataFrame for
    tickers with no usable data).
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    n_fetched = 0
    n_cached = 0
    n_empty = 0
    start_time = time.time()
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        already_cached = os.path.exists(_cache_path(ticker)) and not force

        if already_cached:
            n_cached += 1
            results[ticker] = fetch_ticker_fundamentals(ticker, force=force)
            continue

        remaining = total - i - 1
        elapsed = time.time() - start_time
        eta_str = ""
        if n_fetched >= 5:
            rate = elapsed / n_fetched  # seconds per freshly-fetched ticker
            eta_min = (rate * remaining) / 60.0
            eta_str = f", ETA ~{eta_min:.0f} min"
        print(f"  [gp_fundamentals] fetching {ticker:<8s} "
              f"({i + 1}/{total}, {remaining} remaining{eta_str})...", flush=True)

        df = fetch_ticker_fundamentals(ticker, force=force)
        results[ticker] = df
        n_fetched += 1
        if df.empty:
            n_empty += 1
            print(f"  [gp_fundamentals]   -> {ticker}: no usable data", flush=True)
        else:
            print(f"  [gp_fundamentals]   -> {ticker}: {len(df)} observation(s) "
                  f"({df['period_end'].min()} to {df['period_end'].max()})", flush=True)

        time.sleep(throttle_seconds)

        if n_fetched % 50 == 0:
            elapsed_min = elapsed / 60.0
            print(f"  [gp_fundamentals] === checkpoint: {i + 1}/{total} processed, "
                  f"{n_fetched} freshly fetched ({n_empty} empty), "
                  f"{elapsed_min:.1f} min elapsed ===", flush=True)

    total_elapsed_min = (time.time() - start_time) / 60.0
    print(f"  [gp_fundamentals] Done in {total_elapsed_min:.1f} min. "
          f"{n_fetched} tickers freshly fetched ({n_empty} with no usable data), "
          f"{n_cached} loaded from cache.")
    return results
