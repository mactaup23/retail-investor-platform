"""
Point-in-time DCF replay: reconstructs what the DCF engine's Base-case
valuation gap would have said at a past evaluation date, for backtesting
predictive power against realized forward returns.

Why this needs to exist separately from dcf/valuation.py::run_dcf()
---------------------------------------------------------------------
run_dcf() is a "value the company right now" tool: it reads live current
price/market cap (yfinance .info), a beta from a FIXED historical window
(dashboard.factor.ticker_ff3_profile, hardcoded 2021-01-04..2024-12-31), and
the current risk-free rate (^TNX, last 5 trading days). None of those three
naturally support "what would this have said as of 2018-06-30" — unlike 13F
filings (period-dated) and PEAD earnings events (announcement-dated), DCF's
own live implementation has no time axis at all. This module reconstructs
one, reusing every piece of dcf/valuation.py and dcf/wacc.py that ISN'T
live-data-bound (compute_baseline's math, run_scenario, gordon_terminal_value,
cost_of_equity/cost_of_debt/compute_wacc) and replacing the three live-bound
inputs with point-in-time equivalents:

  1. Fundamentals as of T: fund_df filtered to filed <= T, not period_end <=
     T — a fiscal year ending 2020-12-31 isn't public until its 10-K is
     actually filed, ~60-90 days later; filtering on period_end would leak
     look-ahead. Requires `filed` on fetch_ticker_dcf_fundamentals's output
     (added alongside this module — see dcf/fundamentals.py).
  2. Price/market cap as of T: historical adjusted close at T (via
     pead.prices.fetch_prices — reuses PEAD's ~1,500-ticker price cache
     rather than standing up a second one) x diluted shares from the
     T-truncated fundamentals. This approximates market cap with the most
     recent T-known share count rather than a true point-in-time count
     (no XBRL concept this module pulls gives that directly) — same spirit
     as WACC's existing book-value-of-debt approximation.

     Share-count/price basis bug found and fixed during verification:
     pead.prices' cached price series (both `close` and `adj_close`) is
     split-adjusted to TODAY's share count, but XBRL diluted_shares is the
     TRUE nominal count as originally reported for that period — never
     retroactively adjusted for a LATER split. Dividing equity value by the
     raw XBRL share count and comparing the result to the split-adjusted
     cached price mixes two different per-share bases. Confirmed
     empirically: AAPL as_of=2018-06-29 originally computed a nonsensical
     +601.6% valuation gap — AAPL's FY2017 diluted share count (~5.25B) is
     ~4x today's post-split count, so per-share value came out ~4x too
     HIGH relative to the split-adjusted price it was compared against.
     Fixed in _shares_as_of_basis() below: multiply the raw XBRL share
     count by the product of every split ratio occurring AFTER `as_of`,
     putting it on the same today's-basis the cached price already uses.
  3. Beta as of T: a fresh trailing-window OLS ending at T, reusing
     ticker_ff3_profile's joint 7-factor regression setup verbatim but
     parameterized by end date instead of the fixed 2021-2024 window —
     that fixed window would leak future data into any T before
     2024-12-31 and go stale for any T long before 2021-01-04.
  4. Risk-free rate as of T: dcf.wacc.fetch_risk_free_rate_as_of(T),
     historical ^TNX close on/near T instead of "last 5 trading days."

Score (approved — Base case only, not a Bull/Base/Bear blend)
-----------------------------------------------------------------
    valuation_gap_pct = (base_case_per_share - price_at_T) / price_at_T
Positive means DCF says undervalued at T. A blended score would introduce a
second unvalidated methodology choice (the blend weights) stacked on top of
the reconstruction itself — deferred, same discipline as every other
"don't fit what you can't validate" choice in this codebase.

Scope of this module
---------------------
This is the per-(ticker, as_of) reconstruction engine only — analogous to
dcf/valuation.py + dcf/wacc.py being pure engines while a separate backtest
script loops over the evaluation-date grid and universe and aggregates IC
(mirrors pead/backtest.py's QuarterIC/summarize being reused by
scripts/run_composite_backtest.py's own loop, rather than duplicated).
"""

from __future__ import annotations

import datetime

import pandas as pd
import yfinance as yf

from dcf.exclusions import check_business_model_fit
from dcf.fundamentals import fetch_ticker_dcf_fundamentals
from dcf.valuation import compute_baseline, run_scenario
from dcf.wacc import compute_wacc, cost_of_debt, cost_of_equity, fetch_risk_free_rate_as_of

BETA_LOOKBACK_YEARS = 3   # trailing window for the point-in-time beta regression
MIN_BETA_OBS = 60          # mirrors ticker_ff3_profile's own floor
_PRICE_TOLERANCE_DAYS = 10  # max gap between as_of and the nearest priced row on/before it


def _beta_as_of(
    ticker: str,
    as_of: datetime.date,
    lookback_years: int = BETA_LOOKBACK_YEARS,
    returns: "pd.Series | None" = None,
    factors: "pd.DataFrame | None" = None,
) -> "float | None":
    """
    Trailing-window market-factor beta ending at `as_of`, reusing the same
    FF7-daily-panel joint OLS setup as dashboard.factor.ticker_ff3_profile
    (same 7 factors fit jointly, same beta_market = params["mkt_excess"]
    extraction) so this is the same beta definition, just parameterized by
    end date instead of that function's fixed 2021-2024 window.

    A lookback window starting before GP's own 2013 coverage (e.g. any
    as_of before ~2016) will see its effective window narrowed to GP's
    covered subset after dropna() — the same already-documented behavior
    get_ff7_daily's own docstring describes for any caller doing a single
    joint fit across the full panel. Not a new limitation introduced here.

    returns/factors may be passed in pre-fetched — a full-date-range daily
    log-return Series for this ticker (any adjusted-close-based series is
    fine; factor_engine.data_loader.load_returns' own output works, but so
    does a series derived locally from an already-fetched adjusted-close
    price DataFrame — see scripts/run_dcf_pilot_backtest.py, which
    deliberately derives it from pead.prices' already-batched fetch rather
    than a second independent per-ticker call through data_loader, to cut
    total yfinance call volume at backtest scale) and the FF7 daily factor
    panel (factor_engine.french_data.get_ff7_daily output) spanning the
    whole backtest's date range — so a caller evaluating many as_of dates
    for the same ticker slices the trailing window from an already-loaded
    frame in memory instead of hitting the network on every call. This
    matters more than a usual "pass it in to save a fetch" optimization:
    load_prices' on-disk cache is keyed by the EXACT (ticker, start, end)
    window requested, and a trailing window's start/end both shift every
    quarter — so without pre-fetching, every single (ticker, as_of) pair
    would miss that cache and trigger a fresh fetch, defeating it entirely
    at backtest scale (e.g. 300 tickers x ~48 quarters = ~14,400 avoidable
    fetches). If omitted, falls back to a fresh single-window fetch via
    factor_engine.data_loader (fine for one-off/manual calls, not for a
    backtest loop).

    Returns None for insufficient history (< MIN_BETA_OBS overlapping
    trading days) — same "insufficient data" contract as ticker_ff3_profile.
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    def _date_indexed(s):
        """
        Normalize to a plain datetime.date index. Callers may pass a
        DatetimeIndex series (factor_engine.data_loader / get_ff7_daily's
        own output) or one already indexed by plain datetime.date objects
        (e.g. a return series derived from pead.prices' cached frames,
        which use a plain date Index, not a DatetimeIndex) — the two don't
        align on .join() if mixed (a DatetimeIndex timestamp never equals a
        plain date object for pandas index-matching purposes), so both
        sides must land on the same index type before slicing/joining.
        """
        if isinstance(s.index, pd.DatetimeIndex):
            s = s.copy()
            s.index = s.index.date
        return s

    window_start = as_of - datetime.timedelta(days=365 * lookback_years)

    if returns is None or factors is None:
        from factor_engine.data_loader import load_returns
        from factor_engine.french_data import get_ff7_daily
        try:
            factors_full = get_ff7_daily(window_start.isoformat(), as_of.isoformat())
            rets = load_returns([ticker], window_start.isoformat(), as_of.isoformat())
        except Exception:
            return None
        if ticker not in rets.columns or rets[ticker].isna().all():
            return None
        returns_full = rets[ticker]
    else:
        returns_full, factors_full = returns, factors

    returns_full = _date_indexed(returns_full)
    factors_full = _date_indexed(factors_full)
    stock_returns = returns_full[(returns_full.index >= window_start) & (returns_full.index <= as_of)]
    factors_window = factors_full[(factors_full.index >= window_start) & (factors_full.index <= as_of)]

    aligned = stock_returns.to_frame("stock").join(factors_window, how="inner").dropna()
    if len(aligned) < MIN_BETA_OBS:
        return None

    excess = aligned["stock"] - aligned["rf"]
    factor_cols = ["mkt_excess", "smb", "hml", "rmw", "cma", "mom", "gp"]
    X = add_constant(aligned[factor_cols])
    m = OLS(excess, X).fit()
    return float(m.params["mkt_excess"])


def _shares_as_of_basis(ticker: str, raw_shares: float, as_of: datetime.date, splits: "pd.Series | None" = None) -> float:
    """
    Convert a raw XBRL diluted-share count (true nominal count as originally
    reported for that period) onto the same today's-basis pead.prices'
    cached price series already uses (split-adjusted to the CURRENT share
    count). A split multiplies share count — one pre-split share becomes N
    post-split shares — so a split dated after `as_of` means today's actual
    share count is N times what was truly outstanding at `as_of`; raw_shares
    must grow by that same factor to land on the cached price's basis.
    Multiplies raw_shares by the product of every split ratio
    (yf.Ticker(ticker).splits) dated after `as_of`. No splits after `as_of`
    (or a ticker with no split history) leaves raw_shares unchanged (product
    of an empty set = 1.0).
    """
    if splits is None:
        from yfinance_client import call_with_backoff
        splits = call_with_backoff(lambda: yf.Ticker(ticker).splits)
    if splits is None or splits.empty:
        return raw_shares
    future_splits = splits[splits.index.date > as_of]
    factor = float(future_splits.prod()) if not future_splits.empty else 1.0
    return raw_shares * factor


def _price_as_of(prices: pd.DataFrame, as_of: datetime.date, tolerance_days: int = _PRICE_TOLERANCE_DAYS) -> "float | None":
    """Most recent adjusted close on or before as_of, within tolerance_days — never a future price."""
    eligible = prices[prices.index <= as_of]
    if eligible.empty:
        return None
    last_date = eligible.index.max()
    if (as_of - last_date).days > tolerance_days:
        return None   # data gap (delisted / no coverage near this date) — not a genuine as-of price
    return float(eligible.loc[last_date, "adj_close"])


def compute_point_in_time_dcf(
    ticker: str,
    as_of: datetime.date,
    fund_df: "pd.DataFrame | None" = None,
    prices: "pd.DataFrame | None" = None,
    splits: "pd.Series | None" = None,
    returns: "pd.Series | None" = None,
    factors: "pd.DataFrame | None" = None,
    risk_free_rate: "float | None" = None,
) -> dict:
    """
    Point-in-time DCF Base-case valuation gap for one (ticker, as_of) pair.

    fund_df / prices / splits / returns / factors may be passed in
    pre-fetched — fetch_ticker_dcf_fundamentals's full output / a single
    ticker's pead.prices.fetch_prices frame / yf.Ticker(ticker).splits / a
    full-range daily log-return Series (see _beta_as_of's docstring for
    acceptable sources) / a full-range factor_engine.french_data.get_ff7_daily
    panel — so a caller looping over
    many as_of dates for the same ticker (the normal backtest access
    pattern) doesn't re-fetch any of them on every iteration; all five are
    per-ticker (factors is shared across every ticker), not per-(ticker,
    date). See _beta_as_of's docstring for why pre-fetching returns/factors
    specifically isn't optional at backtest scale. risk_free_rate may also
    be passed in directly (it depends only on `as_of`, not `ticker`, so a
    caller evaluating many tickers at the same as_of should compute it once
    and reuse it rather than calling fetch_risk_free_rate_as_of per ticker).
    Any omitted value is fetched fresh for this single call.

    Returns a dict always containing "ticker" and "as_of"; on any
    data-quality failure also contains "error" (one of:
    unsuitable_business_model, no_xbrl_fundamentals, insufficient_history,
    no_diluted_shares, no_beta, no_price) and no "valuation_gap_pct" —
    same pattern as run_dcf(), callers should check for "error" before
    reading the score.
    """
    unsuitable_reason = check_business_model_fit(ticker)
    if unsuitable_reason is not None:
        return {"ticker": ticker, "as_of": as_of, "error": "unsuitable_business_model", "reason": unsuitable_reason}

    if fund_df is None:
        fund_df = fetch_ticker_dcf_fundamentals(ticker)
    if fund_df.empty:
        return {"ticker": ticker, "as_of": as_of, "error": "no_xbrl_fundamentals"}

    as_of_str = as_of.isoformat()
    known = fund_df[fund_df["filed"].notna() & (fund_df["filed"] <= as_of_str)]
    baseline = compute_baseline(known)
    if baseline is None:
        return {"ticker": ticker, "as_of": as_of, "error": "insufficient_history"}
    if not baseline["diluted_shares"]:
        return {"ticker": ticker, "as_of": as_of, "error": "no_diluted_shares"}

    beta = _beta_as_of(ticker, as_of, returns=returns, factors=factors)
    if beta is None:
        return {"ticker": ticker, "as_of": as_of, "error": "no_beta"}

    if prices is None:
        from pead.prices import fetch_prices
        window_start = as_of - datetime.timedelta(days=_PRICE_TOLERANCE_DAYS + 5)
        fetched = fetch_prices([ticker], window_start, as_of)
        prices = fetched.get(ticker)
    price_at_as_of = _price_as_of(prices, as_of) if prices is not None else None
    if price_at_as_of is None:
        return {"ticker": ticker, "as_of": as_of, "error": "no_price"}

    if risk_free_rate is None:
        risk_free_rate = fetch_risk_free_rate_as_of(as_of)
    if risk_free_rate is None:
        return {"ticker": ticker, "as_of": as_of, "error": "no_risk_free_rate"}

    # See module docstring's "Share-count/price basis bug" note — raw XBRL
    # diluted_shares must be converted onto the same today's-basis the
    # cached price series uses before it's used for market cap OR as the
    # divisor for per-share value, or a ticker with any split between
    # `as_of` and today produces a nonsensical valuation gap.
    baseline = dict(baseline)
    baseline["diluted_shares"] = _shares_as_of_basis(ticker, baseline["diluted_shares"], as_of, splits)

    market_cap = price_at_as_of * baseline["diluted_shares"]
    re = cost_of_equity(beta, risk_free_rate)
    rd = cost_of_debt(baseline["interest_expense"], baseline["total_debt"], baseline["tax_rate"])
    wacc = compute_wacc(market_cap, baseline["total_debt"], re, rd)

    scenario = run_scenario("base", baseline["start_growth"], baseline, wacc, risk_free_rate)
    per_share = scenario["per_share"]
    if per_share is None:
        return {"ticker": ticker, "as_of": as_of, "error": "no_per_share"}

    return {
        "ticker":               ticker,
        "as_of":                as_of,
        "price_at_as_of":       price_at_as_of,
        "beta":                 beta,
        "risk_free_rate":       risk_free_rate,
        "wacc":                 wacc,
        "base_per_share":       per_share,
        "valuation_gap_pct":    (per_share - price_at_as_of) / price_at_as_of,
        "n_annual_observations": baseline["n_annual_observations"],
        "most_recent_period":   baseline["most_recent_period"],
    }
