"""
DCF standalone pilot backtest — does the point-in-time Base-case valuation
gap predict forward returns, and at what horizon?

Context (see CLAUDE.md's DCF Valuation Engine section and dcf/backtest.py's
module docstring): run_dcf() only values a company "as of right now" — no
existing infrastructure gives it a time axis the way 13F filing periods or
PEAD announcement dates naturally have one. dcf/backtest.py builds that
point-in-time reconstruction; this script is the first real backtest run
against it, deliberately scoped to a ~300-ticker PILOT (not the full
~1,500-ticker universe) so a methodology bug — like the share-count/price
basis bug dcf/backtest.py's own verification pass already caught and fixed
(AAPL's original +601.6% valuation gap) — gets caught cheaply rather than
after burning the full universe's compute.

Horizon framing (stated up front, not discovered after a null short-horizon
result)
------------------------------------------------------------------------
DCF/value-based signals are academically documented to work on LONGER
horizons than 13F convergence or PEAD drift — price converging to
fundamental value is a slow, multi-quarter process, the opposite horizon
profile from momentum-style signals. A weak or noisy 1-month IC here is
expected and should NOT be read as "DCF doesn't work" the way it would for
13F/PEAD; the horizon where (if anywhere) an edge shows up is itself part of
what this pilot is checking. Tested across all six of
smart_money/backtest.py's HORIZONS_TRADING_DAYS (21/63/126/168/210/252
trading days = 1/3/6/8/10/12 months), not just PEAD's narrower (21, 63) —
free reuse of an existing horizon set, and gives a full curve rather than
two endpoints.

Universe
--------
Stratified sample from factor_engine.gp_universe's ~1,506-ticker S&P
Composite 1500 universe (same universe PEAD and the GP factor already use),
proportionally sampled across its three size tiers (index_source: sp500 /
sp400 / sp600) so the pilot isn't accidentally skewed toward only large-caps
or only small-caps. Business-model-unsuitable tickers (banks, insurers,
REITs — dcf/exclusions.py) are dropped up front, same as any standalone
DCF run.

Evaluation grid
----------------
Calendar quarter-ends from 2014-06-30 (matching PEAD's and the composite
test's own "cite the representative window" 2014Q2+ convention) through the
most recently completed quarter. No separate knowledge-date lag is layered
on top of the quarter-end itself, unlike 13F's +45 days — every input
dcf/backtest.py's compute_point_in_time_dcf() resolves for a given as_of
(fundamentals truncated by ACTUAL FILING DATE, not period_end; price/beta/
risk-free-rate all historical AS OF that date) is already knowledge-safe by
construction, so the quarter-end itself is a legitimate decision date. Rows
lacking enough forward price history for a given horizon (recent quarters,
delisted tickers) are dropped by the same per-observation coverage gate
every other backtest in this codebase uses — no separate cutoff needed.

Score
-----
valuation_gap_pct only (Base case, not a Bull/Base/Bear blend — approved,
see CLAUDE.md). Positive = DCF says undervalued as of that date.

Performance / rate-limiting note
---------------------------------
Fetches full-date-range prices/returns ONCE per ticker (and the FF7 factor
panel ONCE, total) and reuses them across every quarterly as_of for that
ticker, per dcf/backtest.py's _beta_as_of docstring — a naive per-(ticker,
quarter) fetch would defeat load_prices' (ticker, start, end)-keyed cache
entirely at this scale.

Two fixes landed after the FIRST pilot run here dropped 233/300 tickers
(78%) to Yahoo Finance "Too Many Requests" errors — not a random subset,
but whichever tickers happened to process before the block kicked in
(~50 tickers in under a minute), which would have silently corrupted the
backtest's coverage if the per-ticker error log hadn't been read closely:
  1. Prices for the WHOLE sample are fetched in one batched fetch_prices()
     call (see main()) instead of one call per ticker inside the loop — the
     original per-ticker call defeated pead.prices' own 50-ticker-per-
     yf.download batching entirely, multiplying network round trips ~50x
     for no reason.
  2. Every remaining direct yfinance call site this script's dependency
     chain touches (business-model check's Ticker.info, Ticker.splits,
     load_returns' yf.download) now runs through yfinance_client.py's
     shared process-wide throttle + exponential backoff — the same
     "one shared clock across every call site" pattern edgar_client.py
     already uses for SEC EDGAR, which had no yfinance equivalent until
     this was built.
_load_ticker_data() now returns a specific skip reason (not just a bare
None) so a rerun's coverage can be checked as genuine data gaps (bank/
insurer/REIT exclusions, tickers with no XBRL fundamentals) rather than
another rate-limit dropout wearing an "excluded" label.

A THIRD fix landed after the second run (with the two fixes above) still
throttled to a crawl (~31s/ticker average, 25/300 in 13 minutes, then
stalled further) despite triggering zero rate-limit warnings from
yfinance_client's wrapper. Diagnosis: testing the specific ticker where
progress had stalled (ATO) in a fresh, isolated process returned in under
2 seconds for every call — ruling out a genuine hang or a single bad
ticker. The real cause looks like Yahoo applying a SESSION/ROLLING-WINDOW
soft-throttle (slower responses, not hard errors) that builds up over a
sustained run of many requests — invisible to a quick isolated test, and
not fixable by spacing calls out further, since total request VOLUME
within some window is the likely trigger, not just instantaneous rate.
Fix: cut total call volume, not just its pace. The daily return series
_beta_as_of() needs was previously fetched via a THIRD independent
per-ticker call (factor_engine.data_loader.load_returns, not batched) on
top of the business-model check and splits lookup — redundant, since
prices for the whole sample are already fetched in one batched call (see
fix #1 above). Now derived locally (_log_returns_from_prices()) from that
same already-fetched adjusted-close series instead, cutting total yfinance
call volume by roughly a third. This is a deliberate, flagged methodology
choice, not a silent substitution: dashboard.factor.ticker_ff3_profile
(run_dcf()'s own live beta source) uses factor_engine.data_loader's series,
not pead.prices'; both are standard total-return-adjusted closes and
should track closely, but aren't proven identical — approved as a
call-volume fix, see conversation history, not assumed equivalent without
discussion.

Usage
-----
    .venv/bin/python scripts/run_dcf_pilot_backtest.py [--n-tickers N]
"""

from __future__ import annotations

import argparse
import datetime
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from dcf.backtest import compute_point_in_time_dcf
from dcf.exclusions import check_business_model_fit
from dcf.fundamentals import fetch_ticker_dcf_fundamentals
from dcf.wacc import fetch_risk_free_rate_as_of
from factor_engine.french_data import get_ff7_daily
from factor_engine.gp_universe import get_universe
from pead.backtest import QuarterIC, summarize
from pead.prices import fetch_prices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

N_PILOT_TICKERS_DEFAULT = 300
RANDOM_SEED = 42
_EVAL_START = datetime.date(2014, 6, 30)
_FULL_HISTORY_START = "2010-01-01"   # buffer before eval start covers the 3yr beta lookback
HORIZONS_TRADING_DAYS = (21, 63, 126, 168, 210, 252)   # 1/3/6/8/10/12 months — matches smart_money/backtest.py
_ENTRY_TOLERANCE_DAYS = 10


# ---------------------------------------------------------------------------
# Universe: stratified sample across size tiers
# ---------------------------------------------------------------------------

def _stratified_sample(universe: pd.DataFrame, n: int, seed: int = RANDOM_SEED) -> list[str]:
    """Proportional sample across index_source (sp500/sp400/sp600 size tiers)."""
    rng = random.Random(seed)
    total = len(universe)
    sample: list[str] = []
    for _tier, group in universe.groupby("index_source"):
        tier_n = round(n * len(group) / total)
        tickers = group["ticker"].tolist()
        rng.shuffle(tickers)
        sample.extend(tickers[:tier_n])
    return sorted(sample)


def _quarter_ends(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    return [d.date() for d in pd.date_range(start=start, end=end, freq="QE")]


def _log_returns_from_prices(prices: pd.DataFrame) -> pd.Series:
    """
    Daily log returns from an already-fetched adjusted-close price series —
    same formula as factor_engine.data_loader.load_returns
    (log(p_t / p_t-1), dropna) — derived locally instead of a second
    independent per-ticker yfinance call. See this script's module
    docstring's third fix for why (cutting total call volume was the actual
    fix for the pilot's Yahoo throttling, not just slower pacing).
    """
    return np.log(prices["adj_close"] / prices["adj_close"].shift(1)).dropna()


# ---------------------------------------------------------------------------
# Per-ticker data assembly (one fetch per ticker, reused across every quarter)
# ---------------------------------------------------------------------------

def _load_ticker_data(ticker: str, prices: "pd.DataFrame | None") -> "tuple[dict | None, str | None]":
    """
    Returns (data, None) on success, or (None, reason) if this ticker should
    be skipped entirely — reason is one of "no_price_coverage",
    "unsuitable_business_model", "no_xbrl_fundamentals", "fetch_failure" —
    tracked as a distinct category (not just a single "excluded/failed"
    count) so a rerun's coverage can be verified as genuine data gaps, not
    another rate-limit dropout disguised as an ordinary exclusion.

    `prices` must already be resolved for this ticker from ONE batched
    fetch_prices() call covering the whole sample (see main()) — fetching
    per-ticker here would defeat pead.prices' own 50-ticker-per-yf.download
    batching and is exactly what caused the first pilot run's 78% ticker
    dropout from Yahoo rate-limiting (see this script's module docstring).
    """
    if prices is None or prices.empty:
        return None, "no_price_coverage"
    if check_business_model_fit(ticker) is not None:
        return None, "unsuitable_business_model"

    fund_df = fetch_ticker_dcf_fundamentals(ticker)
    if fund_df.empty:
        return None, "no_xbrl_fundamentals"

    try:
        import yfinance as yf
        from yfinance_client import call_with_backoff
        splits = call_with_backoff(lambda: yf.Ticker(ticker).splits)

        returns = _log_returns_from_prices(prices)
        if returns.empty:
            return None, "fetch_failure"
    except Exception as e:
        log.warning("Skipping %s — fetch failure: %s", ticker, e)
        return None, "fetch_failure"

    return {"fund_df": fund_df, "prices": prices, "splits": splits, "returns": returns}, None


# ---------------------------------------------------------------------------
# Forward returns (mirrors scripts/run_composite_backtest.py's own helpers)
# ---------------------------------------------------------------------------

def _priced_rows_from(prices: pd.DataFrame, start_date: datetime.date, limit: int) -> list[tuple[datetime.date, float]]:
    rows = prices[prices.index >= start_date].sort_index().head(limit)
    return list(zip(rows.index, rows["adj_close"]))


def _forward_return(prices: pd.DataFrame, entry_anchor: datetime.date, horizon_days: int) -> "float | None":
    rows = _priced_rows_from(prices, entry_anchor, horizon_days + 1)
    if not rows:
        return None
    entry_d, entry_price = rows[0]
    if (entry_d - entry_anchor).days > _ENTRY_TOLERANCE_DAYS:
        return None
    if len(rows) <= horizon_days:
        return None
    _, exit_price = rows[horizon_days]
    if entry_price == 0:
        return None
    return exit_price / entry_price - 1.0


# ---------------------------------------------------------------------------
# Cohort IC (reuses pead.backtest's QuarterIC/summarize, same pattern as
# scripts/run_composite_backtest.py)
# ---------------------------------------------------------------------------

def _cohort_ic(cohort_df: pd.DataFrame, fwd_col: str, cohort_label: str, horizon: int, min_obs: int) -> QuarterIC:
    obs = cohort_df[["valuation_gap_pct", fwd_col]].dropna()
    n_candidates = len(cohort_df)
    n_obs = len(obs)
    coverage_pct = (n_obs / n_candidates) if n_candidates else 0.0

    ic = None
    if n_obs >= min_obs:
        rho, _p = spearmanr(obs["valuation_gap_pct"], obs[fwd_col])
        ic = float(rho)

    return QuarterIC(
        quarter_cohort=cohort_label, horizon_days=horizon,
        n_candidates=n_candidates, n_obs=n_obs, coverage_pct=coverage_pct,
        ic=ic, observations=[],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tickers", type=int, default=N_PILOT_TICKERS_DEFAULT,
                         help=f"Pilot sample size (default {N_PILOT_TICKERS_DEFAULT})")
    args = parser.parse_args()

    log.info("Loading universe and drawing stratified pilot sample...")
    universe = get_universe()
    sample = _stratified_sample(universe, args.n_tickers)
    log.info("Pilot sample: %d tickers", len(sample))

    eval_dates = _quarter_ends(_EVAL_START, datetime.date.today())
    log.info("Evaluation grid: %d quarter-ends (%s .. %s)", len(eval_dates), eval_dates[0], eval_dates[-1])

    log.info("Fetching FF7 factor panel (once, shared across all tickers)...")
    factors = get_ff7_daily(_FULL_HISTORY_START, datetime.date.today().isoformat())

    log.info("Fetching risk-free rate per quarter (once, shared across all tickers)...")
    rf_by_date = {d: fetch_risk_free_rate_as_of(d) for d in eval_dates}
    n_missing_rf = sum(1 for v in rf_by_date.values() if v is None)
    if n_missing_rf:
        log.warning("%d/%d quarters have no resolvable risk-free rate", n_missing_rf, len(eval_dates))

    log.info("Fetching prices for all %d tickers in ONE batched call...", len(sample))
    # Must start from _FULL_HISTORY_START, not eval_dates[0] — this same
    # series now also supplies the beta regression's trailing lookback (see
    # _log_returns_from_prices), which needs history from BEFORE the
    # earliest evaluation date (eval_dates[0] - up to BETA_LOOKBACK_YEARS).
    all_prices = fetch_prices(sample, datetime.date.fromisoformat(_FULL_HISTORY_START), datetime.date.today())
    log.info("Price coverage: %d/%d tickers", len(all_prices), len(sample))

    rows = []
    error_counts: dict[str, int] = {}
    load_skip_counts: dict[str, int] = {}

    for i, ticker in enumerate(sample, 1):
        if i % 25 == 0:
            log.info("Progress: %d/%d tickers (%d rows so far)", i, len(sample), len(rows))

        data, skip_reason = _load_ticker_data(ticker, all_prices.get(ticker))
        if data is None:
            load_skip_counts[skip_reason] = load_skip_counts.get(skip_reason, 0) + 1
            continue

        for as_of in eval_dates:
            rf = rf_by_date.get(as_of)
            if rf is None:
                continue
            result = compute_point_in_time_dcf(
                ticker, as_of,
                fund_df=data["fund_df"], prices=data["prices"], splits=data["splits"],
                returns=data["returns"], factors=factors, risk_free_rate=rf,
            )
            if "error" in result:
                error_counts[result["error"]] = error_counts.get(result["error"], 0) + 1
                continue

            row = {"ticker": ticker, "as_of": as_of, "valuation_gap_pct": result["valuation_gap_pct"]}
            for horizon in HORIZONS_TRADING_DAYS:
                row[f"fwd_{horizon}"] = _forward_return(data["prices"], as_of, horizon)
            rows.append(row)

    log.info("Tickers skipped at load, by reason: %s (total %d/%d)",
              load_skip_counts, sum(load_skip_counts.values()), len(sample))
    fetch_failures = load_skip_counts.get("fetch_failure", 0) + load_skip_counts.get("no_price_coverage", 0)
    if fetch_failures / len(sample) > 0.10:
        log.warning(
            "%.0f%% of the sample was dropped by fetch/coverage failures (not legitimate business-model/"
            "no-fundamentals exclusions) — this is the same shape of problem the rate-limited first run had; "
            "treat results with suspicion and check the log above for rate-limit warnings before trusting them.",
            100 * fetch_failures / len(sample),
        )
    log.info("Per-(ticker, quarter) error breakdown: %s", error_counts)

    panel = pd.DataFrame(rows)
    log.info("Scored panel: %d (ticker, quarter) rows across %d tickers", len(panel), panel["ticker"].nunique() if not panel.empty else 0)

    if panel.empty:
        log.error("No scored rows — aborting before backtest.")
        return

    print("\n" + "=" * 78)
    print(f"DCF PILOT BACKTEST — {len(sample)}-ticker stratified sample, 2014Q2+ quarterly grid")
    print("=" * 78)
    print(f"\nScored panel: {len(panel)} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['as_of'].nunique()} quarters")

    min_obs = 10   # mirrors pead.backtest.MIN_COHORT_OBS
    print(f"\n{'Horizon':<10} {'N quarters':>10} {'Mean IC':>9} {'Std IC':>8} {'t-stat':>8} {'Hit rate':>9}")
    for horizon in HORIZONS_TRADING_DAYS:
        label = {21: "1mo", 63: "3mo", 126: "6mo", 168: "8mo", 210: "10mo", 252: "12mo"}[horizon]
        quarter_ics = [
            _cohort_ic(group, f"fwd_{horizon}", as_of.isoformat(), horizon, min_obs)
            for as_of, group in panel.groupby("as_of")
        ]
        summary = summarize(quarter_ics)
        h = summary.horizons[0] if summary.horizons else None
        if h is None or h.mean_ic is None:
            print(f"{label:<10} {'n/a':>10} {'n/a':>9} {'n/a':>8} {'n/a':>8} {'n/a':>9}")
            continue
        print(f"{label:<10} {h.n_quarters:>10} {h.mean_ic:>+9.4f} {h.std_ic:>8.4f} "
              f"{h.t_stat:>+8.2f} {h.hit_rate:>9.1%}")

    panel.to_csv(Path(__file__).parent.parent / "data" / "dcf" / "pilot_backtest_panel.csv", index=False)
    log.info("Panel saved to data/dcf/pilot_backtest_panel.csv")


if __name__ == "__main__":
    main()
