"""
Composite signal backtest — does averaging 13F convergence + PEAD SUE beat either alone?

Context (see CLAUDE.md): the Module 3/4 signal-improvement investigation
concluded the 13F positioning signal (ConvergenceScore.convergence_score,
IC ~0.008-0.02 depending on universe/window) had hit its practical ceiling,
motivating the standalone PEAD signal (pead/, IC ~0.02-0.05, 2014Q2+
primary window). Both are genuinely independent data domains — institutional
positioning vs. earnings surprises — so this script tests the standard next
question: does combining them produce a stronger signal than either alone,
without overfitting the combination itself.

Combination method — fixed 50/50 rank average, not a fitted regression
------------------------------------------------------------------------
composite_score = 0.5 * percentile_rank(conv_score) + 0.5 * percentile_rank(pead_score),
ranks computed within each quarter's paired-intersection cohort. Deliberately
NOT a regression-fit or IC-proportional weighting — same "don't fit what you
can't validate out-of-sample" discipline already applied to every other
design decision this project has made under limited data (see CLAUDE.md's
FF4-vs-FF7 conclusion: "FF4 is kept for parsimony, not proven superiority").
A fixed 50/50 average has zero parameters estimated from this backtest's own
data, so there is nothing here to overfit.

Universe alignment
-------------------
Intersection of (a) tickers with a resolved ticker in ConvergenceScore for a
given 13F period and (b) tickers in the PEAD panel with a non-null score
(score_method != "no_estimate"). A ticker with only one of the two scores is
EXCLUDED from that quarter's composite panel entirely — no imputation, same
philosophy as every other coverage gate in this codebase (returns.py,
smart_money/backtest.py, pead/signal.py's no_estimate exclusion).

Timing alignment — no look-ahead on either side
-------------------------------------------------
13F: knowledge_date(period) = period + 45 days (smart_money/backtest.py).
PEAD: entry_date(announcement_date, session) (pead/backtest.py) — bmo same
day, amc/unknown next day.

Pairing rule: for each (period, ticker), the PEAD event paired to it is the
most recent one with announcement_date <= knowledge_date(period) — i.e. the
freshest SUE actually known by the time the 13F signal itself becomes
knowable. If no such event exists, the (period, ticker) row is dropped.

A second gate caps how stale that "most recent" event may be:
knowledge_date(period) - announcement_date must be <= _MAX_PAIRING_STALENESS_DAYS
(120, ~1 quarter of slack). Found empirically necessary: a small number of
thinly-covered tickers (e.g. BFS, UVV) have almost no scoreable yfinance
earnings history, so without this gate "most recent knowable event" silently
falls back to a single years-old announcement and reuses it as the paired
PEAD score across dozens of unrelated later quarters. Only ~1.5% of raw
pairs exceed 120 days (median gap is 17 days), so this gate is a targeted
exclusion of genuinely stale/meaningless pairings, not a material restriction
of the universe — same "exclude rather than fabricate" philosophy as PEAD's
own no_estimate exclusion.

Shared entry anchor across all three variants (13F-alone, PEAD-alone,
composite) — the one deliberate methodological choice specific to THIS
comparison, not used elsewhere:
    entry_anchor = max(knowledge_date(period), entry_date(paired PEAD event))
Using the later of the two dates for every variant means the only thing that
differs between the three backtests is the SCORE, not the return-measurement
window — otherwise an apples-to-apples "does combining help" comparison would
be confounded by 13F-alone starting its clock earlier than the composite.
This does NOT touch either signal's own canonical standalone backtest
(smart_money/backtest.py, pead/backtest.py) — both remain exactly as
documented; this script recomputes fresh baselines on the shared
intersection subset purely for this comparison.

Backtest scope
--------------
2014Q2+ restricted sample (13F periods >= 2014-06-30), matching the PEAD
signal's own "cite the representative window" discipline (see CLAUDE.md —
the full 1999-2026 PEAD window is inflated by thin 2009-2013 cohorts).
Horizons: 21/63 trading days (1mo/3mo), same as pead/backtest.py.
Spearman IC, MIN_COHORT_OBS=10 gate, mean IC / std / t-stat / hit-rate via
pead.backtest.summarize() (reused directly — same aggregation math, fed a
synthetic quarter_cohort = period.isoformat() per row since this script's
cohorts are 13F periods, not PEAD's own calendar-quarter-of-announcement
grouping).

Success bar: composite's t-stat must exceed BOTH individually-recomputed
baselines (on this same intersection subset) to call combination a real
improvement — a higher point IC alone is not sufficient given how much this
project's other backtests have moved on noise (see the +0.061 IC
investigation in CLAUDE.md).

Usage
-----
    .venv/bin/python scripts/run_composite_backtest.py
"""

from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from pead.backtest import HORIZONS_TRADING_DAYS, MIN_COHORT_OBS, QuarterIC, entry_date
from pead.backtest import summarize as pead_summarize
from pead.prices import fetch_prices
from pead.signal import compute_sue
from pead.surprises import fetch_surprises
from pead.universe import get_universe_tickers
from smart_money.backtest import KNOWLEDGE_LAG_DAYS, knowledge_date
from smart_money.convergence import load_quarter
from smart_money.models import ConvergenceScore, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

_MIN_PERIOD = datetime.date(2014, 4, 1)   # restrict to 13F periods >= 2014Q2 (period-end 2014-06-30)
_ENTRY_TOLERANCE_DAYS = 10                # mirrors pead/backtest.py and smart_money/backtest.py
_MAX_PAIRING_STALENESS_DAYS = 120         # ~1 quarter of slack past knowledge_date; see docstring below
_VARIANTS = ("13F-alone", "PEAD-alone", "Composite")


# ---------------------------------------------------------------------------
# Load the 13F convergence side
# ---------------------------------------------------------------------------

def _load_conv_frame() -> pd.DataFrame:
    """One row per (period, ticker) with a resolved ticker, period >= _MIN_PERIOD."""
    init_db()
    periods = [
        r.period for r in
        ConvergenceScore.select(ConvergenceScore.period).distinct()
        .where(ConvergenceScore.period >= _MIN_PERIOD)
        .order_by(ConvergenceScore.period.asc())
    ]
    log.info("13F periods in scope: %d (%s .. %s)", len(periods), periods[0], periods[-1])

    rows = []
    for period in periods:
        for row in load_quarter(period):
            if row.ticker:
                rows.append({"period": period, "ticker": row.ticker, "conv_score": row.convergence_score})

    df = pd.DataFrame(rows)
    n_before = len(df)
    # A ticker can theoretically resolve from >1 CUSIP in the same quarter (rare).
    # Keep the row with the larger-magnitude conviction, deterministically.
    df["abs_conv"] = df["conv_score"].abs()
    df = (
        df.sort_values("abs_conv", ascending=False)
        .drop_duplicates(subset=["period", "ticker"], keep="first")
        .drop(columns="abs_conv")
    )
    if n_before != len(df):
        log.info("Dropped %d duplicate (period, ticker) rows (multi-CUSIP tickers)", n_before - len(df))
    return df


# ---------------------------------------------------------------------------
# Pair each (period, ticker) to its most recently knowable PEAD event
# ---------------------------------------------------------------------------

def _pead_lookup(pead_panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """ticker -> DataFrame of scored rows, sorted by announcement_date ascending."""
    scored = pead_panel[pead_panel["score"].notna()].sort_values("announcement_date")
    return {t: g.reset_index(drop=True) for t, g in scored.groupby("ticker")}


def _pair(conv_df: pd.DataFrame, pead_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for row in conv_df.itertuples(index=False):
        events = pead_by_ticker.get(row.ticker)
        if events is None:
            continue
        kd = knowledge_date(row.period)
        eligible = events[events["announcement_date"] <= kd]
        if eligible.empty:
            continue
        pead_row = eligible.iloc[-1]   # most recent knowable event
        if (kd - pead_row["announcement_date"]).days > _MAX_PAIRING_STALENESS_DAYS:
            continue   # only "recent knowable" event available is too stale to be a fair pairing
        pead_entry = entry_date(pead_row["announcement_date"], pead_row["session"])
        records.append({
            "period": row.period,
            "ticker": row.ticker,
            "conv_score": row.conv_score,
            "pead_score": pead_row["score"],
            "pead_announcement_date": pead_row["announcement_date"],
            "pead_score_method": pead_row["score_method"],
            "entry_anchor": max(kd, pead_entry),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Forward returns — one shared price source, one shared entry anchor
# ---------------------------------------------------------------------------

def _priced_rows_from(prices: pd.DataFrame, start_date: datetime.date, limit: int) -> list[tuple[datetime.date, float]]:
    rows = prices[prices.index >= start_date].sort_index().head(limit)
    return list(zip(rows.index, rows["adj_close"]))


def _forward_return(prices: pd.DataFrame, entry_anchor: datetime.date, horizon_days: int) -> float | None:
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


def _add_forward_returns(paired: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    for horizon in HORIZONS_TRADING_DAYS:
        col = f"fwd_{horizon}"
        values = []
        for row in paired.itertuples(index=False):
            p = prices.get(row.ticker)
            values.append(_forward_return(p, row.entry_anchor, horizon) if p is not None else None)
        paired[col] = values
    return paired


# ---------------------------------------------------------------------------
# Percentile-rank combination (fixed 50/50, zero fitted parameters)
# ---------------------------------------------------------------------------

def _add_composite_score(paired: pd.DataFrame) -> pd.DataFrame:
    paired["conv_pctl"] = paired.groupby("period")["conv_score"].rank(pct=True)
    paired["pead_pctl"] = paired.groupby("period")["pead_score"].rank(pct=True)
    paired["composite_score"] = 0.5 * paired["conv_pctl"] + 0.5 * paired["pead_pctl"]
    return paired


# ---------------------------------------------------------------------------
# Per-variant Spearman IC, reusing pead.backtest's QuarterIC/summarize
# ---------------------------------------------------------------------------

def _cohort_ic(cohort_df: pd.DataFrame, score_col: str, fwd_col: str, cohort_label: str, horizon: int) -> QuarterIC:
    obs = cohort_df[[score_col, fwd_col]].dropna()
    n_candidates = len(cohort_df)
    n_obs = len(obs)
    coverage_pct = (n_obs / n_candidates) if n_candidates else 0.0

    ic = None
    if n_obs >= MIN_COHORT_OBS:
        rho, _p = spearmanr(obs[score_col], obs[fwd_col])
        ic = float(rho)

    return QuarterIC(
        quarter_cohort=cohort_label,
        horizon_days=horizon,
        n_candidates=n_candidates,
        n_obs=n_obs,
        coverage_pct=coverage_pct,
        ic=ic,
        observations=[],
    )


def _run_variant(paired: pd.DataFrame, score_col: str) -> list[QuarterIC]:
    results = []
    for period, group in paired.groupby("period"):
        label = period.isoformat()
        for horizon in HORIZONS_TRADING_DAYS:
            results.append(_cohort_ic(group, score_col, f"fwd_{horizon}", label, horizon))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Loading 13F convergence scores...")
    conv_df = _load_conv_frame()
    conv_tickers = set(conv_df["ticker"].unique())
    log.info("13F side: %d (period, ticker) rows across %d tickers", len(conv_df), len(conv_tickers))

    log.info("Loading PEAD universe + SUE panel (cached)...")
    pead_tickers = get_universe_tickers()
    surprises = fetch_surprises(pead_tickers)
    pead_panel = compute_sue(surprises)
    pead_by_ticker = _pead_lookup(pead_panel)
    log.info("PEAD side: %d tickers with >=1 scored announcement", len(pead_by_ticker))

    intersection = conv_tickers & set(pead_by_ticker.keys())
    log.info("Intersection universe: %d tickers", len(intersection))
    if len(intersection) < 20:
        log.warning("Intersection is very small — results below will be noisy/unreliable regardless of IC.")

    conv_df = conv_df[conv_df["ticker"].isin(intersection)]

    log.info("Pairing each (period, ticker) to its most recently knowable PEAD event...")
    paired = _pair(conv_df, pead_by_ticker)
    log.info(
        "Paired panel: %d rows (dropped %d with no eligible PEAD event yet)",
        len(paired), len(conv_df) - len(paired),
    )

    log.info("Fetching prices for %d intersection tickers...", paired["ticker"].nunique())
    start_date = paired["period"].min()
    end_date = datetime.date.today()
    prices = fetch_prices(sorted(paired["ticker"].unique()), start_date, end_date)
    log.info("Price coverage: %d/%d tickers", len(prices), paired["ticker"].nunique())

    log.info("Computing forward returns from the shared entry anchor...")
    paired = _add_forward_returns(paired, prices)
    paired = _add_composite_score(paired)

    log.info("Running backtest for all three variants...")
    variant_cols = {"13F-alone": "conv_score", "PEAD-alone": "pead_score", "Composite": "composite_score"}
    summaries = {}
    for variant, col in variant_cols.items():
        quarter_ics = _run_variant(paired, col)
        summaries[variant] = pead_summarize(quarter_ics)

    print("\n" + "=" * 78)
    print("COMPOSITE SIGNAL BACKTEST — 13F convergence + PEAD SUE, 2014Q2+ restricted")
    print("=" * 78)
    print(f"\nIntersection universe: {len(intersection)} tickers")
    print(f"Paired (period, ticker) observations: {len(paired)}")

    for horizon in HORIZONS_TRADING_DAYS:
        label = "1-month" if horizon == 21 else "3-month" if horizon == 63 else f"{horizon}d"
        print(f"\nHorizon: {label} ({horizon} trading days)")
        print(f"  {'Variant':<12} {'N quarters':>10} {'Mean IC':>9} {'Std IC':>8} {'t-stat':>8} {'Hit rate':>9}")
        for variant in _VARIANTS:
            h = next(hs for hs in summaries[variant].horizons if hs.horizon_days == horizon)
            mean_ic = f"{h.mean_ic:+.4f}" if h.mean_ic is not None else "n/a"
            std_ic = f"{h.std_ic:.4f}" if h.std_ic is not None else "n/a"
            t_stat = f"{h.t_stat:+.2f}" if h.t_stat is not None else "n/a"
            hit_rate = f"{h.hit_rate:.1%}" if h.hit_rate is not None else "n/a"
            print(f"  {variant:<12} {h.n_quarters:>10} {mean_ic:>9} {std_ic:>8} {t_stat:>8} {hit_rate:>9}")

    print("\nPer-quarter detail (Composite variant):")
    rows = [
        {"period": q.quarter_cohort, "horizon": q.horizon_days, "n_candidates": q.n_candidates,
         "n_obs": q.n_obs, "coverage_pct": round(q.coverage_pct, 3),
         "ic": round(q.ic, 4) if q.ic is not None else None}
        for q in summaries["Composite"].quarter_ics
    ]
    detail = pd.DataFrame(rows).sort_values(["horizon", "period"])
    print(detail.to_string(index=False))


if __name__ == "__main__":
    main()
