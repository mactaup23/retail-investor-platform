"""
Module 4 (extension) — signal backtesting: does FinalSignal predict returns?

Public interface
----------------
    knowledge_date(period)                          → datetime.date
    run_backtest(periods=None, *, universe="both")   → list[QuarterIC]
    summarize(quarter_ics)                           → BacktestSummary

Signal score
------------
Two parallel universes are scored per quarter, both surfaced side by side:

    "watchlist" — FinalSignal.final_score as persisted.  This is NOT the full
        scored population: signal.combine() drops any cusip whose final_score
        never crossed DISCOVERY_THRESHOLD (0.30) and has no prior watchlist
        row (see signal.py _status()).  The negative / low-conviction tail is
        structurally absent except for the single quarter a name flips to
        EXIT SIGNAL.  IC over this universe is expected to look better than
        the signal's true unrestricted predictive power.

    "full" — every ConvergenceScore row that quarter, re-blended with NLP
        via the same _blend() formula signal.py uses, but WITHOUT the
        discovery filter.  This is the unbiased cross-section.

Subsequent return / look-ahead bias
------------------------------------
A 13F is not public on period_of_report (the quarter-end balance date) — funds
have up to 45 calendar days to file.  Pricing the "entry" at period_of_report
would leak information the investor could not have acted on yet.  Instead:

    knowledge_date(period) = period + KNOWLEDGE_LAG_DAYS (45)

Entry price is the first PriceCache adj_close ON OR AFTER knowledge_date
(never before — the mirror image of returns.py's _adj_close_near, which
looks backward from a quarter-end boundary because it wants the price AS OF
that boundary, not the first tradeable price after public disclosure).

Forward return horizons are expressed in trading days (not calendar days) by
counting rows in the PriceCache series, so weekends/holidays never need to be
modelled: exit price is simply the Nth priced row after entry.

Missing prices
--------------
If there is no priced row within _ENTRY_TOLERANCE_DAYS of knowledge_date, or
fewer than horizon_days rows follow it, that cusip is dropped from that
quarter's observation set (no imputation) — same philosophy as returns.py's
coverage gate.

Information Coefficient
------------------------
Spearman rank correlation between score and forward_return, computed
independently per (period, horizon, universe).  Quarters with fewer than
MIN_QUARTER_OBS observations are recorded (for coverage reporting) but their
`ic` field is None — not enough points for a meaningful rank correlation.

summarize() aggregates per (horizon, universe): mean IC, IC std, a t-stat
(mean / (std / sqrt(n))) analogous to the t-stat/confidence_label convention
already used for skill scores in factor_apply.py, hit rate (fraction of
quarters with same-sign IC), and rolling 4Q / 8Q average IC series.

DB contract
-----------
This module is pure computation over already-persisted tables.  It never
calls init_db(); the caller is responsible for initialising the database.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field

from scipy.stats import spearmanr

from smart_money.convergence import load_quarter
from smart_money.models import ConvergenceScore, FinalSignal, PriceCache
from smart_money.nlp import load_scores
from smart_money.signal import _blend

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWLEDGE_LAG_DAYS = 45   # SEC 13F filing deadline after quarter-end
HORIZONS_TRADING_DAYS = (21, 63, 126, 168, 210, 252)   # ~1, 3, 6, 8, 10, 12 months
MIN_QUARTER_OBS = 10   # minimum observations for a quarter's IC to be meaningful
_ENTRY_TOLERANCE_DAYS = 10   # max gap between knowledge_date and the first priced row

UNIVERSES = ("watchlist", "full")


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class ICObservation:
    cusip: str
    ticker: str | None
    score: float
    forward_return: float


@dataclass
class QuarterIC:
    period: datetime.date
    horizon_days: int
    universe: str          # "watchlist" | "full"
    n_candidates: int       # scored rows before price-availability filtering
    n_obs: int               # observations with both entry and exit prices
    coverage_pct: float      # n_obs / n_candidates, 0 when n_candidates == 0
    ic: float | None         # Spearman rho; None when n_obs < MIN_QUARTER_OBS
    observations: list[ICObservation] = field(default_factory=list)


@dataclass
class HorizonSummary:
    horizon_days: int
    universe: str
    n_quarters: int          # quarters with a computed ic (not None)
    mean_ic: float | None
    std_ic: float | None
    t_stat: float | None
    hit_rate: float | None   # fraction of quarters with same-sign ic as mean_ic
    rolling_4q: list[tuple[datetime.date, float | None]]
    rolling_8q: list[tuple[datetime.date, float | None]]


@dataclass
class BacktestSummary:
    horizons: list[HorizonSummary]
    quarter_ics: list[QuarterIC]


# ---------------------------------------------------------------------------
# Look-ahead-safe dates
# ---------------------------------------------------------------------------

def knowledge_date(period: datetime.date) -> datetime.date:
    """Date the 13F-derived signal for `period` becomes publicly knowable."""
    return period + datetime.timedelta(days=KNOWLEDGE_LAG_DAYS)


# ---------------------------------------------------------------------------
# Price lookup — forward-looking, trading-day-indexed
# ---------------------------------------------------------------------------

def _priced_rows_from(cusip: str, start_date: datetime.date, limit: int) -> list[tuple[datetime.date, float]]:
    """
    Return up to `limit` (date, adj_close) rows for cusip with date >= start_date,
    ordered ascending by date.  Row 0 (if present) is the entry candidate.
    """
    rows = (
        PriceCache.select(PriceCache.date, PriceCache.adj_close)
        .where(
            PriceCache.security_id == cusip,
            PriceCache.date >= start_date,
        )
        .order_by(PriceCache.date.asc())
        .limit(limit)
    )
    return [(r.date, float(r.adj_close)) for r in rows]


def _entry_and_exit(
    cusip: str,
    period: datetime.date,
    horizon_days: int,
    _cache: dict[str, list[tuple[datetime.date, float]]],
) -> float | None:
    """
    Return the forward return over horizon_days trading days from the first
    tradeable price on/after knowledge_date(period), or None if unavailable.
    """
    kd = knowledge_date(period)

    # Fetches exactly what this call's horizon needs, not the global max across
    # HORIZONS_TRADING_DAYS — _cache is always a fresh dict per (period, horizon,
    # universe) call site today (see compute_quarter_ic), so there is no cross-
    # horizon reuse to protect by over-fetching. If a caller ever starts sharing
    # one _cache across multiple horizon_days for the same cusip, this under-fetches
    # for the larger horizon — re-widen to max(...) + 1 if that pattern appears.
    if cusip not in _cache:
        _cache[cusip] = _priced_rows_from(cusip, kd, horizon_days + 1)
    rows = _cache[cusip]

    if not rows:
        return None
    entry_date, entry_price = rows[0]
    if (entry_date - kd).days > _ENTRY_TOLERANCE_DAYS:
        return None   # data gap (delisted / no coverage) — not a genuine entry point
    if len(rows) <= horizon_days:
        return None   # not enough subsequent trading days priced yet
    _, exit_price = rows[horizon_days]
    if entry_price == 0:
        return None
    return exit_price / entry_price - 1.0


# ---------------------------------------------------------------------------
# Per-quarter score universes
# ---------------------------------------------------------------------------

def _watchlist_scores(period: datetime.date) -> dict[str, tuple[float, str | None]]:
    """cusip -> (final_score, ticker) from persisted FinalSignal rows."""
    rows = FinalSignal.select().where(FinalSignal.period == period)
    return {r.cusip: (r.final_score, r.ticker) for r in rows}


def _full_universe_scores(period: datetime.date) -> dict[str, tuple[float, str | None]]:
    """
    cusip -> (blended_score, ticker) for every ConvergenceScore row that
    quarter, re-blended with NLP the same way signal.combine() does but
    without the DISCOVERY_THRESHOLD filter.
    """
    conv_rows: list[ConvergenceScore] = load_quarter(period)
    tickers = [r.ticker for r in conv_rows if r.ticker]
    nlp_map = load_scores(tickers)

    out: dict[str, tuple[float, str | None]] = {}
    for conv in conv_rows:
        nlp = nlp_map.get(conv.ticker) if conv.ticker else None
        if nlp is not None:
            score, _contradicted = _blend(conv.convergence_score, nlp.composite_score)
        else:
            score = conv.convergence_score
        out[conv.cusip] = (score, conv.ticker)
    return out


def _scores_for(period: datetime.date, universe: str) -> dict[str, tuple[float, str | None]]:
    if universe == "watchlist":
        return _watchlist_scores(period)
    if universe == "full":
        return _full_universe_scores(period)
    raise ValueError(f"Unknown universe: {universe!r}")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _available_periods() -> list[datetime.date]:
    rows = (
        ConvergenceScore.select(ConvergenceScore.period)
        .distinct()
        .order_by(ConvergenceScore.period.asc())
    )
    return [r.period for r in rows]


def compute_quarter_ic(
    period: datetime.date,
    horizon_days: int,
    universe: str,
) -> QuarterIC:
    """Compute Spearman IC for one (period, horizon, universe) combination."""
    scores = _scores_for(period, universe)

    price_cache: dict[str, list[tuple[datetime.date, float]]] = {}
    observations: list[ICObservation] = []

    for cusip, (score, ticker) in scores.items():
        fwd_return = _entry_and_exit(cusip, period, horizon_days, price_cache)
        if fwd_return is None:
            continue
        observations.append(ICObservation(cusip=cusip, ticker=ticker, score=score, forward_return=fwd_return))

    n_candidates = len(scores)
    n_obs = len(observations)
    coverage_pct = (n_obs / n_candidates) if n_candidates else 0.0

    ic: float | None = None
    if n_obs >= MIN_QUARTER_OBS:
        rho, _p = spearmanr(
            [o.score for o in observations],
            [o.forward_return for o in observations],
        )
        ic = float(rho)

    return QuarterIC(
        period=period,
        horizon_days=horizon_days,
        universe=universe,
        n_candidates=n_candidates,
        n_obs=n_obs,
        coverage_pct=coverage_pct,
        ic=ic,
        observations=observations,
    )


def run_backtest(
    periods: list[datetime.date] | None = None,
    *,
    universe: str = "both",
) -> list[QuarterIC]:
    """
    Compute QuarterIC rows for every (period, horizon, universe) combination.

    Parameters
    ----------
    periods : list[datetime.date] | None
        Quarters to evaluate. Defaults to every period with ConvergenceScore rows.
    universe : str
        "watchlist", "full", or "both" (default).
    """
    from smart_money.models import init_db
    init_db()

    if periods is None:
        periods = _available_periods()
    universes = UNIVERSES if universe == "both" else (universe,)

    results: list[QuarterIC] = []
    for period in periods:
        for uni in universes:
            for horizon in HORIZONS_TRADING_DAYS:
                results.append(compute_quarter_ic(period, horizon, uni))
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _rolling_avg(ordered_ics: list[tuple[datetime.date, float | None]], window: int) -> list[tuple[datetime.date, float | None]]:
    out: list[tuple[datetime.date, float | None]] = []
    values: list[float] = []
    for p, ic in ordered_ics:
        if ic is not None:
            values.append(ic)
        window_vals = values[-window:]
        avg = sum(window_vals) / len(window_vals) if window_vals else None
        out.append((p, avg))
    return out


def summarize(quarter_ics: list[QuarterIC]) -> BacktestSummary:
    """Aggregate QuarterIC rows into per-(horizon, universe) summaries."""
    groups: dict[tuple[int, str], list[QuarterIC]] = {}
    for q in quarter_ics:
        groups.setdefault((q.horizon_days, q.universe), []).append(q)

    horizon_summaries: list[HorizonSummary] = []
    for (horizon, uni), rows in sorted(groups.items()):
        rows_sorted = sorted(rows, key=lambda r: r.period)
        ic_series = [(r.period, r.ic) for r in rows_sorted]
        valid_ics = [ic for _p, ic in ic_series if ic is not None]

        n = len(valid_ics)
        mean_ic = sum(valid_ics) / n if n else None
        if n > 1:
            variance = sum((x - mean_ic) ** 2 for x in valid_ics) / (n - 1)
            std_ic = variance ** 0.5
        else:
            std_ic = None
        t_stat = (mean_ic / (std_ic / (n ** 0.5))) if (mean_ic is not None and std_ic and std_ic > 0) else None
        hit_rate = (
            sum(1 for x in valid_ics if (x >= 0) == (mean_ic >= 0)) / n
            if (n and mean_ic is not None) else None
        )

        horizon_summaries.append(HorizonSummary(
            horizon_days=horizon,
            universe=uni,
            n_quarters=n,
            mean_ic=mean_ic,
            std_ic=std_ic,
            t_stat=t_stat,
            hit_rate=hit_rate,
            rolling_4q=_rolling_avg(ic_series, 4),
            rolling_8q=_rolling_avg(ic_series, 8),
        ))

    return BacktestSummary(horizons=horizon_summaries, quarter_ics=quarter_ics)
