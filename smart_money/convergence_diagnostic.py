"""
Diagnostic-only convergence scoring: hard-filter small/routine position
changes out of the convergence calculation entirely, instead of the
production continuous down-weighting (_change_mult's tanh curve, which still
lets a 5% trim contribute at 0.55x rather than excluding it).

This exists to answer one question: does requiring a larger position change
before it counts toward convergence improve the signal's 1/3-month IC (see
backtest — those are the two horizons where FF4 already shows the most
(relative) strength: +0.008 IC, t~1.4-1.5)? It is NOT a replacement for
convergence.py's production scan_quarter()/_score_cusip() — this is a
parallel, in-memory-only scoring path for backtesting a methodology variant.
No DB writes, no persisted ConvergenceScore/FinalSignal rows, nothing for a
production pipeline run to pick up.

NEW/CLOSED changes are never filtered — they are already treated as decisive
(change_mult=1.0) in production, and there's no analogous "how big was this
trade" metric for them (no prior_shares baseline to compute a % change
against). Only INCREASED/DECREASED changes are subject to the hard
|shares_pct_change| >= threshold_pct gate; changes that fail it are dropped
from the scoring calculation as if the fund made no move at all that quarter
(not down-weighted to near-zero — excluded, so they also don't count toward
n_funds/breadth).

Reuses unchanged from convergence.py: _collect_all_changes (change collection
doesn't depend on the filter), _primary_change, _skill_weight, _change_mult,
_load_skill_map, _BREADTH_SATURATION. Reuses unchanged from signal.py: _blend
(NLP is independent of position-change filtering), _status, DISCOVERY_THRESHOLD.

Watchlist-universe status chain
--------------------------------
signal.py's _status() needs a convergence_trend argument for its DELTA_NUDGE
(+/-0.05-with-trend-confirmation) refinement, but that refinement only
changes the STRENGTHENING/WEAKENING/HOLDING *label* — never the None-vs-not-None
decision that determines watchlist membership (see _status's source: the
`return None` branch checks only final_score and prior_score, never trend).
Since this diagnostic only needs membership + final_score (not the label) to
compute IC, convergence_trend is passed as None throughout — this is exact
for IC purposes, not an approximation.
"""

from __future__ import annotations

import datetime

from scipy.stats import spearmanr

from smart_money.backtest import (
    ICObservation,
    MIN_QUARTER_OBS,
    QuarterIC,
    _available_periods,
    _entry_and_exit,
    summarize,
)
from smart_money.convergence import (
    _BREADTH_SATURATION,
    _change_mult,
    _collect_all_changes,
    _load_skill_map,
    _primary_change,
    _skill_weight,
)
from smart_money.changes import PositionChange
from smart_money.models import FundSkillResult, Security, init_db
from smart_money.nlp import load_scores
from smart_money.signal import _blend, _status


# ---------------------------------------------------------------------------
# Hard filter + filtered per-CUSIP scorer
# ---------------------------------------------------------------------------

def _passes_filter(primary: PositionChange, threshold_pct: float) -> bool:
    """NEW/CLOSED always pass; INCREASED/DECREASED need |pct_change| >= threshold_pct."""
    if primary["change_type"] in ("NEW", "CLOSED"):
        return True
    pct = abs(primary.get("shares_pct_change") or 0.0)
    return pct >= threshold_pct


def _score_cusip_filtered(
    fund_changes: list[tuple],
    skill_map: dict[int, FundSkillResult],
    threshold_pct: float,
    min_total_weight: float,
) -> tuple[float, int] | None:
    """Same math as convergence._score_cusip's directional*breadth score, but
    changes failing _passes_filter are excluded before they touch bull_w/bear_w
    or the fund count — not down-weighted, dropped."""
    bull_w = bear_w = 0.0
    n_bull = n_bear = 0

    for fund, changes in fund_changes:
        primary = _primary_change(changes)
        if primary["direction"] == "neutral":
            continue
        if not _passes_filter(primary, threshold_pct):
            continue

        sw = _skill_weight(fund, skill_map)
        cm = _change_mult(primary)
        ew = sw * cm

        if primary["direction"] == "bullish_leaning":
            bull_w += ew
            n_bull += 1
        else:
            bear_w += ew
            n_bear += 1

    n_total = n_bull + n_bear
    total_w = bull_w + bear_w
    if n_total < 2 or total_w < min_total_weight:
        return None

    directional = (bull_w - bear_w) / total_w
    breadth = min(n_total / _BREADTH_SATURATION, 1.0)
    return round(directional * breadth, 4), n_total


def score_quarter_filtered(
    all_changes: dict[str, list[tuple]],
    threshold_pct: float,
    skill_map: dict[int, FundSkillResult],
    *,
    min_funds: int = 2,
    min_total_weight: float = 1.0,
) -> dict[str, tuple[float, str | None]]:
    """cusip -> (convergence_score, ticker) under the filtered scoring rule.

    all_changes is the *unfiltered* output of convergence._collect_all_changes
    (candidate gate on raw fund count, same as production scan_quarter, so the
    denominator used for coverage reporting is unaffected by the filter —
    only which of those candidates' moves count toward the score changes).
    """
    candidate_cusips = [c for c, fc in all_changes.items() if len(fc) >= min_funds]
    tickers = {
        row.cusip: row.ticker
        for row in Security.select(Security.cusip, Security.ticker)
        .where(Security.cusip.in_(candidate_cusips))
    } if candidate_cusips else {}

    out: dict[str, tuple[float, str | None]] = {}
    for cusip in candidate_cusips:
        result = _score_cusip_filtered(all_changes[cusip], skill_map, threshold_pct, min_total_weight)
        if result is None:
            continue
        score, _n_total = result
        out[cusip] = (score, tickers.get(cusip))
    return out


# ---------------------------------------------------------------------------
# Backtest over the filtered scoring rule
# ---------------------------------------------------------------------------

def _compute_quarter_ic(
    period: datetime.date,
    horizon_days: int,
    universe: str,
    scores: dict[str, tuple[float, str | None]],
) -> QuarterIC:
    """Same computation as backtest.compute_quarter_ic, sourcing scores from
    an already-built dict instead of _scores_for() (which reads persisted
    ConvergenceScore/FinalSignal — not applicable here, nothing is persisted)."""
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
        period=period, horizon_days=horizon_days, universe=universe,
        n_candidates=n_candidates, n_obs=n_obs, coverage_pct=coverage_pct,
        ic=ic, observations=observations,
    )


def run_filtered_backtest_multi(
    thresholds: list[float],
    horizons: tuple[int, ...] = (21, 63),
) -> dict[float, "BacktestSummary"]:
    """
    Same computation as run_filtered_backtest, for multiple thresholds at
    once, sharing the one expensive step (_collect_all_changes — N-fund
    detect_changes + Holding aggregation per quarter) across all of them
    instead of repeating it once per threshold. _collect_all_changes doesn't
    depend on threshold_pct at all, only the scoring step does.

    Returns {threshold_pct → BacktestSummary}.
    """
    init_db()
    skill_map = _load_skill_map()
    periods = _available_periods()

    quarter_ics_by_threshold: dict[float, list[QuarterIC]] = {t: [] for t in thresholds}
    prior_maps: dict[float, dict[str, float]] = {t: {} for t in thresholds}

    for period in periods:
        all_changes = _collect_all_changes(period)

        for threshold in thresholds:
            filtered_scores = score_quarter_filtered(all_changes, threshold, skill_map)

            tickers = [t for _s, t in filtered_scores.values() if t]
            nlp_map = load_scores(tickers)

            full_scores: dict[str, tuple[float, str | None]] = {}
            watchlist_scores: dict[str, tuple[float, str | None]] = {}
            new_prior_map: dict[str, float] = {}
            prior_map = prior_maps[threshold]

            for cusip, (conv_score, ticker) in filtered_scores.items():
                nlp = nlp_map.get(ticker) if ticker else None
                if nlp is not None:
                    final_score, _contradicted = _blend(conv_score, nlp.composite_score)
                else:
                    final_score = conv_score
                full_scores[cusip] = (final_score, ticker)

                prior_score = prior_map.get(cusip)
                status = _status(final_score, prior_score, None)   # trend=None: exact for membership, see module docstring
                if status is not None:
                    watchlist_scores[cusip] = (final_score, ticker)
                    new_prior_map[cusip] = final_score

            prior_maps[threshold] = new_prior_map

            for horizon in horizons:
                quarter_ics_by_threshold[threshold].append(_compute_quarter_ic(period, horizon, "full", full_scores))
                quarter_ics_by_threshold[threshold].append(_compute_quarter_ic(period, horizon, "watchlist", watchlist_scores))

    return {t: summarize(qics) for t, qics in quarter_ics_by_threshold.items()}


def run_filtered_backtest(threshold_pct: float, horizons: tuple[int, ...] = (21, 63)):
    """
    Run the full/watchlist backtest under the filtered convergence-scoring
    rule (threshold_pct=0.0 reproduces the production/unfiltered rule — use
    it as a sanity check against the real backtest numbers).

    Single-threshold convenience wrapper around run_filtered_backtest_multi.
    Returns a BacktestSummary (same shape as smart_money.backtest.summarize's
    return value) so callers can print it with the same formatting.
    """
    return run_filtered_backtest_multi([threshold_pct], horizons)[threshold_pct]
