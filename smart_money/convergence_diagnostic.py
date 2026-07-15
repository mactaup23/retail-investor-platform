"""
Diagnostic-only convergence scoring: apply a hard include/exclude rule to
each (fund, cusip) move before it enters the convergence calculation, instead
of the production continuous down-weighting (skill_weight x change_mult,
which still lets e.g. a 5% trim or an unscored-but-plausible fund contribute
at a reduced but nonzero weight).

Generalized over a `move_filter(fund, primary_change) -> bool` predicate so
the same scoring/backtest plumbing serves multiple methodology tests:
    size_filter(threshold_pct)   — drop small INCREASED/DECREASED moves
    tier_filter(allowed_fund_ids) — drop funds outside a skill tier entirely

This is NOT a replacement for convergence.py's production
scan_quarter()/_score_cusip() — this is a parallel, in-memory-only scoring
path for backtesting methodology variants. No DB writes, no persisted
ConvergenceScore/FinalSignal rows, nothing for a production pipeline run to
pick up.

Reuses unchanged from convergence.py: _collect_all_changes (change collection
doesn't depend on any filter), _primary_change, _skill_weight, _change_mult,
_load_skill_map, _BREADTH_SATURATION. Reuses unchanged from signal.py: _blend
(NLP is independent of these filters), _status.

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
from typing import Callable

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
from smart_money.models import Fund, FundSkillResult, Security, init_db
from smart_money.nlp import load_scores
from smart_money.signal import _blend, _status

MoveFilter = Callable[[Fund, PositionChange], bool]


# ---------------------------------------------------------------------------
# Move-filter predicates
# ---------------------------------------------------------------------------

def size_filter(threshold_pct: float) -> MoveFilter:
    """NEW/CLOSED always pass; INCREASED/DECREASED need |pct_change| >= threshold_pct.

    There's no analogous "how big was this trade" metric for NEW/CLOSED (no
    prior_shares baseline to compute a % change against), and they're already
    treated as decisive (change_mult=1.0) in production — so they're exempt.
    """
    def _filter(fund: Fund, primary: PositionChange) -> bool:
        if primary["change_type"] in ("NEW", "CLOSED"):
            return True
        pct = abs(primary.get("shares_pct_change") or 0.0)
        return pct >= threshold_pct
    return _filter


def tier_filter(allowed_fund_ids: set[int]) -> MoveFilter:
    """Only funds in allowed_fund_ids contribute a move, regardless of change
    type or size — a fund-identity gate, not a move-characteristic gate."""
    def _filter(fund: Fund, _primary: PositionChange) -> bool:
        return fund.id in allowed_fund_ids
    return _filter


# ---------------------------------------------------------------------------
# Filtered per-CUSIP scorer
# ---------------------------------------------------------------------------

def _score_cusip_variant(
    fund_changes: list[tuple],
    skill_map: dict[int, FundSkillResult],
    move_filter: MoveFilter,
    min_total_weight: float,
) -> tuple[float, int] | None:
    """Same math as convergence._score_cusip's directional*breadth score, but
    moves failing move_filter are excluded before they touch bull_w/bear_w or
    the fund count — not down-weighted, dropped."""
    bull_w = bear_w = 0.0
    n_bull = n_bear = 0

    for fund, changes in fund_changes:
        primary = _primary_change(changes)
        if primary["direction"] == "neutral":
            continue
        if not move_filter(fund, primary):
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


def score_quarter_variant(
    all_changes: dict[str, list[tuple]],
    move_filter: MoveFilter,
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
        result = _score_cusip_variant(all_changes[cusip], skill_map, move_filter, min_total_weight)
        if result is None:
            continue
        score, _n_total = result
        out[cusip] = (score, tickers.get(cusip))
    return out


# ---------------------------------------------------------------------------
# Backtest over a filtered scoring rule
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


def run_variant_backtest_multi(
    variants: dict[str, MoveFilter],
    horizons: tuple[int, ...] = (21, 63),
):
    """
    Run the full/watchlist backtest under multiple move_filter variants at
    once, sharing the one expensive step (_collect_all_changes — N-fund
    detect_changes + Holding aggregation per quarter) across all of them
    instead of repeating it once per variant. _collect_all_changes doesn't
    depend on any filter, only the scoring step does.

    Returns {variant_label → BacktestSummary}.
    """
    init_db()
    skill_map = _load_skill_map()
    periods = _available_periods()

    quarter_ics_by_variant: dict[str, list[QuarterIC]] = {v: [] for v in variants}
    prior_maps: dict[str, dict[str, float]] = {v: {} for v in variants}

    for period in periods:
        all_changes = _collect_all_changes(period)

        for label, move_filter in variants.items():
            filtered_scores = score_quarter_variant(all_changes, move_filter, skill_map)

            tickers = [t for _s, t in filtered_scores.values() if t]
            nlp_map = load_scores(tickers)

            full_scores: dict[str, tuple[float, str | None]] = {}
            watchlist_scores: dict[str, tuple[float, str | None]] = {}
            new_prior_map: dict[str, float] = {}
            prior_map = prior_maps[label]

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

            prior_maps[label] = new_prior_map

            for horizon in horizons:
                quarter_ics_by_variant[label].append(_compute_quarter_ic(period, horizon, "full", full_scores))
                quarter_ics_by_variant[label].append(_compute_quarter_ic(period, horizon, "watchlist", watchlist_scores))

    return {v: summarize(qics) for v, qics in quarter_ics_by_variant.items()}


# ---------------------------------------------------------------------------
# Backward-compatible trade-size-filter entry points
# ---------------------------------------------------------------------------

def run_filtered_backtest_multi(thresholds: list[float], horizons: tuple[int, ...] = (21, 63)):
    """Trade-size-filter convenience wrapper around run_variant_backtest_multi.
    Returns {threshold_pct → BacktestSummary}."""
    variants = {t: size_filter(t) for t in thresholds}
    by_label = run_variant_backtest_multi(variants, horizons)
    return {t: by_label[t] for t in thresholds}


def run_filtered_backtest(threshold_pct: float, horizons: tuple[int, ...] = (21, 63)):
    """Single-threshold convenience wrapper. threshold_pct=0.0 reproduces the
    production/unfiltered rule — use it as a sanity check against the real
    backtest numbers."""
    return run_filtered_backtest_multi([threshold_pct], horizons)[threshold_pct]
