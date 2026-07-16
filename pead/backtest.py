"""
PEAD signal backtesting: does the SUE/percentile score predict forward returns?

Public interface
----------------
    entry_date(announcement_date, session)             -> datetime.date
    run_backtest(panel, prices, *, horizons)             -> list[QuarterIC]
    summarize(quarter_ics)                               -> BacktestSummary

Deliberately mirrors smart_money/backtest.py's IC methodology (Spearman IC,
coverage gate, t-stat/hit-rate/rolling-average summarize()) so the output
format matches the existing 13F signal backtests. Two structural
differences from that module, both driven by PEAD being a different kind
of event:

  1. No 45-day filing lag. A 13F isn't public until up to 45 days after
     quarter-end (smart_money/backtest.py's knowledge_date()); an earnings
     announcement is immediately public the moment it's released. The only
     timing adjustment needed is which trading session first had the
     chance to react — see entry_date() below.

  2. Events aren't aligned to one shared quarter-end grid. 13F periods are
     literally quarter-end dates shared by every fund; PEAD announcements
     are scattered across company-specific fiscal calendars throughout the
     year. Cross-sectional IC groups are formed on quarter_cohort (calendar
     quarter of the announcement date, assigned in pead/signal.py) instead
     of a 13F period.

Entry timing
------------
    entry_date(announcement_date, session):
        "bmo"           -> announcement_date itself (that day's close
                            already reflects the news — released before or
                            during the session)
        "amc"/"unknown" -> announcement_date + 1 calendar day (the next
                            trading day's close is the first to reflect it;
                            "unknown" defaults here conservatively rather
                            than risking a same-day look-ahead)

The actual entry price is the first priced row on/after this anchor
(within _ENTRY_TOLERANCE_DAYS) — never before, mirroring
smart_money/backtest.py's _entry_and_exit exactly for the same reason
(never leak information the investor could not yet have acted on).

Horizons
--------
21 and 63 trading days (~1 and ~3 months) — the two horizons PEAD is most
classically documented at, and a subset of smart_money/backtest.py's
HORIZONS_TRADING_DAYS so the two signals' results are directly comparable.

Missing prices
--------------
If there is no priced row within _ENTRY_TOLERANCE_DAYS of the entry
anchor, or fewer than horizon_days rows follow it (including the common
case of a recent announcement without enough subsequent trading history
yet), that observation is dropped — no imputation, same philosophy as
smart_money/returns.py and smart_money/backtest.py.

No DB
-----
Pure computation over pead/signal.py's panel (a plain DataFrame) and
pead/prices.py's CSV-backed price cache — this package has no database
anywhere in it.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

HORIZONS_TRADING_DAYS = (21, 63)   # ~1 month, ~3 months
MIN_COHORT_OBS = 10                 # minimum observations for a cohort's IC to be meaningful
_ENTRY_TOLERANCE_DAYS = 10           # max gap between entry anchor and the first priced row


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class ICObservation:
    ticker: str
    announcement_date: datetime.date
    score: float
    forward_return: float


@dataclass
class QuarterIC:
    quarter_cohort: str      # e.g. "2024Q3" (calendar quarter of announcement_date)
    horizon_days: int
    n_candidates: int         # scored rows before price-availability filtering
    n_obs: int                 # observations with both entry and exit prices
    coverage_pct: float         # n_obs / n_candidates, 0 when n_candidates == 0
    ic: float | None            # Spearman rho; None when n_obs < MIN_COHORT_OBS
    observations: list[ICObservation] = field(default_factory=list)


@dataclass
class HorizonSummary:
    horizon_days: int
    n_quarters: int           # cohorts with a computed ic (not None)
    mean_ic: float | None
    std_ic: float | None
    t_stat: float | None
    hit_rate: float | None    # fraction of cohorts with same-sign ic as mean_ic
    rolling_4q: list[tuple[str, float | None]]
    rolling_8q: list[tuple[str, float | None]]


@dataclass
class BacktestSummary:
    horizons: list[HorizonSummary]
    quarter_ics: list[QuarterIC]


# ---------------------------------------------------------------------------
# Look-ahead-safe dates
# ---------------------------------------------------------------------------

def entry_date(announcement_date: datetime.date, session: str) -> datetime.date:
    """First calendar day the market could have reacted to this announcement."""
    if session == "bmo":
        return announcement_date
    return announcement_date + datetime.timedelta(days=1)   # "amc" or "unknown"


# ---------------------------------------------------------------------------
# Price lookup — forward-looking, trading-day-indexed
# ---------------------------------------------------------------------------

def _priced_rows_from(prices: pd.DataFrame, start_date: datetime.date, limit: int) -> list[tuple[datetime.date, float]]:
    rows = prices[prices.index >= start_date].sort_index().head(limit)
    return list(zip(rows.index, rows["adj_close"]))


def _entry_and_exit(
    prices: pd.DataFrame,
    announcement_date: datetime.date,
    session: str,
    horizon_days: int,
) -> float | None:
    """Forward return over horizon_days trading days from the first tradeable price on/after entry_date."""
    ed = entry_date(announcement_date, session)
    rows = _priced_rows_from(prices, ed, horizon_days + 1)

    if not rows:
        return None
    entry_d, entry_price = rows[0]
    if (entry_d - ed).days > _ENTRY_TOLERANCE_DAYS:
        return None   # data gap (delisted / no coverage) — not a genuine entry point
    if len(rows) <= horizon_days:
        return None   # not enough subsequent trading days priced yet
    _, exit_price = rows[horizon_days]
    if entry_price == 0:
        return None
    return exit_price / entry_price - 1.0


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_cohort_ic(
    cohort: str,
    horizon_days: int,
    panel: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
) -> QuarterIC:
    """Compute Spearman IC for one (quarter_cohort, horizon) combination."""
    rows = panel[panel["quarter_cohort"] == cohort]
    observations: list[ICObservation] = []

    for _, r in rows.iterrows():
        p = prices.get(r["ticker"])
        if p is None:
            continue
        fwd = _entry_and_exit(p, r["announcement_date"], r["session"], horizon_days)
        if fwd is None:
            continue
        observations.append(ICObservation(
            ticker=r["ticker"],
            announcement_date=r["announcement_date"],
            score=r["score"],
            forward_return=fwd,
        ))

    n_candidates = len(rows)
    n_obs = len(observations)
    coverage_pct = (n_obs / n_candidates) if n_candidates else 0.0

    ic: float | None = None
    if n_obs >= MIN_COHORT_OBS:
        rho, _p = spearmanr(
            [o.score for o in observations],
            [o.forward_return for o in observations],
        )
        ic = float(rho)

    return QuarterIC(
        quarter_cohort=cohort,
        horizon_days=horizon_days,
        n_candidates=n_candidates,
        n_obs=n_obs,
        coverage_pct=coverage_pct,
        ic=ic,
        observations=observations,
    )


def run_backtest(
    panel: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    *,
    horizons: tuple[int, ...] = HORIZONS_TRADING_DAYS,
) -> list[QuarterIC]:
    """
    Compute QuarterIC rows for every (quarter_cohort, horizon) combination.

    Parameters
    ----------
    panel : pd.DataFrame
        Output of pead.signal.compute_sue(). Rows with score_method ==
        "no_estimate" (score is null) are excluded up front.
    prices : dict[ticker, DataFrame]
        Output of pead.prices.fetch_prices().
    """
    scored = panel[panel["score"].notna()]
    cohorts = sorted(scored["quarter_cohort"].unique())

    results: list[QuarterIC] = []
    for cohort in cohorts:
        for horizon in horizons:
            results.append(compute_cohort_ic(cohort, horizon, scored, prices))
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _rolling_avg(ordered_ics: list[tuple[str, float | None]], window: int) -> list[tuple[str, float | None]]:
    out: list[tuple[str, float | None]] = []
    values: list[float] = []
    for cohort, ic in ordered_ics:
        if ic is not None:
            values.append(ic)
        window_vals = values[-window:]
        avg = sum(window_vals) / len(window_vals) if window_vals else None
        out.append((cohort, avg))
    return out


def summarize(quarter_ics: list[QuarterIC]) -> BacktestSummary:
    """Aggregate QuarterIC rows into a per-horizon summary."""
    groups: dict[int, list[QuarterIC]] = {}
    for q in quarter_ics:
        groups.setdefault(q.horizon_days, []).append(q)

    horizon_summaries: list[HorizonSummary] = []
    for horizon, rows in sorted(groups.items()):
        rows_sorted = sorted(rows, key=lambda r: r.quarter_cohort)
        ic_series = [(r.quarter_cohort, r.ic) for r in rows_sorted]
        valid_ics = [ic for _c, ic in ic_series if ic is not None]

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
            n_quarters=n,
            mean_ic=mean_ic,
            std_ic=std_ic,
            t_stat=t_stat,
            hit_rate=hit_rate,
            rolling_4q=_rolling_avg(ic_series, 4),
            rolling_8q=_rolling_avg(ic_series, 8),
        ))

    return BacktestSummary(horizons=horizon_summaries, quarter_ics=quarter_ics)
