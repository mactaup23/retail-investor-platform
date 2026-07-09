"""
Unit tests for factor_engine/factors/gp.py's construction logic.

Regression-recovery tests for compute_factor_loadings() with a synthetic GP
factor live in test_hml_factor.py alongside the other four factors. This file
covers gp.py's own machinery: reporting-lag-aware observation selection,
quarterly rebalance scheduling, and quintile long/short basket assignment —
all deterministic given synthetic fundamentals, so no network calls needed.
"""

import pandas as pd

from factor_engine.factors.gp import (
    MIN_UNIVERSE_FOR_REBALANCE,
    REPORTING_LAG_DAYS,
    _build_rebalance_assignments,
    _quarterly_rebalance_dates,
    _select_gp_ratio,
)


# ── _quarterly_rebalance_dates ──────────────────────────────────────────────

def test_quarterly_rebalance_dates_are_calendar_quarter_starts():
    dates = _quarterly_rebalance_dates(pd.Timestamp("2021-02-15").date(), pd.Timestamp("2022-01-01").date())
    assert dates == [
        pd.Timestamp("2021-04-01"),
        pd.Timestamp("2021-07-01"),
        pd.Timestamp("2021-10-01"),
        pd.Timestamp("2022-01-01"),
    ]


def test_quarterly_rebalance_dates_empty_when_floor_after_ceiling():
    assert _quarterly_rebalance_dates(pd.Timestamp("2022-01-01").date(), pd.Timestamp("2021-01-01").date()) == []


# ── _select_gp_ratio ─────────────────────────────────────────────────────────

def _obs(rows):
    return pd.DataFrame(rows, columns=["period_end", "gp_ratio"])


def test_select_gp_ratio_respects_reporting_lag():
    obs = _obs([
        {"period_end": "2021-12-31", "gp_ratio": 0.10},
        {"period_end": "2022-03-31", "gp_ratio": 0.20},
    ])
    just_inside_lag = pd.Timestamp("2022-03-31") + pd.Timedelta(days=REPORTING_LAG_DAYS)
    just_before_lag = just_inside_lag - pd.Timedelta(days=1)

    # Q1 statement not yet "reportable" the day before its lag elapses —
    # should fall back to the prior (Q4) observation.
    assert _select_gp_ratio(obs, just_before_lag) == 0.10
    # Exactly on the lag boundary, the newer observation becomes eligible.
    assert _select_gp_ratio(obs, just_inside_lag) == 0.20


def test_select_gp_ratio_none_when_nothing_eligible():
    obs = _obs([{"period_end": "2024-01-01", "gp_ratio": 0.15}])
    assert _select_gp_ratio(obs, pd.Timestamp("2024-01-05")) is None  # lag hasn't elapsed


def test_select_gp_ratio_none_for_empty_or_missing_observations():
    assert _select_gp_ratio(pd.DataFrame(), pd.Timestamp("2024-01-01")) is None
    assert _select_gp_ratio(None, pd.Timestamp("2024-01-01")) is None


def test_select_gp_ratio_picks_most_recent_eligible():
    obs = _obs([
        {"period_end": "2021-01-01", "gp_ratio": 0.05},
        {"period_end": "2021-06-01", "gp_ratio": 0.10},
        {"period_end": "2021-09-01", "gp_ratio": 0.30},
    ])
    asof = pd.Timestamp("2021-09-01") + pd.Timedelta(days=REPORTING_LAG_DAYS + 10)
    assert _select_gp_ratio(obs, asof) == 0.30


# ── _build_rebalance_assignments ────────────────────────────────────────────

def _fundamentals_with_ratios(ratios: dict[str, float], period_end="2021-01-01") -> dict[str, pd.DataFrame]:
    return {t: _obs([{"period_end": period_end, "gp_ratio": r}]) for t, r in ratios.items()}


def test_quintile_assignment_splits_top_and_bottom():
    ratios = {f"T{i:03d}": float(i) for i in range(150)}  # 0..149, evenly spread
    fundamentals = _fundamentals_with_ratios(ratios, period_end="2020-01-01")
    rb_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=REPORTING_LAG_DAYS + 5)

    assignments = _build_rebalance_assignments(fundamentals, [rb_date])
    assert len(assignments) == 1
    a = assignments[0]

    n = len(ratios)
    q = int(n * 0.20)
    assert len(a["long"]) == q
    assert len(a["short"]) == q
    # Long basket must be the highest-ratio names, short the lowest.
    assert set(a["long"])  == {f"T{i:03d}" for i in range(n - q, n)}
    assert set(a["short"]) == {f"T{i:03d}" for i in range(0, q)}
    # No overlap between long and short baskets.
    assert not (set(a["long"]) & set(a["short"]))


def test_rebalance_skipped_when_universe_too_small():
    ratios = {f"T{i:03d}": float(i) for i in range(MIN_UNIVERSE_FOR_REBALANCE - 1)}
    fundamentals = _fundamentals_with_ratios(ratios, period_end="2020-01-01")
    rb_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=REPORTING_LAG_DAYS + 5)

    assignments = _build_rebalance_assignments(fundamentals, [rb_date])
    assert assignments == []


def test_rebalance_skipped_before_reporting_lag_elapses():
    ratios = {f"T{i:03d}": float(i) for i in range(150)}
    fundamentals = _fundamentals_with_ratios(ratios, period_end="2020-01-01")
    too_early = pd.Timestamp("2020-01-01") + pd.Timedelta(days=REPORTING_LAG_DAYS - 5)

    assignments = _build_rebalance_assignments(fundamentals, [too_early])
    assert assignments == []
