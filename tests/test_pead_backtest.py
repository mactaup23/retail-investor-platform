"""
Unit tests for pead/backtest.py — entry-date session anchoring and the
Spearman IC / coverage-gate computation. All synthetic, deterministic, no
network calls (mirrors the style of tests/test_gp_factor.py).
"""

import datetime

import pandas as pd
import pytest

from pead.backtest import (
    MIN_COHORT_OBS,
    compute_cohort_ic,
    entry_date,
)


# ── entry_date session anchoring ─────────────────────────────────────────────

def test_entry_date_bmo_is_same_day():
    d = datetime.date(2024, 5, 1)
    assert entry_date(d, "bmo") == d


def test_entry_date_amc_is_next_day():
    d = datetime.date(2024, 5, 1)
    assert entry_date(d, "amc") == datetime.date(2024, 5, 2)


def test_entry_date_unknown_defaults_to_next_day_conservatively():
    # No timestamp at all -- must not risk a same-day look-ahead assumption.
    d = datetime.date(2024, 5, 1)
    assert entry_date(d, "unknown") == entry_date(d, "amc")


# ── compute_cohort_ic ─────────────────────────────────────────────────────────

def _panel(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _flat_prices(start: datetime.date, days: int, daily_return: float) -> pd.DataFrame:
    """Synthetic price series compounding at a fixed daily return."""
    dates = pd.bdate_range(start, periods=days).date
    prices = [100.0 * (1 + daily_return) ** i for i in range(days)]
    return pd.DataFrame({"adj_close": prices}, index=pd.Index(dates, name="date"))


def test_ic_is_none_below_min_cohort_obs():
    rows = [
        {
            "ticker": f"T{i}", "announcement_date": datetime.date(2024, 1, 15),
            "session": "bmo", "score": float(i), "quarter_cohort": "2024Q1",
        }
        for i in range(MIN_COHORT_OBS - 1)   # one short of the threshold
    ]
    panel = _panel(rows)
    prices = {r["ticker"]: _flat_prices(datetime.date(2024, 1, 15), 100, 0.001) for r in rows}

    q = compute_cohort_ic("2024Q1", 21, panel, prices)
    assert q.n_obs == MIN_COHORT_OBS - 1
    assert q.ic is None


def test_ic_computed_at_min_cohort_obs_and_detects_perfect_rank_correlation():
    # Higher score -> higher daily return -> higher forward return. Spearman
    # rho should be (near) +1.0 for a monotonic relationship.
    rows = []
    prices = {}
    for i in range(MIN_COHORT_OBS):
        ticker = f"T{i}"
        rows.append({
            "ticker": ticker,
            "announcement_date": datetime.date(2024, 1, 15),
            "session": "bmo",
            "score": float(i),
            "quarter_cohort": "2024Q1",
        })
        prices[ticker] = _flat_prices(datetime.date(2024, 1, 15), 100, 0.0005 * i)

    panel = _panel(rows)
    q = compute_cohort_ic("2024Q1", 21, panel, prices)
    assert q.n_obs == MIN_COHORT_OBS
    assert q.ic == pytest.approx(1.0)   # strictly monotonic score <-> forward-return relationship


def test_missing_price_data_drops_observation_not_crashes():
    rows = [
        {"ticker": "HAS_PRICE", "announcement_date": datetime.date(2024, 1, 15), "session": "bmo", "score": 1.0, "quarter_cohort": "2024Q1"},
        {"ticker": "NO_PRICE", "announcement_date": datetime.date(2024, 1, 15), "session": "bmo", "score": 2.0, "quarter_cohort": "2024Q1"},
    ]
    panel = _panel(rows)
    prices = {"HAS_PRICE": _flat_prices(datetime.date(2024, 1, 15), 100, 0.001)}   # NO_PRICE absent

    q = compute_cohort_ic("2024Q1", 21, panel, prices)
    assert q.n_candidates == 2
    assert q.n_obs == 1   # NO_PRICE dropped, not imputed
