"""
Sanity checks for factor_engine/risk_metrics.py.

Tests use synthetic return series so no network call is needed.
"""

import numpy as np
import pandas as pd
import pytest

from factor_engine.risk_metrics import (
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    compute_risk_metrics,
)


def _dates(n, start="2022-01-03"):
    return pd.date_range(start, periods=n, freq="B")


def test_annualized_volatility_matches_formula():
    idx = _dates(500)
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)

    result = annualized_volatility(rets)

    assert result == pytest.approx(rets.std() * np.sqrt(252))


def test_max_drawdown_known_scenario():
    # Flat, then a clean -20% drop over 10 days, then flat recovery partway.
    idx = _dates(30)
    daily_drop = np.log(0.8) / 10  # 10 equal log-return steps compounding to -20%
    rets = pd.Series(0.0, index=idx)
    rets.iloc[10:20] = daily_drop
    rets.iloc[20:] = 0.001  # small recovery, doesn't erase the drawdown

    result = max_drawdown(rets)

    assert result["max_drawdown"] == pytest.approx(-0.20, abs=1e-6)
    # Trough should land at the 20th row (end of the drop), peak at day 10 (row index 9)
    assert result["trough_date"] == str(idx[19].date())
    assert result["peak_date"] == str(idx[9].date())


def test_max_drawdown_monotonic_gain_is_zero():
    idx = _dates(50)
    rets = pd.Series(0.001, index=idx)  # strictly positive every day

    result = max_drawdown(rets)

    assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-9)


def test_sharpe_ratio_zero_vol_returns_none():
    idx = _dates(100)
    rets = pd.Series(0.0005, index=idx)  # constant → zero volatility
    rf = pd.Series(0.00001, index=idx)

    result = sharpe_ratio(rets, rf)

    assert result["annualized_vol"] == pytest.approx(0.0, abs=1e-12)
    assert result["sharpe_ratio"] is None


def test_sharpe_ratio_known_values():
    idx = _dates(252)
    rng = np.random.default_rng(7)
    rets = pd.Series(rng.normal(0.0006, 0.008, len(idx)), index=idx)
    rf = pd.Series(0.00002, index=idx)

    result = sharpe_ratio(rets, rf)

    ann_return = np.expm1(rets.sum() * (252 / len(rets)))
    ann_rf = rf.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    expected_sharpe = (ann_return - ann_rf) / ann_vol

    assert result["sharpe_ratio"] == pytest.approx(expected_sharpe, rel=1e-6)
    assert result["annualized_rf"] == pytest.approx(ann_rf, rel=1e-6)


def test_sharpe_ratio_aligns_mismatched_index():
    idx1 = _dates(100)
    idx2 = _dates(90, start=str(idx1[10].date()))
    rets = pd.Series(0.0005, index=idx1)
    rf = pd.Series(0.00001, index=idx2)

    result = sharpe_ratio(rets, rf)

    # Only the overlapping dates should survive the inner join
    assert result["n_obs"] == len(idx1.intersection(idx2))


def test_sortino_ratio_no_downside_days_returns_none():
    idx = _dates(100)
    rets = pd.Series(0.001, index=idx)   # always above rf
    rf = pd.Series(0.00001, index=idx)

    result = sortino_ratio(rets, rf)

    assert result["n_downside_days"] == 0
    assert result["downside_deviation"] == pytest.approx(0.0)
    assert result["sortino_ratio"] is None


def test_sortino_ratio_downside_deviation_only_uses_negative_excess():
    idx = _dates(10)
    # Excess returns: some positive, some negative, known values
    rf = pd.Series(0.0, index=idx)
    rets = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, -0.01, 0.02, 0.01, -0.03, 0.01], index=idx)

    result = sortino_ratio(rets, rf)

    downside_excess = rets[rets < 0]
    expected_dd = downside_excess.std(ddof=0) * np.sqrt(252)

    assert result["n_downside_days"] == len(downside_excess)
    assert result["downside_deviation"] == pytest.approx(expected_dd, rel=1e-6)


def test_sortino_greater_than_sharpe_when_downside_is_smaller_share_of_total_vol():
    # A return series with most volatility coming from upside moves should
    # have a smaller downside deviation than total vol, so Sortino > Sharpe
    # for the same positive excess return.
    idx = _dates(252)
    rng = np.random.default_rng(3)
    upside_heavy = rng.normal(0, 0.005, len(idx))
    upside_heavy[upside_heavy < 0] *= 0.3  # shrink the downside moves specifically
    rets = pd.Series(0.0008 + upside_heavy, index=idx)
    rf = pd.Series(0.00001, index=idx)

    sharpe = sharpe_ratio(rets, rf)["sharpe_ratio"]
    sortino = sortino_ratio(rets, rf)["sortino_ratio"]

    assert sortino > sharpe


def test_compute_risk_metrics_bundles_all_fields():
    idx = _dates(300)
    rng = np.random.default_rng(11)
    combined_rets = pd.Series(rng.normal(0.0005, 0.009, len(idx)), index=idx)
    factors = pd.DataFrame({"rf": 0.00002}, index=idx)

    result = compute_risk_metrics(combined_rets, factors)

    expected_keys = {
        "annualized_volatility", "max_drawdown", "max_drawdown_peak_date",
        "max_drawdown_trough_date", "sharpe_ratio", "sortino_ratio",
        "annualized_return", "annualized_rf", "downside_deviation", "n_obs",
    }
    assert expected_keys == set(result.keys())
    assert result["n_obs"] == len(idx)
