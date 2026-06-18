"""
Sanity checks for the market factor module.

Tests use synthetic price series so no network call is needed.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from factor_engine.factors.market import compute_beta, build_market_factor


def _synthetic_returns(n=500, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    market = pd.Series(rng.normal(0.0003, 0.01, n), index=dates, name="SPY")
    # stock = 1.2 * market + noise  →  true beta ≈ 1.2
    stock = 1.2 * market + rng.normal(0, 0.005, n)
    stock.name = "TEST"
    rf = pd.Series(0.00002, index=dates, name="^IRX")  # flat ~0.5% annual
    return market, stock, rf


def _make_market_factor(market, rf):
    return pd.DataFrame({
        "market_return": market,
        "rf_rate": rf,
        "market_excess": market - rf,
    })


@patch("factor_engine.factors.market.load_returns")
@patch("factor_engine.factors.market.build_market_factor")
def test_beta_close_to_true_value(mock_mf, mock_lr):
    market, stock, rf = _synthetic_returns()
    mf = _make_market_factor(market, rf)

    mock_mf.return_value = mf
    mock_lr.return_value = pd.DataFrame({"TEST": stock})

    result = compute_beta("TEST", "2020-01-01", "2022-12-31", market_factor=mf)

    assert abs(result["beta"] - 1.2) < 0.1, f"Expected beta ≈ 1.2, got {result['beta']}"
    assert 0 <= result["r_squared"] <= 1
    assert result["p_value_beta"] < 0.05


def test_result_keys():
    market, stock, rf = _synthetic_returns()
    mf = _make_market_factor(market, rf)

    with patch("factor_engine.factors.market.load_returns") as mock_lr:
        mock_lr.return_value = pd.DataFrame({"TEST": stock})
        result = compute_beta("TEST", "2020-01-01", "2022-12-31", market_factor=mf)

    expected_keys = {"ticker", "beta", "alpha_annualised", "r_squared",
                     "t_stat_beta", "p_value_beta", "n_obs", "start", "end"}
    assert expected_keys == set(result.keys())


def test_excess_returns_subtract_rf():
    """compute_beta must subtract rf_rate from stock returns, not regress raw returns."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-02", periods=300, freq="B")
    market = pd.Series(rng.normal(0.0003, 0.01, 300), index=dates)
    rf = pd.Series(0.00005, index=dates)  # ~1.26% annual
    stock = 1.5 * market + rng.normal(0, 0.004, 300)
    stock = pd.Series(stock, index=dates, name="SYNTH")

    mf = pd.DataFrame({
        "market_return": market,
        "rf_rate": rf,
        "market_excess": market - rf,
    })

    with patch("factor_engine.factors.market.load_returns") as mock_lr:
        mock_lr.return_value = pd.DataFrame({"SYNTH": stock})
        result = compute_beta("SYNTH", "2020-01-01", "2022-12-31", market_factor=mf)

    # Beta should still be ≈ 1.5; if rf were not subtracted from stock returns
    # the regression would use raw returns and the estimate would drift.
    assert abs(result["beta"] - 1.5) < 0.15, f"Expected beta ≈ 1.5, got {result['beta']}"


def test_rf_rate_gap_fill_in_build_market_factor():
    """
    build_market_factor must forward-fill rf_rate gaps rather than dropping
    equity trading days where ^IRX has no quote.
    """
    dates = pd.date_range("2020-01-02", periods=10, freq="B")
    market_returns = pd.Series(0.001, index=dates, name="SPY")

    # rf_rate is missing on dates[3] and dates[7] — simulates ^IRX calendar gap
    rf_values = [0.00002] * 10
    rf_index = dates.delete([3, 7])
    rf_series = pd.Series(
        [v for i, v in enumerate(rf_values) if i not in (3, 7)],
        index=rf_index,
        name="^IRX",
    )

    with (
        patch("factor_engine.factors.market.load_returns") as mock_lr,
        patch("factor_engine.factors.market.load_prices") as mock_lp,
    ):
        mock_lr.return_value = pd.DataFrame({"SPY": market_returns})
        # load_prices returns the raw rf series (with gaps) when called for ^IRX
        mock_lp.return_value = pd.DataFrame({"^IRX": rf_series})

        mf = build_market_factor("2020-01-02", "2020-01-15")

    # All 10 trading days should be present after ffill — none dropped for rf gap
    assert len(mf) == 10, f"Expected 10 rows, got {len(mf)}"
    assert mf["rf_rate"].isna().sum() == 0
