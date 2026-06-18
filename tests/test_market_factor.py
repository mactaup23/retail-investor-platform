"""
Basic sanity checks for the market factor module.

These tests use a small synthetic price series so no network call is needed.
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
