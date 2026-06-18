"""
Sanity checks for the SMB factor module.

Tests use synthetic return series so no network call is needed.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from factor_engine.factors.smb import build_smb_factor, compute_smb_loading


def _make_dates(n=500):
    return pd.date_range("2020-01-02", periods=n, freq="B")


def _synthetic_factors(n=500, seed=42):
    rng = np.random.default_rng(seed)
    dates = _make_dates(n)

    market = pd.Series(rng.normal(0.0003, 0.010, n), index=dates)
    rf = pd.Series(0.00002, index=dates)
    # True small-cap premium: small caps drift slightly above large caps
    small = pd.Series(rng.normal(0.00035, 0.011, n), index=dates, name="IWM")
    large = pd.Series(rng.normal(0.00020, 0.008, n), index=dates, name="IWB")

    market_factor = pd.DataFrame({
        "market_return": market,
        "rf_rate": rf,
        "market_excess": market - rf,
    })
    smb_factor = pd.DataFrame({
        "small_return": small,
        "large_return": large,
        "smb": small - large,
    })
    return market_factor, smb_factor, rf


def _synthetic_stock(market_factor, smb_factor, beta_mkt=1.1, beta_smb=0.6, seed=7, n=500):
    """Stock whose true loadings are known: r = α + β_mkt·mkt + β_smb·smb + ε."""
    rng = np.random.default_rng(seed)
    dates = _make_dates(n)
    noise = rng.normal(0, 0.004, n)
    stock_excess = (
        beta_mkt * market_factor["market_excess"].values
        + beta_smb * smb_factor["smb"].values
        + noise
    )
    stock = pd.Series(stock_excess + market_factor["rf_rate"].values, index=dates, name="TEST")
    return stock


@patch("factor_engine.factors.smb.load_returns")
def test_build_smb_factor_columns(mock_lr):
    dates = _make_dates(10)
    mock_lr.return_value = pd.DataFrame({
        "IWM": pd.Series(0.001, index=dates),
        "IWB": pd.Series(0.0005, index=dates),
    })
    smb = build_smb_factor("2020-01-01", "2020-01-15")
    assert set(smb.columns) == {"small_return", "large_return", "smb"}
    assert len(smb) == 10


@patch("factor_engine.factors.smb.load_returns")
def test_smb_equals_small_minus_large(mock_lr):
    dates = _make_dates(10)
    small = pd.Series(np.linspace(0.001, 0.002, 10), index=dates)
    large = pd.Series(np.linspace(0.0005, 0.0015, 10), index=dates)
    mock_lr.return_value = pd.DataFrame({"IWM": small, "IWB": large})

    smb = build_smb_factor("2020-01-01", "2020-01-15")
    pd.testing.assert_series_equal(smb["smb"], small - large, check_names=False)


@patch("factor_engine.factors.smb.load_returns")
def test_smb_loading_recovers_true_betas(mock_lr):
    """Joint 2-factor OLS should recover β_mkt ≈ 1.1 and β_smb ≈ 0.6."""
    n = 500
    mf, sf, _ = _synthetic_factors(n=n)
    stock = _synthetic_stock(mf, sf, beta_mkt=1.1, beta_smb=0.6, n=n)

    mock_lr.return_value = pd.DataFrame({"TEST": stock})
    result = compute_smb_loading("TEST", "2020-01-01", "2022-12-31",
                                 market_factor=mf, smb_factor=sf)

    assert abs(result["beta_market"] - 1.1) < 0.12, f"Expected β_mkt ≈ 1.1, got {result['beta_market']}"
    assert abs(result["beta_smb"] - 0.6) < 0.12, f"Expected β_smb ≈ 0.6, got {result['beta_smb']}"


@patch("factor_engine.factors.smb.load_returns")
def test_result_keys(mock_lr):
    n = 500
    mf, sf, _ = _synthetic_factors(n=n)
    stock = _synthetic_stock(mf, sf, n=n)
    mock_lr.return_value = pd.DataFrame({"TEST": stock})

    result = compute_smb_loading("TEST", "2020-01-01", "2022-12-31",
                                 market_factor=mf, smb_factor=sf)

    expected = {
        "ticker", "beta_market", "beta_smb", "alpha_annualised", "r_squared",
        "t_stat_market", "t_stat_smb", "p_value_market", "p_value_smb",
        "n_obs", "start", "end",
    }
    assert expected == set(result.keys())


@patch("factor_engine.factors.smb.load_returns")
def test_negative_smb_loading_for_large_cap_stock(mock_lr):
    """A stock that perfectly tracks IWB should have β_smb < 0."""
    n = 500
    mf, sf, rf = _synthetic_factors(n=n)
    # Perfect large-cap stock: tracks large_return exactly
    stock = pd.Series(
        sf["large_return"].values + rf.values,
        index=_make_dates(n),
        name="BIGCO",
    )
    mock_lr.return_value = pd.DataFrame({"BIGCO": stock})

    result = compute_smb_loading("BIGCO", "2020-01-01", "2022-12-31",
                                 market_factor=mf, smb_factor=sf)
    assert result["beta_smb"] < 0, f"Expected negative β_smb for large-cap, got {result['beta_smb']}"


@patch("factor_engine.factors.smb.load_returns")
def test_smb_p_values_significant_for_clean_signal(mock_lr):
    n = 500
    mf, sf, _ = _synthetic_factors(n=n)
    stock = _synthetic_stock(mf, sf, beta_mkt=1.0, beta_smb=0.8, n=n)
    mock_lr.return_value = pd.DataFrame({"TEST": stock})

    result = compute_smb_loading("TEST", "2020-01-01", "2022-12-31",
                                 market_factor=mf, smb_factor=sf)
    assert result["p_value_smb"] < 0.05
    assert result["p_value_market"] < 0.05
