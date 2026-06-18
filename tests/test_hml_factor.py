"""
Unit tests for the HML factor module and the unified 3-factor loading function.

Tests use synthetic return series — no network calls needed.
"""

import numpy as np
import pandas as pd
from unittest.mock import patch

from factor_engine.factors.hml import build_hml_factor, compute_factor_loadings


def _make_dates(n=500):
    return pd.date_range("2020-01-02", periods=n, freq="B")


def _synthetic_factors(n=500, seed=42):
    """Return synthetic market, SMB, and HML factor DataFrames."""
    rng = np.random.default_rng(seed)
    dates = _make_dates(n)

    market = pd.Series(rng.normal(0.0003, 0.010, n), index=dates)
    rf     = pd.Series(0.00002, index=dates)
    small  = pd.Series(rng.normal(0.00035, 0.011, n), index=dates)
    large  = pd.Series(rng.normal(0.00020, 0.008, n), index=dates)
    value  = pd.Series(rng.normal(0.00030, 0.009, n), index=dates)
    growth = pd.Series(rng.normal(0.00015, 0.009, n), index=dates)

    market_factor = pd.DataFrame({
        "market_return": market,
        "rf_rate":       rf,
        "market_excess": market - rf,
    })
    smb_factor = pd.DataFrame({
        "small_return": small,
        "large_return": large,
        "smb":          small - large,
    })
    hml_factor = pd.DataFrame({
        "value_return":  value,
        "growth_return": growth,
        "hml":           value - growth,
    })
    return market_factor, smb_factor, hml_factor, rf


def _synthetic_stock(
    market_factor,
    smb_factor,
    hml_factor,
    beta_mkt=1.1,
    beta_smb=0.5,
    beta_hml=0.4,
    seed=7,
    n=500,
):
    """Stock with known true loadings: r = α + β_mkt·mkt + β_smb·smb + β_hml·hml + ε."""
    rng = np.random.default_rng(seed)
    dates = _make_dates(n)
    noise = rng.normal(0, 0.004, n)
    stock_excess = (
        beta_mkt * market_factor["market_excess"].values
        + beta_smb * smb_factor["smb"].values
        + beta_hml * hml_factor["hml"].values
        + noise
    )
    stock = pd.Series(
        stock_excess + market_factor["rf_rate"].values,
        index=dates,
        name="TEST",
    )
    return stock


# ── build_hml_factor ──────────────────────────────────────────────────────────

@patch("factor_engine.factors.hml.load_returns")
def test_build_hml_factor_columns(mock_lr):
    dates = _make_dates(10)
    mock_lr.return_value = pd.DataFrame({
        "IWD": pd.Series(0.0010, index=dates),
        "IWF": pd.Series(0.0012, index=dates),
        "IWN": pd.Series(0.0008, index=dates),
        "IWO": pd.Series(0.0014, index=dates),
    })
    hml = build_hml_factor("2020-01-01", "2020-01-15")
    assert set(hml.columns) == {"value_return", "growth_return", "hml"}
    assert len(hml) == 10


@patch("factor_engine.factors.hml.load_returns")
def test_hml_4etf_averaging(mock_lr):
    """HML must be 0.5*(IWD+IWN) - 0.5*(IWF+IWO), not a 2-ETF difference."""
    dates = _make_dates(5)
    iwd = pd.Series([0.01, 0.02, -0.01, 0.00,  0.03], index=dates)
    iwf = pd.Series([0.02, 0.01,  0.01, 0.01,  0.02], index=dates)
    iwn = pd.Series([0.00, 0.03, -0.02, 0.01,  0.01], index=dates)
    iwo = pd.Series([0.03, 0.02,  0.00, 0.02,  0.04], index=dates)
    mock_lr.return_value = pd.DataFrame({"IWD": iwd, "IWF": iwf, "IWN": iwn, "IWO": iwo})

    hml = build_hml_factor("2020-01-01", "2020-01-08")

    expected_value  = 0.5 * (iwd + iwn)
    expected_growth = 0.5 * (iwf + iwo)
    expected_hml    = expected_value - expected_growth

    pd.testing.assert_series_equal(hml["value_return"],  expected_value,  check_names=False)
    pd.testing.assert_series_equal(hml["growth_return"], expected_growth, check_names=False)
    pd.testing.assert_series_equal(hml["hml"],           expected_hml,    check_names=False)


# ── compute_factor_loadings ───────────────────────────────────────────────────

@patch("factor_engine.factors.hml.load_returns")
def test_compute_factor_loadings_recovers_true_betas(mock_lr):
    """3-factor OLS should recover β_mkt ≈ 1.1, β_smb ≈ 0.5, β_hml ≈ 0.4."""
    n = 500
    mf, sf, hf, _ = _synthetic_factors(n=n)
    stock = _synthetic_stock(mf, sf, hf, beta_mkt=1.1, beta_smb=0.5, beta_hml=0.4, n=n)

    mock_lr.return_value = pd.DataFrame({"TEST": stock})
    result = compute_factor_loadings(
        "TEST", "2020-01-01", "2022-12-31",
        market_factor=mf, smb_factor=sf, hml_factor=hf,
    )

    assert abs(result["beta_market"] - 1.1) < 0.12, f"β_mkt: expected ≈1.1, got {result['beta_market']}"
    assert abs(result["beta_smb"]    - 0.5) < 0.12, f"β_smb: expected ≈0.5, got {result['beta_smb']}"
    assert abs(result["beta_hml"]    - 0.4) < 0.12, f"β_hml: expected ≈0.4, got {result['beta_hml']}"


@patch("factor_engine.factors.hml.load_returns")
def test_result_keys(mock_lr):
    n = 500
    mf, sf, hf, _ = _synthetic_factors(n=n)
    stock = _synthetic_stock(mf, sf, hf, n=n)
    mock_lr.return_value = pd.DataFrame({"TEST": stock})

    result = compute_factor_loadings(
        "TEST", "2020-01-01", "2022-12-31",
        market_factor=mf, smb_factor=sf, hml_factor=hf,
    )

    expected = {
        "ticker", "beta_market", "beta_smb", "beta_hml",
        "alpha_annualised", "r_squared",
        "t_stat_market", "t_stat_smb", "t_stat_hml",
        "p_value_market", "p_value_smb", "p_value_hml",
        "n_obs", "start", "end",
    }
    assert expected == set(result.keys())


@patch("factor_engine.factors.hml.load_returns")
def test_negative_hml_for_growth_stock(mock_lr):
    """A stock tracking the growth ETF basket should have β_hml < 0."""
    n = 500
    mf, sf, hf, rf = _synthetic_factors(n=n)
    # Perfect growth stock: tracks growth_return exactly
    stock = pd.Series(
        hf["growth_return"].values + rf.values,
        index=_make_dates(n),
        name="GROWCO",
    )
    mock_lr.return_value = pd.DataFrame({"GROWCO": stock})

    result = compute_factor_loadings(
        "GROWCO", "2020-01-01", "2022-12-31",
        market_factor=mf, smb_factor=sf, hml_factor=hf,
    )
    assert result["beta_hml"] < 0, f"Expected β_hml < 0 for growth stock, got {result['beta_hml']}"


@patch("factor_engine.factors.hml.load_returns")
def test_positive_hml_for_value_stock(mock_lr):
    """A stock tracking the value ETF basket should have β_hml > 0."""
    n = 500
    mf, sf, hf, rf = _synthetic_factors(n=n)
    stock = pd.Series(
        hf["value_return"].values + rf.values,
        index=_make_dates(n),
        name="VALCO",
    )
    mock_lr.return_value = pd.DataFrame({"VALCO": stock})

    result = compute_factor_loadings(
        "VALCO", "2020-01-01", "2022-12-31",
        market_factor=mf, smb_factor=sf, hml_factor=hf,
    )
    assert result["beta_hml"] > 0, f"Expected β_hml > 0 for value stock, got {result['beta_hml']}"


@patch("factor_engine.factors.hml.load_returns")
def test_negative_smb_for_large_cap_stock(mock_lr):
    """β_smb should remain negative for a large-cap stock in the 3-factor model."""
    n = 500
    mf, sf, hf, rf = _synthetic_factors(n=n)
    stock = pd.Series(
        sf["large_return"].values + rf.values,
        index=_make_dates(n),
        name="BIGCO",
    )
    mock_lr.return_value = pd.DataFrame({"BIGCO": stock})

    result = compute_factor_loadings(
        "BIGCO", "2020-01-01", "2022-12-31",
        market_factor=mf, smb_factor=sf, hml_factor=hf,
    )
    assert result["beta_smb"] < 0, f"Expected β_smb < 0 for large-cap stock, got {result['beta_smb']}"


@patch("factor_engine.factors.hml.load_returns")
def test_p_values_significant_for_clean_signal(mock_lr):
    """With strong synthetic factor signal all three betas should be significant."""
    n = 500
    mf, sf, hf, _ = _synthetic_factors(n=n)
    stock = _synthetic_stock(mf, sf, hf, beta_mkt=1.0, beta_smb=0.7, beta_hml=0.6, n=n)
    mock_lr.return_value = pd.DataFrame({"TEST": stock})

    result = compute_factor_loadings(
        "TEST", "2020-01-01", "2022-12-31",
        market_factor=mf, smb_factor=sf, hml_factor=hf,
    )
    assert result["p_value_market"] < 0.05
    assert result["p_value_smb"]    < 0.05
    assert result["p_value_hml"]    < 0.05
