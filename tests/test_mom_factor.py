"""
Unit tests for the MOM factor build function.

Loading / regression tests for compute_factor_loadings() (the unified 4-factor
OLS function) live in test_hml_factor.py. This file additionally verifies the
directional sanity check for momentum specifically: a synthetic "winner" stock
(tracks the momentum leg) must recover a positive beta_mom, and a synthetic
"loser" stock (tracks the inverse) must recover a negative one. Using synthetic
data with known composition — rather than a live ticker like NVDA — keeps this
deterministic: a real ticker's trailing return can flip sign between when this
test is written and when it runs.
"""

import numpy as np
import pandas as pd
from unittest.mock import patch

from factor_engine.factors.hml import compute_factor_loadings
from factor_engine.factors.mom import build_mom_factor
from tests.test_hml_factor import _synthetic_factors


def _make_dates(n=10):
    return pd.date_range("2020-01-02", periods=n, freq="B")


@patch("factor_engine.factors.mom.load_returns")
def test_build_mom_factor_columns(mock_lr):
    dates = _make_dates(10)
    mock_lr.return_value = pd.DataFrame({
        "MTUM": pd.Series(0.0012, index=dates),
        "IWB":  pd.Series(0.0006, index=dates),
    })
    mom = build_mom_factor("2020-01-01", "2020-01-15")
    assert set(mom.columns) == {"momentum_return", "benchmark_return", "mom"}
    assert len(mom) == 10


@patch("factor_engine.factors.mom.load_returns")
def test_mom_equals_momentum_minus_benchmark(mock_lr):
    dates = _make_dates(10)
    momentum  = pd.Series(np.linspace(0.001, 0.003, 10), index=dates)
    benchmark = pd.Series(np.linspace(0.0005, 0.0015, 10), index=dates)
    mock_lr.return_value = pd.DataFrame({"MTUM": momentum, "IWB": benchmark})

    mom = build_mom_factor("2020-01-01", "2020-01-15")
    pd.testing.assert_series_equal(mom["mom"], momentum - benchmark, check_names=False)


@patch("factor_engine.factors.hml.load_returns")
def test_positive_mom_for_winner_stock(mock_lr):
    """A stock tracking the momentum ETF leg exactly should have β_mom > 0."""
    n = 500
    mf, sf, hf, mmf, rf = _synthetic_factors(n=n)
    stock = pd.Series(
        mmf["momentum_return"].values + rf.values,
        index=_make_dates(n),
        name="WINCO",
    )
    mock_lr.return_value = pd.DataFrame({"WINCO": stock})

    result = compute_factor_loadings(
        "WINCO", "2020-01-01", "2022-12-31",
        market_factor=mf, smb_factor=sf, hml_factor=hf, mom_factor=mmf,
    )
    assert result["beta_mom"] > 0, f"Expected β_mom > 0 for winner stock, got {result['beta_mom']}"


@patch("factor_engine.factors.hml.load_returns")
def test_negative_mom_for_loser_stock(mock_lr):
    """A stock tracking the benchmark leg exactly (avoiding recent winners) should have β_mom < 0."""
    n = 500
    mf, sf, hf, mmf, rf = _synthetic_factors(n=n)
    stock = pd.Series(
        mmf["benchmark_return"].values + rf.values,
        index=_make_dates(n),
        name="LOSECO",
    )
    mock_lr.return_value = pd.DataFrame({"LOSECO": stock})

    result = compute_factor_loadings(
        "LOSECO", "2020-01-01", "2022-12-31",
        market_factor=mf, smb_factor=sf, hml_factor=hf, mom_factor=mmf,
    )
    assert result["beta_mom"] < 0, f"Expected β_mom < 0 for loser stock, got {result['beta_mom']}"
