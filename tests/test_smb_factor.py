"""
Unit tests for the SMB factor build function.

Loading / regression tests live in test_hml_factor.py alongside
compute_factor_loadings(), which is the unified 4-factor OLS function.
"""

import numpy as np
import pandas as pd
from unittest.mock import patch

from factor_engine.factors.smb import build_smb_factor


def _make_dates(n=10):
    return pd.date_range("2020-01-02", periods=n, freq="B")


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
