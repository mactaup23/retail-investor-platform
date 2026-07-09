"""
SMB (Small Minus Big) size factor — Fama-French style daily return series.

Construction
------------
Small-cap proxy : IWM  (iShares Russell 2000 ETF)
Large-cap proxy : IWB  (iShares Russell 1000 ETF)
Daily SMB       : log_return(IWM) − log_return(IWB)

The Russell 2000/1000 boundary is a natural size breakpoint that reconstitutes
annually in late June — matching FF's own June-end portfolio rebalance
convention.

ETF proxy vs. pure Fama-French SMB
------------------------------------
The academic FF SMB averages return spreads across three B/M buckets within
each size tier so that value/growth tilts cancel.  IWM and IWB are
cap-weighted within their respective Russell indices and carry residual value
exposure (Russell 2000 tilts value vs. Russell 1000).  Empirical correlation
between this ETF-based series and the FF-published daily SMB factor is
approximately 0.85–0.90, which is appropriate for a retail investor platform
but should be noted when comparing factor loadings to academic benchmarks.

Factor loading regression
--------------------------
Factor loading estimation uses the full Fama-French-Carhart 4-factor OLS defined
in factor_engine.factors.hml.compute_factor_loadings():

    r_i − r_f = α + β_mkt·(Mkt-RF) + β_smb·SMB + β_hml·HML + β_mom·MOM + ε

All four betas are estimated jointly.  A paired regression omitting any one of
them would bias the estimates of the rest via correlated regressors.
"""

import pandas as pd

from factor_engine.data_loader import load_returns

SMALL_CAP_ETF = "IWM"   # iShares Russell 2000
LARGE_CAP_ETF = "IWB"   # iShares Russell 1000


def build_smb_factor(start: str, end: str) -> pd.DataFrame:
    """
    Construct the daily SMB factor series.

    Returns a DataFrame with columns:
        small_return  — IWM daily log return
        large_return  — IWB daily log return
        smb           — small_return − large_return  (the SMB factor)
    """
    returns = load_returns([SMALL_CAP_ETF, LARGE_CAP_ETF], start, end)
    return pd.DataFrame({
        "small_return": returns[SMALL_CAP_ETF],
        "large_return": returns[LARGE_CAP_ETF],
        "smb": returns[SMALL_CAP_ETF] - returns[LARGE_CAP_ETF],
    }).dropna()
