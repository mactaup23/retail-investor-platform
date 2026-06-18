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
compute_smb_loading() fits a joint 2-factor OLS:

    r_i − r_f = α + β_mkt·(Mkt-RF) + β_smb·SMB + ε

Both market beta (β_mkt) and SMB loading (β_smb) are returned from a single
regression.  Running separate single-factor regressions would omit a correlated
regressor and bias both estimates.
"""

import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from factor_engine.data_loader import load_returns
from factor_engine.factors.market import build_market_factor

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


def compute_smb_loading(
    ticker: str,
    start: str,
    end: str,
    market_factor: pd.DataFrame | None = None,
    smb_factor: pd.DataFrame | None = None,
) -> dict:
    """
    Estimate a stock's market beta and SMB loading via joint 2-factor OLS.

    Parameters
    ----------
    ticker : str
    start, end : str  ISO dates
    market_factor : optional pre-built market factor DataFrame
        Columns: market_return, rf_rate, market_excess.
    smb_factor : optional pre-built SMB factor DataFrame
        Columns: small_return, large_return, smb.

    Returns
    -------
    dict with keys:
        ticker, beta_market, beta_smb,
        alpha_annualised, r_squared,
        t_stat_market, t_stat_smb,
        p_value_market, p_value_smb,
        n_obs, start, end
    """
    if market_factor is None:
        market_factor = build_market_factor(start, end)
    if smb_factor is None:
        smb_factor = build_smb_factor(start, end)

    stock_returns = load_returns([ticker], start, end)[ticker]

    combined = pd.DataFrame({
        "stock_return": stock_returns,
        "rf_rate": market_factor["rf_rate"],
        "mkt_excess": market_factor["market_excess"],
        "smb": smb_factor["smb"],
    }).dropna()

    stock_excess = combined["stock_return"] - combined["rf_rate"]
    X = add_constant(combined[["mkt_excess", "smb"]])
    model = OLS(stock_excess, X).fit()

    return {
        "ticker": ticker,
        "beta_market": round(model.params["mkt_excess"], 4),
        "beta_smb": round(model.params["smb"], 4),
        "alpha_annualised": round(model.params["const"] * 252, 4),
        "r_squared": round(model.rsquared, 4),
        "t_stat_market": round(model.tvalues["mkt_excess"], 4),
        "t_stat_smb": round(model.tvalues["smb"], 4),
        "p_value_market": round(model.pvalues["mkt_excess"], 6),
        "p_value_smb": round(model.pvalues["smb"], 6),
        "n_obs": int(model.nobs),
        "start": start,
        "end": end,
    }
