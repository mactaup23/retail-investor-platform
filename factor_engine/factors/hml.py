"""
HML (High Minus Low) value factor — Fama-French style daily return series.

Construction
------------
Value proxy  : average of IWD (Russell 1000 Value) and IWN (Russell 2000 Value)
Growth proxy : average of IWF (Russell 1000 Growth) and IWO (Russell 2000 Growth)
Daily HML    : 0.5·(log_ret(IWD) + log_ret(IWN)) − 0.5·(log_ret(IWF) + log_ret(IWO))

Fama-French sort all stocks across both size tiers to cancel the size effect:
    HML_FF = ½(Small Value + Big Value) − ½(Small Growth + Big Growth)

The 4-ETF averaging structure mirrors that construction: two value ETFs (one large-cap,
one small-cap) minus two growth ETFs (one large-cap, one small-cap).  Averaging within
each side eliminates most of the size tilt that a single large-cap-only proxy (e.g. IVE
vs IVW) would carry, and keeps corr(HML, SMB) low by design.

Russell indices reconstitute annually in late June — matching FF's own June-end
portfolio rebalance convention.

ETF proxy vs. pure Fama-French HML
------------------------------------
The academic FF HML uses pure Book-to-Market (B/M) breakpoints from CRSP.  Russell
value/growth classification uses B/M plus analyst I/B/E/S growth forecasts plus
historical sales-per-share growth — a B/M-plus-quality blend rather than a pure B/M
sort.  Empirical correlation between this ETF-based series and the FF-published daily
HML factor is approximately 0.80–0.88 (lower than the ~0.85–0.90 we see for the
IWM/IWB-based SMB series).  This is appropriate for a retail investor platform but
should be noted when comparing factor loadings to academic benchmarks.

Factor loading regression (Fama-French-Carhart 4-factor model)
------------------------------------------------------------------
compute_factor_loadings() fits a joint 4-factor OLS:

    r_i − r_f = α + β_mkt·(Mkt-RF) + β_smb·SMB + β_hml·HML + β_mom·MOM + ε

All four betas are estimated in a single regression.  Running separate or paired
regressions would omit correlated regressors and bias all estimates.  See
factor_engine/factors/mom.py for the momentum factor construction and its
ETF-proxy caveats.
"""

import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from factor_engine.data_loader import load_returns
from factor_engine.factors.market import build_market_factor
from factor_engine.factors.mom import build_mom_factor
from factor_engine.factors.smb import build_smb_factor

VALUE_LARGE_ETF  = "IWD"   # iShares Russell 1000 Value
GROWTH_LARGE_ETF = "IWF"   # iShares Russell 1000 Growth
VALUE_SMALL_ETF  = "IWN"   # iShares Russell 2000 Value
GROWTH_SMALL_ETF = "IWO"   # iShares Russell 2000 Growth


def build_hml_factor(start: str, end: str) -> pd.DataFrame:
    """
    Construct the daily HML factor series.

    Returns a DataFrame with columns:
        value_return  — average of IWD and IWN daily log returns
        growth_return — average of IWF and IWO daily log returns
        hml           — value_return − growth_return  (the HML factor)
    """
    tickers = [VALUE_LARGE_ETF, GROWTH_LARGE_ETF, VALUE_SMALL_ETF, GROWTH_SMALL_ETF]
    returns = load_returns(tickers, start, end)
    value_return  = 0.5 * (returns[VALUE_LARGE_ETF]  + returns[VALUE_SMALL_ETF])
    growth_return = 0.5 * (returns[GROWTH_LARGE_ETF] + returns[GROWTH_SMALL_ETF])
    return pd.DataFrame({
        "value_return":  value_return,
        "growth_return": growth_return,
        "hml":           value_return - growth_return,
    }).dropna()


def compute_factor_loadings(
    ticker: str,
    start: str,
    end: str,
    market_factor: pd.DataFrame | None = None,
    smb_factor: pd.DataFrame | None = None,
    hml_factor: pd.DataFrame | None = None,
    mom_factor: pd.DataFrame | None = None,
) -> dict:
    """
    Estimate a stock's Fama-French-Carhart 4-factor loadings via joint OLS.

    Parameters
    ----------
    ticker : str
    start, end : str  ISO dates
    market_factor : optional pre-built market factor DataFrame
        Columns: market_return, rf_rate, market_excess.
    smb_factor : optional pre-built SMB factor DataFrame
        Columns: small_return, large_return, smb.
    hml_factor : optional pre-built HML factor DataFrame
        Columns: value_return, growth_return, hml.
    mom_factor : optional pre-built MOM factor DataFrame
        Columns: momentum_return, benchmark_return, mom.

    Returns
    -------
    dict with keys:
        ticker, beta_market, beta_smb, beta_hml, beta_mom,
        alpha_annualised, r_squared,
        t_stat_market, t_stat_smb, t_stat_hml, t_stat_mom,
        p_value_market, p_value_smb, p_value_hml, p_value_mom,
        n_obs, start, end
    """
    if market_factor is None:
        market_factor = build_market_factor(start, end)
    if smb_factor is None:
        smb_factor = build_smb_factor(start, end)
    if hml_factor is None:
        hml_factor = build_hml_factor(start, end)
    if mom_factor is None:
        mom_factor = build_mom_factor(start, end)

    stock_returns = load_returns([ticker], start, end)[ticker]

    combined = pd.DataFrame({
        "stock_return": stock_returns,
        "rf_rate":      market_factor["rf_rate"],
        "mkt_excess":   market_factor["market_excess"],
        "smb":          smb_factor["smb"],
        "hml":          hml_factor["hml"],
        "mom":          mom_factor["mom"],
    }).dropna()

    stock_excess = combined["stock_return"] - combined["rf_rate"]
    X = add_constant(combined[["mkt_excess", "smb", "hml", "mom"]])
    model = OLS(stock_excess, X).fit()

    return {
        "ticker":           ticker,
        "beta_market":      round(model.params["mkt_excess"], 4),
        "beta_smb":         round(model.params["smb"], 4),
        "beta_hml":         round(model.params["hml"], 4),
        "beta_mom":         round(model.params["mom"], 4),
        "alpha_annualised": round(model.params["const"] * 252, 4),
        "r_squared":        round(model.rsquared, 4),
        "t_stat_market":    round(model.tvalues["mkt_excess"], 4),
        "t_stat_smb":       round(model.tvalues["smb"], 4),
        "t_stat_hml":       round(model.tvalues["hml"], 4),
        "t_stat_mom":       round(model.tvalues["mom"], 4),
        "p_value_market":   round(model.pvalues["mkt_excess"], 6),
        "p_value_smb":      round(model.pvalues["smb"], 6),
        "p_value_hml":      round(model.pvalues["hml"], 6),
        "p_value_mom":      round(model.pvalues["mom"], 6),
        "n_obs":            int(model.nobs),
        "start":            start,
        "end":              end,
    }
