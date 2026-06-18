"""
Market factor (CAPM beta).

The market factor is the simplest Fama-French factor: excess return of a
broad market portfolio over the risk-free rate.  Here we use SPY as the
market proxy and the 13-week T-bill (^IRX) as the risk-free rate.

`compute_beta` runs an OLS regression of a stock's excess returns on the
market's excess returns and returns the slope (beta), intercept (alpha),
and summary statistics.
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from factor_engine.data_loader import load_returns, load_prices

MARKET_PROXY = "SPY"
RISK_FREE_PROXY = "^IRX"  # annualised 13-week T-bill yield in percent


def _load_risk_free_rate(start: str, end: str) -> pd.Series:
    """Return a daily risk-free rate aligned to trading days, as a decimal."""
    rf_annual_pct = load_prices([RISK_FREE_PROXY], start, end)[RISK_FREE_PROXY]
    # ^IRX is quoted as an annualised percent; convert to a daily decimal rate
    rf_daily = rf_annual_pct / 100 / 252
    return rf_daily


def build_market_factor(start: str, end: str) -> pd.DataFrame:
    """
    Construct the daily market excess-return factor.

    Returns a DataFrame with columns:
        market_return  — SPY log return
        rf_rate        — daily risk-free rate
        market_excess  — market_return minus rf_rate  (this is the factor)
    """
    market_returns = load_returns([MARKET_PROXY], start, end)[MARKET_PROXY]
    rf_rate = _load_risk_free_rate(start, end)

    aligned = pd.DataFrame({
        "market_return": market_returns,
        "rf_rate": rf_rate,
    }).dropna()

    aligned["market_excess"] = aligned["market_return"] - aligned["rf_rate"]
    return aligned


def compute_beta(
    ticker: str,
    start: str,
    end: str,
    market_factor: pd.DataFrame | None = None,
) -> dict:
    """
    Estimate CAPM beta for `ticker` over the given date range.

    Parameters
    ----------
    ticker : str
    start, end : str  ISO dates
    market_factor : optional pre-built market factor DataFrame
        Pass this in to avoid re-fetching market data when analysing many tickers.

    Returns
    -------
    dict with keys:
        ticker, beta, alpha_annualised, r_squared, t_stat_beta, p_value_beta,
        n_obs, start, end
    """
    if market_factor is None:
        market_factor = build_market_factor(start, end)

    stock_returns = load_returns([ticker], start, end)[ticker]

    combined = pd.DataFrame({
        "stock_return": stock_returns,
        "rf_rate": market_factor["rf_rate"],
        "market_excess": market_factor["market_excess"],
    }).dropna()

    stock_excess = combined["stock_return"] - combined["rf_rate"]
    X = add_constant(combined["market_excess"])
    model = OLS(stock_excess, X).fit()

    return {
        "ticker": ticker,
        "beta": round(model.params["market_excess"], 4),
        "alpha_annualised": round(model.params["const"] * 252, 4),
        "r_squared": round(model.rsquared, 4),
        "t_stat_beta": round(model.tvalues["market_excess"], 4),
        "p_value_beta": round(model.pvalues["market_excess"], 6),
        "n_obs": int(model.nobs),
        "start": start,
        "end": end,
    }
