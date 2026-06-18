"""
Market factor (CAPM beta) using proper excess returns.

Both the stock and the market return have the risk-free rate subtracted
before the OLS regression, consistent with the standard CAPM specification:

    r_i - r_f = α + β(r_m - r_f) + ε

Risk-free rate source: ^IRX (CBOE 13-Week Treasury Bill index) via Yahoo
Finance.  This is the standard 3-month T-bill proxy used in practitioner
factor models.  The official Fama-French RF series uses the 1-month T-bill
from CRSP, but the spread between the two is typically < 5 bp — negligible
at daily frequency.  ^IRX requires no API key and is already accessible
through the project's yfinance data layer.

Conversion: annualised bank-discount percent → daily decimal
    rf_daily = (annualised_pct / 100) / 252
The /252 divisor (252 trading days per year) is the same convention Fama-
French uses when publishing their daily factor series.
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from factor_engine.data_loader import load_returns, load_prices

MARKET_PROXY = "SPY"
RISK_FREE_TICKER = "^IRX"  # CBOE 13-week T-Bill index, annualised percent


def _load_risk_free_rate(start: str, end: str) -> pd.Series:
    """
    Return the 13-week T-bill yield as a daily decimal risk-free rate.

    Uses ffill so minor calendar gaps in ^IRX (days when equities trade but
    the CBOE index has no quote) don't silently drop equity trading days.
    """
    rf_annual_pct = load_prices([RISK_FREE_TICKER], start, end, ffill=True)[RISK_FREE_TICKER]
    return rf_annual_pct / 100 / 252


def build_market_factor(start: str, end: str) -> pd.DataFrame:
    """
    Construct the daily market excess-return factor (Mkt-RF).

    Returns a DataFrame with columns:
        market_return  — SPY daily log return
        rf_rate        — daily risk-free rate (decimal)
        market_excess  — market_return − rf_rate  (the Mkt-RF factor)
    """
    market_returns = load_returns([MARKET_PROXY], start, end)[MARKET_PROXY]
    rf_rate = _load_risk_free_rate(start, end)

    # Reindex rf_rate to the equity calendar; forward-fill any remaining gaps
    # (^IRX may be absent on some days SPY trades, e.g. certain US holidays).
    rf_aligned = rf_rate.reindex(market_returns.index).ffill()

    return pd.DataFrame({
        "market_return": market_returns,
        "rf_rate": rf_aligned,
        "market_excess": market_returns - rf_aligned,
    }).dropna()


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
