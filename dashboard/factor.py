"""
Factor engine integration for the dashboard.

Public functions:
    portfolio_ff3_betas()   — portfolio headline FF3 betas (cache_resource: once per process)
    ticker_ff3_profile(t)   — single-ticker FF3 OLS regression (cache_data: 24h)

portfolio_ff3_betas() is a cache_resource singleton so it runs exactly once per
server process regardless of how many sessions are open.  pre-warming it in
streamlit_app.py ensures the Factor Profile tab is always instant.

Fast path for portfolio betas: reads data/portfolio_analysis_cache.json when the
Portfolio page has already run the analysis.  Slow path: calls analyze_portfolio()
directly (~10–30s, downloads non-cached ticker CSVs once).
"""
from __future__ import annotations

import streamlit as st

_ANALYSIS_START = "2021-01-04"
_ANALYSIS_END   = "2024-12-31"


@st.cache_resource(show_spinner=False)
def portfolio_ff3_betas() -> dict | None:
    """
    Portfolio headline FF3 betas.  Runs once per server process.

    Returns the same dict structure as factor_engine.portfolio._run_ff3_ols():
        beta_market, beta_smb, beta_hml, alpha_annualised, r_squared,
        t_stat_market, t_stat_smb, t_stat_hml, n_obs, start, end
    Returns None on failure (missing dependencies, network error, etc.).
    """
    # Fast path — Portfolio page already computed and saved this
    try:
        from dashboard.cache import load as _load_disk
        cached = _load_disk()
        if cached and cached.get("headline"):
            return cached["headline"]
    except Exception:
        pass

    # Slow path — download and compute; also saves disk cache for next time
    try:
        from factor_engine.portfolio import analyze_portfolio
        result = analyze_portfolio(start=_ANALYSIS_START, end=_ANALYSIS_END)
        try:
            from dashboard.cache import save as _save_disk
            _save_disk(result, [])   # stress_tests populated by Portfolio page later
        except Exception:
            pass
        return result["headline"]
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def ticker_ff3_profile(ticker: str) -> dict | None:
    """
    Fama-French 3-factor regression for a single ticker.

    Uses the French FF3 daily factor file (data/french/us_ff3_daily.csv, already
    cached on disk) and loads ticker price history via data_loader (CSV cache then
    yfinance fallback).  The loaded CSV is saved to data/ so subsequent calls are
    instantaneous.

    Returns None gracefully for: new IPOs, delisted tickers, < 60 trading days
    of overlapping data, or any other failure.
    """
    try:
        from factor_engine.french_data import get_ff3_daily
        from factor_engine.data_loader import load_returns
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant

        factors = get_ff3_daily(_ANALYSIS_START, _ANALYSIS_END)
        rets    = load_returns([ticker], _ANALYSIS_START, _ANALYSIS_END)
        if ticker not in rets.columns or rets[ticker].isna().all():
            return None

        aligned = (
            rets[ticker]
            .to_frame("stock")
            .join(factors, how="inner")
            .dropna()
        )
        if len(aligned) < 60:
            return None

        excess = aligned["stock"] - aligned["rf"]
        X = add_constant(aligned[["mkt_excess", "smb", "hml"]])
        m = OLS(excess, X).fit()

        return {
            "ticker":           ticker,
            "beta_market":      round(m.params["mkt_excess"], 3),
            "beta_smb":         round(m.params["smb"],        3),
            "beta_hml":         round(m.params["hml"],        3),
            "alpha_annualized": round(m.params["const"] * 252, 4),
            "r_squared":        round(m.rsquared,             3),
            "t_stat_market":    round(m.tvalues["mkt_excess"], 2),
            "t_stat_smb":       round(m.tvalues["smb"],        2),
            "t_stat_hml":       round(m.tvalues["hml"],        2),
            "n_obs":            int(m.nobs),
            "start":            _ANALYSIS_START,
            "end":              _ANALYSIS_END,
        }
    except Exception:
        return None
