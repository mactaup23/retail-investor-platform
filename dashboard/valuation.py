"""
DCF valuation engine integration for the dashboard.

Public function:
    ticker_dcf_valuation(t) — full DCF run for one ticker (cache_data: 24h)

Same TTL as dashboard.factor.ticker_ff3_profile (the beta this engine
reuses) — a company's underlying financials, beta, and 10-year Treasury
yield don't meaningfully change intraday, so a daily cache is appropriate
for a per-ticker, on-demand computation like this (unlike GP/PEAD, which
batch-process the full universe, this is only ever run for whichever
ticker a card or tab is actually viewing).
"""
from __future__ import annotations

import streamlit as st


@st.cache_data(ttl=86400, show_spinner=False)
def ticker_dcf_valuation(ticker: str) -> dict:
    """
    Full DCF run for one ticker (dcf.valuation.run_dcf) — business-model-fit
    check, WACC, and Bull/Base/Bear scenarios.

    Always returns a dict containing "ticker"; check for "error" before
    reading "scenarios" — see dcf.valuation.run_dcf's docstring for the
    possible error values (unsuitable_business_model, no_xbrl_fundamentals,
    insufficient_history, no_diluted_shares, no_beta, no_market_data).
    Wraps run_dcf in a try/except so an unexpected failure (e.g. a network
    timeout) degrades to an "unexpected_failure" error dict rather than
    crashing the dashboard — run_dcf itself already handles every known
    data-quality gap without raising.
    """
    from dcf.valuation import run_dcf
    try:
        return run_dcf(ticker)
    except Exception as e:
        return {"ticker": ticker, "error": "unexpected_failure", "reason": str(e)}
