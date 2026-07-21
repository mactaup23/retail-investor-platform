"""
Quality & Health metrics integration for the dashboard (DuPont, Altman Z'',
Piotroski F-Score, Beneish M-Score — see quality/ package).

Same caching rationale as dashboard.valuation.ticker_dcf_valuation: a
per-ticker, on-demand computation from annual fundamentals that don't
change intraday, not a full-universe batch job — a daily cache is
appropriate.
"""
from __future__ import annotations

import streamlit as st


@st.cache_data(ttl=86400, show_spinner=False)
def ticker_dupont(ticker: str) -> dict:
    from quality.dupont import compute_dupont
    try:
        return compute_dupont(ticker)
    except Exception as e:
        return {"status": "unexpected_failure", "reason": str(e)}


@st.cache_data(ttl=86400, show_spinner=False)
def ticker_altman_z(ticker: str) -> dict:
    from quality.altman import compute_altman_z
    try:
        return compute_altman_z(ticker)
    except Exception as e:
        return {"status": "unexpected_failure", "reason": str(e)}


@st.cache_data(ttl=86400, show_spinner=False)
def ticker_piotroski_f(ticker: str) -> dict:
    from quality.piotroski import compute_piotroski_f
    try:
        return compute_piotroski_f(ticker)
    except Exception as e:
        return {"status": "unexpected_failure", "reason": str(e)}


@st.cache_data(ttl=86400, show_spinner=False)
def ticker_beneish_m(ticker: str) -> dict:
    from quality.beneish import compute_beneish_m
    try:
        return compute_beneish_m(ticker)
    except Exception as e:
        return {"status": "unexpected_failure", "reason": str(e)}
