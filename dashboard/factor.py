"""
Factor engine integration for the dashboard.

Public functions:
    portfolio_ff3_betas(weights)   — portfolio headline 7-factor betas for given weights (cache_resource)
    current_portfolio_betas()      — portfolio_ff3_betas() for the user's saved portfolio (data/user_prefs.json)
    ticker_ff3_profile(t)          — single-ticker 7-factor OLS regression (cache_data: 24h)
    portfolio_impact_full(t, key)  — full blended-portfolio vol/Sharpe/correlation impact of
                                      adding ticker t at 5% (cache_data: 24h, keyed to the
                                      Portfolio page's cache timestamp — see docstring below)

Function names kept as "_ff3_" for continuity with existing dashboard callers;
both now run the full 7-factor model: Fama-French 5 (market, size, value,
profitability, investment) + Carhart momentum + this platform's proprietary
GP (Gross Profitability) factor.  GP now has full history back to 2013 (EDGAR
XBRL-sourced, see factor_engine/factors/gp.py) matching the platform's other
six factors — label any GP-specific display "Gross Profitability (2013-present)".

portfolio_ff3_betas() is a cache_resource singleton so it runs exactly once per
server process regardless of how many sessions are open.  pre-warming it in
streamlit_app.py ensures the Factor Profile tab is always instant.

Fast path for portfolio betas: reads data/portfolio_analysis_cache.json when the
Portfolio page has already run the analysis *for the same weights*.  Slow path:
calls analyze_portfolio() directly (~10–30s, downloads non-cached ticker CSVs
once).

Portfolio weights come from the user's saved holdings (data/user_prefs.json),
not a hardcoded constant — see streamlit_app.py's prewarm call — so this
reflects whatever portfolio the user actually owns.
"""
from __future__ import annotations

import streamlit as st

_ANALYSIS_START = "2021-01-04"
_ANALYSIS_END   = "2024-12-31"


@st.cache_resource(show_spinner=False)
def portfolio_ff3_betas(weights: dict[str, float] | None = None) -> dict | None:
    """
    Portfolio headline 7-factor betas for the given weights.  Cached per distinct
    weights dict (once per server process for that portfolio).

    Parameters
    ----------
    weights : ticker -> raw weight dict. Defaults to the hardcoded example
        portfolio (factor_engine.portfolio.WEIGHTS) when omitted.

    Returns the same dict structure as factor_engine.portfolio._run_ff7_ols():
        beta_market, beta_smb, beta_hml, beta_rmw, beta_cma, beta_mom, beta_gp,
        alpha_annualised, r_squared, t_stat_market, t_stat_smb, t_stat_hml,
        t_stat_rmw, t_stat_cma, t_stat_mom, t_stat_gp, n_obs, start, end
    Returns None on failure (missing dependencies, network error, etc.).
    """
    from dashboard.holdings import normalize_weights_dict
    from factor_engine.portfolio import _RAW_WEIGHTS as _DEFAULT_WEIGHTS
    weights = weights if weights is not None else dict(_DEFAULT_WEIGHTS)
    # Normalize here so this always compares/caches against the same canonical
    # representation the Portfolio page uses (dashboard.cache stores whatever
    # was passed to analyze_portfolio, which the Portfolio page always
    # normalizes before calling).
    weights = normalize_weights_dict(weights)

    # Fast path — Portfolio page already computed and saved this exact portfolio
    try:
        from dashboard.cache import load as _load_disk
        from dashboard.holdings import weights_match
        cached = _load_disk()
        if cached and cached.get("headline") and weights_match(cached.get("raw_weights"), weights):
            return cached["headline"]
    except Exception:
        pass

    # Slow path — download and compute; also saves disk cache for next time
    try:
        from factor_engine.portfolio import analyze_portfolio
        result = analyze_portfolio(start=_ANALYSIS_START, end=_ANALYSIS_END, weights=weights)
        try:
            from dashboard.cache import save as _save_disk
            _save_disk(result, [])   # stress_tests populated by Portfolio page later
        except Exception:
            pass
        return result["headline"]
    except Exception:
        return None


def current_portfolio_betas() -> dict | None:
    """portfolio_ff3_betas() for the user's currently saved portfolio (data/user_prefs.json)."""
    from dashboard.holdings import weights_dict
    from dashboard.prefs import load as _load_prefs
    weights = weights_dict(_load_prefs()["portfolio"])
    return portfolio_ff3_betas(weights)


@st.cache_data(ttl=86400, show_spinner=False)
def ticker_ff3_profile(ticker: str) -> dict | None:
    """
    Full 7-factor regression for a single ticker (Fama-French 5 + Carhart
    momentum + proprietary GP).

    Uses get_ff7_daily() (Ken French series cached at data/french/, GP series
    cached at data/gp/) and loads ticker price history via data_loader (CSV
    cache then yfinance fallback).  The loaded CSV is saved to data/ so
    subsequent calls are instantaneous.

    Returns None gracefully for: new IPOs, delisted tickers, < 60 trading days
    of overlapping data, or any other failure.  GP's own coverage now spans
    2013-present (EDGAR XBRL-sourced), fully covering the default
    _ANALYSIS_START of 2021-01-04 — no trim from GP specifically.
    """
    try:
        from factor_engine.french_data import get_ff7_daily
        from factor_engine.data_loader import load_returns
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant

        factors = get_ff7_daily(_ANALYSIS_START, _ANALYSIS_END)
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
        factor_cols = ["mkt_excess", "smb", "hml", "rmw", "cma", "mom", "gp"]
        X = add_constant(aligned[factor_cols])
        m = OLS(excess, X).fit()

        profile = {
            "ticker":           ticker,
            "beta_market":      round(m.params["mkt_excess"], 3),
            "beta_smb":         round(m.params["smb"],        3),
            "beta_hml":         round(m.params["hml"],        3),
            "beta_rmw":         round(m.params["rmw"],        3),
            "beta_cma":         round(m.params["cma"],        3),
            "beta_mom":         round(m.params["mom"],        3),
            "beta_gp":          round(m.params["gp"],         3),
            "alpha_annualized": round(m.params["const"] * 252, 4),
            "r_squared":        round(m.rsquared,             3),
            "n_obs":            int(m.nobs),
            "start":            _ANALYSIS_START,
            "end":              _ANALYSIS_END,
        }
        for col, key in (("mkt_excess", "market"), ("smb", "smb"), ("hml", "hml"),
                         ("rmw", "rmw"), ("cma", "cma"), ("mom", "mom"), ("gp", "gp")):
            profile[f"t_stat_{key}"] = round(m.tvalues[col], 2)
        return profile
    except Exception:
        return None


_IMPACT_WEIGHT_PCT = 0.05


@st.cache_data(
    ttl=86400,
    show_spinner="Computing full portfolio impact (volatility, Sharpe, correlation)…",
)
def portfolio_impact_full(ticker: str, portfolio_cache_key: str) -> dict:
    """
    Full recompute (not an approximation) of the current portfolio's volatility,
    Sharpe ratio, and trailing correlation profile with `ticker` added at
    _IMPACT_WEIGHT_PCT (5%), scaling existing positions down proportionally —
    same 95/5 blend as _portfolio_impact_line's beta sentence in
    app_pages/signals.py, just applied to the actual per-holding return series
    (via analyze_portfolio()) instead of pre-computed betas, so the new
    position's real cross-correlation with existing holdings is captured, not
    just a linear blend of two independently-estimated beta vectors.

    portfolio_cache_key should be the current disk cache's `cached_at` timestamp
    (dashboard.cache.load()) — passed explicitly as a cache-busting argument so
    Streamlit's cache_data invalidates whenever the user re-runs the Portfolio
    page's analysis, rather than serving a stale before/after comparison for up
    to the full 24h TTL.

    Returns {"error": <code>} on failure rather than a bare None — same
    convention as dashboard.valuation.ticker_dcf_valuation — so the caller can
    show an accurate reason instead of a generic message. Confirmed necessary
    in practice: CRWV (a 2025 IPO) has no price history back to a typical
    2021-2024 analysis window, which is a DATA problem specific to the
    candidate ticker, not a stale/missing Portfolio-page cache — conflating the
    two would incorrectly tell the user to "refresh the Portfolio page" for a
    problem refreshing it can't fix.

    Error codes:
        no_portfolio_cache      — no current, up-to-date Portfolio-page analysis
                                   cached for the exact holdings in
                                   data/user_prefs.json (missing cache, stale
                                   weights, or predates risk_metrics/
                                   concentration). Caller should direct the user
                                   to run/refresh the Portfolio page.
        insufficient_candidate_data — the candidate ticker itself lacks enough
                                   price history over the cached analysis
                                   window (new IPO, delisted, thin overlap).
                                   Caller should NOT suggest refreshing the
                                   Portfolio page — that isn't the cause.
    """
    from dashboard.cache import load as _load_disk
    from dashboard.holdings import normalize_weights_dict, weights_dict, weights_match
    from dashboard.prefs import load as _load_prefs
    from factor_engine.concentration import trailing_correlation_matrix
    from factor_engine.portfolio import analyze_portfolio
    from factor_engine.risk_metrics import compute_risk_metrics

    cached = _load_disk()
    current_weights = normalize_weights_dict(weights_dict(_load_prefs()["portfolio"]))
    if (
        cached is None
        or not cached.get("risk_metrics")
        or not cached.get("concentration")
        or not weights_match(cached.get("raw_weights"), current_weights)
    ):
        return {"error": "no_portfolio_cache"}

    # Scale existing (already-normalized) weights down by 5%, add the candidate
    # at 5% — or top up its existing position if it's already a holding, rather
    # than the dict merge silently clobbering it.
    blended = {t: (1 - _IMPACT_WEIGHT_PCT) * w for t, w in cached["weights"].items()}
    blended[ticker] = blended.get(ticker, 0.0) + _IMPACT_WEIGHT_PCT

    try:
        result = analyze_portfolio(start=cached["start"], end=cached["end"], weights=blended)
    except Exception:
        return {"error": "insufficient_candidate_data"}

    blended_risk = compute_risk_metrics(result["combined_rets"], result["factors"])
    blended_corr = trailing_correlation_matrix(result["all_returns"])
    if ticker not in blended_corr.columns:
        return {"error": "insufficient_candidate_data"}
    corr_row = blended_corr[ticker]

    threshold = cached["concentration"].get("correlation_threshold", 0.70)
    existing_tickers = [t for t in cached["weights"] if t != ticker]
    correlated_with = [
        t for t in existing_tickers if t in corr_row.index and corr_row[t] > threshold
    ]

    # Does the candidate join an existing multi-holding clique (every member
    # >threshold correlated with it), i.e. deepen an already-established
    # mutual-overlap group — vs. just a one-off pairwise correlation?
    deepened_clique = None
    for clique in cached["concentration"].get("trailing_cliques", []):
        members = clique["tickers"]
        if len(members) >= 2 and all(
            m in corr_row.index and corr_row[m] > threshold for m in members
        ):
            deepened_clique = members
            break

    if deepened_clique:
        tier = "deepens"
    elif correlated_with:
        tier = "neutral"
    else:
        tier = "diversifies"

    return {
        "current_risk":    cached["risk_metrics"],
        "blended_risk":    blended_risk,
        "tier":            tier,
        "deepened_clique": deepened_clique,
        "correlated_with": correlated_with,
    }
