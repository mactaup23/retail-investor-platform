"""
Portfolio page — Fama-French 3-factor analysis of the real portfolio.

Layout
------
Header row:  date range inputs + Refresh button + last-analysed timestamp
Factor row:  β_mkt · β_smb · β_hml · R² · Alpha metric cards
Summary:     plain-English interpretation from the factor engine
Attribution: stacked bar chart (weighted beta contributions per holding)
             + full per-holding table
Stress tests: three scenario cards (2008 · COVID · 2022 Rate Hikes)

Data flow
---------
Results are expensive (~20s on first run — network + OLS).  They are cached
to data/portfolio_analysis_cache.json and served from there on subsequent
page loads.  The Refresh button reruns the analysis and overwrites the cache.

PLACEHOLDER — Edit holdings
---------------------------
# TODO: "Edit holdings" section — allow updating portfolio weights without
# touching factor_engine/portfolio.py directly. Planned as a future polish
# item. Currently weights are hardcoded in factor_engine/portfolio.py.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import altair as alt

import dashboard.cache as analysis_cache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:+.2f}%"


def _render_stress_card(scenario: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**{scenario['label']}**")
        st.caption(scenario["description"])
        col_p, col_spy, col_d = st.columns(3)
        with col_p:
            v = scenario["period_return"]
            color = "green" if v >= 0 else "red"
            st.metric("Portfolio (est.)", f"{v * 100:+.1f}%")
        with col_spy:
            spy = scenario["spy_return"]
            st.metric("SPY actual", f"{spy * 100:+.1f}%")
        with col_d:
            diff = scenario["diff_vs_spy"]
            st.metric("vs SPY", f"{diff * 100:+.1f}%")

        # Factor decomposition
        mkt  = scenario.get("mkt_contrib", 0)
        smb  = scenario.get("smb_contrib", 0)
        hml  = scenario.get("hml_contrib", 0)
        rf   = scenario.get("rf_contrib", 0)
        alph = scenario.get("alpha_contrib", 0)
        st.caption(
            f"Decomposition — Market: {mkt * 100:+.1f}%  "
            f"SMB: {smb * 100:+.1f}%  "
            f"HML: {hml * 100:+.1f}%  "
            f"RF: {rf * 100:+.1f}%"
        )
        st.caption(
            ":gray[Model-based risk characterisation using current betas applied to "
            "historical factor returns. Not a backtest — the portfolio didn't exist then.]"
        )


def _run_analysis(start: str, end: str) -> dict:
    """Run the full factor analysis and stress tests. Returns a merged results dict."""
    from factor_engine.portfolio import analyze_portfolio
    from factor_engine.stress_test import run_stress_tests

    results = analyze_portfolio(start=start, end=end)
    h = results["headline"]
    stress = run_stress_tests(
        beta_market=h["beta_market"],
        beta_smb=h["beta_smb"],
        beta_hml=h["beta_hml"],
        alpha_daily=h["alpha_daily"],
    )
    analysis_cache.save(results, stress)
    return {**results, "stress_tests": stress}


# ---------------------------------------------------------------------------
# Sidebar — date range and refresh control
# ---------------------------------------------------------------------------

with st.sidebar:
    st.divider()
    with st.form("portfolio_date_form", border=False):
        st.caption("Analysis period")
        col_s, col_e = st.columns(2)
        with col_s:
            start_date = st.text_input("Start", value="2021-01-04", key="pf_start")
        with col_e:
            end_date = st.text_input("End", value="2024-12-31", key="pf_end")
        refresh = st.form_submit_button(
            ":material/refresh: Refresh analysis", type="primary"
        )


# ---------------------------------------------------------------------------
# Load or run analysis
# ---------------------------------------------------------------------------

st.header(":material/bar_chart: Portfolio")

cached = analysis_cache.load()

if refresh:
    with st.spinner("Running factor analysis and stress tests (~20 seconds)…"):
        try:
            data = _run_analysis(start_date, end_date)
            st.success("Analysis complete.", icon=":material/check_circle:")
        except Exception as e:
            st.error(f"Analysis failed: {e}", icon=":material/error:")
            st.stop()
elif cached is not None:
    data = cached
else:
    with st.spinner("Running factor analysis on first load (~20 seconds)…"):
        try:
            data = _run_analysis(start_date, end_date)
            st.rerun()
        except Exception as e:
            st.error(f"Analysis failed: {e}", icon=":material/error:")
            st.stop()

# Show cache age
age = analysis_cache.age_str(data)
st.caption(
    f"Analysis period: {data['start']} → {data['end']}  ·  Last run: {age}  ·  "
    f"Click **Refresh analysis** in the sidebar to update."
)


# ---------------------------------------------------------------------------
# Factor exposure cards
# ---------------------------------------------------------------------------

headline = data["headline"]

st.subheader("Factor exposures")
with st.container(horizontal=True):
    st.metric("β market",    f"{headline['beta_market']:+.3f}",  border=True)
    st.metric("β SMB",       f"{headline['beta_smb']:+.3f}",     border=True)
    st.metric("β HML",       f"{headline['beta_hml']:+.3f}",     border=True)
    st.metric("R²",          f"{headline['r_squared']:.3f}",     border=True)
    st.metric("Alpha (ann.)", _pct(headline["alpha_annualised"]), border=True)

st.caption(
    f"FF3 model fit: R² = {headline['r_squared']:.3f} explains "
    f"{headline['r_squared'] * 100:.1f}% of daily portfolio variance  ·  "
    f"n = {headline['n_obs']} trading days  ·  "
    f"Period: {data['start']} → {data['end']}"
)

# Plain-English summary
st.subheader("What the numbers mean")
st.info(data["summary_text"], icon=":material/lightbulb:")


# ---------------------------------------------------------------------------
# Per-holding attribution
# ---------------------------------------------------------------------------

st.subheader("Per-holding attribution")

per_holding = data["per_holding"]
ph_df = pd.DataFrame(per_holding)[
    ["ticker", "weight", "beta_market", "beta_smb", "beta_hml",
     "wtd_beta_market", "wtd_beta_smb", "wtd_beta_hml",
     "alpha_annualised", "r_squared", "factor_basis"]
].copy()
ph_df["weight_pct"] = (ph_df["weight"] * 100).round(2)
ph_df = ph_df.sort_values("weight", ascending=False)

# Stacked bar chart — weighted beta contributions
chart_df = pd.DataFrame([
    row
    for r in per_holding
    for row in [
        {"ticker": r["ticker"], "factor": "β market", "contribution": r["wtd_beta_market"]},
        {"ticker": r["ticker"], "factor": "β SMB",    "contribution": r["wtd_beta_smb"]},
        {"ticker": r["ticker"], "factor": "β HML",    "contribution": r["wtd_beta_hml"]},
    ]
])

bar_chart = (
    alt.Chart(chart_df)
    .mark_bar()
    .encode(
        x=alt.X("ticker:N", sort="-y", title=None),
        y=alt.Y("contribution:Q", title="Weighted beta contribution"),
        color=alt.Color(
            "factor:N",
            scale=alt.Scale(
                domain=["β market", "β SMB", "β HML"],
                range=["#60A5FA", "#34D399", "#A78BFA"],
            ),
            legend=alt.Legend(title=None),
        ),
        tooltip=["ticker:N", "factor:N", alt.Tooltip("contribution:Q", format="+.4f")],
    )
    .properties(height=280)
)
st.altair_chart(bar_chart, use_container_width=True)

# Attribution table
st.dataframe(
    ph_df.rename(columns={
        "ticker":          "Ticker",
        "weight_pct":      "Weight %",
        "beta_market":     "β market",
        "beta_smb":        "β SMB",
        "beta_hml":        "β HML",
        "wtd_beta_market": "Wtd β mkt",
        "wtd_beta_smb":    "Wtd β SMB",
        "wtd_beta_hml":    "Wtd β HML",
        "alpha_annualised":"Alpha (ann.)",
        "r_squared":       "R²",
        "factor_basis":    "Factor basis",
    }),
    hide_index=True,
    column_config={
        "Weight %":     st.column_config.NumberColumn(format="%.2f%%"),
        "β market":     st.column_config.NumberColumn(format="%.4f"),
        "β SMB":        st.column_config.NumberColumn(format="%.4f"),
        "β HML":        st.column_config.NumberColumn(format="%.4f"),
        "Wtd β mkt":    st.column_config.NumberColumn(format="%.4f"),
        "Wtd β SMB":    st.column_config.NumberColumn(format="%.4f"),
        "Wtd β HML":    st.column_config.NumberColumn(format="%.4f"),
        "Alpha (ann.)": st.column_config.NumberColumn(format="%+.4f"),
        "R²":           st.column_config.NumberColumn(format="%.4f"),
    },
)
st.caption(
    "Weighted betas (Wtd β) = individual ticker beta × portfolio weight. "
    "Their sum approximates — but won't exactly match — the headline portfolio betas, "
    "because independent regressions share the same factor matrix but not the same residual structure. "
    "VXUS uses US FF3 factors as an approximation (international ETF — labeled in Factor basis column)."
)


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

st.subheader("Stress scenarios")

stress = data.get("stress_tests", [])
if not stress:
    with st.spinner("Computing stress scenarios…"):
        try:
            from factor_engine.stress_test import run_stress_tests
            import json
            h = data["headline"]
            stress = run_stress_tests(
                beta_market=h["beta_market"],
                beta_smb=h["beta_smb"],
                beta_hml=h["beta_hml"],
                alpha_daily=h["alpha_daily"],
            )
            data["stress_tests"] = stress
            analysis_cache.CACHE_PATH.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            st.caption(":gray[Stress test results unavailable. Click Refresh analysis to recompute.]")

if stress:
    cols = st.columns(len(stress))
    for col, scenario in zip(cols, stress):
        with col:
            _render_stress_card(scenario)
