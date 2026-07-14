"""
Portfolio page — 7-factor analysis (Fama-French 5 + Carhart momentum + proprietary
Gross Profitability) of the user's editable portfolio.

Layout
------
Holdings:    manual ticker/weight editor (add, remove, CSV import) — persisted
             to data/user_prefs.json under the "portfolio" key
Header row:  date range inputs + Refresh button + last-analysed timestamp
Factor row:  β_mkt · β_smb · β_hml · β_rmw · β_cma · β_mom · β_gp · R² · Alpha metric cards
             (β_gp labeled "Gross Profitability, 2013–present" — EDGAR XBRL-sourced,
             matching the other six factors' history)
Summary:     plain-English interpretation from the factor engine
Attribution: stacked bar chart (weighted beta contributions per holding)
             + full per-holding table
Stress tests: three scenario cards (2008 · COVID · 2022 Rate Hikes) — GP's contribution
             is omitted (not zeroed) for scenarios predating its coverage

Data flow
---------
Results are expensive (~20s on first run — network + OLS).  They are cached
to data/portfolio_analysis_cache.json and served from there on subsequent
page loads.  Editing holdings, clicking Refresh, or clicking Run Portfolio
Analysis reruns the analysis and overwrites the cache. If the cache was
computed for a different set of holdings than currently saved, it's treated
as stale and recomputed automatically.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import altair as alt

import dashboard.cache as analysis_cache
import dashboard.holdings as holdings_module
import dashboard.prefs as prefs_module

# ---------------------------------------------------------------------------
# Holdings state
# ---------------------------------------------------------------------------

def _save_holdings(holdings: list[dict]) -> None:
    p = prefs_module.load()
    p["portfolio"] = holdings
    prefs_module.save(p)


if "pf_holdings" not in st.session_state:
    st.session_state.pf_holdings = prefs_module.load()["portfolio"]


def _current_weights_normalized() -> tuple[dict[str, float], bool]:
    """Return (normalized weights dict, was_normalized) for the current holdings."""
    w = holdings_module.weights_dict(st.session_state.pf_holdings)
    total = sum(w.values())
    if total <= 0 or abs(total - 1.0) < 1e-6:
        return w, False
    return holdings_module.normalize_weights_dict(w), True


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
        rmw  = scenario.get("rmw_contrib", 0)
        cma  = scenario.get("cma_contrib", 0)
        mom  = scenario.get("mom_contrib", 0)
        rf   = scenario.get("rf_contrib", 0)
        alph = scenario.get("alpha_contrib", 0)
        gp_available = scenario.get("gp_available", False)
        gp   = scenario.get("gp_contrib")
        decomp_line = (
            f"Decomposition — Market: {mkt * 100:+.1f}%  "
            f"SMB: {smb * 100:+.1f}%  "
            f"HML: {hml * 100:+.1f}%  "
            f"RMW: {rmw * 100:+.1f}%  "
            f"CMA: {cma * 100:+.1f}%  "
            f"MOM: {mom * 100:+.1f}%  "
            f"RF: {rf * 100:+.1f}%"
        )
        if gp_available and gp is not None:
            decomp_line += f"  Gross Profitability (2013–present): {gp * 100:+.1f}%"
        st.caption(decomp_line)
        if not gp_available:
            st.caption(
                ":gray[Gross Profitability (2013–present) has insufficient history to cover "
                "this scenario — omitted from the estimate above rather than treated as zero.]"
            )
        st.caption(
            ":gray[Model-based risk characterisation using current betas applied to "
            "historical factor returns. Not a backtest — the portfolio didn't exist then.]"
        )


def _run_analysis(start: str, end: str, weights: dict[str, float]) -> dict:
    """Run the full factor analysis and stress tests. Returns a merged results dict."""
    from factor_engine.portfolio import analyze_portfolio
    from factor_engine.stress_test import run_stress_tests

    results = analyze_portfolio(start=start, end=end, weights=weights)
    h = results["headline"]
    stress = run_stress_tests(
        beta_market=h["beta_market"],
        beta_smb=h["beta_smb"],
        beta_hml=h["beta_hml"],
        beta_rmw=h["beta_rmw"],
        beta_cma=h["beta_cma"],
        beta_mom=h["beta_mom"],
        beta_gp=h["beta_gp"],
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


st.header(":material/bar_chart: Portfolio")


# ---------------------------------------------------------------------------
# Holdings editor
# ---------------------------------------------------------------------------

st.subheader("Holdings")

with st.form("add_position_form", clear_on_submit=True, border=False):
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        new_ticker = st.text_input(
            "Ticker", key="pf_new_ticker", label_visibility="collapsed",
            placeholder="Ticker (e.g. AAPL)",
        )
    with c2:
        new_weight = st.number_input(
            "Weight %", key="pf_new_weight", label_visibility="collapsed",
            min_value=0.0, max_value=100.0, value=5.0, step=0.1, format="%.1f",
        )
    with c3:
        add_clicked = st.form_submit_button(":material/add: Add Position", width="stretch")

if add_clicked:
    ticker = (new_ticker or "").strip().upper()
    existing_tickers = {h["ticker"] for h in st.session_state.pf_holdings}
    if not ticker:
        st.error("Enter a ticker symbol.", icon=":material/error:")
    elif ticker in existing_tickers:
        st.error(f"{ticker} is already in your portfolio.", icon=":material/error:")
    elif len(st.session_state.pf_holdings) >= holdings_module.MAX_POSITIONS:
        st.error(
            f"Maximum {holdings_module.MAX_POSITIONS} positions reached.",
            icon=":material/error:",
        )
    elif not (holdings_module.MIN_WEIGHT_PCT <= new_weight <= holdings_module.MAX_WEIGHT_PCT):
        st.error(
            f"Weight must be between {holdings_module.MIN_WEIGHT_PCT}% and "
            f"{holdings_module.MAX_WEIGHT_PCT}%.",
            icon=":material/error:",
        )
    else:
        with st.spinner(f"Validating {ticker}…"):
            result = holdings_module.lookup_ticker(ticker)
        if result["valid"]:
            st.session_state.pf_holdings.append({"ticker": ticker, "weight": new_weight / 100})
            _save_holdings(st.session_state.pf_holdings)
            st.success(f"Added {ticker} — {result['name']}", icon=":material/check_circle:")
            st.rerun()
        else:
            st.error(result["error"], icon=":material/error:")

if st.session_state.pf_holdings:
    hdr = st.columns([2, 4, 2, 1])
    hdr[0].caption("**Ticker**")
    hdr[1].caption("**Company**")
    hdr[2].caption("**Weight %**")
    hdr[3].caption("")
    for h in list(st.session_state.pf_holdings):
        ticker = h["ticker"]
        info = holdings_module.lookup_ticker(ticker)
        row = st.columns([2, 4, 2, 1])
        row[0].write(ticker)
        row[1].write(info["name"] if info["valid"] and info["name"] else ":gray[unknown]")
        row[2].write(f"{h['weight'] * 100:.2f}%")
        if row[3].button(":material/delete:", key=f"remove_{ticker}", help=f"Remove {ticker}"):
            st.session_state.pf_holdings = [
                x for x in st.session_state.pf_holdings if x["ticker"] != ticker
            ]
            _save_holdings(st.session_state.pf_holdings)
            st.rerun()

    total_pct = holdings_module.total_weight_pct(st.session_state.pf_holdings)
    if abs(total_pct - 100) < 0.01:
        st.markdown(f":green[**Total weight: {total_pct:.2f}%**]")
    elif abs(total_pct - 100) <= 2:
        st.markdown(f":orange[**Total weight: {total_pct:.2f}%**]  ·  :gray[will be normalized to 100% on run]")
    else:
        st.markdown(f":red[**Total weight: {total_pct:.2f}%**]  ·  :gray[will be normalized to 100% on run]")
else:
    st.info("Add at least one position to run a factor analysis.", icon=":material/info:")

with st.expander("Import from CSV", icon=":material/upload:"):
    st.download_button(
        "Download CSV template",
        data=holdings_module.csv_template_bytes(),
        file_name="portfolio_template.csv",
        mime="text/csv",
    )
    st.caption(
        "Two columns: ticker, weight. Weight may be a decimal (0.2437) or a "
        "percentage (24.37) — the format is detected automatically. Importing "
        "replaces your current holdings table."
    )
    uploaded = st.file_uploader(
        "Upload ticker/weight CSV", type=["csv"], key="pf_csv_upload",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        rows, errors = holdings_module.parse_csv(uploaded.getvalue())
        if errors:
            for e in errors:
                st.error(e, icon=":material/error:")
        else:
            with st.spinner(f"Validating {len(rows)} ticker(s)…"):
                validated, invalid = [], []
                for r in rows:
                    info = holdings_module.lookup_ticker(r["ticker"])
                    if info["valid"]:
                        validated.append({**r, "name": info["name"]})
                    else:
                        invalid.append(r["ticker"])

            preview_df = pd.DataFrame([
                {
                    "Ticker": r["ticker"],
                    "Company": r.get("name", "—"),
                    "Weight %": f"{r['weight'] * 100:.2f}%",
                    "Status": "✓ valid" if r["ticker"] not in invalid else "✗ not found",
                }
                for r in rows
            ])
            st.dataframe(preview_df, hide_index=True, width="stretch")

            if invalid:
                st.error(
                    "Ticker(s) not found — please check the symbol and try again: "
                    + ", ".join(invalid),
                    icon=":material/error:",
                )
            elif st.button("Apply import", type="primary", key="pf_apply_import"):
                st.session_state.pf_holdings = [
                    {"ticker": r["ticker"], "weight": r["weight"]} for r in validated
                ]
                _save_holdings(st.session_state.pf_holdings)
                st.success(f"Imported {len(validated)} position(s).", icon=":material/check_circle:")
                st.rerun()

run_clicked = st.button(
    ":material/play_arrow: Run Portfolio Analysis",
    type="primary",
    disabled=not st.session_state.pf_holdings,
)


# ---------------------------------------------------------------------------
# Load or run analysis
# ---------------------------------------------------------------------------

trigger_run = refresh or run_clicked

if trigger_run and not st.session_state.pf_holdings:
    st.error("Add at least one position before running analysis.", icon=":material/error:")
    st.stop()

cached = analysis_cache.load()
current_weights_normalized, _ = _current_weights_normalized()
cache_is_current = cached is not None and holdings_module.weights_match(
    cached.get("raw_weights"), current_weights_normalized
)

if trigger_run:
    weights, was_normalized = _current_weights_normalized()
    with st.spinner("Running factor analysis and stress tests (~20 seconds)…"):
        try:
            data = _run_analysis(start_date, end_date, weights)
            if was_normalized:
                st.caption(":gray[Weights normalized to 100%]")
            st.success("Analysis complete.", icon=":material/check_circle:")
        except Exception as e:
            st.error(f"Analysis failed: {e}", icon=":material/error:")
            st.stop()
elif cache_is_current:
    data = cached
else:
    if not st.session_state.pf_holdings:
        st.info(
            "Add at least one position above, then click **Run Portfolio Analysis**.",
            icon=":material/info:",
        )
        st.stop()
    weights, was_normalized = _current_weights_normalized()
    with st.spinner("Running factor analysis on first load (~20 seconds)…"):
        try:
            data = _run_analysis(start_date, end_date, weights)
            st.rerun()
        except Exception as e:
            st.error(f"Analysis failed: {e}", icon=":material/error:")
            st.stop()

# Show cache age
age = analysis_cache.age_str(data)
st.caption(
    f"Analysis period: {data['start']} → {data['end']}  ·  Last run: {age}  ·  "
    f"Click **Refresh analysis** in the sidebar or **Run Portfolio Analysis** above to update."
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
    st.metric("β RMW",       f"{headline['beta_rmw']:+.3f}",     border=True)
    st.metric("β CMA",       f"{headline['beta_cma']:+.3f}",     border=True)
    st.metric("β MOM",       f"{headline['beta_mom']:+.3f}",     border=True)
    st.metric(
        "β GP", f"{headline['beta_gp']:+.3f}", border=True,
        help="Gross Profitability (2013–present) — proprietary factor, sourced from "
             "SEC EDGAR XBRL, matching the other six factors' history.",
    )
    st.metric("R²",          f"{headline['r_squared']:.3f}",     border=True)
    st.metric("Alpha (ann.)", _pct(headline["alpha_annualised"]), border=True)

st.caption(
    f"7-factor model fit: R² = {headline['r_squared']:.3f} explains "
    f"{headline['r_squared'] * 100:.1f}% of daily portfolio variance  ·  "
    f"n = {headline['n_obs']} trading days  ·  "
    f"Period: {data['start']} → {data['end']}  ·  "
    f"GP is Gross Profitability (2013–present)"
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
    ["ticker", "weight", "beta_market", "beta_smb", "beta_hml", "beta_rmw", "beta_cma",
     "beta_mom", "beta_gp",
     "wtd_beta_market", "wtd_beta_smb", "wtd_beta_hml", "wtd_beta_rmw", "wtd_beta_cma",
     "wtd_beta_mom", "wtd_beta_gp",
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
        {"ticker": r["ticker"], "factor": "β RMW",    "contribution": r["wtd_beta_rmw"]},
        {"ticker": r["ticker"], "factor": "β CMA",    "contribution": r["wtd_beta_cma"]},
        {"ticker": r["ticker"], "factor": "β MOM",    "contribution": r["wtd_beta_mom"]},
        {"ticker": r["ticker"], "factor": "β GP (Gross Profitability, 2013–present)",
         "contribution": r["wtd_beta_gp"]},
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
                domain=["β market", "β SMB", "β HML", "β RMW", "β CMA", "β MOM",
                        "β GP (Gross Profitability, 2013–present)"],
                range=["#60A5FA", "#34D399", "#A78BFA", "#F472B6", "#38BDF8", "#FBBF24", "#FB923C"],
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
        "beta_rmw":        "β RMW",
        "beta_cma":        "β CMA",
        "beta_mom":        "β MOM",
        "beta_gp":         "β GP (2013–present)",
        "wtd_beta_market": "Wtd β mkt",
        "wtd_beta_smb":    "Wtd β SMB",
        "wtd_beta_hml":    "Wtd β HML",
        "wtd_beta_rmw":    "Wtd β RMW",
        "wtd_beta_cma":    "Wtd β CMA",
        "wtd_beta_mom":    "Wtd β MOM",
        "wtd_beta_gp":     "Wtd β GP (2013–present)",
        "alpha_annualised":"Alpha (ann.)",
        "r_squared":       "R²",
        "factor_basis":    "Factor basis",
    }),
    hide_index=True,
    column_config={
        "Weight %":              st.column_config.NumberColumn(format="%.2f%%"),
        "β market":              st.column_config.NumberColumn(format="%.4f"),
        "β SMB":                 st.column_config.NumberColumn(format="%.4f"),
        "β HML":                 st.column_config.NumberColumn(format="%.4f"),
        "β RMW":                 st.column_config.NumberColumn(format="%.4f"),
        "β CMA":                 st.column_config.NumberColumn(format="%.4f"),
        "β MOM":                 st.column_config.NumberColumn(format="%.4f"),
        "β GP (2013–present)":   st.column_config.NumberColumn(format="%.4f"),
        "Wtd β mkt":             st.column_config.NumberColumn(format="%.4f"),
        "Wtd β SMB":             st.column_config.NumberColumn(format="%.4f"),
        "Wtd β HML":             st.column_config.NumberColumn(format="%.4f"),
        "Wtd β RMW":             st.column_config.NumberColumn(format="%.4f"),
        "Wtd β CMA":             st.column_config.NumberColumn(format="%.4f"),
        "Wtd β MOM":             st.column_config.NumberColumn(format="%.4f"),
        "Wtd β GP (2013–present)": st.column_config.NumberColumn(format="%.4f"),
        "Alpha (ann.)":          st.column_config.NumberColumn(format="%+.4f"),
        "R²":                    st.column_config.NumberColumn(format="%.4f"),
    },
)
_intl_tickers = [r["ticker"] for r in per_holding if "intl" in r["factor_basis"]]
_intl_note = (
    f" {', '.join(_intl_tickers)} {'uses' if len(_intl_tickers) == 1 else 'use'} US FF7 factors "
    "as an approximation (international ETF — labeled in Factor basis column)."
    if _intl_tickers else ""
)
st.caption(
    "Weighted betas (Wtd β) = individual ticker beta × portfolio weight. "
    "Their sum approximates — but won't exactly match — the headline portfolio betas, "
    "because independent regressions share the same factor matrix but not the same residual structure. "
    "GP (Gross Profitability) is labeled '2013–present' throughout — sourced from SEC EDGAR "
    "XBRL, matching this platform's other 2013-era history floors (13F filings, MTUM ETF "
    "inception), though still short of RMW/CMA/HML's multi-decade Ken French history."
    + _intl_note
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
                beta_rmw=h["beta_rmw"],
                beta_cma=h["beta_cma"],
                beta_mom=h["beta_mom"],
                beta_gp=h["beta_gp"],
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
