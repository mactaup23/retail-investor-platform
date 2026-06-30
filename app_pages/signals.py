"""
Signals page — smart-money convergence tracker.

Layout
------
Sidebar:  Quarter selector, pipeline status + run command
Main:     Alert strip (EXIT SIGNALs, new STRENGTHENING, near-LT lots)
          ├── Watchlist tab  — scored active entries, add/remove inline
          └── Discovery tab  — full FinalSignal list, card layout with inline deep-dive

Deep-dive panel (6 tabs)
------------------------
1. Signal      — fund moves, NLP dimensions, convergence breakdown (DB data)
2. Price       — 1-year chart, period returns, volume, beta
3. Valuation   — multiples, analyst consensus, short interest
4. Financials  — revenue/margin charts, income table, FCF, balance sheet
5. Earnings    — beat/miss bar chart, EPS history
6. Factor      — FF3 profile vs portfolio (Module 1 engine)
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.db import (
    get_available_periods,
    get_pipeline_status,
    load_convergence_detail,
    load_fund_skill_map,
    load_nlp_detail,
    load_signals,
    load_watchlist_scored,
    load_watchlist_tickers,
    watchlist_add,
    watchlist_remove,
)
from dashboard.factor import portfolio_ff3_betas, ticker_ff3_profile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATUS_COLOR = {
    "STRENGTHENING": "green",
    "HOLDING":       "gray",
    "WEAKENING":     "orange",
    "EXIT SIGNAL":   "red",
}

_STATUS_ICON = {
    "STRENGTHENING": ":material/trending_up:",
    "HOLDING":       ":material/trending_flat:",
    "WEAKENING":     ":material/trending_down:",
    "EXIT SIGNAL":   ":material/exit_to_app:",
}

_TREND_COLOR = {
    "new":          "green",
    "accelerating": "green",
    "stable":       "gray",
    "fading":       "orange",
}

_DISC_MAX_CARDS = 60


# ---------------------------------------------------------------------------
# Conviction / NLP color helpers
# ---------------------------------------------------------------------------

def _conviction_info(v: float | None) -> tuple[str, str]:
    """Return (label, streamlit_color) for a score value."""
    if v is None:
        return "—", "gray"
    if v >= 0.70:
        return "Very High Conviction", "green"
    if v >= 0.50:
        return "High Conviction", "green"
    if v >= 0.35:
        return "Moderate", "orange"
    if v >= 0.20:
        return "Weak Signal", "gray"
    if v < 0:
        return "Bearish", "red"
    return "Weak Signal", "gray"


def _nlp_color(v: float | None) -> str:
    if v is None:
        return "gray"
    if v >= 0.20:
        return "green"
    if v >= 0.05:
        return "orange"
    if v < 0:
        return "red"
    return "gray"


# ---------------------------------------------------------------------------
# Shared card-level helpers
# ---------------------------------------------------------------------------

def _badge(status: str | None) -> None:
    if not status:
        st.caption(":gray[No signal this quarter]")
        return
    st.badge(status, icon=_STATUS_ICON.get(status, ""), color=_STATUS_COLOR.get(status, "gray"))


def _render_score_cell(fs: float | None, conv: float | None) -> None:
    """Conviction label as primary; raw scores as captions."""
    if fs is None:
        st.caption(":gray[No signal this quarter]")
        return
    label, color = _conviction_info(fs)
    st.markdown(f":{color}[●] **{label}**")
    st.caption(f"Score: {fs:+.3f}")
    if conv is not None:
        _, conv_color = _conviction_info(conv)
        st.caption(f"Conv: :{conv_color}[{conv:+.3f}]")


def _render_nlp_inline(nlp_available: bool, nlp_score: float | None) -> None:
    """Badge only in the main card — reasoning is in the Signal tab."""
    if not nlp_available or nlp_score is None:
        st.caption(":gray[NLP: not yet run]")
        return
    color = _nlp_color(nlp_score)
    st.caption(f"NLP: :{color}[{nlp_score:+.3f}]")


# ---------------------------------------------------------------------------
# yfinance cached fetchers  (lazy, 1-hour TTL)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _yf_info(ticker: str) -> dict:
    try:
        import yfinance as yf
        return dict(yf.Ticker(ticker).info)
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _yf_history(ticker: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        return yf.Ticker(ticker).history(period="1y")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _yf_financials(ticker: str, annual: bool = False) -> pd.DataFrame:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.financials if annual else t.quarterly_financials
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _yf_cashflow(ticker: str, annual: bool = False) -> pd.DataFrame:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.cashflow if annual else t.quarterly_cashflow
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _yf_balance_sheet(ticker: str, annual: bool = False) -> pd.DataFrame:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.balance_sheet if annual else t.quarterly_balance_sheet
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _yf_earnings(ticker: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        return yf.Ticker(ticker).earnings_dates
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_row(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for c in candidates:
        if c in df.index:
            return df.loc[c]
    return None


def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except Exception:
        return None


def _fmt_b(v) -> str:
    f = _safe_float(v)
    if f is None:
        return "—"
    abs_f = abs(f)
    if abs_f >= 1e12:
        return f"${f / 1e12:.1f}T"
    if abs_f >= 1e9:
        return f"${f / 1e9:.1f}B"
    if abs_f >= 1e6:
        return f"${f / 1e6:.0f}M"
    return f"${f:,.0f}"


def _fmt_pct(v) -> str:
    f = _safe_float(v)
    return f"{f:.1f}%" if f is not None else "—"


def _to_quarter_label(s: str) -> str:
    """Convert '2026-04' or '2026-04-01' to 'Q2 2026'."""
    try:
        year, month = int(s[:4]), int(s[5:7])
        return f"Q{(month - 1) // 3 + 1} {year}"
    except (ValueError, IndexError):
        return s


# ---------------------------------------------------------------------------
# Tab 1 — Signal
# ---------------------------------------------------------------------------

_NLP_DIMENSIONS = [
    ("Guidance",               "guidance_delta",                0.25),
    ("Confidence",             "confidence_delta",              0.20),
    ("Customer Demand",        "customer_demand_delta",         0.20),
    ("Competitive Position",   "competitive_positioning_delta", 0.15),
    ("Operational Efficiency", "operational_efficiency_delta",  0.10),
    ("Risk Factors",           "risk_factors_delta",            0.05),
    ("Capital Allocation",     "capital_allocation_delta",      0.05),
]


def _fund_moves_df(moves: list[dict], skill_map: dict) -> pd.DataFrame:
    rows = []
    for m in moves:
        skill = skill_map.get(m["fund_name"], {})
        alpha = skill.get("alpha_annualized")
        reliable = skill.get("is_reliable")
        n_q = skill.get("n_quarters", 0)
        if alpha is not None and n_q > 0:
            rel_tag = "reliable" if reliable else "low-conf"
            skill_str = f"α {alpha * 100:+.1f}% ({n_q}q, {rel_tag})"
        else:
            skill_str = "unscored"
        rows.append({
            "Fund":       m["fund_name"],
            "Change":     m["change_type"],
            "Pos %":      f"{m['portfolio_pct']:.2f}%" if m.get("portfolio_pct") else "—",
            "Value":      f"${m['current_value_usd'] / 1e6:.0f}M" if m.get("current_value_usd") else "—",
            "Skill Wt":   f"{m['skill_weight']:.2f}",
            "Skill":      skill_str,
        })
    return pd.DataFrame(rows)


def _portfolio_impact_line(ticker: str) -> str | None:
    """
    Return a one-line English description of adding `ticker` at 5% to the current portfolio,
    or None if factor data is unavailable.
    """
    profile = ticker_ff3_profile(ticker)
    port    = portfolio_ff3_betas()
    if profile is None or port is None:
        return None

    pm,   psmb,  phml  = port.get("beta_market", 0), port.get("beta_smb", 0), port.get("beta_hml", 0)
    bm,   bsmb,  bhml  = profile["beta_market"], profile["beta_smb"], profile["beta_hml"]
    new_m, new_smb, new_hml = 0.95 * pm + 0.05 * bm, 0.95 * psmb + 0.05 * bsmb, 0.95 * phml + 0.05 * bhml

    dm = new_m - pm
    if dm > 0.01:
        market_part = f"increase your market beta from {pm:.2f} to {new_m:.2f}"
    elif dm < -0.01:
        market_part = f"reduce your market beta from {pm:.2f} to {new_m:.2f}"
    else:
        market_part = f"leave your market beta roughly flat ({pm:.2f} → {new_m:.2f})"

    d_smb = abs(new_smb - psmb)
    d_hml = abs(new_hml - phml)
    sec_part = ""
    if max(d_smb, d_hml) > 0.02:
        if d_hml >= d_smb:
            if new_hml < phml:
                desc = "deepen your existing growth tilt" if phml < -0.05 else "add a growth tilt"
            else:
                desc = "reinforce your existing value tilt" if phml > 0.05 else "add a value tilt"
            sec_part = f" and {desc} (HML {phml:+.2f} → {new_hml:+.2f})"
        else:
            if new_smb > psmb:
                desc = "increase small-cap exposure"
            else:
                desc = "reduce small-cap exposure"
            sec_part = f" and {desc} (SMB {psmb:+.2f} → {new_smb:+.2f})"

    return f"Adding {ticker} at 5% would {market_part}{sec_part}. Full factor breakdown in the Factor Profile tab."


def _render_signal_tab(row: dict, period_str: str) -> None:
    cusip  = row.get("cusip")
    ticker = row.get("ticker")

    # Portfolio impact callout for high-conviction signals only
    final_score = row.get("final_score")
    if final_score is not None and final_score >= 0.50 and ticker:
        impact = _portfolio_impact_line(ticker)
        if impact:
            st.info(impact, icon=":material/science:")

    # --- Convergence score breakdown ---
    st.markdown("#### Convergence Signal")
    detail = load_convergence_detail(cusip, period_str) if cusip else None

    conv  = row.get("convergence_score")
    trend = row.get("convergence_trend")
    bulls = row.get("n_funds_bullish") or 0
    bears = row.get("n_funds_bearish") or 0

    c1, c2, c3, c4, c5 = st.columns(5)
    _, conv_color = _conviction_info(conv)
    c1.metric("Conv Score", f"{conv:+.3f}" if conv is not None else "—")
    if trend:
        t_color = _TREND_COLOR.get(trend, "gray")
        c2.metric("Trend", f":{t_color}[{trend.title()}]")
    else:
        c2.metric("Trend", "—")
    c3.metric("Bullish Funds", str(bulls))
    c4.metric("Bearish Funds", str(bears))

    if detail:
        dir_val = detail.get("directional")
        c5.metric("Directional", f"{dir_val:+.3f}" if dir_val is not None else "—")

        avg_pct = detail.get("avg_position_pct")
        sc = detail.get("sector_concentration")
        n_hold = detail.get("n_holding")
        univ = detail.get("universe_size", 38)

        sub_parts = []
        if avg_pct is not None:
            sub_parts.append(f"Avg position: **{avg_pct:.1f}%** of fund portfolio")
        if sc is not None:
            sub_parts.append(f"Sector concentration: **{sc:.2f}**")
        if sub_parts:
            st.caption("  ·  ".join(sub_parts))

        if n_hold is not None:
            name = row.get("display_name") or ticker or "this stock"
            st.caption(f":material/group: **{n_hold} of {univ}** tracked funds hold {name} this quarter.")

    # --- Fund moves ---
    if detail and detail.get("fund_moves"):
        skill_map = load_fund_skill_map()
        moves = detail["fund_moves"]
        bullish_moves = [m for m in moves if m["direction"] == "bullish_leaning"]
        bearish_moves = [m for m in moves if m["direction"] == "bearish_leaning"]

        if bullish_moves:
            st.markdown(f"**:green[▲ Bullish / Accumulating ({len(bullish_moves)} funds)]**")
            st.dataframe(
                _fund_moves_df(bullish_moves, skill_map),
                hide_index=True, width="stretch",
            )
        if bearish_moves:
            st.markdown(f"**:red[▼ Bearish / Trimming ({len(bearish_moves)} funds)]**")
            st.dataframe(
                _fund_moves_df(bearish_moves, skill_map),
                hide_index=True, width="stretch",
            )
    elif not detail:
        st.caption(":gray[No convergence detail found for this CUSIP / quarter.]")

    # --- NLP language analysis ---
    st.divider()
    st.markdown("#### NLP Language Analysis (MD&A)")

    nlp_available = row.get("nlp_available", False)
    nlp_score     = row.get("nlp_composite_score")

    nlp = load_nlp_detail(ticker) if ticker else None

    if nlp is None:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.info(
                "NLP scoring requires an Anthropic API key. "
                "Set `ANTHROPIC_API_KEY` in your `.env` file and restart the app.",
                icon=":material/key:",
            )
        elif not ticker:
            st.caption(":gray[No ticker symbol — cannot run NLP analysis.]")
        else:
            st.markdown(":gray[MD&A language analysis not yet run for this ticker.]")
            if st.button(
                "Run NLP Analysis",
                key=f"nlp_run_{ticker}",
                icon=":material/play_arrow:",
                help="Score this ticker's MD&A language shifts via Claude API (~30–60s)",
            ):
                with st.spinner(f"Scoring MD&A filings for {ticker}…"):
                    try:
                        from smart_money.nlp import score_ticker as _score_ticker
                        result = _score_ticker(ticker)
                        if result is None:
                            st.warning(
                                f"NLP returned no result for {ticker}. "
                                "The ticker may not have two qualifying filings in EDGAR.",
                                icon=":material/warning:",
                            )
                        else:
                            load_nlp_detail.clear()
                            st.rerun()
                    except Exception as _nlp_err:
                        st.error(f"NLP scoring failed: {_nlp_err}", icon=":material/error:")
        return

    color = _nlp_color(nlp["composite_score"])
    st.metric(
        "NLP Composite Score",
        f":{color}[{nlp['composite_score']:+.3f}]",
        help="Weighted sum of 7 MD&A dimension deltas (current vs prior filing).",
    )

    # 7-dimension table
    dim_rows = []
    for label, key, weight in _NLP_DIMENSIONS:
        delta = nlp.get(key)
        if delta is not None:
            direction = "↑" if delta > 0.05 else "↓" if delta < -0.05 else "→"
            contrib = delta * weight
        else:
            direction, contrib = "—", None
        dim_rows.append({
            "Dimension":    label,
            "Weight":       f"{weight * 100:.0f}%",
            "Delta":        f"{delta:+.2f}" if delta is not None else "—",
            "Dir":          direction,
            "Contribution": f"{contrib:+.3f}" if contrib is not None else "—",
        })
    st.dataframe(pd.DataFrame(dim_rows), hide_index=True, width="stretch")

    # Full reasoning text
    if nlp.get("reasoning"):
        st.markdown("**MD&A Analysis Reasoning:**")
        for para in nlp["reasoning"].split("\n"):
            if para.strip():
                st.markdown(f"> {para.strip()}")


# ---------------------------------------------------------------------------
# Tab 2 — Price & Chart
# ---------------------------------------------------------------------------

def _render_price_tab(ticker: str, info: dict, hist: pd.DataFrame) -> None:
    import datetime

    cur    = info.get("currentPrice") or info.get("regularMarketPrice")
    high52 = info.get("fiftyTwoWeekHigh")
    low52  = info.get("fiftyTwoWeekLow")
    prev   = info.get("previousClose")
    beta   = info.get("beta")
    vol    = info.get("volume")
    avg_vol = info.get("averageVolume")

    pct_today = ((cur - prev) / prev * 100) if cur and prev else None
    from_high = ((cur - high52) / high52 * 100) if cur and high52 else None

    # Row 1: price metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Price",
        f"${cur:,.2f}" if cur else "—",
        f"{pct_today:+.2f}%" if pct_today is not None else None,
    )
    c2.metric("52w High", f"${high52:,.2f}" if high52 else "—")
    c3.metric("52w Low",  f"${low52:,.2f}"  if low52  else "—")
    c4.metric("From 52w High", f"{from_high:+.1f}%" if from_high is not None else "—")
    c5.metric("Beta", f"{beta:.2f}" if beta else "—")

    # Row 2: volume + period returns
    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Volume", f"{vol / 1e6:.1f}M" if vol else "—")
    c7.metric("Avg Volume", f"{avg_vol / 1e6:.1f}M" if avg_vol else "—")

    if not hist.empty and "Close" in hist.columns:
        prices = hist["Close"].dropna()
        today_val = float(prices.iloc[-1]) if cur is None else cur
        def _period_return(days: int) -> str | None:
            cutoff = prices.index[-1] - pd.Timedelta(days=days)
            prior = prices[prices.index <= cutoff]
            if prior.empty:
                return None
            pct = (today_val - float(prior.iloc[-1])) / float(prior.iloc[-1]) * 100
            return f"{pct:+.1f}%"

        c8.metric("1M Return",  _period_return(30)  or "—")
        c9.metric("3M Return",  _period_return(90)  or "—")
        c10.metric("6M Return", _period_return(180) or "—")

        # 1-year chart
        chart = hist[["Close"]].rename(columns={"Close": "Price"})
        chart.index = pd.to_datetime(chart.index).tz_localize(None)
        st.line_chart(chart, width="stretch")
    else:
        st.caption(":gray[No price history available]")


# ---------------------------------------------------------------------------
# Tab 3 — Valuation
# ---------------------------------------------------------------------------

def _render_valuation_tab(ticker: str, info: dict) -> None:
    def _n(v, fmt=".1f"):
        f = _safe_float(v)
        return f"{f:{fmt}}" if f is not None else "N/A"

    def _b(v):
        f = _safe_float(v)
        if f is None:
            return "N/A"
        abs_f = abs(f)
        if abs_f >= 1e12:
            return f"${f / 1e12:.2f}T"
        if abs_f >= 1e9:
            return f"${f / 1e9:.1f}B"
        return f"${f / 1e6:.0f}M"

    def _card(col, label: str, value: str) -> None:
        col.markdown(f":gray[{label}]")
        col.markdown(f"**{value}**")

    rec_map = {
        "strong_buy":  "Strong Buy",
        "buy":         "Buy",
        "hold":        "Hold",
        "sell":        "Sell",
        "strong_sell": "Strong Sell",
    }

    # ── Trading Multiples ──────────────────────────────────────────────────
    st.markdown("**Trading Multiples**")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    _card(c1, "P/E (TTM)",  _n(info.get("trailingPE")))
    _card(c2, "P/E (Fwd)",  _n(info.get("forwardPE")))
    _card(c3, "PEG Ratio",  _n(info.get("pegRatio"), ".2f"))
    _card(c4, "P/B",        _n(info.get("priceToBook"), ".2f"))
    _card(c5, "P/S (TTM)",  _n(info.get("priceToSalesTrailing12Months"), ".2f"))
    _card(c6, "EV/EBITDA",  _n(info.get("enterpriseToEbitda")))

    st.markdown("---")

    # ── FCF Metrics ────────────────────────────────────────────────────────
    st.markdown("**Free Cash Flow Metrics**")

    _cf  = _yf_cashflow(ticker, annual=False)
    _fin = _yf_financials(ticker, annual=False)

    _fcf_row = _find_row(_cf, ["Free Cash Flow"]) if not _cf.empty else None
    _rev_row = _find_row(_fin, ["Total Revenue", "Operating Revenue"]) if not _fin.empty else None

    # TTM FCF: sum of up to 4 most recent quarters (avoids single-quarter × 4 seasonality distortion)
    _fcf_vals_q = [_safe_float(_fcf_row.iloc[i]) for i in range(min(4, len(_fcf_row)))] if _fcf_row is not None else []
    _fcf_valid  = [v for v in _fcf_vals_q if v is not None]
    _fcf_ttm    = sum(_fcf_valid) if _fcf_valid else None
    _n_fcf_qtrs = len(_fcf_valid)

    # TTM Revenue: sum of up to 4 most recent quarters (consistent denominator for margin)
    _rev_vals_q = [_safe_float(_rev_row.iloc[i]) for i in range(min(4, len(_rev_row)))] if _rev_row is not None else []
    _rev_valid_q = [v for v in _rev_vals_q if v is not None]
    _rev_ttm    = sum(_rev_valid_q) if _rev_valid_q else None

    _ev = _safe_float(info.get("enterpriseValue"))

    _fcf_yield  = _fcf_ttm / _ev * 100 if (_fcf_ttm is not None and _ev and _ev != 0) else None
    _fcf_margin = _fcf_ttm / _rev_ttm * 100 if (_fcf_ttm is not None and _rev_ttm and _rev_ttm != 0) else None

    # 3-year historical FCF yield average: annual indices 1-3 (prior years, excluding most recent)
    _cf_annual = _yf_cashflow(ticker, annual=True)
    _fcf_hist_avg: float | None = None
    if not _cf_annual.empty and _ev and _ev != 0:
        _fcf_annual_row = _find_row(_cf_annual, ["Free Cash Flow"])
        if _fcf_annual_row is not None and len(_fcf_annual_row) >= 4:
            _hist_fcf   = [_safe_float(_fcf_annual_row.iloc[i]) for i in range(1, 4)]
            _hist_valid = [v for v in _hist_fcf if v is not None]
            if len(_hist_valid) >= 3:
                _fcf_hist_avg = sum(v / _ev * 100 for v in _hist_valid) / len(_hist_valid)

    fcf_c1, fcf_c2 = st.columns(2)

    with fcf_c1:
        st.markdown(":gray[FCF Yield (TTM)]")
        if _fcf_yield is not None:
            if _fcf_hist_avg is not None:
                _fy_color = (
                    "green"  if _fcf_yield > _fcf_hist_avg * 1.15 else
                    "red"    if _fcf_yield < _fcf_hist_avg * 0.85 else
                    "orange"
                )
            else:
                _fy_color = "green" if _fcf_yield >= 4 else "orange" if _fcf_yield >= 1.5 else "red"
            st.markdown(f"### :{_fy_color}[{_fcf_yield:.1f}%]")
        else:
            st.markdown("### N/A")
        with st.expander("ℹ️ Methodology"):
            if _fcf_hist_avg is not None:
                st.caption(
                    f"Compared against 3-year historical average ({_fcf_hist_avg:.1f}%), "
                    "excluding the most recent fiscal year to avoid overlap bias. "
                    "See About page for full methodology."
                )
            else:
                st.caption(
                    "Insufficient history for historical comparison — using general benchmark. "
                    "See About page for full methodology."
                )

    with fcf_c2:
        st.markdown(":gray[FCF Margin (TTM)]")
        if _fcf_margin is not None:
            _fm_color = "green" if _fcf_margin >= 15 else "orange" if _fcf_margin >= 5 else "red"
            st.markdown(f"### :{_fm_color}[{_fcf_margin:.1f}%]")
        else:
            st.markdown("### N/A")
        st.caption(
            "What percentage of revenue becomes free cash flow — "
            "a measure of capital efficiency. TTM FCF ÷ TTM Revenue."
        )

    if _fcf_ttm is not None:
        st.caption(f"TTM FCF ({_n_fcf_qtrs}q sum): {_fmt_b(_fcf_ttm)}")
    elif _fcf_row is None:
        st.caption(":gray[Free Cash Flow row not found in yfinance cashflow data.]")

    st.markdown("---")

    # ── Size & Capital Structure ───────────────────────────────────────────
    st.markdown("**Size & Capital Structure**")
    short_pct = _safe_float(info.get("shortPercentOfFloat"))
    c7, c8, c9 = st.columns(3)
    _card(c7, "Market Cap",       _b(info.get("marketCap")))
    _card(c8, "Enterprise Value", _b(info.get("enterpriseValue")))
    _card(c9, "Short % Float",    f"{short_pct * 100:.1f}%" if short_pct else "N/A")

    st.markdown("---")

    # ── Analyst Consensus ─────────────────────────────────────────────────
    st.markdown("**Analyst Consensus**")
    target    = _safe_float(info.get("targetMeanPrice"))
    n_anal    = info.get("numberOfAnalystOpinions")
    rec_key   = info.get("recommendationKey") or ""
    rec_label = rec_map.get(rec_key, rec_key.replace("_", " ").title() if rec_key else None)
    cur       = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))

    if rec_label:
        badge_color = (
            "green"  if rec_key in ("strong_buy", "buy") else
            "orange" if rec_key == "hold" else
            "red"    if rec_key in ("sell", "strong_sell") else
            "gray"
        )
        st.badge(rec_label.upper(), color=badge_color)

    parts: list[str] = []
    if target:
        parts.append(f"**${target:,.2f}** target")
    if cur:
        parts.append(f"**${cur:,.2f}** current")
        if target:
            upside = (target - cur) / cur * 100
            up_col = "green" if upside >= 0 else "red"
            parts.append(f":{up_col}[**{upside:+.1f}%** upside]")
    if n_anal:
        parts.append(f"{n_anal} analysts")

    if parts:
        st.markdown("  ·  ".join(parts))
    elif not rec_label:
        st.caption(":gray[No analyst consensus data available]")


# ---------------------------------------------------------------------------
# Tab 4 — Financials
# ---------------------------------------------------------------------------

def _render_financials_tab(ticker: str) -> None:
    import matplotlib.pyplot as plt

    # ── Annual / Quarterly toggle ──────────────────────────────────────────
    period_type = st.radio(
        "View period",
        ["Quarterly", "Annual"],
        horizontal=True,
        index=0,
        label_visibility="collapsed",
    )
    annual    = period_type == "Annual"
    chg_label = "YoY" if annual else "QoQ"

    fin = _yf_financials(ticker, annual=annual)
    cf  = _yf_cashflow(ticker, annual=annual)
    bs  = _yf_balance_sheet(ticker, annual=annual)

    if fin.empty:
        st.caption(":gray[No financial data available from yfinance]")
        return

    n_p = min(fin.shape[1], 4)
    raw_periods = [str(c)[:7] for c in fin.columns[:n_p]]
    period_hdrs = [p[:4] for p in raw_periods] if annual else [_to_quarter_label(p) for p in raw_periods]

    rev_row = _find_row(fin, ["Total Revenue", "Operating Revenue"])
    gp_row  = _find_row(fin, ["Gross Profit"])
    oi_row  = _find_row(fin, ["Operating Income", "EBIT", "Total Operating Income As Reported"])
    ni_row  = _find_row(fin, ["Net Income", "Net Income Common Stockholders"])

    def _raw(row, n) -> list[float | None]:
        if row is None:
            return [None] * n
        return [_safe_float(row.iloc[i]) for i in range(min(n, len(row)))]

    rev_vals = _raw(rev_row, n_p)
    gp_vals  = _raw(gp_row,  n_p)
    oi_vals  = _raw(oi_row,  n_p)
    ni_vals  = _raw(ni_row,  n_p)

    # ── Revenue vs Net Income bar chart ───────────────────────────────────
    valid_mask = [i for i in range(n_p) if rev_vals[i] is not None and ni_vals[i] is not None]
    if valid_mask:
        xs = list(range(len(valid_mask)))
        qs = [period_hdrs[i] for i in valid_mask]
        rv = [rev_vals[i] / 1e9 for i in valid_mask]
        ni = [ni_vals[i] / 1e9 for i in valid_mask]

        def _lbl_b(v_b: float) -> str:
            return f"${v_b:.1f}B" if abs(v_b) >= 1 else f"${v_b * 1000:.0f}M"

        fig, ax = plt.subplots(figsize=(9, 3.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#fafafa")
        w = 0.38
        bars_rv = ax.bar([x - w / 2 for x in xs], rv, w, label="Revenue",    color="#2563eb", alpha=0.85)
        bars_ni = ax.bar([x + w / 2 for x in xs], ni, w, label="Net Income",  color="#16a34a", alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels(qs, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("$ Billions", fontsize=10)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9)
        y_max = max((v for v in rv + ni if v is not None), default=1)
        for bar in list(bars_rv) + list(bars_ni):
            h = bar.get_height()
            if h != 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + y_max * 0.015,
                    _lbl_b(h),
                    ha="center", va="bottom", fontsize=7, color="#374151",
                )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Margin table ─────────────────────────────────────────────────────
    def _margins(num_v, denom_v, n):
        return [
            num_v[i] / denom_v[i] * 100
            if num_v[i] is not None and denom_v[i] and denom_v[i] != 0
            else None
            for i in range(n)
        ]

    gm = _margins(gp_vals, rev_vals, n_p)
    om = _margins(oi_vals, rev_vals, n_p)
    nm = _margins(ni_vals, rev_vals, n_p)

    # Shared style constants — reused by income and cash flow tables
    _DARK  = "color: #111827"
    _GREEN = f"background-color: #dcfce7; {_DARK}"
    _RED   = f"background-color: #fee2e2; {_DARK}"
    _ALT   = f"background-color: #f8f9fa; {_DARK}"

    mrg_rows_raw  = []
    mrg_rows_disp = []
    for lbl, vals in [("Gross Margin", gm), ("Operating Margin", om), ("Net Margin", nm)]:
        if all(v is None for v in vals):
            continue
        mrg_rows_raw.append({"Metric": lbl,  **{period_hdrs[i]: vals[i] for i in range(len(period_hdrs))}})
        mrg_rows_disp.append({"Metric": lbl, **{
            period_hdrs[i]: f"{vals[i]:.1f}%" if vals[i] is not None else "—"
            for i in range(len(period_hdrs))
        }})

    if mrg_rows_disp:
        raw_mrg  = pd.DataFrame(mrg_rows_raw).set_index("Metric")[period_hdrs]
        disp_mrg = pd.DataFrame(mrg_rows_disp).set_index("Metric")[period_hdrs]

        def _margin_style(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            cols = list(df.columns)
            for i, idx in enumerate(df.index):
                base = _ALT if i % 2 == 0 else _DARK
                for j, col in enumerate(cols):
                    v    = raw_mrg.loc[idx, col] if idx in raw_mrg.index else None
                    prev = raw_mrg.loc[idx, cols[j + 1]] if j + 1 < len(cols) and idx in raw_mrg.index else None
                    if v is not None and prev is not None and pd.notna(v) and pd.notna(prev):
                        styles.loc[idx, col] = _GREEN if v > prev else _RED if v < prev else base
                    else:
                        styles.loc[idx, col] = base
            return styles

        st.markdown("**Margins**")
        st.dataframe(disp_mrg.style.apply(_margin_style, axis=None), width="stretch")

    # ── Income Statement table ────────────────────────────────────────────
    st.markdown(f"**Income Statement** :gray[· {chg_label} Δ in cell color and (Δ%)]")

    def _cell_with_delta(v: float | None, prev: float | None) -> str:
        base = _fmt_b(v)
        if v is not None and prev is not None and prev != 0:
            return f"{base} ({(v - prev) / abs(prev) * 100:+.1f}%)"
        return base

    income_rows_data = [
        ("Revenue",          rev_vals),
        ("Gross Profit",     gp_vals),
        ("Operating Income", oi_vals),
        ("Net Income",       ni_vals),
    ]
    raw_inc_rows  = []
    disp_inc_rows = []
    for lbl, vals in income_rows_data:
        if all(v is None for v in vals):
            continue
        raw_inc_rows.append(
            {"Metric": lbl, **{period_hdrs[i]: vals[i] for i in range(len(period_hdrs))}}
        )
        disp_inc_rows.append(
            {"Metric": lbl, **{
                period_hdrs[i]: _cell_with_delta(vals[i], vals[i + 1] if i + 1 < len(vals) else None)
                for i in range(len(period_hdrs))
            }}
        )

    if disp_inc_rows:
        raw_inc  = pd.DataFrame(raw_inc_rows).set_index("Metric")[period_hdrs]
        disp_inc = pd.DataFrame(disp_inc_rows).set_index("Metric")[period_hdrs]

        def _income_style(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            cols = list(df.columns)
            for i, idx in enumerate(df.index):
                alt_bg = _ALT if i % 2 == 0 else _DARK
                for j, col in enumerate(cols):
                    v    = raw_inc.loc[idx, col] if idx in raw_inc.index else None
                    prev = raw_inc.loc[idx, cols[j + 1]] if j + 1 < len(cols) and idx in raw_inc.index else None
                    if v is not None and prev is not None and pd.notna(v) and pd.notna(prev) and prev != 0:
                        styles.loc[idx, col] = (
                            _GREEN if v > prev else
                            _RED   if v < prev else alt_bg
                        )
                    else:
                        styles.loc[idx, col] = alt_bg
            return styles

        st.dataframe(disp_inc.style.apply(_income_style, axis=None), width="stretch")

    # ── Key Margin Ratios ─────────────────────────────────────────────────
    st.markdown("**Margin & FCF Summary** :gray[(most recent period · arrow = direction vs prior)]")
    fcf_row   = _find_row(cf, ["Free Cash Flow"]) if not cf.empty else None
    fcf_cf    = [_safe_float(fcf_row.iloc[i]) for i in range(min(n_p, len(fcf_row)))] if fcf_row is not None else [None] * n_p
    fcf_m     = [
        fcf_cf[i] / rev_vals[i] * 100
        if fcf_cf[i] is not None and rev_vals[i] and rev_vals[i] != 0
        else None
        for i in range(n_p)
    ]
    margin_badges = [
        ("Gross",     gm[0] if gm else None,   gm[1]   if len(gm) > 1   else None),
        ("Operating", om[0] if om else None,   om[1]   if len(om) > 1   else None),
        ("Net",       nm[0] if nm else None,   nm[1]   if len(nm) > 1   else None),
        ("FCF",       fcf_m[0] if fcf_m else None, fcf_m[1] if len(fcf_m) > 1 else None),
    ]
    badge_cols = st.columns(4)
    for col, (name, latest, prior) in zip(badge_cols, margin_badges):
        if latest is not None:
            improving  = (latest > prior) if prior is not None else None
            badge_color = "green" if improving else "red" if improving is False else "gray"
            arrow = " ↑" if improving else " ↓" if improving is False else ""
            col.badge(f"{name}: {latest:.1f}%{arrow}", color=badge_color)
        else:
            col.caption(f":gray[{name}: N/A]")

    # ── Cash Flow ─────────────────────────────────────────────────────────
    st.markdown("**Cash Flow**")
    if cf.empty:
        st.caption(":gray[Cash flow data unavailable]")
    else:
        cf_n = min(cf.shape[1], 4)
        cf_raw_p = [str(c)[:7] for c in cf.columns[:cf_n]]
        cf_hdrs  = [p[:4] for p in cf_raw_p] if annual else [_to_quarter_label(p) for p in cf_raw_p]

        opcf_row  = _find_row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Total Cash From Operating Activities"])
        capex_row = _find_row(cf, ["Capital Expenditure", "Capital Expenditures", "Purchase Of Property Plant And Equipment"])

        cf_raw_rows  = []
        cf_disp_rows = []
        for lbl, row_data in [("Operating CF", opcf_row), ("CapEx", capex_row), ("Free Cash Flow", fcf_row)]:
            if row_data is None:
                continue
            cf_vals = [_safe_float(row_data.iloc[i]) for i in range(min(cf_n, len(row_data)))]
            if all(v is None for v in cf_vals):
                continue
            raw_r  = {"Metric": lbl}
            disp_r = {"Metric": lbl}
            for i, ph in enumerate(cf_hdrs):
                v    = cf_vals[i] if i < len(cf_vals) else None
                prev = cf_vals[i + 1] if i + 1 < len(cf_vals) else None
                raw_r[ph]  = v
                disp_r[ph] = _cell_with_delta(v, prev)
            cf_raw_rows.append(raw_r)
            cf_disp_rows.append(disp_r)

        if cf_disp_rows:
            cf_raw_df  = pd.DataFrame(cf_raw_rows).set_index("Metric")[cf_hdrs]
            cf_disp_df = pd.DataFrame(cf_disp_rows).set_index("Metric")[cf_hdrs]

            def _cf_style(df):
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                cols = list(df.columns)
                for i, idx in enumerate(df.index):
                    alt_bg = _ALT if i % 2 == 0 else _DARK
                    for j, col in enumerate(cols):
                        v    = cf_raw_df.loc[idx, col] if idx in cf_raw_df.index else None
                        prev = cf_raw_df.loc[idx, cols[j + 1]] if j + 1 < len(cols) and idx in cf_raw_df.index else None
                        if v is not None and prev is not None and pd.notna(v) and pd.notna(prev) and prev != 0:
                            styles.loc[idx, col] = (
                                _GREEN if v > prev else
                                _RED   if v < prev else alt_bg
                            )
                        else:
                            styles.loc[idx, col] = alt_bg
                return styles

            st.dataframe(cf_disp_df.style.apply(_cf_style, axis=None), width="stretch")

    # ── Balance Sheet ─────────────────────────────────────────────────────
    st.markdown("**Balance Sheet** :gray[(most recent period)]")
    if not bs.empty:
        cash_row   = _find_row(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])
        debt_row   = _find_row(bs, ["Total Debt"])
        equity_row = _find_row(bs, ["Common Stock Equity", "Stockholders Equity", "Total Equity Gross Minority Interest"])

        cash   = _safe_float(cash_row.iloc[0])   if cash_row   is not None else None
        debt   = _safe_float(debt_row.iloc[0])   if debt_row   is not None else None
        equity = _safe_float(equity_row.iloc[0]) if equity_row is not None else None
        de     = debt / equity if debt and equity and equity != 0 else None

        net_cash     = (cash - debt) if cash is not None and debt is not None else None
        net_cash_lbl = "Net Cash" if net_cash is not None and net_cash >= 0 else "Net Debt"
        net_cash_val = _fmt_b(abs(net_cash)) if net_cash is not None else "N/A"

        bc1, bc2, bc3, bc4 = st.columns(4)
        for col, lbl, val in [
            (bc1, "Cash & Equivalents", _fmt_b(cash)),
            (bc2, "Total Debt",         _fmt_b(debt)),
            (bc3, net_cash_lbl,         net_cash_val),
            (bc4, "Debt / Equity",      f"{de:.2f}×" if de is not None else "N/A"),
        ]:
            col.markdown(f":gray[{lbl}]")
            col.markdown(f"**{val}**")
    else:
        st.caption(":gray[Balance sheet data unavailable]")


# ---------------------------------------------------------------------------
# Tab 5 — Earnings
# ---------------------------------------------------------------------------

def _render_earnings_tab(ticker: str, info: dict) -> None:
    import datetime, matplotlib.pyplot as plt

    # Next earnings date
    ts = info.get("earningsTimestamp")
    if ts:
        try:
            next_date = datetime.datetime.fromtimestamp(int(ts)).strftime("%B %d, %Y")
            st.info(f":material/event: **Next earnings: {next_date}**")
        except Exception:
            pass

    ed = _yf_earnings(ticker)
    if ed is None or (hasattr(ed, "empty") and ed.empty):
        st.caption(":gray[Historical EPS data unavailable]")
        return

    today = str(datetime.date.today())
    past: list[dict] = []
    for dt_idx, erow in ed.iterrows():
        date_str = str(dt_idx)[:10]
        if date_str <= today:
            past.append({
                "date": date_str[:7],
                "est":  _safe_float(erow.get("EPS Estimate")),
                "act":  _safe_float(erow.get("Reported EPS")),
                "surp": _safe_float(erow.get("Surprise(%)")),
            })
        if len(past) >= 8:
            break

    if not past:
        st.caption(":gray[No past earnings records found]")
        return

    past = list(reversed(past))  # chronological order for chart

    # --- EPS beat/miss bar chart ---
    xs      = list(range(len(past)))
    labels  = [r["date"] for r in past]
    ests    = [r["est"] or 0 for r in past]
    acts    = [r["act"] or 0 for r in past]
    colors  = [
        "#2ecc71" if (r["act"] is not None and r["est"] is not None and r["act"] >= r["est"])
        else "#e74c3c"
        for r in past
    ]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    w = 0.35
    ax.bar([x - w / 2 for x in xs], ests, w, label="Estimate", color="#94a3b8", alpha=0.85)
    ax.bar([x + w / 2 for x in xs], acts, w, label="Actual",   color=colors,    alpha=0.95)

    for i, r in enumerate(past):
        surp = r.get("surp")
        act  = r.get("act")
        if surp is not None and act is not None:
            s_color = "#15803d" if surp >= 0 else "#dc2626"
            ax.annotate(
                f"{surp:+.0f}%",
                xy=(i + w / 2, act),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=s_color,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels([_to_quarter_label(q) for q in labels], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("EPS ($)", fontsize=10)
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Detail table
    table = [
        {
            "Quarter":  r["date"],
            "EPS Est":  f"${r['est']:.2f}" if r["est"] is not None else "—",
            "EPS Act":  f"${r['act']:.2f}" if r["act"] is not None else "—",
            "Surprise": f"{r['surp']:+.1f}%" if r["surp"] is not None else "—",
            "Result":   "Beat ✓" if (r["act"] and r["est"] and r["act"] >= r["est"]) else "Miss ✗",
        }
        for r in reversed(past)  # newest first in table
    ]
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Tab 6 — Factor Profile
# ---------------------------------------------------------------------------

def _render_factor_tab(ticker: str) -> None:
    from factor_engine.portfolio import (
        _interpret_beta_market,
        _interpret_beta_smb,
        _interpret_beta_hml,
    )

    with st.spinner(f"Computing FF3 factor profile for {ticker}…"):
        profile = ticker_ff3_profile(ticker)

    if profile is None:
        st.info(
            f"FF3 factor profile not available for {ticker}. "
            "This may be a new IPO, a delisted security, or a ticker with insufficient "
            "price history in the 2021–2024 analysis window.",
            icon=":material/info:",
        )
        return

    bm  = profile["beta_market"]
    bsmb = profile["beta_smb"]
    bhml = profile["beta_hml"]
    r2   = profile["r_squared"]
    alpha = profile["alpha_annualized"]

    # --- Metrics grid ---
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(
        "Market β", f"{bm:+.3f}",
        help=(
            "Market Beta measures how sensitive this stock is to overall market movements. "
            "A beta of 1.0 means it moves in line with the market. Above 1.0 means more volatile "
            "than the market — a 10% market move implies a larger move in this stock. Below 1.0 "
            "means less sensitive. Negative beta means it tends to move opposite to the market."
        ),
    )
    c2.metric(
        "SMB β", f"{bsmb:+.3f}",
        help=(
            "Size Factor (Small Minus Big) measures exposure to the size premium. Positive values "
            "mean the stock behaves more like small-cap stocks. Negative values mean it behaves "
            "more like large-cap stocks."
        ),
    )
    c3.metric(
        "HML β", f"{bhml:+.3f}",
        help=(
            "Value/Growth Factor (High Minus Low book-to-market) measures the stock's style tilt. "
            "Positive values indicate a value tilt — cheap relative to book value. Negative values "
            "indicate a growth tilt — priced on future earnings rather than current assets. "
            "Near zero means style-neutral."
        ),
    )
    c4.metric(
        "R²", f"{r2:.3f}",
        help=(
            "R-Squared measures how much of this stock's daily return variance is explained by "
            "the three Fama-French factors (market, size, value). Higher R² means the stock is "
            "more factor-driven; lower means more stock-specific."
        ),
    )
    c5.metric(
        "Alpha (ann.)", f"{alpha * 100:+.2f}%",
        help=(
            "Annualized Alpha is the return unexplained by factor exposures — the stock's "
            "performance above or below what its market, size, and value tilts would predict. "
            "Positive alpha suggests historical outperformance versus its factor benchmark; "
            "negative means underperformance. This is a historical measure, not a prediction "
            "of future returns."
        ),
    )
    c6.metric(
        "n_obs", str(profile["n_obs"]),
        help=(
            "Number of trading days used in the regression. More observations generally produce "
            "more reliable factor estimates. The analysis uses daily returns over the selected "
            "date range."
        ),
    )

    st.caption(
        f"{profile['start']} → {profile['end']} · "
        f"Ken French US FF3 factors"
    )

    # --- Plain-English interpretation ---
    st.markdown("**Factor Interpretation:**")
    st.markdown(f"- **Market:** {_interpret_beta_market(bm)}")
    st.markdown(f"- **Size (SMB):** {_interpret_beta_smb(bsmb)}")
    st.markdown(f"- **Value/Growth (HML):** {_interpret_beta_hml(bhml)}")
    st.markdown(
        f"- **Model fit:** FF3 explains **{r2 * 100:.1f}%** of this stock's daily return variance. "
        + ("High — most price movement is factor-driven." if r2 > 0.60
           else "Moderate — substantial idiosyncratic component." if r2 > 0.35
           else "Low — heavily idiosyncratic or sector-specific.")
    )

    # --- Portfolio comparison ---
    st.divider()
    st.markdown("**Portfolio Comparison**")

    port = portfolio_ff3_betas()
    if port is None:
        st.caption(":gray[Portfolio betas not available — run the Portfolio page to compute them.]")
        return

    pm   = port.get("beta_market", 0)
    psmb = port.get("beta_smb", 0)
    phml = port.get("beta_hml", 0)

    alloc_pct = st.select_slider(
        "Allocation size",
        options=[1, 2, 5, 10, 15, 20],
        value=5,
        format_func=lambda x: f"{x}%",
        key=f"factor_alloc_{ticker}",
    )
    w = alloc_pct / 100
    new_m   = (1 - w) * pm   + w * bm
    new_smb = (1 - w) * psmb + w * bsmb
    new_hml = (1 - w) * phml + w * bhml

    def _delta_icon(delta: float) -> str:
        return ":green[↑]" if delta > 0.02 else ":red[↓]" if delta < -0.02 else ":gray[→]"

    impact_col = f"+{alloc_pct}% Impact"
    comp_data = [
        {
            "Factor":     "Market β",
            "This Stock": f"{bm:+.3f}",
            "Portfolio":  f"{pm:+.3f}",
            impact_col:   f"{new_m:+.3f} ({new_m - pm:+.3f})",
            "":           _delta_icon(new_m - pm),
        },
        {
            "Factor":     "SMB β (size)",
            "This Stock": f"{bsmb:+.3f}",
            "Portfolio":  f"{psmb:+.3f}",
            impact_col:   f"{new_smb:+.3f} ({new_smb - psmb:+.3f})",
            "":           _delta_icon(new_smb - psmb),
        },
        {
            "Factor":     "HML β (value/growth)",
            "This Stock": f"{bhml:+.3f}",
            "Portfolio":  f"{phml:+.3f}",
            impact_col:   f"{new_hml:+.3f} ({new_hml - phml:+.3f})",
            "":           _delta_icon(new_hml - phml),
        },
    ]
    st.dataframe(pd.DataFrame(comp_data), hide_index=True, width="stretch")

    # Diversification / concentration verdict
    verdicts = []
    if abs(bm - pm) < 0.15:
        verdicts.append("market-neutral (similar beta)")
    elif bm > pm:
        verdicts.append("increases market sensitivity")
    else:
        verdicts.append("reduces market sensitivity (defensive)")

    if abs(bsmb - psmb) < 0.10:
        verdicts.append("size-neutral")
    elif bsmb > psmb:
        verdicts.append("adds small-cap exposure")
    else:
        verdicts.append("reinforces large-cap tilt")

    if abs(bhml - phml) < 0.10:
        verdicts.append("style-neutral")
    elif bhml > phml:
        verdicts.append("adds value tilt")
    else:
        verdicts.append("deepens growth tilt")

    st.caption(f":material/balance: Adding {ticker} at {alloc_pct}%: " + " · ".join(verdicts) + ".")
    st.caption(
        f"Portfolio analysis period: {port.get('start', '?')} → {port.get('end', '?')} · "
        f"R² = {port.get('r_squared', 0):.3f}"
    )


# ---------------------------------------------------------------------------
# Deep-dive panel  (6 tabs)
# ---------------------------------------------------------------------------

def _render_deep_dive(ticker: str, row: dict, period_str: str) -> None:
    # Pre-fetch yfinance data used by multiple tabs
    with st.spinner(f"Loading {ticker}…"):
        info = _yf_info(ticker)
        hist = _yf_history(ticker)

    # Business description above tabs
    desc = info.get("longBusinessSummary", "")
    if desc:
        short = desc[:500] + ("…" if len(desc) > 500 else "")
        st.caption(short)

    tab_sig, tab_price, tab_val, tab_fin, tab_earn, tab_factor = st.tabs([
        ":material/radar: Signal",
        ":material/show_chart: Price",
        ":material/analytics: Valuation",
        ":material/table_chart: Financials",
        ":material/calendar_month: Earnings",
        ":material/science: Factor Profile",
    ])

    with tab_sig:
        _render_signal_tab(row, period_str)

    with tab_price:
        _render_price_tab(ticker, info, hist)

    with tab_val:
        _render_valuation_tab(ticker, info)

    with tab_fin:
        _render_financials_tab(ticker)

    with tab_earn:
        _render_earnings_tab(ticker, info)

    with tab_factor:
        _render_factor_tab(ticker)


# ---------------------------------------------------------------------------
# Shared card renderer
# ---------------------------------------------------------------------------

def _render_card(
    row: dict,
    expand_key: str,
    watched: set[str],
    period_str: str,
    show_remove: bool = False,
) -> None:
    """
    Render one signal card and, when expanded, the 6-tab deep-dive panel below.

    expand_key  — session_state key that tracks the currently expanded display_name
    period_str  — passed through to Signal tab for convergence DB lookup
    show_remove — watchlist mode (Remove button) vs discovery mode (Watch button)
    """
    display = row["display_name"]
    ticker  = row.get("ticker")
    issuer  = row.get("issuer_name", "") or ""
    row_id  = row.get("cusip") or display

    is_expanded = st.session_state.get(expand_key) == display

    with st.container(border=True):
        col_name, col_score, col_status, col_act = st.columns(
            [3, 2, 2, 1.5], vertical_alignment="center"
        )

        with col_name:
            st.markdown(f"**{display}**")
            if issuer and issuer.lower() != display.lower():
                st.caption(issuer)
            detail_parts = []
            if row.get("sector"):
                detail_parts.append(row["sector"])
            if row.get("date_added"):
                detail_parts.append(f"Added {row['date_added']}")
            if detail_parts:
                st.caption("  ·  ".join(detail_parts))

        with col_score:
            _render_score_cell(row.get("final_score"), row.get("convergence_score"))

        with col_status:
            _badge(row.get("status"))
            trend = row.get("convergence_trend")
            bulls = row.get("n_funds_bullish")
            bears = row.get("n_funds_bearish")
            if trend and trend != "stable":
                t_color = _TREND_COLOR.get(trend, "gray")
                st.caption(f"Trend: :{t_color}[{trend}]")
            if bulls is not None:
                st.caption(f":green[{bulls} bullish]  :red[{bears} bearish]")

        with col_act:
            if show_remove:
                if st.button("Remove", key=f"rm_{row_id}"):
                    identifier = row.get("ticker") or row.get("cusip")
                    if identifier:
                        watchlist_remove(identifier)
                        st.toast(f"Removed {display} from watchlist")
                        st.rerun()
            else:
                if display in watched:
                    st.badge("★ Watching", color="blue")
                elif ticker:
                    if st.button("+ Watch", key=f"add_{row_id}", type="primary"):
                        watchlist_add(ticker)
                        st.toast(f"Added {display} to watchlist")
                        st.rerun()

        # Full-width: NLP badge + signal drivers
        _render_nlp_inline(row.get("nlp_available", False), row.get("nlp_composite_score"))
        if row.get("signal_drivers"):
            st.caption(f":material/info: {row['signal_drivers']}")
        if row.get("note"):
            st.caption(f":material/edit_note: {row['note']}")

        # Deep-dive expand toggle
        if ticker:
            expand_icon = "▼" if is_expanded else "▶"
            if st.button(
                f"{expand_icon} Deep dive",
                key=f"exp_{expand_key}_{row_id}",
                type="secondary",
            ):
                st.session_state[expand_key] = None if is_expanded else display
                st.rerun()

    if is_expanded and ticker:
        with st.container(border=True):
            _render_deep_dive(ticker, row, period_str)


# ---------------------------------------------------------------------------
# Alert strip
# ---------------------------------------------------------------------------

def _render_alerts(watchlist_rows: list[dict], period_str: str) -> None:
    """Surface EXIT SIGNALs, new STRENGTHENING discoveries, and near-LT lots."""
    exit_signals = [r for r in watchlist_rows if r["status"] == "EXIT SIGNAL"]

    try:
        all_sigs = load_signals(period_str)
        watched  = {r["display_name"] for r in watchlist_rows}
        new_strong = [
            r for r in all_sigs
            if r["status"] == "STRENGTHENING"
            and r["display_name"] not in watched
            and r.get("convergence_trend") in ("new", "accelerating")
        ]
    except Exception:
        new_strong = []

    try:
        from dashboard.db import load_tax_lots
        from dashboard.prefs import load as load_prefs
        lots   = load_tax_lots(load_prefs()["account_id"])
        near_lt = [l for l in lots if l["near_lt"] and (l["unrealized_gl"] or 0) < 0]
    except Exception:
        near_lt = []

    if not exit_signals and not new_strong and not near_lt:
        return

    for row in exit_signals:
        st.error(
            f"**Exit signal — {row['display_name']}:** "
            f"{row['signal_drivers'] or 'Convergence score has dropped below the discovery threshold.'}",
            icon=":material/exit_to_app:",
        )

    if new_strong:
        names = ", ".join(r["display_name"] for r in new_strong[:5])
        extra = f" (+{len(new_strong) - 5} more)" if len(new_strong) > 5 else ""
        st.success(
            f"**New strengthening signals not yet on your watchlist:** {names}{extra}",
            icon=":material/trending_up:",
        )

    if near_lt:
        tickers = ", ".join(sorted({l["ticker"] for l in near_lt}))
        n = len(near_lt)
        st.warning(
            f"**{n} loss lot{'s' if n > 1 else ''} within 30 days of long-term threshold** — "
            f"consider holding before harvesting: {tickers}. See the Tax lots page.",
            icon=":material/schedule:",
        )

    st.space("small")


# ---------------------------------------------------------------------------
# Watchlist tab
# ---------------------------------------------------------------------------

def _render_watchlist(period_str: str) -> None:
    rows = load_watchlist_scored(period_str)
    st.session_state.setdefault("expanded_wl", None)

    if not rows:
        st.info(
            "Your watchlist is empty. Switch to the Discovery tab to find signals and add tickers.",
            icon=":material/playlist_add:",
        )
        return

    for row in rows:
        _render_card(row, expand_key="expanded_wl", watched=set(),
                     period_str=period_str, show_remove=True)


# ---------------------------------------------------------------------------
# Discovery tab
# ---------------------------------------------------------------------------

def _render_discovery(period_str: str) -> None:
    all_signals = load_signals(period_str)
    watched     = load_watchlist_tickers()
    st.session_state.setdefault("expanded_disc", None)

    if not all_signals:
        st.info(
            "No signals found for this quarter. Run the pipeline to generate them.",
            icon=":material/info:",
        )
        st.code(".venv/bin/python -m smart_money.pipeline")
        return

    # Filters
    col_f1, col_f2, col_f3 = st.columns([3, 3, 3])
    with col_f1:
        sectors = sorted({r["sector"] for r in all_signals if r["sector"]})
        sel_sectors = st.multiselect(
            "Sector", sectors, placeholder="All sectors", key="disc_sectors"
        )
    with col_f2:
        sel_status = st.multiselect(
            "Status",
            ["STRENGTHENING", "HOLDING", "WEAKENING", "EXIT SIGNAL"],
            placeholder="All statuses",
            key="disc_status",
        )
    with col_f3:
        min_score = st.slider("Min score", 0.0, 1.0, 0.30, 0.05, key="disc_min_score")

    col_f4, _ = st.columns([3, 6])
    with col_f4:
        show_wl_only = st.toggle("Watchlist only", key="disc_wl_only")

    # Apply filters
    filtered = all_signals
    if sel_sectors:
        filtered = [r for r in filtered if r["sector"] in sel_sectors]
    if sel_status:
        filtered = [r for r in filtered if r["status"] in sel_status]
    filtered = [
        r for r in filtered
        if r["final_score"] >= min_score or r["status"] == "EXIT SIGNAL"
    ]
    if show_wl_only:
        filtered = [r for r in filtered if r["display_name"] in watched]

    total = len(filtered)
    shown = filtered[:_DISC_MAX_CARDS]
    st.caption(f"{total} signals · {period_str}")

    if total > _DISC_MAX_CARDS:
        st.info(
            f"Showing top {_DISC_MAX_CARDS} of {total} results — use the filters above to narrow.",
            icon=":material/filter_list:",
        )

    if not filtered:
        st.info("No signals match the current filters.")
        return

    for row in shown:
        _render_card(row, expand_key="expanded_disc", watched=watched,
                     period_str=period_str, show_remove=False)


# ---------------------------------------------------------------------------
# Sidebar — period selector and pipeline info
# ---------------------------------------------------------------------------

periods = get_available_periods()

with st.sidebar:
    st.divider()
    if periods:
        st.session_state.setdefault("signals_period", periods[0])
        cur = st.session_state["signals_period"]
        idx = periods.index(cur) if cur in periods else 0
        st.session_state["signals_period"] = st.selectbox(
            "Quarter", periods, index=idx, key="signals_period_sel"
        )
        period_str: str | None = st.session_state["signals_period"]
    else:
        period_str = None
        st.caption(":gray[No signal data in DB]")

    info = get_pipeline_status()
    if info["last_computed"]:
        ts = str(info["last_computed"])[:16].replace("T", " ")
        st.caption(f"Data as of {ts}")
    else:
        st.caption(":orange[Pipeline has never run]")

    exp = st.expander("Run pipeline", icon=":material/terminal:")
    if exp.open:
        with exp:
            st.caption("Standard run:")
            st.code(".venv/bin/python -m smart_money.pipeline")
            st.caption("Full refresh (re-fetches all prices):")
            st.code(".venv/bin/python -m smart_money.pipeline --refresh")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.header(":material/radar: Signals")

if not period_str:
    st.warning(
        "No signal data found. Run the pipeline to get started.",
        icon=":material/warning:",
    )
    st.code(".venv/bin/python -m smart_money.pipeline")
    st.stop()

watchlist_rows = load_watchlist_scored(period_str)
_render_alerts(watchlist_rows, period_str)

tab_wl, tab_disc = st.tabs(
    [":material/bookmark: Watchlist", ":material/search: Discovery"],
    on_change="rerun",
)

if tab_wl.open:
    with tab_wl:
        _render_watchlist(period_str)

if tab_disc.open:
    with tab_disc:
        _render_discovery(period_str)
