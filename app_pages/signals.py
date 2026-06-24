"""
Signals page — smart-money convergence tracker.

Layout
------
Sidebar:  Quarter selector, pipeline status + run command
Main:     Alert strip (EXIT SIGNALs, new STRENGTHENING, near-LT lots)
          ├── Watchlist tab  — scored active entries, add/remove inline
          └── Discovery tab  — full FinalSignal list with filters
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.db import (
    get_available_periods,
    get_pipeline_status,
    load_signals,
    load_watchlist_scored,
    load_watchlist_tickers,
    watchlist_add,
    watchlist_remove,
)

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


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _badge(status: str | None) -> None:
    if not status:
        st.caption(":gray[No signal this quarter]")
        return
    st.badge(status, icon=_STATUS_ICON.get(status, ""), color=_STATUS_COLOR.get(status, "gray"))


def _color_score(v: float | None) -> str:
    """Return colored markdown string for a score value."""
    if v is None:
        return ":gray[—]"
    if v >= 0.10:
        return f":green[{v:+.3f}]"
    if v <= -0.10:
        return f":red[{v:+.3f}]"
    return f":gray[{v:+.3f}]"


# ---------------------------------------------------------------------------
# Alert strip
# ---------------------------------------------------------------------------

def _render_alerts(watchlist_rows: list[dict], period_str: str) -> None:
    """Surface EXIT SIGNALs, new STRENGTHENING discoveries, and near-LT lots."""
    exit_signals = [r for r in watchlist_rows if r["status"] == "EXIT SIGNAL"]

    # STRENGTHENING signals not yet on watchlist — proactive discovery
    try:
        all_sigs = load_signals(period_str)
        watched = {r["display_name"] for r in watchlist_rows}
        new_strong = [
            r for r in all_sigs
            if r["status"] == "STRENGTHENING"
            and r["display_name"] not in watched
            and r.get("convergence_trend") in ("new", "accelerating")
        ]
    except Exception:
        new_strong = []

    # Near-LT lots with unrealised losses (time-sensitive)
    try:
        from dashboard.db import load_tax_lots
        from dashboard.prefs import load as load_prefs
        lots = load_tax_lots(load_prefs()["account_id"])
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

    if not rows:
        st.info(
            "Your watchlist is empty. Switch to the Discovery tab to find signals and add tickers.",
            icon=":material/playlist_add:",
        )
        return

    for row in rows:
        with st.container(border=True):
            col_name, col_score, col_status, col_act = st.columns(
                [3, 2, 2, 1], vertical_alignment="center"
            )

            with col_name:
                st.markdown(f"**{row['display_name']}**")
                detail_parts = []
                if row.get("sector"):
                    detail_parts.append(row["sector"])
                if row.get("date_added"):
                    detail_parts.append(f"Added {row['date_added']}")
                if detail_parts:
                    st.caption("  ·  ".join(detail_parts))

            with col_score:
                fs = row.get("final_score")
                if fs is not None:
                    st.markdown(f"### {_color_score(fs)}")
                    conv = row.get("convergence_score")
                    nlp  = row.get("nlp_composite_score")
                    st.caption(
                        f"Conv {_color_score(conv)}"
                        + (f"  ·  NLP {_color_score(nlp)}" if nlp is not None else "  ·  NLP :gray[—]")
                    )
                else:
                    st.caption(":gray[No signal this quarter]")

            with col_status:
                _badge(row.get("status"))
                trend = row.get("convergence_trend")
                bulls = row.get("n_funds_bullish")
                bears = row.get("n_funds_bearish")
                if trend and trend != "stable":
                    st.caption(f"Trend: {trend}")
                if bulls is not None:
                    st.caption(f":green[{bulls} bullish]  :red[{bears} bearish]")

            with col_act:
                key = f"rm_{row.get('cusip') or row.get('ticker')}"
                if st.button("Remove", key=key):
                    identifier = row.get("ticker") or row.get("cusip")
                    if identifier:
                        watchlist_remove(identifier)
                        st.toast(f"Removed {row['display_name']} from watchlist")
                        st.rerun()

            # Signal drivers — the most useful line after the headline numbers
            if row.get("signal_drivers"):
                st.caption(f":material/info: {row['signal_drivers']}")
            if row.get("note"):
                st.caption(f":material/edit_note: {row['note']}")


# ---------------------------------------------------------------------------
# Discovery tab
# ---------------------------------------------------------------------------

def _render_discovery(period_str: str) -> None:
    all_signals = load_signals(period_str)
    watched = load_watchlist_tickers()

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

    col_f4, col_f5 = st.columns([3, 6])
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

    st.caption(f"{len(filtered)} of {len(all_signals)} signals · {period_str}")

    if not filtered:
        st.info("No signals match the current filters.")
        return

    # Build display DataFrame
    rows_data = []
    for r in filtered:
        rows_data.append({
            "★":             "★" if r["display_name"] in watched else "",
            "Ticker":        r["display_name"],
            "Status":        r["status"] or "",
            "Score":         r["final_score"],
            "Conv":          r["convergence_score"],
            "NLP":           r["nlp_composite_score"],
            "▲":             r["n_funds_bullish"],
            "▼":             r["n_funds_bearish"],
            "Trend":         r["convergence_trend"] or "—",
            "Sector":        r["sector"] or "—",
            "Signal drivers": r["signal_drivers"] or "",
        })

    df = pd.DataFrame(rows_data)
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "★":              st.column_config.TextColumn("", width=30),
            "Ticker":         st.column_config.TextColumn("Ticker", width="small"),
            "Status":         st.column_config.TextColumn("Status", width="medium"),
            "Score":          st.column_config.ProgressColumn(
                                  "Score", min_value=-1.0, max_value=1.0, format="%.3f"
                              ),
            "Conv":           st.column_config.NumberColumn("Conv", format="%.3f"),
            "NLP":            st.column_config.NumberColumn("NLP", format="%.3f"),
            "▲":              st.column_config.NumberColumn("▲ Funds", width="small"),
            "▼":              st.column_config.NumberColumn("▼ Funds", width="small"),
            "Trend":          st.column_config.TextColumn("Trend", width="small"),
            "Sector":         st.column_config.TextColumn("Sector"),
            "Signal drivers": st.column_config.TextColumn("Signal drivers", width="large"),
        },
    )

    # Add to watchlist
    st.space("small")
    unwatched_tickers = [
        r["display_name"] for r in filtered
        if r["display_name"] not in watched and r.get("ticker")
    ]
    if unwatched_tickers:
        with st.container(horizontal=True):
            pick = st.selectbox(
                "Watch a ticker",
                options=[""] + unwatched_tickers,
                key="disc_pick",
                label_visibility="collapsed",
            )
            if st.button("Add to watchlist", type="primary", disabled=not pick):
                watchlist_add(pick)
                st.toast(f"Added {pick} to watchlist")
                st.rerun()


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
