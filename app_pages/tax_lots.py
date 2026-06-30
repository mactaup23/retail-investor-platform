"""
Tax Lots page — cost basis inventory, harvest candidates, and sell modeler.

Layout
------
Settings expander:  Tax rates (st_rate, lt_rate, state_rate, NIIT) + account_id
CSV upload:         Drag-and-drop; auto-detects Fidelity / Schwab / IBKR
Three tabs:
  Inventory         All lots grouped by ticker with G/L, LT/ST status, flags
  Harvest           Loss lots with estimated tax savings and wash-sale notes
  Sell Modeler      Ticker → quantity → FIFO / LIFO / MIN_TAX side-by-side
"""
from __future__ import annotations

import datetime

import streamlit as st
import pandas as pd

import dashboard.prefs as prefs_module
from dashboard.db import load_tax_lots, bust_tax_lot_cache

# ---------------------------------------------------------------------------
# Settings expander
# ---------------------------------------------------------------------------

def _load_rates_from_prefs(p: dict):
    from smart_money.taxlot import TaxRates
    return TaxRates(
        st_rate    = p["st_rate"],
        lt_rate    = p["lt_rate"],
        state_rate = p["state_rate"],
        niit       = p["niit"],
    )


st.header(":material/account_balance: Tax lots")

prefs = prefs_module.load()

with st.expander("Settings", icon=":material/tune:"):
    with st.form("tax_prefs_form", border=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st_rate = st.number_input(
                "ST rate", min_value=0.0, max_value=1.0,
                value=float(prefs["st_rate"]), step=0.01, format="%.2f",
                help="Federal short-term / ordinary income marginal rate",
            )
        with c2:
            lt_rate = st.number_input(
                "LT rate", min_value=0.0, max_value=1.0,
                value=float(prefs["lt_rate"]), step=0.01, format="%.2f",
                help="Federal long-term capital gains rate (0 / 0.15 / 0.20)",
            )
        with c3:
            state_rate = st.number_input(
                "State rate", min_value=0.0, max_value=0.20,
                value=float(prefs["state_rate"]), step=0.005, format="%.3f",
                help="State rate (added to both ST and LT)",
            )
        with c4:
            niit = st.toggle(
                "NIIT (+3.8%)", value=bool(prefs["niit"]),
                help="Net Investment Income Tax surcharge",
            )
        with c5:
            account_id = st.text_input(
                "Account ID", value=str(prefs["account_id"]),
                help="Label for this account's lots (e.g. fidelity-taxable)",
            )
        if st.form_submit_button("Save settings", type="primary"):
            prefs_module.save({
                "st_rate":    st_rate,
                "lt_rate":    lt_rate,
                "state_rate": state_rate,
                "niit":       niit,
                "account_id": account_id,
            })
            prefs = prefs_module.load()
            st.toast("Settings saved")
            st.rerun()

# Always derive rates from current prefs (may have just been saved)
prefs = prefs_module.load()
rates = _load_rates_from_prefs(prefs)
account_id = prefs["account_id"]

# ---------------------------------------------------------------------------
# CSV upload
# ---------------------------------------------------------------------------

lots = load_tax_lots(account_id)

with st.expander(
    f"Upload cost-basis CSV  ·  {len(lots)} lots loaded" if lots else "Upload cost-basis CSV",
    icon=":material/upload:",
    expanded=not bool(lots),
):
    uploaded = st.file_uploader(
        "Drag-and-drop a brokerage CSV export",
        type=["csv"],
        label_visibility="collapsed",
        key="tax_upload",
    )
    st.caption(
        "Supported formats auto-detected: Fidelity (Lot-Level Detail), "
        "Schwab (Portfolio Lot Detail), IBKR (Open Lots report). "
        "Account ID from settings is used as the lot label."
    )
    if uploaded:
        with st.spinner("Ingesting lots…"):
            try:
                from smart_money.taxlot import ingest
                csv_bytes = uploaded.read()
                ingested = ingest(
                    csv_bytes,
                    account_id=account_id,
                    fetch_prices=True,
                )
                bust_tax_lot_cache()
                lots = load_tax_lots(account_id)
                st.success(
                    f"Ingested {len(ingested)} lots for account **{account_id}**.",
                    icon=":material/check_circle:",
                )
                st.rerun()
            except Exception as e:
                st.error(f"Ingest failed: {e}", icon=":material/error:")

_EMPTY_PLACEHOLDER = (
    "Upload a cost-basis CSV above to unlock this view. "
    "The Inventory tab shows all your lots with unrealized G/L, "
    "Harvest Candidates surfaces loss lots with estimated tax savings, "
    "and the Sell Modeler compares FIFO, LIFO, and MIN_TAX side-by-side before you act."
)

if not lots:
    tab_inv, tab_harv, tab_sell = st.tabs(
        [
            ":material/table_chart: Inventory",
            ":material/savings: Harvest candidates",
            ":material/calculate: Sell modeler",
        ],
        on_change="rerun",
    )
    with tab_inv:
        st.info(_EMPTY_PLACEHOLDER, icon=":material/upload:")
    with tab_harv:
        st.info(_EMPTY_PLACEHOLDER, icon=":material/upload:")
    with tab_sell:
        st.info(_EMPTY_PLACEHOLDER, icon=":material/upload:")
    st.stop()

# Show valuation date staleness
val_dates = {l["valuation_date"] for l in lots}
val_date_str = max(val_dates) if val_dates else "unknown"
st.caption(
    f"Prices as of {val_date_str}  ·  Re-upload CSV to refresh current prices  ·  "
    f"Effective rates: ST {(rates.effective_st_rate * 100):.1f}%  "
    f"LT {(rates.effective_lt_rate * 100):.1f}%"
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_inv, tab_harv, tab_sell = st.tabs(
    [
        ":material/table_chart: Inventory",
        ":material/savings: Harvest candidates",
        ":material/calculate: Sell modeler",
    ],
    on_change="rerun",
)


# ---------------------------------------------------------------------------
# Inventory tab
# ---------------------------------------------------------------------------

def _render_inventory(lots: list[dict]) -> None:
    tickers = sorted({l["ticker"] for l in lots})
    total_cost   = sum((l["total_cost_basis"] or 0) for l in lots)
    total_value  = sum((l["current_value"] or 0) for l in lots)
    total_gl     = total_value - total_cost

    # Portfolio summary row
    with st.container(horizontal=True):
        st.metric("Tickers", len(tickers), border=True)
        st.metric("Lots", len(lots), border=True)
        st.metric("Total cost basis", f"${total_cost:,.0f}", border=True)
        st.metric("Current value", f"${total_value:,.0f}", border=True)
        delta_str = f"${total_gl:+,.0f}"
        st.metric("Unrealised G/L", delta_str, border=True)

    # Per-ticker groups
    for ticker in tickers:
        ticker_lots = [l for l in lots if l["ticker"] == ticker]
        ticker_cost  = sum((l["total_cost_basis"] or 0) for l in ticker_lots)
        ticker_value = sum((l["current_value"] or 0) for l in ticker_lots)
        ticker_gl    = ticker_value - ticker_cost
        lt_count     = sum(1 for l in ticker_lots if l["is_long_term"])
        st_count     = len(ticker_lots) - lt_count
        near_lt_flag = any(l["near_lt"] for l in ticker_lots)
        wash_flag    = any(l["wash_sale_risk"] for l in ticker_lots)

        gl_color = "green" if ticker_gl >= 0 else "red"
        label_parts = [f"**{ticker}**  ·  {len(ticker_lots)} lots"]
        if near_lt_flag:
            label_parts.append(":orange[⚠ near-LT]")
        if wash_flag:
            label_parts.append(":red[⚠ wash-sale risk]")

        with st.expander("  ".join(label_parts)):
            # Summary line for this ticker
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Cost basis", f"${ticker_cost:,.0f}")
            with c2:
                st.metric("Value", f"${ticker_value:,.0f}")
            with c3:
                gl_pct = (ticker_gl / ticker_cost * 100) if ticker_cost else 0
                st.metric("G/L", f"${ticker_gl:+,.0f}  ({gl_pct:+.1f}%)")
            with c4:
                st.metric("LT / ST", f"{lt_count} / {st_count}")

            # Lot detail table
            df = pd.DataFrame([
                {
                    "Acquired":      l["acquisition_date"],
                    "Qty":           l["quantity"],
                    "Cost/sh":       l["cost_basis_per_share"],
                    "Price":         l["current_price"],
                    "Cost basis":    l["total_cost_basis"],
                    "Value":         l["current_value"],
                    "G/L $":         l["unrealized_gl"],
                    "G/L %":         (l["unrealized_gl_pct"] or 0),
                    "Days":          l["holding_days"],
                    "LT/ST":         "LT" if l["is_long_term"] else "ST",
                    "→LT in":        f"{l['days_to_lt']}d" if not l["is_long_term"] else "—",
                    "Wash risk":     "⚠" if l["wash_sale_risk"] else "",
                }
                for l in ticker_lots
            ])
            st.dataframe(
                df,
                hide_index=True,
                column_config={
                    "Qty":        st.column_config.NumberColumn(format="%.4f"),
                    "Cost/sh":    st.column_config.NumberColumn(format="$%.4f"),
                    "Price":      st.column_config.NumberColumn(format="$%.2f"),
                    "Cost basis": st.column_config.NumberColumn(format="$%,.2f"),
                    "Value":      st.column_config.NumberColumn(format="$%,.2f"),
                    "G/L $":      st.column_config.NumberColumn(format="$%+,.2f"),
                    "G/L %":      st.column_config.NumberColumn(format="%+.2f%%"),
                    "Days":       st.column_config.NumberColumn(format="%d"),
                },
            )


# ---------------------------------------------------------------------------
# Harvest candidates tab
# ---------------------------------------------------------------------------

def _render_harvest(lots: list[dict], rates) -> None:
    from smart_money.taxlot import harvest_candidates, TaxRates

    try:
        candidates = harvest_candidates(account_id=account_id, rates=rates)
    except Exception as e:
        st.error(f"Could not compute harvest candidates: {e}", icon=":material/error:")
        return

    if not candidates:
        st.info(
            "No harvestable losses in the current inventory.",
            icon=":material/check_circle:",
        )
        return

    total_savings = sum(c.tax_savings or 0 for c in candidates)
    st.metric(
        "Estimated total tax savings",
        f"${total_savings:,.0f}",
        help="Sum of estimated tax savings across all harvestable loss lots at your configured rates",
    )
    st.caption(
        ":gray[Tax savings = abs(unrealised loss) × applicable rate. "
        "Consult a tax professional before acting on these estimates.]"
    )

    rows = []
    for c in candidates:
        rows.append({
            "Ticker":       c.ticker,
            "Acquired":     str(c.acquisition_date),
            "Qty":          c.quantity,
            "Loss":         c.unrealized_gl,
            "Type":         c.gain_type.replace("_", " ").title(),
            "Est. savings": c.tax_savings,
            "→LT in":       f"{c.days_to_lt}d" if not c.near_lt else f":orange[{c.days_to_lt}d]",
            "Wash risk":    "⚠ DISALLOWED" if (c.wash_sale_flag and c.wash_sale_flag.kind == "disallowed") else
                            ("⚠ warning" if c.wash_sale_flag else ""),
            "Recommendation": c.recommendation,
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "Qty":          st.column_config.NumberColumn(format="%.4f"),
            "Loss":         st.column_config.NumberColumn(format="$%+,.2f"),
            "Est. savings": st.column_config.NumberColumn(format="$%,.0f"),
        },
    )

    # Show wash-sale explanations for disallowed lots
    disallowed = [c for c in candidates if c.wash_sale_flag and c.wash_sale_flag.kind == "disallowed"]
    if disallowed:
        st.warning(
            f"{len(disallowed)} lot(s) have disallowed losses due to the wash-sale rule. "
            "See details below.",
            icon=":material/warning:",
        )
        for c in disallowed:
            st.caption(f"**{c.ticker}:** {c.wash_sale_flag.explanation}")


# ---------------------------------------------------------------------------
# Sell Modeler tab
# ---------------------------------------------------------------------------

def _render_sell_modeler(lots: list[dict], rates) -> None:
    from smart_money.taxlot import model_sell

    tickers = sorted({l["ticker"] for l in lots})

    with st.form("sell_model_form", border=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            ticker = st.selectbox("Ticker", tickers, key="sell_ticker")
        with c2:
            ticker_lots = [l for l in lots if l["ticker"] == (ticker or "")]
            max_qty     = sum(l["quantity"] for l in ticker_lots)
            quantity    = st.number_input(
                "Quantity to sell",
                min_value=0.0001,
                max_value=float(max_qty),
                value=float(max_qty),
                step=1.0,
                format="%.4f",
            )
        with c3:
            cur_price = next(
                (l["current_price"] for l in ticker_lots if l["current_price"]),
                None
            )
            sell_price = st.number_input(
                "Sell price ($/sh)",
                min_value=0.01,
                value=float(cur_price) if cur_price else 100.0,
                step=0.01,
                format="%.2f",
            )
        run = st.form_submit_button("Model sell", type="primary")

    if not run:
        st.caption("Select a ticker, enter quantity and price, then click Model sell.")
        return

    with st.spinner(f"Modelling sell of {quantity:.4f} {ticker}…"):
        try:
            decisions = model_sell(
                ticker,
                quantity,
                account_id=account_id,
                sell_price=sell_price,
                rates=rates,
                sell_date=datetime.date.today(),
            )
        except Exception as e:
            st.error(f"Model sell failed: {e}", icon=":material/error:")
            return

    if not decisions:
        st.warning("No sell decisions returned — check that sufficient lots exist.")
        return

    # Comparison table — methods as columns, metrics as rows
    methods = ["FIFO", "LIFO", "MIN_TAX"]
    metric_labels = [
        ("Total proceeds",       "total_proceeds",      "$%,.2f"),
        ("ST gain recognised",   "total_st_gain",       "$%,.2f"),
        ("LT gain recognised",   "total_lt_gain",       "$%,.2f"),
        ("ST loss recognised",   "total_st_loss",       "$%,.2f"),
        ("LT loss recognised",   "total_lt_loss",       "$%,.2f"),
        ("Tax owed (est.)",      "net_tax_owed",        "$%,.2f"),
        ("Effective tax rate",   "effective_tax_rate",  "%.1f%%"),
        ("After-tax proceeds",   "after_tax_proceeds",  "$%,.2f"),
    ]

    table_rows = []
    for label, attr, fmt in metric_labels:
        row = {"Metric": label}
        for m in methods:
            d = decisions.get(m)
            if d is None:
                row[m] = "—"
            else:
                v = getattr(d, attr)
                if "%" in fmt:
                    row[m] = fmt % (v * 100)
                elif "$" in fmt:
                    row[m] = fmt % v
                else:
                    row[m] = str(v)
        table_rows.append(row)

    comp_df = pd.DataFrame(table_rows).set_index("Metric")
    st.dataframe(comp_df)

    # Highlight best after-tax proceeds
    at_proceeds = {m: getattr(d, "after_tax_proceeds", 0) for m, d in decisions.items() if d}
    if at_proceeds:
        best_method = max(at_proceeds, key=at_proceeds.get)
        st.caption(
            f":green[**{best_method}**] produces the highest after-tax proceeds "
            f"(${at_proceeds[best_method]:,.2f}) for this transaction."
        )

    # Wash-sale warnings
    all_flags = []
    for d in decisions.values():
        if d:
            all_flags.extend(d.wash_sale_flags)

    if all_flags:
        unique_expl = {f.explanation for f in all_flags}
        for expl in unique_expl:
            kind = "error" if "disallowed" in expl.lower() else "warning"
            if kind == "error":
                st.error(expl, icon=":material/warning:")
            else:
                st.warning(expl, icon=":material/info:")

    # MIN_TAX vs LIFO note
    st.caption(
        ":gray[**MIN_TAX vs LIFO:** LIFO may numerically show lower tax when disallowed-loss lots "
        "are selected. MIN_TAX deliberately excludes those lots — the LIFO 'savings' are deferred "
        "into the replacement lot's cost basis, not realised. MIN_TAX is the economically correct "
        "choice for harvesting. Consult a tax professional before executing any sale.]"
    )

    # Lot-level detail
    best_d = decisions.get(best_method) if at_proceeds else None
    if best_d and best_d.lots_sold:
        with st.expander(f"Lot detail — {best_method}", icon=":material/list:"):
            lot_rows = [
                {
                    "Lot acquired":  str(sl.acquisition_date),
                    "Qty":           sl.quantity_sold,
                    "Cost/sh":       sl.cost_basis_per_share,
                    "Proceeds":      sl.proceeds,
                    "G/L":           sl.gain_loss,
                    "Type":          sl.gain_type.replace("_", " ").title(),
                    "Tax":           sl.tax_owed,
                    "After-tax":     sl.after_tax_proceeds,
                }
                for sl in best_d.lots_sold
            ]
            st.dataframe(
                pd.DataFrame(lot_rows),
                hide_index=True,
                column_config={
                    "Qty":       st.column_config.NumberColumn(format="%.4f"),
                    "Cost/sh":   st.column_config.NumberColumn(format="$%.4f"),
                    "Proceeds":  st.column_config.NumberColumn(format="$%,.2f"),
                    "G/L":       st.column_config.NumberColumn(format="$%+,.2f"),
                    "Tax":       st.column_config.NumberColumn(format="$%,.2f"),
                    "After-tax": st.column_config.NumberColumn(format="$%,.2f"),
                },
            )


# ---------------------------------------------------------------------------
# Render tabs
# ---------------------------------------------------------------------------

if tab_inv.open:
    with tab_inv:
        _render_inventory(lots)

if tab_harv.open:
    with tab_harv:
        _render_harvest(lots, rates)

if tab_sell.open:
    with tab_sell:
        _render_sell_modeler(lots, rates)
