"""
Backtest page — does the FinalSignal / convergence signal predict returns?

Layout
------
Header + methodology caveats (45-day knowledge lag, watchlist vs full universe)
Horizon selector (1M / 1Q / 2Q trading-day forward return)
Stat tiles: mean IC, t-stat, hit rate, coverage — full universe vs watchlist
IC time series: per-quarter bars (diverging by sign) + rolling 4Q average line
Scatter: score vs. forward return for a selected quarter, sanity-check view

All data comes from smart_money.backtest via dashboard.db cache wrappers.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.db import load_backtest, load_backtest_observations

_HORIZON_LABELS = {21: "1 month (21 trading days)", 63: "1 quarter (63 trading days)", 126: "2 quarters (126 trading days)"}
_UNIVERSE_LABELS = {"full": "Full universe", "watchlist": "Watchlist (FinalSignal)"}
_POS_COLOR = "#34D399"   # green — reused from portfolio.py's factor palette
_NEG_COLOR = "#F87171"   # red
_LINE_COLOR = "#94A3B8"  # muted slate — rolling average, never competes with the bars


st.header(":material/query_stats: Signal Backtest")

st.markdown(
    "Measures whether the convergence + NLP signal actually predicts subsequent returns, "
    "via the **Information Coefficient** (Spearman rank correlation between signal score "
    "and forward return) computed per quarter."
)

with st.expander("Methodology and caveats", expanded=False):
    st.markdown("""
- **Look-ahead safety** — a 13F is not public on its quarter-end balance date; funds have
  up to 45 calendar days to file. Every return window starts from `knowledge_date = period + 45 days`,
  priced at the first tradeable close on or after that date — never before.
- **Two universes, shown side by side**:
    - **Full universe** — every scored cusip that quarter, re-blended with NLP the same way
      the live signal does, but *without* the watchlist discovery filter. This is the unbiased
      cross-section.
    - **Watchlist** — `FinalSignal.final_score` as actually persisted and shown on the Signals
      page. This table structurally excludes the negative / low-conviction tail (a cusip only
      appears once its score crosses the 0.30 discovery threshold, or briefly on exit) — so IC
      measured here is expected to look better than the signal's true unrestricted predictive
      power. Treat it as "how good does the *tool you actually see* look," not ground truth.
- **Missing prices are dropped, not imputed.** A quarter's IC is only computed when at least
  10 cusips have both an entry and exit price; otherwise it's reported as "not enough data."
- **Coverage %** is the fraction of that quarter's scored cusips that had usable prices — a low
  coverage % means the IC for that quarter is based on a thin, non-representative slice.
""")

data = load_backtest()
quarter_df = pd.DataFrame(data["quarter_ics"])
horizons_df = pd.DataFrame(data["horizons"])

if quarter_df.empty:
    st.info(
        "No ConvergenceScore data found. Run the pipeline (`smart_money.pipeline`) before "
        "backtesting.",
        icon=":material/info:",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Horizon selector
# ---------------------------------------------------------------------------

horizon = st.selectbox(
    "Forward-return horizon",
    options=list(_HORIZON_LABELS.keys()),
    format_func=lambda h: _HORIZON_LABELS[h],
)

quarter_df = quarter_df[quarter_df["horizon_days"] == horizon].copy()
horizons_df = horizons_df[horizons_df["horizon_days"] == horizon].copy()

n_periods = quarter_df["period"].nunique()
if n_periods < 2:
    st.warning(
        f"Only {n_periods} quarter{'s' if n_periods != 1 else ''} of data available for this "
        "horizon so far — an IC time series needs several quarters of pipeline history to be "
        "meaningful. Numbers below will fill in as more quarters are run through the pipeline "
        "and `signal.combine()`.",
        icon=":material/hourglass_empty:",
    )

# ---------------------------------------------------------------------------
# Stat tiles — full vs watchlist
# ---------------------------------------------------------------------------

st.subheader("Summary")
cols = st.columns(2)
for col, universe in zip(cols, ("full", "watchlist")):
    row = horizons_df[horizons_df["universe"] == universe]
    with col:
        st.markdown(f"**{_UNIVERSE_LABELS[universe]}**")
        if row.empty or row.iloc[0]["n_quarters"] == 0:
            st.caption("Not enough quarters with a computable IC yet.")
            continue
        r = row.iloc[0]
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Mean IC", f"{r['mean_ic']:+.3f}" if r["mean_ic"] is not None else "—")
        t2.metric("t-stat", f"{r['t_stat']:+.2f}" if r["t_stat"] is not None else "—")
        t3.metric("Hit rate", f"{r['hit_rate']:.0%}" if r["hit_rate"] is not None else "—")
        t4.metric("Quarters", int(r["n_quarters"]))

st.caption(
    "t-stat = mean IC / (IC std / √n) — a rough significance check, not a formal hypothesis "
    "test; treat values below ~2 as directional, not conclusive, given how few quarters "
    "of pipeline history exist so far."
)

# ---------------------------------------------------------------------------
# IC time series — one small-multiple chart per universe
# ---------------------------------------------------------------------------

st.subheader("IC by quarter")

for universe in ("full", "watchlist"):
    uni_df = quarter_df[quarter_df["universe"] == universe].sort_values("period").copy()
    if uni_df.empty:
        continue

    st.markdown(f"**{_UNIVERSE_LABELS[universe]}**")

    plot_df = uni_df.dropna(subset=["ic"]).copy()
    if plot_df.empty:
        st.caption("No quarter yet has enough observations (≥10) for a computable IC.")
        continue

    plot_df["sign"] = plot_df["ic"].apply(lambda v: "Positive" if v >= 0 else "Negative")

    rolling = next(
        (h["rolling_4q"] for h in data["horizons"]
         if h["horizon_days"] == horizon and h["universe"] == universe),
        [],
    )
    rolling_df = pd.DataFrame(rolling, columns=["period", "rolling_ic"]).dropna()

    zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#64748B", strokeDash=[2, 2]).encode(y="y:Q")

    bars = (
        alt.Chart(plot_df)
        .mark_bar(size=18)
        .encode(
            x=alt.X("period:N", title=None, sort=list(plot_df["period"])),
            y=alt.Y("ic:Q", title="Information Coefficient"),
            color=alt.Color(
                "sign:N",
                scale=alt.Scale(domain=["Positive", "Negative"], range=[_POS_COLOR, _NEG_COLOR]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("period:N", title="Quarter"),
                alt.Tooltip("ic:Q", format="+.3f", title="IC"),
                alt.Tooltip("n_obs:Q", title="Observations"),
                alt.Tooltip("coverage_pct:Q", format=".0%", title="Coverage"),
            ],
        )
    )

    layers = [zero_rule, bars]
    if not rolling_df.empty:
        line = (
            alt.Chart(rolling_df)
            .mark_line(color=_LINE_COLOR, strokeWidth=2, point=alt.OverlayMarkDef(color=_LINE_COLOR, size=30))
            .encode(
                x=alt.X("period:N", sort=list(plot_df["period"])),
                y=alt.Y("rolling_ic:Q"),
                tooltip=[alt.Tooltip("period:N", title="Quarter"), alt.Tooltip("rolling_ic:Q", format="+.3f", title="4Q avg IC")],
            )
        )
        layers.append(line)

    st.altair_chart(alt.layer(*layers).properties(height=240), use_container_width=True)

st.caption("Bars = per-quarter IC (green = positive, red = negative). Gray line = trailing 4-quarter average IC.")

# ---------------------------------------------------------------------------
# Scatter — score vs. forward return, sanity-check view
# ---------------------------------------------------------------------------

st.subheader("Score vs. forward return")

available_periods = sorted(quarter_df["period"].unique(), reverse=True)
scatter_universe = st.radio("Universe", options=["full", "watchlist"], format_func=lambda u: _UNIVERSE_LABELS[u], horizontal=True)
scatter_period = st.selectbox("Quarter", options=available_periods)

obs = load_backtest_observations(scatter_period, int(horizon), scatter_universe)
obs_df = pd.DataFrame(obs)

if obs_df.empty:
    st.caption("No priced observations for this quarter / horizon / universe combination.")
else:
    scatter = (
        alt.Chart(obs_df)
        .mark_circle(size=40, opacity=0.6, color="#60A5FA")
        .encode(
            x=alt.X("score:Q", title="Signal score"),
            y=alt.Y("forward_return:Q", title="Forward return", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("ticker:N", title="Ticker"),
                alt.Tooltip("score:Q", format="+.3f"),
                alt.Tooltip("forward_return:Q", format="+.2%"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(scatter, use_container_width=True)
    st.caption(f"{len(obs_df)} cusips priced for {scatter_period}, {_HORIZON_LABELS[horizon]}, {_UNIVERSE_LABELS[scatter_universe]}.")
