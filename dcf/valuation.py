"""
DCF valuation engine: baseline construction, growth fade, FCF projection,
Gordon Growth terminal value, and Bull/Base/Bear scenario assembly.

Free cash flow (approved simplification, stated explicitly rather than
silently applied)
------------------------------------------------------------------------
    FCF = EBIT x (1 - effective_tax_rate) + D&A - Capex

Change in net working capital is NOT modeled (implicitly assumed zero). A
granular NWC build-up (receivables, inventory, payables) would inherit the
same XBRL tag-coverage gaps the GP factor's NIBCL work already documented —
only 76%/65% standalone AP/accrued-liabilities coverage, 20% resolving
neither. Rather than build a second fragile reconstruction on the same weak
tag coverage, NWC is dropped and this is flagged as a limitation everywhere
the result is surfaced (see STATED_LIMITATIONS below) — this is a stated
scope decision, not an oversight.

Baseline margin / D&A% / capex% (approved: 60/40 blend)
---------------------------------------------------------
Each of EBIT margin, D&A-as-%-of-revenue, and capex-as-%-of-revenue is
computed as 60% x (most recent annual observation) + 40% x (trailing
3-year average of the same ratio), then held FLAT across the entire 10-year
projection — one blend, reused for every projected year, rather than a
second fade curve stacked on top of the growth fade. This balances
responsiveness to a company's current economics against smoothing out one
unusual year, without adding a second independently-tunable trend
assumption on top of growth.

Growth fade (approved)
----------------------
Year-1 growth = the company's own trailing revenue CAGR (over the last
{CAGR_WINDOW_YEARS} annual observations available, capped by however much
history actually exists), clamped to [GROWTH_CLAMP_MIN, GROWTH_CLAMP_MAX] to
prevent one distortive year (a COVID trough, a large acquisition) from
being extrapolated indefinitely — same discipline as PEAD's dollar-based
SUE standardization and GP's ratio recalibrations. Linearly interpolated
down (or up) to a terminal growth rate by year 10.

Terminal growth = the current 10-year Treasury yield (dcf.wacc
.fetch_risk_free_rate()), not a hardcoded ~2.5% GDP-growth constant — more
theoretically grounded (a mature company's terminal growth shouldn't
persistently exceed the economy's long-run risk-free rate) and needs no
manual updates as rates move. Guarded to stay below WACC (gordon_terminal_
value below), since Gordon Growth is undefined/explodes otherwise.

Bull / Base / Bear (approved — NOT Monte Carlo)
------------------------------------------------
Three explicit scenarios varying only the Year-1 starting growth rate — all
three fade to the SAME terminal growth rate by year 10, since the economic
argument for fading to a risk-free-rate-level terminal growth is that
competitive forces erode any company's today's growth premium by year 10
regardless of starting point. A Monte Carlo simulation would sample from an
assumed probability distribution that is itself just a guess — implying a
statistical rigor this model doesn't actually have. Three labeled,
explainable scenarios are the more honest framing.

Bug found and fixed: bull/bear used to be computed as
clamp(base_growth +/- GROWTH_SPREAD, GROWTH_CLAMP_MIN, GROWTH_CLAMP_MAX) —
i.e. spread around the ALREADY-CLAMPED base value, reusing base's own
ceiling. For a name whose raw growth already exceeds the base ceiling (e.g.
NVDA: raw unclamped 5yr revenue CAGR of 66.9% against a 30% base ceiling),
base saturates at 30%, and bull (30% + 5pp, reclamped to the same 30%
ceiling) collapses onto it — no bull/base differentiation for exactly the
highest-growth names, where a reader would most want to see the bull case
reasoned through.

Considered a purely proportional/multiplicative fix (bull = base x 1.3) as
originally suggested, but that has a real correctness problem, not just a
stylistic one: for a negative or near-zero base growth rate, multiplying by
a factor > 1 either inverts the direction (a -5% base x 1.3 = -6.5%, making
"bull" WORSE than base) or produces a negligible spread (a 1% base barely
moves). The actual root cause isn't the additive-vs-proportional choice —
it's that bull/bear share base's ceiling at all. Fixed by deriving the
spread from the RAW, unclamped CAGR (not the already-clamped base value)
and giving bull/bear their own wider clamp bands
(BULL_GROWTH_CLAMP_MAX / BEAR_GROWTH_CLAMP_MIN, each GROWTH_CLAMP_HEADROOM
beyond the base band). This stays additive (simple, sign-robust, no
multiplicative blow-up or inversion) while fixing the actual bug: NVDA now
gets bear=base=30% (raw CAGR minus the spread is still above the base
ceiling, so bear saturates at the same conservative figure as base — there
is no more-bearish story the data supports) and bull=45% (genuinely
differentiated, reaching toward but still well below its true 66.9%
trailing growth). The asymmetry — bull differentiates from base more often
than bear does, for high-growth names — is a real, expected property of
this fix, not a residual bug: it reflects that the base clamp is already a
conservative read for such names, so there's little room for an
"even more conservative" bear case using the same mechanism.
"""

import statistics

import pandas as pd

from dcf.exclusions import check_business_model_fit
from dcf.fundamentals import fetch_ticker_dcf_fundamentals
from dcf.wacc import compute_wacc, cost_of_debt, cost_of_equity, fetch_risk_free_rate

PROJECTION_YEARS = 10
MARGIN_BLEND_WEIGHT_TTM = 0.60
CAGR_WINDOW_YEARS = 5          # uses up to this many trailing intervals (CAGR_WINDOW_YEARS+1 observations)
GROWTH_CLAMP_MIN = -0.15
GROWTH_CLAMP_MAX = 0.30
GROWTH_CLAMP_HEADROOM = 0.15   # extra room bull/bear get beyond the base band — see module docstring's clamp-collapse fix
BULL_GROWTH_CLAMP_MAX = GROWTH_CLAMP_MAX + GROWTH_CLAMP_HEADROOM   # 0.45
BEAR_GROWTH_CLAMP_MIN = GROWTH_CLAMP_MIN - GROWTH_CLAMP_HEADROOM   # -0.30
GROWTH_SPREAD = 0.05           # +/- applied to the RAW (unclamped) CAGR for Bull/Bear, not to the already-clamped Base
MIN_ANNUAL_OBSERVATIONS = 3    # floor for a reliable 3yr-average baseline

STATED_LIMITATIONS = (
    "Working-capital changes are not modeled (assumed zero) — a known "
    "simplification, not an oversight; see dcf/valuation.py module "
    "docstring. EBIT margin, D&A%, and capex% are held flat across the "
    "10-year projection at a blended 60% TTM / 40% trailing-3-year "
    "baseline, not separately faded. Bull/Base/Bear are three explicit, "
    "labeled scenarios, not a statistical (Monte Carlo) distribution."
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _cagr(start_value: float, end_value: float, periods: int) -> "float | None":
    if start_value <= 0 or end_value <= 0 or periods <= 0:
        return None
    return (end_value / start_value) ** (1.0 / periods) - 1.0


def compute_baseline(fund_df: pd.DataFrame) -> "dict | None":
    """
    Blended baseline (EBIT margin, D&A%, capex%, tax rate, Year-1 growth) plus
    the balance-sheet inputs (revenue, total debt, cash, diluted shares,
    interest expense) a DCF run needs, all from the most recent annual
    observation or a trailing window of them.

    Returns None if fewer than MIN_ANNUAL_OBSERVATIONS annual observations
    are available — a 3-year-average baseline isn't meaningful with less
    history than that (same "exclude rather than fabricate" floor used
    throughout this codebase, e.g. PEAD's MIN_QUARTERS).
    """
    if len(fund_df) < MIN_ANNUAL_OBSERVATIONS:
        return None

    df = fund_df.sort_values("period_end").reset_index(drop=True)
    ttm = df.iloc[-1]
    if not ttm["revenue"]:
        return None

    window3 = df.iloc[-3:]

    def _blended_ratio(numerator_col: str) -> float:
        ttm_ratio = ttm[numerator_col] / ttm["revenue"]
        window_ratios = [
            row[numerator_col] / row["revenue"]
            for _, row in window3.iterrows()
            if row["revenue"]
        ]
        avg_3yr = statistics.mean(window_ratios) if window_ratios else ttm_ratio
        return MARGIN_BLEND_WEIGHT_TTM * ttm_ratio + (1 - MARGIN_BLEND_WEIGHT_TTM) * avg_3yr

    ebit_margin = _blended_ratio("ebit")
    da_pct = _blended_ratio("da")
    capex_pct = _blended_ratio("capex")

    cagr_window = df.iloc[-(CAGR_WINDOW_YEARS + 1):] if len(df) > CAGR_WINDOW_YEARS else df
    raw_cagr = _cagr(
        cagr_window.iloc[0]["revenue"],
        cagr_window.iloc[-1]["revenue"],
        len(cagr_window) - 1,
    )
    raw_cagr = raw_cagr if raw_cagr is not None else 0.0
    start_growth = _clamp(raw_cagr, GROWTH_CLAMP_MIN, GROWTH_CLAMP_MAX)

    return {
        "ttm_revenue":        float(ttm["revenue"]),
        "ebit_margin":        ebit_margin,
        "da_pct":             da_pct,
        "capex_pct":          capex_pct,
        "tax_rate":           float(ttm["effective_tax_rate"]),
        "tax_rate_source":    ttm["tax_rate_source"],
        "raw_growth_cagr":    raw_cagr,
        "start_growth":       start_growth,
        "total_debt":         float(ttm["total_debt"]),
        "debt_source":        ttm["debt_source"],
        "cash":               float(ttm["cash"]),
        "interest_expense":   float(ttm["interest_expense"]),
        "interest_expense_source": ttm["interest_expense_source"],
        "diluted_shares":     float(ttm["diluted_shares"]) if pd.notna(ttm["diluted_shares"]) else None,
        "n_annual_observations": len(df),
        "most_recent_period": str(ttm["period_end"]),
    }


def fade_growth_path(start_growth: float, terminal_growth: float, years: int = PROJECTION_YEARS) -> list[float]:
    """Linear interpolation from start_growth (year 1) to terminal_growth (year `years`)."""
    return [
        start_growth + (terminal_growth - start_growth) * (t / years)
        for t in range(1, years + 1)
    ]


def project_fcf(
    revenue_start: float,
    growth_path: list[float],
    ebit_margin: float,
    da_pct: float,
    capex_pct: float,
    tax_rate: float,
) -> list[float]:
    """Year-by-year unlevered FCF given a starting (TTM) revenue and a per-year growth path."""
    revenue = revenue_start
    fcfs = []
    for g in growth_path:
        revenue *= (1 + g)
        ebit = revenue * ebit_margin
        da = revenue * da_pct
        capex = revenue * capex_pct
        fcfs.append(ebit * (1 - tax_rate) + da - capex)
    return fcfs


def discount_cash_flows(values: list[float], rate: float) -> list[float]:
    return [v / (1 + rate) ** (t + 1) for t, v in enumerate(values)]


def gordon_terminal_value(terminal_fcf: float, wacc: float, terminal_growth: float) -> tuple[float, float]:
    """
    Gordon Growth terminal value: TV = FCF_terminal x (1+g) / (WACC - g).
    Returns (terminal_value, terminal_growth_actually_used) — g is capped at
    WACC - 0.005 if it would otherwise equal or exceed WACC (undefined /
    sign-flipped result), so the cap event is visible to the caller rather
    than silently producing nonsense.
    """
    g = terminal_growth
    if g >= wacc:
        g = wacc - 0.005
    return terminal_fcf * (1 + g) / (wacc - g), g


def run_scenario(
    name: str,
    start_growth: float,
    baseline: dict,
    wacc: float,
    terminal_growth: float,
    years: int = PROJECTION_YEARS,
) -> dict:
    growth_path = fade_growth_path(start_growth, terminal_growth, years)
    fcfs = project_fcf(
        baseline["ttm_revenue"], growth_path,
        baseline["ebit_margin"], baseline["da_pct"], baseline["capex_pct"], baseline["tax_rate"],
    )
    discounted = discount_cash_flows(fcfs, wacc)
    pv_explicit = sum(discounted)
    terminal_value, terminal_growth_used = gordon_terminal_value(fcfs[-1], wacc, terminal_growth)
    pv_terminal_value = terminal_value / (1 + wacc) ** years
    enterprise_value = pv_explicit + pv_terminal_value
    equity_value = enterprise_value - baseline["total_debt"] + baseline["cash"]

    diluted_shares = baseline["diluted_shares"]
    per_share = equity_value / diluted_shares if diluted_shares else None
    pct_from_terminal_value = pv_terminal_value / enterprise_value if enterprise_value else None

    return {
        "scenario":                name,
        "start_growth":            start_growth,
        "growth_path":             growth_path,
        "fcf_projection":          fcfs,
        "enterprise_value":        enterprise_value,
        "equity_value":            equity_value,
        "per_share":               per_share,
        "pct_from_terminal_value": pct_from_terminal_value,
        "terminal_growth_used":    terminal_growth_used,
    }


def _market_data(ticker: str) -> "tuple[float | None, float | None]":
    """(market_cap, current_price) from yfinance's info dict — live market data, not financial-statement history."""
    import yfinance as yf

    info = yf.Ticker(ticker).info
    market_cap = info.get("marketCap")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    return (
        float(market_cap) if market_cap else None,
        float(current_price) if current_price else None,
    )


def run_dcf(ticker: str) -> dict:
    """
    Full DCF run for one ticker: fundamentals -> baseline -> WACC ->
    Bull/Base/Bear scenarios -> per-share intrinsic value + upside/downside
    vs. current price.

    Returns a dict always containing "ticker"; on any data-quality failure,
    also contains "error" (one of: unsuitable_business_model,
    no_xbrl_fundamentals, insufficient_history, no_diluted_shares, no_beta,
    no_market_data) and no "scenarios" key — callers should check for
    "error" before reading scenario results, same pattern as
    ticker_ff3_profile returning None for insufficient data.

    unsuitable_business_model (see dcf/exclusions.py) is checked first,
    before any fetch work — standard unlevered-FCF DCF is a poor
    methodological fit for banks, insurers, and REITs regardless of data
    quality, so there's no point spending a fetch on a number that
    shouldn't be produced at all.
    """
    unsuitable_reason = check_business_model_fit(ticker)
    if unsuitable_reason is not None:
        return {"ticker": ticker, "error": "unsuitable_business_model", "reason": unsuitable_reason}

    fund_df = fetch_ticker_dcf_fundamentals(ticker)
    if fund_df.empty:
        return {"ticker": ticker, "error": "no_xbrl_fundamentals"}

    baseline = compute_baseline(fund_df)
    if baseline is None:
        return {"ticker": ticker, "error": "insufficient_history"}
    if not baseline["diluted_shares"]:
        return {"ticker": ticker, "error": "no_diluted_shares"}

    from dashboard.factor import ticker_ff3_profile
    profile = ticker_ff3_profile(ticker)
    if profile is None or profile.get("beta_market") is None:
        return {"ticker": ticker, "error": "no_beta"}
    beta = profile["beta_market"]

    market_cap, current_price = _market_data(ticker)
    if market_cap is None:
        return {"ticker": ticker, "error": "no_market_data"}

    risk_free_rate = fetch_risk_free_rate()
    re = cost_of_equity(beta, risk_free_rate)
    rd = cost_of_debt(baseline["interest_expense"], baseline["total_debt"], baseline["tax_rate"])
    wacc = compute_wacc(market_cap, baseline["total_debt"], re, rd)

    # Spread is applied to the RAW (unclamped) CAGR, and bull/bear each get
    # their own wider clamp band — NOT base's band re-applied — so a
    # name whose raw growth already exceeds the base ceiling (e.g. NVDA)
    # still shows genuine bull/base differentiation instead of collapsing.
    # See module docstring's "Bug found and fixed" note.
    raw_cagr = baseline["raw_growth_cagr"]
    scenario_bounds = {
        "bear": (raw_cagr - GROWTH_SPREAD, BEAR_GROWTH_CLAMP_MIN, GROWTH_CLAMP_MAX),
        "base": (baseline["start_growth"], GROWTH_CLAMP_MIN, GROWTH_CLAMP_MAX),
        "bull": (raw_cagr + GROWTH_SPREAD, GROWTH_CLAMP_MIN, BULL_GROWTH_CLAMP_MAX),
    }

    scenarios = {}
    for name, (raw_value, lo, hi) in scenario_bounds.items():
        start_growth = _clamp(raw_value, lo, hi)
        scenario = run_scenario(name, start_growth, baseline, wacc, risk_free_rate)
        if scenario["per_share"] is not None and current_price:
            scenario["upside_pct"] = (scenario["per_share"] - current_price) / current_price * 100
        else:
            scenario["upside_pct"] = None
        scenarios[name] = scenario

    return {
        "ticker":            ticker,
        "current_price":     current_price,
        "market_cap":        market_cap,
        "beta":              beta,
        "risk_free_rate":    risk_free_rate,
        "cost_of_equity":    re,
        "cost_of_debt":      rd,
        "wacc":              wacc,
        "baseline":          baseline,
        "scenarios":         scenarios,
        "stated_limitations": STATED_LIMITATIONS,
    }
