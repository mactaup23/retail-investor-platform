"""
WACC (weighted average cost of capital) for the DCF engine.

Cost of equity: CAPM, reusing the platform's own beta from Module 1's factor
model (dashboard.factor.ticker_ff3_profile — the FF7-daily OLS regression)
rather than computing a second, separate beta.

Risk-free rate: the current 10-year US Treasury yield (^TNX), NOT the
3-month-bill-based rate the platform's factor regressions use
(factor_engine/french_data.py get_ff4_daily/get_ff7_daily's "rf" column,
sourced from Ken French's daily series / historically ^IRX — see CLAUDE.md's
"Ken French RF swap" polish item). That short-duration rate is the right
choice for daily factor regressions; it is duration-mismatched for a 10-year
DCF cash flow stream and its perpetuity growth rate, so this module
deliberately fetches its own long-duration rate rather than reusing the
factor model's. Also used directly as the terminal growth rate (see
valuation.py) — avoids a hardcoded "2.5%-ish GDP growth" constant that would
need manual updates as rates move.

Cost of debt: interest expense / total debt (book cost of existing debt),
after-tax. Deliberately simple for v1 — equity value is far more sensitive
to cost of equity than cost of debt in a properly-weighted WACC (approved
as a v1 simplification), so no separate credit-spread/rating-based model.

Weights: E = current market value of equity (price x shares — market cap is
observable and correct for equity weight); D = book value of total debt
(market value of debt isn't practically observable here; book value is the
standard proxy).
"""

import yfinance as yf

EQUITY_RISK_PREMIUM = 0.05   # documented assumption (Damodaran long-run US implied ERP range), not platform-derived


def fetch_risk_free_rate() -> float:
    """
    Current 10-year US Treasury yield, as a decimal (e.g. 0.045 for 4.5%).
    Raises if ^TNX has no recent price data — callers should treat this as a
    hard dependency, not silently default a risk-free rate.
    """
    hist = yf.Ticker("^TNX").history(period="5d")
    if hist.empty:
        raise ValueError("Could not fetch 10-year Treasury yield (^TNX) — no recent price data")
    return float(hist["Close"].iloc[-1]) / 100.0


def cost_of_equity(beta: float, risk_free_rate: float, erp: float = EQUITY_RISK_PREMIUM) -> float:
    """CAPM: Re = Rf + beta x ERP."""
    return risk_free_rate + beta * erp


def cost_of_debt(interest_expense: float, total_debt: float, effective_tax_rate: float) -> "float | None":
    """
    After-tax cost of debt: (interest_expense / total_debt) x (1 - tax_rate).
    Returns None for a debt-free (or near-zero-debt) company — cost of debt
    is undefined there, and its WACC weight will be ~0 regardless.
    """
    if total_debt <= 0:
        return None
    pretax_rd = interest_expense / total_debt
    return pretax_rd * (1 - effective_tax_rate)


def compute_wacc(
    market_cap: float,
    total_debt: float,
    cost_of_equity_value: float,
    cost_of_debt_value: "float | None",
) -> float:
    """
    Weighted average cost of capital: (E/V) x Re + (D/V) x Rd, where V = E + D.
    total_debt is floored at 0 (a negative XBRL-derived figure would only
    ever indicate a data problem, never a real negative debt balance).
    """
    total_debt = max(total_debt, 0.0)
    total_value = market_cap + total_debt
    if total_value <= 0:
        raise ValueError("market_cap + total_debt must be positive to compute WACC")

    equity_weight = market_cap / total_value
    debt_weight = total_debt / total_value
    rd = cost_of_debt_value if cost_of_debt_value is not None else 0.0

    return equity_weight * cost_of_equity_value + debt_weight * rd
