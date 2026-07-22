"""
Portfolio-level volatility / Sharpe / Sortino / max-drawdown metrics.

All four metrics are computed from data `analyze_portfolio()` already
produces — no new fetch. The portfolio's combined daily log-return series
(factor_engine/portfolio.py::build_combined_return_series()) supplies vol and
drawdown; the "rf" column already present in the get_ff7_daily() factor panel
(Ken French's official daily risk-free series) supplies the risk-free rate for
Sharpe/Sortino.

Risk-free rate choice: this is deliberately the same short-duration daily rf
already used throughout the factor regressions, NOT the 10-year Treasury rate
built for the DCF engine (dcf/wacc.py::fetch_risk_free_rate()). Sharpe/Sortino
here annualize a daily return series, so the short-duration rate is
duration-matched — reusing the 10yr rate would repeat, in the opposite
direction, the exact duration mismatch dcf/wacc.py's own docstring warns
against for a decade-long DCF cash flow stream discounted at a 3-month rate.

Sortino's downside deviation is computed on daily *excess* returns (return -
rf) below zero, not raw returns below zero — this keeps Sharpe and Sortino
directly comparable: both have the same numerator (annualized excess return),
differing only in whether the denominator is full or downside-only dispersion.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Floating-point floor below which volatility/downside-deviation is treated as
# zero. A nominally-constant return series' .std() is not exactly 0.0 due to
# floating-point rounding (e.g. ~1e-16 on identical float64 inputs) — dividing
# by that residual noise instead of guarding against it produces a nonsensical
# huge ratio rather than the intended "undefined, no volatility" None.
_ZERO_VOL_EPSILON = 1e-10


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Standard deviation of daily log returns, annualized by sqrt(252)."""
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(daily_returns: pd.Series) -> dict:
    """
    Largest peak-to-trough decline over the return series.

    Builds a cumulative value path from daily log returns (starting at 1.0),
    tracks the running peak, and finds the deepest (value / peak - 1). Returns
    the trough date and the peak date that preceded it, not just the magnitude,
    since "when did this happen and how long was the drawdown" matters as much
    as the number itself for a retail reader.

    peak_date is the LAST date at the peak value on or before the trough, not
    the first — pandas idxmax() returns the first occurrence on ties, which
    would misreport the peak as the start of a flat run at the high-water mark
    rather than the day the decline actually began.
    """
    cum_value = np.exp(daily_returns.cumsum())
    running_peak = cum_value.cummax()
    drawdown = cum_value / running_peak - 1.0

    trough_date = drawdown.idxmin()
    max_dd = float(drawdown.loc[trough_date])
    peak_value = running_peak.loc[trough_date]
    pre_trough = cum_value.loc[:trough_date]
    peak_date = pre_trough[pre_trough == peak_value].index[-1]

    return {
        "max_drawdown": max_dd,
        "peak_date": str(peak_date.date()),
        "trough_date": str(trough_date.date()),
    }


def _annualized_return(daily_returns: pd.Series) -> float:
    """Geometric annualized return from daily log returns."""
    n = len(daily_returns)
    if n == 0:
        return 0.0
    total_log_return = daily_returns.sum()
    return float(np.expm1(total_log_return * (TRADING_DAYS_PER_YEAR / n)))


def sharpe_ratio(daily_returns: pd.Series, rf_daily: pd.Series) -> dict:
    """
    Sharpe ratio: (annualized return - annualized rf) / annualized volatility.

    rf_daily is aligned to daily_returns' index (inner join) before
    annualizing, so both figures cover exactly the same trading days.
    """
    aligned = pd.DataFrame({"r": daily_returns, "rf": rf_daily}).dropna()
    ann_return = _annualized_return(aligned["r"])
    ann_rf = float(aligned["rf"].mean() * TRADING_DAYS_PER_YEAR)
    ann_vol = annualized_volatility(aligned["r"])

    sharpe = (ann_return - ann_rf) / ann_vol if ann_vol > _ZERO_VOL_EPSILON else None

    return {
        "sharpe_ratio": sharpe,
        "annualized_return": ann_return,
        "annualized_rf": ann_rf,
        "annualized_vol": ann_vol,
        "n_obs": int(len(aligned)),
    }


def sortino_ratio(daily_returns: pd.Series, rf_daily: pd.Series) -> dict:
    """
    Sortino ratio: same numerator as Sharpe (annualized excess return over rf),
    denominator is annualized downside deviation of daily excess returns below
    zero — see module docstring for why excess (not raw) returns are used.

    Returns downside_deviation = 0.0 (and sortino_ratio = None) in the
    edge case of no negative-excess-return days in the window, rather than
    dividing by zero.
    """
    aligned = pd.DataFrame({"r": daily_returns, "rf": rf_daily}).dropna()
    daily_excess = aligned["r"] - aligned["rf"]
    ann_return = _annualized_return(aligned["r"])
    ann_rf = float(aligned["rf"].mean() * TRADING_DAYS_PER_YEAR)

    downside = daily_excess[daily_excess < 0]
    downside_deviation = float(downside.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(downside) > 0 else 0.0

    sortino = (ann_return - ann_rf) / downside_deviation if downside_deviation > _ZERO_VOL_EPSILON else None

    return {
        "sortino_ratio": sortino,
        "downside_deviation": downside_deviation,
        "n_downside_days": int(len(downside)),
        "n_obs": int(len(aligned)),
    }


def compute_risk_metrics(combined_rets: pd.Series, factors: pd.DataFrame) -> dict:
    """
    Top-level entry point: bundles volatility, max drawdown, Sharpe, and
    Sortino for the portfolio's combined daily return series.

    Parameters
    ----------
    combined_rets : the portfolio's weighted daily log-return series
        (factor_engine/portfolio.py::build_combined_return_series() output).
    factors : the FF7 daily factor panel (get_ff7_daily() output) — only the
        "rf" column is used here.
    """
    rf_daily = factors["rf"]
    dd = max_drawdown(combined_rets)
    sharpe = sharpe_ratio(combined_rets, rf_daily)
    sortino = sortino_ratio(combined_rets, rf_daily)

    return {
        "annualized_volatility": annualized_volatility(combined_rets),
        "max_drawdown": dd["max_drawdown"],
        "max_drawdown_peak_date": dd["peak_date"],
        "max_drawdown_trough_date": dd["trough_date"],
        "sharpe_ratio": sharpe["sharpe_ratio"],
        "sortino_ratio": sortino["sortino_ratio"],
        "annualized_return": sharpe["annualized_return"],
        "annualized_rf": sharpe["annualized_rf"],
        "downside_deviation": sortino["downside_deviation"],
        "n_obs": sharpe["n_obs"],
    }
