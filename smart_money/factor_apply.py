"""
FF3 skill-vs-beta decomposition for Module 3 — 13F Smart-Money Positioning & Skill Tracker.

Public interface
---------------
    score_from_returns(fund_cik, fund_name, valid_quarters, *, min_quarters)
        → FundSkillScore | None

    score_fund(fund, *, coverage_threshold)
        → FundSkillScore | None

    score_all_funds(funds, *, coverage_threshold)
        → list[FundSkillScore]

Power limitation
----------------
This module is a directional decomposition tool. With typical hedge fund quarterly
data, achieving p < 0.05 on alpha requires approximately 100 observations — far more
than available here. The t-stat and alpha are directional signals, not significance
tests. Use confidence_label and alpha_t_stat to communicate uncertainty to end users.

Alignment
---------
Fund quarterly returns from returns.py use adj_close prices at period_start (BOQ)
and period_end (EOQ): R_fund = p_eoq / p_boq - 1.  The corresponding FF3 window
covers dates strictly after period_start through period_end — French's daily factor
for date D represents the return from D-1's close to D's close, so period_start
itself is not a return day.  Daily factors are compounded geometrically, consistent
with the fund return computation.

Regression
----------
Quarterly excess fund return = α + β_mkt · MktExcess_q + β_smb · SMB_q + β_hml · HML_q + ε

where:
    excess_fund_q = reconstructed_return − RF_q        (both arithmetic, quarterly)
    MktExcess_q   = (1+mkt_d1)·(1+mkt_d2)·...·(1+mkt_dT) − 1  compounded over quarter
    SMB_q, HML_q  = same compounding applied to smb, hml daily series
    RF_q          = same compounding applied to rf daily series

Skill score
-----------
alpha_annualized = alpha_quarterly × 4

Return decomposition (historical attribution, sample-period averages):
    avg_excess_return_q ≈ alpha_quarterly
                        + β_mkt · mean(MktExcess_q)   [return_from_market]
                        + β_smb · mean(SMB_q)          [return_from_smb]
                        + β_hml · mean(HML_q)          [return_from_hml]

By OLS construction the residual mean is zero, so these four components sum exactly
to avg_excess_return_q.  This is historical attribution over the regression window,
not a forward-looking projection.

Reliability
-----------
MIN_QUARTERS_REG      = 8   minimum to compute any score (returns None below this)
MIN_QUARTERS_RELIABLE = 12  sets is_reliable=True; corresponds to 3 years of data
"""

from typing import TypedDict

import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from factor_engine.french_data import get_ff3_daily
from smart_money.models import Fund
from smart_money.returns import COVERAGE_THRESHOLD, FundQuarterReturn, reconstruct_all_quarters

MIN_QUARTERS_REG      = 8
MIN_QUARTERS_RELIABLE = 12


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

class FundSkillScore(TypedDict):
    fund_cik:             str
    fund_name:            str
    n_quarters:           int
    is_reliable:          bool       # n_quarters >= MIN_QUARTERS_RELIABLE
    confidence_label:     str        # plain-English for dashboard display
    quarters_used:        list[str]  # e.g. ["2023Q1", ..., "2026Q1"]
    # Regression outputs
    alpha_quarterly:      float      # OLS intercept (quarterly)
    alpha_annualized:     float      # alpha_quarterly * 4
    alpha_t_stat:         float
    alpha_p_value:        float
    beta_market:          float
    beta_smb:             float
    beta_hml:             float
    t_stat_market:        float
    t_stat_smb:           float
    t_stat_hml:           float
    r_squared:            float
    # Historical attribution (sample-period averages, quarterly)
    # These four sum exactly to avg_excess_return_q by OLS construction.
    avg_excess_return_q:  float      # mean(R_fund_q − RF_q) over regression window
    return_from_market:   float      # beta_market * mean(MktExcess_q)
    return_from_smb:      float      # beta_smb    * mean(SMB_q)
    return_from_hml:      float      # beta_hml    * mean(HML_q)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _aggregate_quarter_factors(
    ff3_daily: pd.DataFrame,
    period_start: str,
    period_end: str,
) -> "dict[str, float] | None":
    """
    Compound daily FF3 factors to a single quarterly observation.

    Window: dates strictly after period_start through period_end (inclusive).
    Returns None if no trading days fall in the window (e.g. data gap).
    """
    start  = pd.Timestamp(period_start)
    end    = pd.Timestamp(period_end)
    mask   = (ff3_daily.index > start) & (ff3_daily.index <= end)
    window = ff3_daily.loc[mask]
    if window.empty:
        return None
    return {
        "mkt_excess": float((1.0 + window["mkt_excess"]).prod() - 1.0),
        "smb":        float((1.0 + window["smb"]).prod()        - 1.0),
        "hml":        float((1.0 + window["hml"]).prod()        - 1.0),
        "rf":         float((1.0 + window["rf"]).prod()         - 1.0),
    }


def _confidence_label(n: int, alpha_t: float) -> str:
    """Plain-English confidence string for dashboard display."""
    if n < MIN_QUARTERS_REG:
        return "Insufficient data (< 8 quarters)"
    if n >= MIN_QUARTERS_RELIABLE and abs(alpha_t) > 1.5:
        return f"High ({n} quarters, |t| = {abs(alpha_t):.1f})"
    if n >= MIN_QUARTERS_RELIABLE:
        return f"Moderate ({n} quarters, |t| = {abs(alpha_t):.1f} < 1.5)"
    return f"Low ({n} quarters, below 12-quarter reliability threshold)"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def score_from_returns(
    fund_cik: str,
    fund_name: str,
    valid_quarters: "list[FundQuarterReturn]",
    *,
    min_quarters: int = MIN_QUARTERS_REG,
) -> "FundSkillScore | None":
    """
    Compute FF3 skill decomposition from a pre-built list of valid quarterly returns.

    This is the computational core — it loads FF3 data, aggregates factors to
    quarterly frequency, and runs OLS.  It is database-free and directly testable.
    score_fund() is the DB-backed wrapper that fetches and filters quarters first.

    Parameters
    ----------
    fund_cik : str
    fund_name : str
    valid_quarters : list[FundQuarterReturn]
        Pre-filtered to is_valid=True.  Sorted by period_end ascending.
    min_quarters : int
        Minimum quarters required to attempt a regression.  Callers should not
        pass a value lower than 4 (the OLS degree-of-freedom floor with 4 params).
        Override below MIN_QUARTERS_REG=8 only in testing contexts.

    Returns
    -------
    FundSkillScore | None
        None when len(valid_quarters) < min_quarters or when fewer than
        min_quarters quarters survive the FF3 alignment step.
    """
    if len(valid_quarters) < min_quarters:
        return None

    # Load FF3 daily data spanning the full sample in one call.
    earliest = min(q["period_start"] for q in valid_quarters)
    latest   = max(q["period_end"]   for q in valid_quarters)
    ff3_daily = get_ff3_daily(earliest, latest)

    # Build per-quarter rows: compound daily factors + fund excess return.
    rows = []
    for q in valid_quarters:
        factors_q = _aggregate_quarter_factors(ff3_daily, q["period_start"], q["period_end"])
        if factors_q is None:
            continue
        rows.append({
            "quarter":     q["quarter"],
            "fund_excess": q["reconstructed_return"] - factors_q["rf"],
            "mkt_excess":  factors_q["mkt_excess"],
            "smb":         factors_q["smb"],
            "hml":         factors_q["hml"],
        })

    if len(rows) < min_quarters:
        return None

    df = pd.DataFrame(rows).set_index("quarter")

    # OLS: fund_excess = alpha + beta_mkt*mkt_excess + beta_smb*smb + beta_hml*hml
    X      = add_constant(df[["mkt_excess", "smb", "hml"]])
    result = OLS(df["fund_excess"], X).fit()

    alpha_q  = float(result.params["const"])
    alpha_t  = float(result.tvalues["const"])
    alpha_p  = float(result.pvalues["const"])
    beta_mkt = float(result.params["mkt_excess"])
    beta_smb = float(result.params["smb"])
    beta_hml = float(result.params["hml"])

    # Historical attribution over the regression sample window.
    mean_mkt = float(df["mkt_excess"].mean())
    mean_smb = float(df["smb"].mean())
    mean_hml = float(df["hml"].mean())

    n      = len(rows)
    is_rel = n >= MIN_QUARTERS_RELIABLE
    conf   = _confidence_label(n, alpha_t)

    return FundSkillScore(
        fund_cik             = fund_cik,
        fund_name            = fund_name,
        n_quarters           = n,
        is_reliable          = is_rel,
        confidence_label     = conf,
        quarters_used        = list(df.index),
        alpha_quarterly      = round(alpha_q, 6),
        alpha_annualized     = round(alpha_q * 4, 6),
        alpha_t_stat         = round(alpha_t, 4),
        alpha_p_value        = round(alpha_p, 6),
        beta_market          = round(beta_mkt, 4),
        beta_smb             = round(beta_smb, 4),
        beta_hml             = round(beta_hml, 4),
        t_stat_market        = round(float(result.tvalues["mkt_excess"]), 4),
        t_stat_smb           = round(float(result.tvalues["smb"]), 4),
        t_stat_hml           = round(float(result.tvalues["hml"]), 4),
        r_squared            = round(float(result.rsquared), 4),
        avg_excess_return_q  = round(float(df["fund_excess"].mean()), 6),
        return_from_market   = round(beta_mkt * mean_mkt, 6),
        return_from_smb      = round(beta_smb * mean_smb, 6),
        return_from_hml      = round(beta_hml * mean_hml, 6),
    )


def score_fund(
    fund: Fund,
    *,
    coverage_threshold: float = COVERAGE_THRESHOLD,
) -> "FundSkillScore | None":
    """
    Compute FF3 skill decomposition for a fund using the DB.

    Fetches all quarterly returns from the DB via reconstruct_all_quarters,
    filters to is_valid=True, and delegates to score_from_returns.

    Returns None if fewer than MIN_QUARTERS_REG valid quarters are available.

    The database must be initialised (init_db() called) before invoking this.
    """
    all_quarters = reconstruct_all_quarters(fund, coverage_threshold=coverage_threshold)
    valid        = [q for q in all_quarters if q["is_valid"]]
    return score_from_returns(fund.cik or "", fund.name, valid)


def score_all_funds(
    funds: "list[Fund]",
    *,
    coverage_threshold: float = COVERAGE_THRESHOLD,
) -> "list[FundSkillScore]":
    """
    Score every fund in the list; silently skip funds with insufficient data.

    Returns results sorted by alpha_annualized descending (best skill first).
    """
    scores = []
    for fund in funds:
        s = score_fund(fund, coverage_threshold=coverage_threshold)
        if s is not None:
            scores.append(s)
    scores.sort(key=lambda s: s["alpha_annualized"], reverse=True)
    return scores
