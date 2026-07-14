"""
FF4 skill-vs-beta decomposition for Module 3 — 13F Smart-Money Positioning &
Skill Tracker.

Public interface
---------------
    score_from_returns(fund_cik, fund_name, valid_quarters, *, min_quarters)
        → FundSkillScore | None

    score_fund(fund, *, coverage_threshold)
        → FundSkillScore | None

    score_all_funds(funds, *, coverage_threshold)
        → list[FundSkillScore]

Why FF4, not FF7 — do not re-upgrade this module without re-validating
------------------------------------------------------------------------
This module ran a two-tier 7-factor model (mkt+smb+hml+mom+rmw+cma, plus a
GP-restricted secondary fit) for a period. That upgrade degraded the
platform's 3-month backtest IC from +0.061 (t=3.24, 99% significant, FF4) to
+0.007-0.008 (t≈1.3-1.5, not significant, FF7) — see app_pages/about.py's
"Signal Validation" section for the full investigation.

The root cause was empirically isolated, not assumed: three separate
diagnostic backtests, each adding exactly ONE of RMW, CMA, or GP to FF4 (see
smart_money/factor_apply_diagnostic.py and scripts/run_ff4_plus_one_diagnostic.py),
all independently degraded 3-month IC to the same ~0.0075-0.0076 — nearly
identical to each other and to the full FF7 result. If any one factor's
economic content or data quality were the cause, adding it alone should have
looked different from adding the others. It didn't. This rules out GP's
shorter history, RMW/CMA's construction, or any single factor's data quality
as the cause, and confirms a bias-variance/overfitting problem: with only
12-40 quarters of 13F history per fund, adding a 5th regressor of ANY kind
overfits the alpha estimate to noise rather than capturing genuine
stock-picking skill, regardless of which factor it is.

This is why Module 3/4 uses FF4 while Module 2 (portfolio-level risk
characterization, factor_engine/portfolio.py + dashboard/factor.py) keeps
FF7 — Module 2 analyzes one portfolio's long daily return history, not
dozens of funds' short quarterly panels, so it doesn't have the same
degrees-of-freedom constraint. More factors there add real risk-decomposition
value without the penalty seen here. Do not casually re-add RMW/CMA/GP to
THIS module without re-running the isolated single-factor backtest
diagnostic first — the evidence that FF4 is correct here is specific and
already established; reintroducing any of the three without fresh evidence
would be repeating a change already shown to hurt this module's validated
predictive power.

Power limitation
----------------
This module is a directional decomposition tool. With typical hedge fund quarterly
data, achieving p < 0.05 on alpha requires approximately 100 observations — far more
than available here. The t-stat and alpha are directional signals, not significance
tests. Use confidence_label and alpha_t_stat to communicate uncertainty to end users.

Alignment
---------
Fund quarterly returns from returns.py use adj_close prices at period_start (BOQ)
and period_end (EOQ): R_fund = p_eoq / p_boq - 1.  The corresponding factor window
covers dates strictly after period_start through period_end — French's daily factor
for date D represents the return from D-1's close to D's close, so period_start
itself is not a return day.  Daily factors are compounded geometrically, consistent
with the fund return computation.

Regression
----------
Quarterly excess fund return =
    α + β_mkt · MktExcess_q + β_smb · SMB_q + β_hml · HML_q + β_mom · MOM_q + ε

where:
    excess_fund_q = reconstructed_return − RF_q        (both arithmetic, quarterly)
    MktExcess_q   = (1+mkt_d1)·(1+mkt_d2)·...·(1+mkt_dT) − 1  compounded over quarter
    SMB_q, HML_q, MOM_q = same compounding applied to the respective daily series
    RF_q          = same compounding applied to rf daily series

MOM_q comes from Ken French's official daily series (get_ff4_daily()), not
an ETF proxy — see factor_engine/french_data.py. Momentum matters most for
growth/momentum-tilted funds: a manager who structurally holds recent
winners will show inflated alpha under a model missing momentum, because
that return component has nowhere to go but the residual. An explicit beta
gives a cleaner separation of stock-picking skill from factor beta — the
platform's stated thesis.

Skill score
-----------
alpha_annualized = alpha_quarterly × 4

Return decomposition (historical attribution, sample-period averages):
    avg_excess_return_q ≈ alpha_quarterly
                        + β_mkt · mean(MktExcess_q)   [return_from_market]
                        + β_smb · mean(SMB_q)          [return_from_smb]
                        + β_hml · mean(HML_q)          [return_from_hml]
                        + β_mom · mean(MOM_q)           [return_from_mom]

By OLS construction the residual mean is zero, so these five components sum
exactly to avg_excess_return_q — this is historical attribution over the
regression window, not a forward-looking projection.

Reliability
-----------
MIN_QUARTERS_REG      = 8   minimum to compute any score (returns None below this)
MIN_QUARTERS_RELIABLE = 12  sets is_reliable=True; corresponds to 3 years of data

RMW/CMA/GP fields
-----------------
beta_rmw/beta_cma/beta_gp (and their t-stats/return_from_*) are always None
under FF4 — they remain in FundSkillScore/FundSkillResult (nullable columns)
so the schema doesn't need to change and so smart_money/factor_apply_diagnostic.py's
single-factor isolation tests can still write to the same table. No
dashboard code currently reads these fields for Module 3/4 display (verified
— see the "why FF4" note above); if that ever changes, it must handle None.
"""

from typing import TypedDict

import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from factor_engine.french_data import get_ff4_daily
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
    # Regression outputs (mkt+smb+hml+mom, full fund history)
    alpha_quarterly:      float      # OLS intercept (quarterly)
    alpha_annualized:     float      # alpha_quarterly * 4
    alpha_t_stat:         float
    alpha_p_value:        float
    beta_market:          float
    beta_smb:             float
    beta_hml:             float
    beta_mom:             float
    t_stat_market:        float
    t_stat_smb:           float
    t_stat_hml:           float
    t_stat_mom:           float
    r_squared:            float
    # Historical attribution (sample-period averages, quarterly)
    # These five sum exactly to avg_excess_return_q by OLS construction.
    avg_excess_return_q:  float      # mean(R_fund_q − RF_q) over regression window
    return_from_market:   float      # beta_market * mean(MktExcess_q)
    return_from_smb:      float      # beta_smb    * mean(SMB_q)
    return_from_hml:      float      # beta_hml    * mean(HML_q)
    return_from_mom:      float      # beta_mom    * mean(MOM_q)
    # Always None under FF4 — see module docstring "RMW/CMA/GP fields".
    beta_rmw:              "float | None"
    beta_cma:              "float | None"
    beta_gp:                "float | None"
    t_stat_rmw:             "float | None"
    t_stat_cma:             "float | None"
    t_stat_gp:               "float | None"
    return_from_rmw:        "float | None"
    return_from_cma:        "float | None"
    return_from_gp:         "float | None"
    n_quarters_gp:           int        # always 0 under FF4


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _aggregate_quarter_factors(
    ff4_daily: pd.DataFrame,
    period_start: str,
    period_end: str,
) -> "dict[str, float] | None":
    """
    Compound daily factors to a single quarterly observation.

    Window: dates strictly after period_start through period_end (inclusive).
    Returns None if no trading days fall in the window (e.g. data gap).
    mkt_excess/smb/hml/mom/rf are always compounded (Ken French has full history).
    """
    start  = pd.Timestamp(period_start)
    end    = pd.Timestamp(period_end)
    mask   = (ff4_daily.index > start) & (ff4_daily.index <= end)
    window = ff4_daily.loc[mask]
    if window.empty:
        return None

    return {
        "mkt_excess": float((1.0 + window["mkt_excess"]).prod() - 1.0),
        "smb":        float((1.0 + window["smb"]).prod()        - 1.0),
        "hml":        float((1.0 + window["hml"]).prod()        - 1.0),
        "mom":        float((1.0 + window["mom"]).prod()        - 1.0),
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
    Compute the FF4 skill decomposition from a pre-built list of valid
    quarterly returns.

    This is the computational core — it loads factor data, aggregates to
    quarterly frequency, and runs OLS.  It is database-free and directly testable.
    score_fund() is the DB-backed wrapper that fetches and filters quarters first.

    Parameters
    ----------
    fund_cik : str
    fund_name : str
    valid_quarters : list[FundQuarterReturn]
        Pre-filtered to is_valid=True.  Sorted by period_end ascending.
    min_quarters : int
        Minimum quarters required to attempt the regression. Callers should
        not pass a value lower than 6 (the OLS degree-of-freedom floor with
        4 params + const). Override below MIN_QUARTERS_REG=8 only in testing
        contexts.

    Returns
    -------
    FundSkillScore | None
        None when len(valid_quarters) < min_quarters or when fewer than
        min_quarters quarters survive the factor alignment step.
    """
    if len(valid_quarters) < min_quarters:
        return None

    earliest = min(q["period_start"] for q in valid_quarters)
    latest   = max(q["period_end"]   for q in valid_quarters)
    ff4_daily = get_ff4_daily(earliest, latest)

    rows = []
    for q in valid_quarters:
        factors_q = _aggregate_quarter_factors(ff4_daily, q["period_start"], q["period_end"])
        if factors_q is None:
            continue
        rows.append({
            "quarter":     q["quarter"],
            "fund_excess": q["reconstructed_return"] - factors_q["rf"],
            "mkt_excess":  factors_q["mkt_excess"],
            "smb":         factors_q["smb"],
            "hml":         factors_q["hml"],
            "mom":         factors_q["mom"],
        })

    if len(rows) < min_quarters:
        return None

    df = pd.DataFrame(rows).set_index("quarter")

    X      = add_constant(df[["mkt_excess", "smb", "hml", "mom"]])
    result = OLS(df["fund_excess"], X).fit()

    alpha_q  = float(result.params["const"])
    alpha_t  = float(result.tvalues["const"])
    alpha_p  = float(result.pvalues["const"])
    beta_mkt = float(result.params["mkt_excess"])
    beta_smb = float(result.params["smb"])
    beta_hml = float(result.params["hml"])
    beta_mom = float(result.params["mom"])

    mean_mkt = float(df["mkt_excess"].mean())
    mean_smb = float(df["smb"].mean())
    mean_hml = float(df["hml"].mean())
    mean_mom = float(df["mom"].mean())

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
        beta_mom             = round(beta_mom, 4),
        t_stat_market        = round(float(result.tvalues["mkt_excess"]), 4),
        t_stat_smb           = round(float(result.tvalues["smb"]), 4),
        t_stat_hml           = round(float(result.tvalues["hml"]), 4),
        t_stat_mom           = round(float(result.tvalues["mom"]), 4),
        r_squared            = round(float(result.rsquared), 4),
        avg_excess_return_q  = round(float(df["fund_excess"].mean()), 6),
        return_from_market   = round(beta_mkt * mean_mkt, 6),
        return_from_smb      = round(beta_smb * mean_smb, 6),
        return_from_hml      = round(beta_hml * mean_hml, 6),
        return_from_mom      = round(beta_mom * mean_mom, 6),
        beta_rmw             = None,
        beta_cma             = None,
        beta_gp              = None,
        t_stat_rmw           = None,
        t_stat_cma           = None,
        t_stat_gp            = None,
        return_from_rmw      = None,
        return_from_cma      = None,
        return_from_gp       = None,
        n_quarters_gp        = 0,
    )


def score_fund(
    fund: Fund,
    *,
    coverage_threshold: float = COVERAGE_THRESHOLD,
) -> "FundSkillScore | None":
    """
    Compute the FF4 skill decomposition for a fund using the DB.

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
