"""
7-factor skill-vs-beta decomposition for Module 3 — 13F Smart-Money
Positioning & Skill Tracker.

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
and period_end (EOQ): R_fund = p_eoq / p_boq - 1.  The corresponding factor window
covers dates strictly after period_start through period_end — French's daily factor
for date D represents the return from D-1's close to D's close, so period_start
itself is not a return day.  Daily factors are compounded geometrically, consistent
with the fund return computation.

Two-tier regression design — why this module doesn't run one joint 7-factor fit
----------------------------------------------------------------------------------
mkt, smb, hml, mom, rmw, cma all come from Ken French's daily series with full
history; a fund's regression sample is limited only by how many valid 13F
quarters it has (often back to 2013, per the EDGAR XML cutoff). GP (Gross
Profitability, factor_engine/factors/gp.py) is this platform's proprietary
factor built from yfinance fundamentals, which only supports ~2021-present
coverage — a hard limitation of the free data source, not an engineering gap.
A single joint regression that includes gp as a regressor can only use rows
where gp is non-null, which would silently truncate every fund's pre-2021
quarters out of the ENTIRE regression — including alpha and the six other
betas — even though those six have no such limitation. That would be a real
methodology regression versus the prior FF4/FF6 spec's sample depth.

Instead:
  PRIMARY regression (mkt+smb+hml+mom+rmw+cma, 6 factors): runs over the
    fund's full available valid-quarter history, exactly as FF4/FF6 did
    before — unchanged sample depth. Produces alpha, alpha_t_stat, and all
    six non-GP betas/t-stats. avg_excess_return_q and the five return_from_*
    components below sum to it exactly, per the OLS residual-mean-zero
    property — this decomposition is unaffected by GP's shorter history.
  SECONDARY regression (mkt+smb+hml+mom+rmw+cma+gp, 7 factors): runs ONLY
    over the subset of valid quarters that fall within GP's coverage window.
    Only beta_gp/t_stat_gp/return_from_gp are taken from this fit — its
    alpha and other six betas are discarded (the primary fit's full-sample
    versions are better estimates and are what's reported). If fewer than
    min_quarters quarters survive this filter, beta_gp/t_stat_gp/
    return_from_gp are None and n_quarters_gp records how many were
    available. GP loadings are therefore inherently less statistically
    reliable than the other six — always surface n_quarters_gp alongside
    beta_gp, and label GP "Gross Profitability (2021-present)" in any
    display.

Regression (primary tier)
--------------------------
Quarterly excess fund return =
    α + β_mkt · MktExcess_q + β_smb · SMB_q + β_hml · HML_q
      + β_rmw · RMW_q + β_cma · CMA_q + β_mom · MOM_q + ε

where:
    excess_fund_q = reconstructed_return − RF_q        (both arithmetic, quarterly)
    MktExcess_q   = (1+mkt_d1)·(1+mkt_d2)·...·(1+mkt_dT) − 1  compounded over quarter
    SMB_q, HML_q, RMW_q, CMA_q, MOM_q = same compounding applied to the respective
        daily series
    RF_q          = same compounding applied to rf daily series

MOM_q, RMW_q, CMA_q all come from Ken French's official daily series
(get_ff6_daily()), not ETF proxies — see factor_engine/french_data.py. Adding
momentum (and now RMW/CMA) is important for growth/momentum-tilted funds and
funds with strong profitability/investment tilts: a manager who structurally
holds recent winners (or high-margin compounders, or disciplined capital
allocators) will show inflated alpha under a model missing the relevant
factor because that return component has nowhere to go but the residual.
Explicit betas give a cleaner separation of stock-picking skill from factor
beta — the platform's stated thesis.

Skill score
-----------
alpha_annualized = alpha_quarterly × 4

Return decomposition (historical attribution, sample-period averages, primary tier):
    avg_excess_return_q ≈ alpha_quarterly
                        + β_mkt · mean(MktExcess_q)   [return_from_market]
                        + β_smb · mean(SMB_q)          [return_from_smb]
                        + β_hml · mean(HML_q)          [return_from_hml]
                        + β_rmw · mean(RMW_q)          [return_from_rmw]
                        + β_cma · mean(CMA_q)          [return_from_cma]
                        + β_mom · mean(MOM_q)           [return_from_mom]

By OLS construction the residual mean is zero, so these seven components sum exactly
to avg_excess_return_q. This is historical attribution over the primary regression
window, not a forward-looking projection. return_from_gp (secondary tier) is NOT
part of this exact-sum decomposition — it's a supplementary, shorter-window estimate.

Reliability
-----------
MIN_QUARTERS_REG      = 8   minimum to compute any score (returns None below this)
MIN_QUARTERS_RELIABLE = 12  sets is_reliable=True; corresponds to 3 years of data

The same MIN_QUARTERS_REG floor gates the secondary GP-only regression —
funds without at least 8 valid quarters inside GP's ~2021-present coverage
window get beta_gp=None rather than a fit on too few observations.
"""

from typing import TypedDict

import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from factor_engine.french_data import get_ff7_daily
from smart_money.models import Fund
from smart_money.returns import COVERAGE_THRESHOLD, FundQuarterReturn, reconstruct_all_quarters

MIN_QUARTERS_REG      = 8
MIN_QUARTERS_RELIABLE = 12
# The secondary GP-only fit has 7 regressors (vs. the primary's 6), so its
# floor is one quarter higher than MIN_QUARTERS_REG — otherwise n_gp could
# equal the parameter count (const + 7 betas = 8) exactly, leaving zero
# residual degrees of freedom and producing nan/inf t-stats rather than a
# clean "insufficient data" None.
MIN_QUARTERS_GP        = MIN_QUARTERS_REG + 1


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
    # Regression outputs (primary tier: mkt+smb+hml+mom+rmw+cma, full fund history)
    alpha_quarterly:      float      # OLS intercept (quarterly)
    alpha_annualized:     float      # alpha_quarterly * 4
    alpha_t_stat:         float
    alpha_p_value:        float
    beta_market:          float
    beta_smb:             float
    beta_hml:             float
    beta_mom:             float
    beta_rmw:             float
    beta_cma:             float
    t_stat_market:        float
    t_stat_smb:           float
    t_stat_hml:           float
    t_stat_mom:           float
    t_stat_rmw:            float
    t_stat_cma:            float
    r_squared:            float
    # Historical attribution (sample-period averages, quarterly, primary tier)
    # These seven sum exactly to avg_excess_return_q by OLS construction.
    avg_excess_return_q:  float      # mean(R_fund_q − RF_q) over regression window
    return_from_market:   float      # beta_market * mean(MktExcess_q)
    return_from_smb:      float      # beta_smb    * mean(SMB_q)
    return_from_hml:      float      # beta_hml    * mean(HML_q)
    return_from_mom:      float      # beta_mom    * mean(MOM_q)
    return_from_rmw:      float      # beta_rmw    * mean(RMW_q)
    return_from_cma:      float      # beta_cma    * mean(CMA_q)
    # Secondary tier: GP-only fit restricted to GP's coverage window — NOT
    # part of the exact-sum decomposition above. None when n_quarters_gp < MIN_QUARTERS_REG.
    beta_gp:              "float | None"
    t_stat_gp:             "float | None"
    return_from_gp:       "float | None"
    n_quarters_gp:         int        # quarters feeding the secondary GP fit (0 if none)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _aggregate_quarter_factors(
    ff7_daily: pd.DataFrame,
    period_start: str,
    period_end: str,
) -> "dict[str, float | None] | None":
    """
    Compound daily factors to a single quarterly observation.

    Window: dates strictly after period_start through period_end (inclusive).
    Returns None if no trading days fall in the window (e.g. data gap).

    mkt_excess/smb/hml/rmw/cma/mom/rf are always compounded (Ken French has
    full history). gp is compounded ONLY if every day in the window has a
    non-null gp value — i.e. the whole quarter falls within GP's coverage
    window; a quarter straddling GP's coverage boundary is treated as
    entirely uncovered (gp=None) rather than partially compounded, since
    prod(skipna=True) would silently understate the true compounded return
    by dropping the missing days instead of flagging them.
    """
    start  = pd.Timestamp(period_start)
    end    = pd.Timestamp(period_end)
    mask   = (ff7_daily.index > start) & (ff7_daily.index <= end)
    window = ff7_daily.loc[mask]
    if window.empty:
        return None

    gp_covered = "gp" in window.columns and not window["gp"].isna().any()

    return {
        "mkt_excess": float((1.0 + window["mkt_excess"]).prod() - 1.0),
        "smb":        float((1.0 + window["smb"]).prod()        - 1.0),
        "hml":        float((1.0 + window["hml"]).prod()        - 1.0),
        "rmw":        float((1.0 + window["rmw"]).prod()        - 1.0),
        "cma":        float((1.0 + window["cma"]).prod()        - 1.0),
        "mom":        float((1.0 + window["mom"]).prod()        - 1.0),
        "rf":         float((1.0 + window["rf"]).prod()         - 1.0),
        "gp":         float((1.0 + window["gp"]).prod() - 1.0) if gp_covered else None,
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
    Compute the two-tier 7-factor skill decomposition from a pre-built list of
    valid quarterly returns (see module docstring for why it's two-tier).

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
        Minimum quarters required to attempt the primary regression. Callers
        should not pass a value lower than 7 (the OLS degree-of-freedom floor
        with 6 params + const). Override below MIN_QUARTERS_REG=8 only in
        testing contexts. The secondary GP-only fit uses its own higher floor
        (MIN_QUARTERS_GP) regardless of what's passed here.

    Returns
    -------
    FundSkillScore | None
        None when len(valid_quarters) < min_quarters or when fewer than
        min_quarters quarters survive the factor alignment step.
    """
    if len(valid_quarters) < min_quarters:
        return None

    # Load the full 7-factor panel spanning the sample in one call. mkt/smb/
    # hml/rmw/cma/mom have full history; gp is NaN before ~2021 (see
    # factor_engine/french_data.py::get_ff7_daily()).
    earliest = min(q["period_start"] for q in valid_quarters)
    latest   = max(q["period_end"]   for q in valid_quarters)
    ff7_daily = get_ff7_daily(earliest, latest)

    # Build per-quarter rows: compound daily factors + fund excess return.
    rows = []
    for q in valid_quarters:
        factors_q = _aggregate_quarter_factors(ff7_daily, q["period_start"], q["period_end"])
        if factors_q is None:
            continue
        rows.append({
            "quarter":     q["quarter"],
            "fund_excess": q["reconstructed_return"] - factors_q["rf"],
            "mkt_excess":  factors_q["mkt_excess"],
            "smb":         factors_q["smb"],
            "hml":         factors_q["hml"],
            "rmw":         factors_q["rmw"],
            "cma":         factors_q["cma"],
            "mom":         factors_q["mom"],
            "gp":          factors_q["gp"],   # None outside GP's coverage window
        })

    if len(rows) < min_quarters:
        return None

    df = pd.DataFrame(rows).set_index("quarter")

    # PRIMARY tier: 6-factor fit over the fund's FULL available history —
    # unaffected by gp's shorter coverage since it isn't a regressor here.
    X      = add_constant(df[["mkt_excess", "smb", "hml", "rmw", "cma", "mom"]])
    result = OLS(df["fund_excess"], X).fit()

    alpha_q  = float(result.params["const"])
    alpha_t  = float(result.tvalues["const"])
    alpha_p  = float(result.pvalues["const"])
    beta_mkt = float(result.params["mkt_excess"])
    beta_smb = float(result.params["smb"])
    beta_hml = float(result.params["hml"])
    beta_rmw = float(result.params["rmw"])
    beta_cma = float(result.params["cma"])
    beta_mom = float(result.params["mom"])

    # Historical attribution over the primary regression sample window.
    mean_mkt = float(df["mkt_excess"].mean())
    mean_smb = float(df["smb"].mean())
    mean_hml = float(df["hml"].mean())
    mean_rmw = float(df["rmw"].mean())
    mean_cma = float(df["cma"].mean())
    mean_mom = float(df["mom"].mean())

    n      = len(rows)
    is_rel = n >= MIN_QUARTERS_RELIABLE
    conf   = _confidence_label(n, alpha_t)

    # SECONDARY tier: 7-factor fit restricted to GP-covered quarters. Only
    # beta_gp/t_stat_gp/return_from_gp are taken from this fit — its alpha
    # and other six betas are discarded in favor of the primary fit's
    # full-sample estimates.
    gp_df  = df.dropna(subset=["gp"])
    n_gp   = len(gp_df)
    beta_gp: "float | None" = None
    t_stat_gp: "float | None" = None
    return_from_gp: "float | None" = None
    if n_gp >= MIN_QUARTERS_GP:
        Xg = add_constant(gp_df[["mkt_excess", "smb", "hml", "rmw", "cma", "mom", "gp"]])
        gp_result = OLS(gp_df["fund_excess"], Xg).fit()
        beta_gp_val = float(gp_result.params["gp"])
        beta_gp = round(beta_gp_val, 4)
        t_stat_gp = round(float(gp_result.tvalues["gp"]), 4)
        return_from_gp = round(beta_gp_val * float(gp_df["gp"].mean()), 6)

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
        beta_rmw             = round(beta_rmw, 4),
        beta_cma             = round(beta_cma, 4),
        beta_mom             = round(beta_mom, 4),
        t_stat_market        = round(float(result.tvalues["mkt_excess"]), 4),
        t_stat_smb           = round(float(result.tvalues["smb"]), 4),
        t_stat_hml           = round(float(result.tvalues["hml"]), 4),
        t_stat_rmw           = round(float(result.tvalues["rmw"]), 4),
        t_stat_cma           = round(float(result.tvalues["cma"]), 4),
        t_stat_mom           = round(float(result.tvalues["mom"]), 4),
        r_squared            = round(float(result.rsquared), 4),
        avg_excess_return_q  = round(float(df["fund_excess"].mean()), 6),
        return_from_market   = round(beta_mkt * mean_mkt, 6),
        return_from_smb      = round(beta_smb * mean_smb, 6),
        return_from_hml      = round(beta_hml * mean_hml, 6),
        return_from_rmw      = round(beta_rmw * mean_rmw, 6),
        return_from_cma      = round(beta_cma * mean_cma, 6),
        return_from_mom      = round(beta_mom * mean_mom, 6),
        beta_gp              = beta_gp,
        t_stat_gp            = t_stat_gp,
        return_from_gp       = return_from_gp,
        n_quarters_gp        = n_gp,
    )


def score_fund(
    fund: Fund,
    *,
    coverage_threshold: float = COVERAGE_THRESHOLD,
) -> "FundSkillScore | None":
    """
    Compute the two-tier 7-factor skill decomposition for a fund using the DB.

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
