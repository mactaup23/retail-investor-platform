"""
Diagnostic-only skill scoring: FF4 + exactly one of {RMW, CMA, GP}.

This exists to answer one question: of the three factors FF7 added on top
of FF4 (RMW, CMA, GP), which one is actually responsible for the backtest
IC degradation seen going from FF4 (+0.061 IC) to FF7 (+0.007, later +0.008
on the 60-fund universe)? It is NOT a replacement for factor_apply.py's
production score_from_returns() — that function's two-tier design exists to
reconcile RMW/CMA/GP's differing coverage windows *simultaneously* in one
7-factor model. Testing one added factor at a time sidesteps that problem
entirely: there's only ever one factor whose coverage might be shorter than
FF4's, so a single flat OLS restricted to that factor's covered quarters is
sufffient — no two-tier split needed.

Reuses factor_apply.py's labeling helper (_confidence_label,
MIN_QUARTERS_REG/RELIABLE) and factor_engine.french_data.get_ff7_daily()
(already has all 7 factors in one panel, including the 2013-2026 GP series).
Factor-quarter aggregation is a LOCAL copy (_aggregate_quarter_factors_ff7
below), not imported from factor_apply.py — since that module reverted to
FF4-only (see its module docstring), its aggregation helper no longer
carries rmw/cma/gp columns. This module needs the full 7-factor panel
regardless of what the production model currently uses, so it stays
self-contained rather than depending on production internals that may
change again.

Persists to the SAME FundSkillResult table the production pipeline writes
(via .on_conflict_replace(), same as pipeline.py's phase 5) so
ConvergenceScore/FinalSignal rebuild and backtest scripts work completely
unchanged. beta/t_stat/return_from for the two UNTESTED factors (of
rmw/cma/gp) are written as genuine NULL (models.py already allows this),
not a fake placeholder. This is destructive to the live FundSkillResult
state — see scripts/run_ff4_plus_one_diagnostic.py for the backup/restore
wrapper that makes running three variants sequentially safe.
"""

import datetime
import json

import pandas as pd
from statsmodels.api import OLS, add_constant

from factor_engine.french_data import get_ff7_daily
from smart_money.factor_apply import (
    MIN_QUARTERS_REG,
    MIN_QUARTERS_RELIABLE,
    FundSkillScore,
    _confidence_label,
)
from smart_money.models import Fund, FundSkillResult
from smart_money.returns import reconstruct_all_quarters

_EXTRA_FACTORS = ("rmw", "cma", "gp")


def _aggregate_quarter_factors_ff7(
    ff7_daily: pd.DataFrame,
    period_start: str,
    period_end: str,
) -> "dict[str, float | None] | None":
    """
    Compound daily factors to a single quarterly observation, full 7-factor
    panel. Local copy of the pre-FF4-revert factor_apply.py helper — see
    module docstring for why this doesn't import from factor_apply.py.

    gp is compounded ONLY if every day in the window has a non-null gp value
    (i.e. the whole quarter falls within GP's coverage window); a quarter
    straddling GP's coverage boundary is treated as entirely uncovered
    (gp=None) rather than partially compounded.
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


def score_from_returns_single_factor(
    fund_cik: str,
    fund_name: str,
    valid_quarters: "list",
    extra_factor: str,
    *,
    min_quarters: int = MIN_QUARTERS_REG,
) -> "FundSkillScore | None":
    """
    FF4 (mkt+smb+hml+mom) + exactly one of {rmw, cma, gp}, one flat OLS.

    Restricted to quarters where extra_factor has data — a no-op filter for
    rmw/cma (full Ken French history), the operative constraint for gp
    (coverage-bounded). Returns None below min_quarters valid quarters, same
    contract as the production score_from_returns().
    """
    if extra_factor not in _EXTRA_FACTORS:
        raise ValueError(f"extra_factor must be one of {_EXTRA_FACTORS}, got {extra_factor!r}")
    if len(valid_quarters) < min_quarters:
        return None

    earliest = min(q["period_start"] for q in valid_quarters)
    latest   = max(q["period_end"]   for q in valid_quarters)
    ff7_daily = get_ff7_daily(earliest, latest)

    rows = []
    for q in valid_quarters:
        factors_q = _aggregate_quarter_factors_ff7(ff7_daily, q["period_start"], q["period_end"])
        if factors_q is None:
            continue
        if factors_q[extra_factor] is None:
            continue   # outside this factor's coverage window (relevant for gp only)
        rows.append({
            "quarter":     q["quarter"],
            "fund_excess": q["reconstructed_return"] - factors_q["rf"],
            "mkt_excess":  factors_q["mkt_excess"],
            "smb":         factors_q["smb"],
            "hml":         factors_q["hml"],
            "mom":         factors_q["mom"],
            "extra":       factors_q[extra_factor],
        })

    if len(rows) < min_quarters:
        return None

    df = pd.DataFrame(rows).set_index("quarter")
    X  = add_constant(df[["mkt_excess", "smb", "hml", "mom", "extra"]])
    result = OLS(df["fund_excess"], X).fit()

    alpha_q  = float(result.params["const"])
    alpha_t  = float(result.tvalues["const"])
    n        = len(rows)
    is_rel   = n >= MIN_QUARTERS_RELIABLE
    conf     = _confidence_label(n, alpha_t)

    beta_extra = round(float(result.params["extra"]), 4)
    t_extra    = round(float(result.tvalues["extra"]), 4)
    return_extra = round(beta_extra * float(df["extra"].mean()), 6)

    score: FundSkillScore = FundSkillScore(
        fund_cik             = fund_cik,
        fund_name            = fund_name,
        n_quarters           = n,
        is_reliable          = is_rel,
        confidence_label     = conf,
        quarters_used        = list(df.index),
        alpha_quarterly      = round(alpha_q, 6),
        alpha_annualized     = round(alpha_q * 4, 6),
        alpha_t_stat         = round(alpha_t, 4),
        alpha_p_value        = round(float(result.pvalues["const"]), 6),
        beta_market          = round(float(result.params["mkt_excess"]), 4),
        beta_smb             = round(float(result.params["smb"]), 4),
        beta_hml             = round(float(result.params["hml"]), 4),
        beta_mom             = round(float(result.params["mom"]), 4),
        beta_rmw             = beta_extra if extra_factor == "rmw" else None,
        beta_cma             = beta_extra if extra_factor == "cma" else None,
        t_stat_market        = round(float(result.tvalues["mkt_excess"]), 4),
        t_stat_smb           = round(float(result.tvalues["smb"]), 4),
        t_stat_hml           = round(float(result.tvalues["hml"]), 4),
        t_stat_mom           = round(float(result.tvalues["mom"]), 4),
        t_stat_rmw           = t_extra if extra_factor == "rmw" else None,
        t_stat_cma           = t_extra if extra_factor == "cma" else None,
        r_squared            = round(float(result.rsquared), 4),
        avg_excess_return_q  = round(float(df["fund_excess"].mean()), 6),
        return_from_market   = round(float(result.params["mkt_excess"]) * float(df["mkt_excess"].mean()), 6),
        return_from_smb      = round(float(result.params["smb"]) * float(df["smb"].mean()), 6),
        return_from_hml      = round(float(result.params["hml"]) * float(df["hml"].mean()), 6),
        return_from_mom      = round(float(result.params["mom"]) * float(df["mom"].mean()), 6),
        return_from_rmw      = return_extra if extra_factor == "rmw" else None,
        return_from_cma      = return_extra if extra_factor == "cma" else None,
        beta_gp              = beta_extra if extra_factor == "gp" else None,
        t_stat_gp            = t_extra if extra_factor == "gp" else None,
        return_from_gp       = return_extra if extra_factor == "gp" else None,
        n_quarters_gp        = n if extra_factor == "gp" else 0,
    )
    return score


def score_fund_single_factor(fund: Fund, extra_factor: str) -> "FundSkillScore | None":
    all_quarters = reconstruct_all_quarters(fund)
    valid = [q for q in all_quarters if q["is_valid"]]
    return score_from_returns_single_factor(fund.cik or "", fund.name, valid, extra_factor)


def run_diagnostic_variant(extra_factor: str, active_funds: "list[Fund]") -> tuple[int, int]:
    """
    Score every fund in active_funds with FF4+extra_factor and persist to
    FundSkillResult (same table, same on_conflict_replace semantics as the
    production pipeline). Returns (n_scored, n_insufficient).
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    n_scored = 0
    n_insufficient = 0

    for fund in active_funds:
        try:
            result = score_fund_single_factor(fund, extra_factor)
        except Exception as e:
            print(f"  [diagnostic:{extra_factor}] ERROR  {fund.name}  {e}")
            n_insufficient += 1
            continue

        if result is None:
            n_insufficient += 1
            continue

        (FundSkillResult
         .insert(
             fund=fund,
             scored_at=now,
             n_quarters=result["n_quarters"],
             is_reliable=result["is_reliable"],
             confidence_label=result["confidence_label"],
             quarters_used=json.dumps(result["quarters_used"]),
             alpha_quarterly=result["alpha_quarterly"],
             alpha_annualized=result["alpha_annualized"],
             alpha_t_stat=result["alpha_t_stat"],
             alpha_p_value=result["alpha_p_value"],
             beta_market=result["beta_market"],
             beta_smb=result["beta_smb"],
             beta_hml=result["beta_hml"],
             beta_mom=result["beta_mom"],
             beta_rmw=result["beta_rmw"],
             beta_cma=result["beta_cma"],
             beta_gp=result["beta_gp"],
             t_stat_market=result["t_stat_market"],
             t_stat_smb=result["t_stat_smb"],
             t_stat_hml=result["t_stat_hml"],
             t_stat_mom=result["t_stat_mom"],
             t_stat_rmw=result["t_stat_rmw"],
             t_stat_cma=result["t_stat_cma"],
             t_stat_gp=result["t_stat_gp"],
             r_squared=result["r_squared"],
             avg_excess_return_q=result["avg_excess_return_q"],
             return_from_market=result["return_from_market"],
             return_from_smb=result["return_from_smb"],
             return_from_hml=result["return_from_hml"],
             return_from_mom=result["return_from_mom"],
             return_from_rmw=result["return_from_rmw"],
             return_from_cma=result["return_from_cma"],
             return_from_gp=result["return_from_gp"],
             n_quarters_gp=result["n_quarters_gp"],
         )
         .on_conflict_replace()
         .execute())
        n_scored += 1

    return n_scored, n_insufficient
