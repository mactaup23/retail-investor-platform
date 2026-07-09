"""
Historical scenario stress tests for the portfolio.

Methodology
-----------
We apply the portfolio's estimated 7-factor betas to the *actual* factor
returns that occurred during each stress scenario.  This answers: "Given this
portfolio's current factor exposures, what return would the factor model have
predicted during that historical episode?"

This is not a backtested return (the portfolio didn't exist in 2008).  It is a
model-based sensitivity estimate — how much pain (or cushion) the portfolio's
specific market/size/value/profitability/investment/momentum/GP loading
structure would have produced under those macro factor shocks.

Daily portfolio return estimate:
    r_t ≈ rf_t + α_daily + β_mkt·mkt_excess_t + β_smb·smb_t + β_hml·hml_t
               + β_rmw·rmw_t + β_cma·cma_t + β_mom·mom_t + β_gp·gp_t

Period return (compounded):
    R = exp(Σ r_t) − 1

Factor decomposition (additive approximation):
    market contribution        = β_mkt × Σ mkt_excess_t
    size contribution          = β_smb × Σ smb_t
    value contribution         = β_hml × Σ hml_t
    profitability contribution = β_rmw × Σ rmw_t
    investment contribution    = β_cma × Σ cma_t
    momentum contribution      = β_mom × Σ mom_t
    GP contribution            = β_gp  × Σ gp_t   (only when GP has full coverage — see below)
    alpha contribution         = α_daily × n_days
    rf contribution            = Σ rf_t

Momentum/RMW/CMA factor returns come from Ken French's official daily series
(factor_engine/french_data.py::get_ff7_daily()), not ETF proxies — the
official series has full history back to the 1920s/1963, whereas the ETF
proxy (MTUM) only exists from 2013 onward and could not cover the 2008
scenario, and there is no ETF proxy at all for RMW/CMA.

GP coverage gap — by design, not a bug
-----------------------------------------
GP (this platform's proprietary Gross Profitability factor,
factor_engine/factors/gp.py) has only ~2021-present coverage, a hard
limitation of the free yfinance fundamentals data it's built from — it
structurally CANNOT cover the 2008 or 2020 scenarios below, and even the 2022
scenario is marginal.  Each scenario's GP contribution is computed only if
GP has data for every trading day in that scenario's window; otherwise
gp_contrib is None and gp_available is False in the result dict, rather than
silently treating the missing exposure as zero.  RMW/CMA have no such gap and
apply cleanly to all three scenarios.  Callers must check gp_available before
displaying gp_contrib and should show an explicit "insufficient GP history for
this period" caveat when False, exactly as this platform already does for
other structural data gaps (quant fund coverage ceiling, Baupost confidential
treatment, etc.).

SPY actual return for each period serves as the market benchmark.
"""

import numpy as np
import pandas as pd

from factor_engine.data_loader import load_returns
from factor_engine.french_data import get_ff7_daily

SCENARIOS: dict[str, dict] = {
    "2008_financial_crisis": {
        "label":       "2008 Financial Crisis",
        "start":       "2008-09-01",
        "end":         "2009-03-31",
        "description": "Lehman Brothers collapse through S&P 500 trough",
    },
    "2020_covid_crash": {
        "label":       "2020 COVID Crash",
        "start":       "2020-02-19",
        "end":         "2020-03-23",
        "description": "S&P 500 peak to trough (fastest 30%+ decline in history)",
    },
    "2022_rate_hike_bear": {
        "label":       "2022 Rate Hike Bear Market",
        "start":       "2022-01-03",
        "end":         "2022-12-30",
        "description": "Full calendar year 2022; Fed raised rates 425 bp",
    },
}


def _estimate_scenario_return(
    factors: pd.DataFrame,
    beta_market: float,
    beta_smb: float,
    beta_hml: float,
    beta_rmw: float,
    beta_cma: float,
    beta_mom: float,
    beta_gp: float,
    alpha_daily: float,
) -> dict:
    """
    Estimate portfolio performance in a scenario using daily factor returns.

    Returns headline period return (compounded) plus an additive factor
    decomposition.  GP is included in both the daily-return estimate and the
    decomposition only if every day in the window has a GP value — see
    module docstring.  When GP isn't available, the headline period_return
    simply omits that term (equivalent to beta_gp treated as not-yet-estimable
    for this window) rather than silently substituting zero.
    """
    n = len(factors)
    gp_available = "gp" in factors.columns and not factors["gp"].isna().any()

    # Daily estimated portfolio log-returns
    daily_r = (
        factors["rf"]
        + alpha_daily
        + beta_market * factors["mkt_excess"]
        + beta_smb    * factors["smb"]
        + beta_hml    * factors["hml"]
        + beta_rmw    * factors["rmw"]
        + beta_cma    * factors["cma"]
        + beta_mom    * factors["mom"]
    )
    if gp_available:
        daily_r = daily_r + beta_gp * factors["gp"]
    period_return = float(np.expm1(daily_r.sum()))

    # Additive factor contributions (sum of log-return components)
    mkt_contrib   = beta_market * factors["mkt_excess"].sum()
    smb_contrib   = beta_smb    * factors["smb"].sum()
    hml_contrib   = beta_hml    * factors["hml"].sum()
    rmw_contrib   = beta_rmw    * factors["rmw"].sum()
    cma_contrib   = beta_cma    * factors["cma"].sum()
    mom_contrib   = beta_mom    * factors["mom"].sum()
    gp_contrib    = float(beta_gp * factors["gp"].sum()) if gp_available else None
    alpha_contrib = alpha_daily * n
    rf_contrib    = factors["rf"].sum()

    return {
        "period_return":   period_return,
        "n_days":          n,
        "mkt_contrib":     mkt_contrib,
        "smb_contrib":     smb_contrib,
        "hml_contrib":     hml_contrib,
        "rmw_contrib":     rmw_contrib,
        "cma_contrib":     cma_contrib,
        "mom_contrib":     mom_contrib,
        "gp_contrib":      gp_contrib,
        "gp_available":    gp_available,
        "alpha_contrib":   alpha_contrib,
        "rf_contrib":      rf_contrib,
    }


def _get_spy_return(start: str, end: str) -> float:
    """Actual SPY log-return (compounded to simple) over the period."""
    try:
        rets = load_returns(["SPY"], start, end)["SPY"]
        return float(np.expm1(rets.sum()))
    except Exception:
        return float("nan")


def run_stress_tests(
    beta_market: float,
    beta_smb: float,
    beta_hml: float,
    beta_rmw: float,
    beta_cma: float,
    beta_mom: float,
    beta_gp: float,
    alpha_daily: float,
) -> list[dict]:
    """
    Run all three stress scenarios against the given portfolio betas.

    Parameters
    ----------
    beta_market, beta_smb, beta_hml, beta_rmw, beta_cma, beta_mom, beta_gp : float
        Portfolio-level factor loadings from the headline regression.
        beta_gp's contribution will be None for scenarios predating GP's
        ~2021-present coverage (2008, 2020; likely also 2022) — see module
        docstring. Pass 0.0 if the caller doesn't have a beta_gp estimate.
    alpha_daily : float
        Daily intercept from the headline regression.

    Returns
    -------
    List of dicts, one per scenario, with all estimation results. Each dict
    includes gp_available: bool — check this before displaying gp_contrib.
    """
    results = []

    for key, scenario in SCENARIOS.items():
        s, e = scenario["start"], scenario["end"]
        print(f"  Fetching factors for {scenario['label']} ({s} → {e})...")

        factors = get_ff7_daily(s, e)
        if factors.empty:
            print(f"  Warning: no factor data for {key}; skipping.")
            continue

        est = _estimate_scenario_return(
            factors, beta_market, beta_smb, beta_hml, beta_rmw, beta_cma, beta_mom, beta_gp, alpha_daily
        )
        if not est["gp_available"]:
            print(f"  Note: GP has insufficient history for {scenario['label']}; "
                  f"gp_contrib omitted from this scenario's estimate.")

        print(f"  Fetching SPY actual return for {scenario['label']}...")
        spy_return = _get_spy_return(s, e)

        results.append({
            "key":            key,
            "label":          scenario["label"],
            "description":    scenario["description"],
            "start":          s,
            "end":            e,
            **est,
            "spy_return":     spy_return,
            "diff_vs_spy":    est["period_return"] - spy_return,
        })

    return results
