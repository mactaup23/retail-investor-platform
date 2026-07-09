"""
Historical scenario stress tests for the portfolio.

Methodology
-----------
We apply the portfolio's estimated FF4 betas to the *actual* factor returns
that occurred during each stress scenario.  This answers: "Given this portfolio's
current factor exposures, what return would the factor model have predicted during
that historical episode?"

This is not a backtested return (the portfolio didn't exist in 2008).  It is a
model-based sensitivity estimate — how much pain (or cushion) the portfolio's
specific market/size/value/momentum loading structure would have produced under
those macro factor shocks.

Daily portfolio return estimate:
    r_t ≈ rf_t + α_daily + β_mkt·mkt_excess_t + β_smb·smb_t + β_hml·hml_t + β_mom·mom_t

Period return (compounded):
    R = exp(Σ r_t) − 1

Factor decomposition (additive approximation):
    market contribution    = β_mkt × Σ mkt_excess_t
    size contribution      = β_smb × Σ smb_t
    value contribution     = β_hml × Σ hml_t
    momentum contribution  = β_mom × Σ mom_t
    alpha contribution     = α_daily × n_days
    rf contribution        = Σ rf_t

Momentum factor returns come from Ken French's official daily series
(factor_engine/french_data.py::get_ff4_daily()), not the ETF proxy — the
official series has full history back to the 1920s, whereas the ETF proxy
(MTUM) only exists from 2013 onward and could not cover the 2008 scenario.

SPY actual return for each period serves as the market benchmark.
"""

import numpy as np
import pandas as pd

from factor_engine.data_loader import load_returns
from factor_engine.french_data import get_ff4_daily

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
    beta_mom: float,
    alpha_daily: float,
) -> dict:
    """
    Estimate portfolio performance in a scenario using daily factor returns.

    Returns headline period return (compounded) plus an additive factor
    decomposition.
    """
    n = len(factors)

    # Daily estimated portfolio log-returns
    daily_r = (
        factors["rf"]
        + alpha_daily
        + beta_market * factors["mkt_excess"]
        + beta_smb    * factors["smb"]
        + beta_hml    * factors["hml"]
        + beta_mom    * factors["mom"]
    )
    period_return = float(np.expm1(daily_r.sum()))

    # Additive factor contributions (sum of log-return components)
    mkt_contrib   = beta_market * factors["mkt_excess"].sum()
    smb_contrib   = beta_smb    * factors["smb"].sum()
    hml_contrib   = beta_hml    * factors["hml"].sum()
    mom_contrib   = beta_mom    * factors["mom"].sum()
    alpha_contrib = alpha_daily * n
    rf_contrib    = factors["rf"].sum()

    return {
        "period_return":   period_return,
        "n_days":          n,
        "mkt_contrib":     mkt_contrib,
        "smb_contrib":     smb_contrib,
        "hml_contrib":     hml_contrib,
        "mom_contrib":     mom_contrib,
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
    beta_mom: float,
    alpha_daily: float,
) -> list[dict]:
    """
    Run all three stress scenarios against the given portfolio betas.

    Parameters
    ----------
    beta_market, beta_smb, beta_hml, beta_mom : float
        Portfolio-level factor loadings from the headline regression.
    alpha_daily : float
        Daily intercept from the headline regression.

    Returns
    -------
    List of dicts, one per scenario, with all estimation results.
    """
    results = []

    for key, scenario in SCENARIOS.items():
        s, e = scenario["start"], scenario["end"]
        print(f"  Fetching factors for {scenario['label']} ({s} → {e})...")

        factors = get_ff4_daily(s, e)
        if factors.empty:
            print(f"  Warning: no factor data for {key}; skipping.")
            continue

        est = _estimate_scenario_return(factors, beta_market, beta_smb, beta_hml, beta_mom, alpha_daily)

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
