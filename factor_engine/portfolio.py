"""
Portfolio-level 7-factor analysis (Fama-French 5 + Carhart momentum + GP).

Two complementary views are produced:

1. Combined return series (Tier 1 headline)
   The portfolio's daily return is constructed as the weight-averaged sum of
   individual holding log returns.  One joint 7-factor regression on that
   combined series gives the portfolio's effective factor exposures,
   capturing the diversification effect (cross-holding correlations).

2. Per-holding attribution (Tier 2)
   The 7-factor model is run independently on each holding.  Weighted-average
   contributions (weight × beta) explain *which positions drive each factor
   tilt*.  The sum of weighted betas approximates — but will not exactly
   match — the headline betas because the combined-series regression and
   independent regressions share the same factor matrix but not the same
   residual structure.

Factor data source: official Ken French daily series for mkt/smb/hml/rmw/cma/mom
(factor_engine/french_data.py::get_ff7_daily()) rather than ETF proxies, for
maximum accuracy, plus this platform's proprietary GP (Gross Profitability)
factor (factor_engine/factors/gp.py) — there is no Ken French analog for GP.

GP coverage note: GP's own history now spans 2013-present (EDGAR
XBRL-sourced), fully covering this module's default analysis window start
(2021-01-04) — no special handling is needed here beyond the existing
dropna() in run_headline_regression()/run_per_holding_regressions(), which
naturally trims to whatever rows have
data for all seven factors.  Callers requesting a start before 2013 should
expect the effective sample to begin wherever GP coverage actually starts,
not the requested date. (smart_money/factor_apply.py — Module 3/4 fund skill
scoring — uses FF4 only and doesn't include GP at all; see that module's
docstring for why.)
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from factor_engine.data_loader import load_returns
from factor_engine.french_data import get_ff7_daily

# ---------------------------------------------------------------------------
# Portfolio definition
# ---------------------------------------------------------------------------

_RAW_WEIGHTS: dict[str, float] = {
    "VTI":   0.2437,
    "QQQM":  0.1140,
    "SCHD":  0.1178,
    "VXUS":  0.1558,
    "NVDA":  0.0294,
    "GOOGL": 0.0518,
    "QTUM":  0.1027,
    "VTV":   0.0821,
    "XLI":   0.0512,
}
# Raw weights sum to 0.9485 (the remaining 5.15% was unallocated cash).
# Normalize so they sum to exactly 1.0 for factor analysis.
_TOTAL = sum(_RAW_WEIGHTS.values())
WEIGHTS: dict[str, float] = {t: w / _TOTAL for t, w in _RAW_WEIGHTS.items()}

# Common international / global-ex-US equity ETFs.  See factor_engine/french_data.py
# for the full methodology note on why these use US FF7 factors rather than a
# regional blend (Ken French doesn't publish a matching "developed ex-US" daily
# series; correlation with US factors is r ≈ 0.70–0.85). This is a maintained
# list rather than live category lookups, since yfinance's category/fund-family
# metadata is inconsistent across issuers and would make basis labeling
# non-deterministic.
_INTERNATIONAL_TICKERS: frozenset[str] = frozenset({
    "VXUS", "IXUS", "EFA", "IEFA", "VEU", "ACWX", "VT", "ACWI", "URTH",
    "VWO", "IEMG", "EEM", "SCHF", "SPDW", "SCZ", "GWX", "EFAV", "DLS", "IDEV",
    "FNDF", "VSS", "SCHE", "SCHC", "IXUS", "VGK", "VPL", "EWJ", "FEZ",
})


def _is_international_ticker(ticker: str) -> bool:
    return ticker.upper() in _INTERNATIONAL_TICKERS


def _factor_basis_label(ticker: str) -> str:
    return "US FF7 (intl. approx.)" if _is_international_ticker(ticker) else "US FF7"


FACTOR_BASIS_LABEL: dict[str, str] = {t: _factor_basis_label(t) for t in WEIGHTS}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _align_returns_and_factors(
    ticker_returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join ticker returns with factor data on the date index."""
    combined = ticker_returns.join(factors, how="inner").dropna()
    return combined


def _run_ff7_ols(
    excess_returns: pd.Series,
    factors: pd.DataFrame,
) -> dict:
    """
    Fit: excess_return = α + β_mkt·mkt_excess + β_smb·smb + β_hml·hml
                        + β_rmw·rmw + β_cma·cma + β_mom·mom + β_gp·gp + ε

    Returns a dict with all regression outputs.
    """
    factor_cols = ["mkt_excess", "smb", "hml", "rmw", "cma", "mom", "gp"]
    X = add_constant(factors[factor_cols])
    model = OLS(excess_returns, X).fit()
    result = {
        "beta_market":      round(model.params["mkt_excess"], 4),
        "beta_smb":         round(model.params["smb"], 4),
        "beta_hml":         round(model.params["hml"], 4),
        "beta_rmw":         round(model.params["rmw"], 4),
        "beta_cma":         round(model.params["cma"], 4),
        "beta_mom":         round(model.params["mom"], 4),
        "beta_gp":          round(model.params["gp"], 4),
        "alpha_daily":      model.params["const"],
        "alpha_annualised": round(model.params["const"] * 252, 4),
        "r_squared":        round(model.rsquared, 4),
        "n_obs":            int(model.nobs),
    }
    for col, key in (("mkt_excess", "market"), ("smb", "smb"), ("hml", "hml"),
                     ("rmw", "rmw"), ("cma", "cma"), ("mom", "mom"), ("gp", "gp")):
        result[f"t_stat_{key}"] = round(model.tvalues[col], 4)
        result[f"p_value_{key}"] = round(model.pvalues[col], 6)
    return result


# ---------------------------------------------------------------------------
# Plain-English summary
# ---------------------------------------------------------------------------

def _interpret_beta_market(b: float) -> str:
    if b < 0.80:
        return f"defensive ({b:.2f}x) — dampens market swings"
    if b < 1.00:
        return f"slightly below market ({b:.2f}x) — moves with the market but with a small buffer"
    if b < 1.15:
        return f"market-like ({b:.2f}x) — tracks the broad market closely"
    if b < 1.30:
        return f"moderately aggressive ({b:.2f}x) — amplifies market moves"
    return f"notably aggressive ({b:.2f}x) — significantly amplifies market moves"


def _interpret_beta_smb(b: float) -> str:
    if b < -0.25:
        return f"strong large-cap tilt ({b:+.2f})"
    if b < -0.10:
        return f"mild large-cap tilt ({b:+.2f})"
    if b <= 0.10:
        return f"size-neutral ({b:+.2f})"
    if b <= 0.25:
        return f"mild small-cap tilt ({b:+.2f})"
    return f"notable small-cap tilt ({b:+.2f})"


def _interpret_beta_mom(b: float) -> str:
    if b < -0.20:
        return f"strong contrarian tilt ({b:+.2f}) — leans toward recent losers"
    if b < -0.05:
        return f"mild contrarian tilt ({b:+.2f})"
    if b <= 0.05:
        return f"momentum-neutral ({b:+.2f})"
    if b <= 0.20:
        return f"mild momentum tilt ({b:+.2f})"
    return f"strong momentum tilt ({b:+.2f}) — leans toward recent winners"


def _interpret_beta_hml(b: float) -> str:
    if b < -0.30:
        return f"significant growth tilt ({b:+.2f}) — headwind in rising-rate environments"
    if b < -0.10:
        return f"moderate growth tilt ({b:+.2f})"
    if b <= 0.10:
        return f"style-neutral ({b:+.2f})"
    if b <= 0.30:
        return f"moderate value tilt ({b:+.2f})"
    return f"significant value tilt ({b:+.2f}) — tailwind in rising-rate environments"


def _interpret_beta_rmw(b: float) -> str:
    if b < -0.20:
        return f"weak-profitability tilt ({b:+.2f}) — leans toward lower-margin businesses"
    if b < -0.05:
        return f"mild weak-profitability tilt ({b:+.2f})"
    if b <= 0.05:
        return f"profitability-neutral ({b:+.2f})"
    if b <= 0.20:
        return f"mild robust-profitability tilt ({b:+.2f})"
    return f"strong robust-profitability tilt ({b:+.2f}) — leans toward highly profitable businesses"


def _interpret_beta_cma(b: float) -> str:
    if b < -0.20:
        return f"aggressive-investment tilt ({b:+.2f}) — leans toward high-capex/acquisitive companies"
    if b < -0.05:
        return f"mild aggressive-investment tilt ({b:+.2f})"
    if b <= 0.05:
        return f"investment-neutral ({b:+.2f})"
    if b <= 0.20:
        return f"mild conservative-investment tilt ({b:+.2f})"
    return f"strong conservative-investment tilt ({b:+.2f}) — leans toward disciplined capital allocators"


def _interpret_beta_gp(b: float) -> str:
    if b < -0.20:
        return f"low-gross-margin tilt ({b:+.2f}) — leans toward commodity-economics businesses"
    if b < -0.05:
        return f"mild low-gross-margin tilt ({b:+.2f})"
    if b <= 0.05:
        return f"gross-margin-neutral ({b:+.2f})"
    if b <= 0.20:
        return f"mild high-gross-margin tilt ({b:+.2f})"
    return f"strong high-gross-margin tilt ({b:+.2f}) — leans toward businesses with superior unit economics"


def generate_plain_english_summary(headline: dict, per_holding: list[dict]) -> str:
    bm  = headline["beta_market"]
    bsmb = headline["beta_smb"]
    bhml = headline["beta_hml"]
    brmw = headline["beta_rmw"]
    bcma = headline["beta_cma"]
    bmom = headline["beta_mom"]
    bgp  = headline["beta_gp"]
    alpha_pct = headline["alpha_annualised"] * 100
    r2 = headline["r_squared"]

    # Identify the largest drivers for each factor
    def top_contributors(factor_key: str, n: int = 3) -> str:
        rows = sorted(per_holding, key=lambda r: abs(r[f"wtd_{factor_key}"]), reverse=True)
        return ", ".join(r["ticker"] for r in rows[:n])

    mkt_drivers  = top_contributors("beta_market")
    smb_drivers  = top_contributors("beta_smb")
    hml_drivers  = top_contributors("beta_hml")
    rmw_drivers  = top_contributors("beta_rmw")
    cma_drivers  = top_contributors("beta_cma")
    mom_drivers  = top_contributors("beta_mom")
    gp_drivers   = top_contributors("beta_gp")

    # Growth/value split for HML narrative
    value_holders  = [r["ticker"] for r in per_holding if r["beta_hml"] > 0.10]
    growth_holders = [r["ticker"] for r in per_holding if r["beta_hml"] < -0.10]

    lines = [
        "Market exposure:",
        f"  {_interpret_beta_market(bm)}.  Primary contributors: {mkt_drivers}.",
        "",
        "Size tilt:",
        f"  {_interpret_beta_smb(bsmb)}.  Driven by: {smb_drivers}.",
        "",
        "Value / growth tilt:",
        f"  {_interpret_beta_hml(bhml)}.",
        "",
        "Profitability tilt (RMW):",
        f"  {_interpret_beta_rmw(brmw)}.  Driven by: {rmw_drivers}.",
        "",
        "Investment tilt (CMA):",
        f"  {_interpret_beta_cma(bcma)}.  Driven by: {cma_drivers}.",
        "",
        "Momentum tilt:",
        f"  {_interpret_beta_mom(bmom)}.  Driven by: {mom_drivers}.",
        "",
        "Gross profitability tilt (GP, proprietary):",
        f"  {_interpret_beta_gp(bgp)}.  Driven by: {gp_drivers}.",
        f"  Note: GP has 2013-present coverage, sourced from SEC EDGAR XBRL, matching the other six factors' history.",
    ]
    if value_holders and growth_holders:
        lines.append(
            f"  Value anchors ({', '.join(value_holders)}) partially offset "
            f"growth drivers ({', '.join(growth_holders)})."
        )
    elif growth_holders:
        lines.append(f"  Growth drivers: {', '.join(growth_holders)}.")
    elif value_holders:
        lines.append(f"  Value holders: {', '.join(value_holders)}.")

    lines += [
        "",
        f"Model fit: R² = {r2:.3f}  |  the 7-factor model explains {r2 * 100:.1f}% of daily portfolio variance.",
        f"Alpha: {alpha_pct:+.2f}% annualised (unexplained excess return above factor exposures).",
    ]

    if bhml < -0.20:
        lines += [
            "",
            "Risk flag: The growth tilt (negative HML) was a significant headwind in 2022.",
            "The stress test below quantifies the exposure.",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def build_combined_return_series(
    returns: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Construct the portfolio's daily log-return series as a weighted sum.

    For a daily-rebalanced portfolio, weighted-sum of log returns is equivalent
    to the simple return computation at daily frequency (log ≈ simple for small r).
    """
    aligned = returns[[t for t in weights if t in returns.columns]].dropna()
    w_series = pd.Series({t: weights[t] for t in aligned.columns})
    portfolio_return = aligned @ w_series
    portfolio_return.name = "portfolio"
    return portfolio_return


def run_headline_regression(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    start: str,
    end: str,
) -> dict:
    """7-factor regression on the combined portfolio return series (Tier 1)."""
    combined = portfolio_returns.to_frame().join(factors, how="inner").dropna()
    excess = combined["portfolio"] - combined["rf"]
    result = _run_ff7_ols(excess, combined)
    result.update({"start": start, "end": end})
    return result


def run_per_holding_regressions(
    factors: pd.DataFrame,
    start: str,
    end: str,
    weights: dict[str, float],
) -> list[dict]:
    """
    7-factor regression for each holding individually (Tier 2 attribution).

    Returns a list of dicts, one per ticker, including the weighted beta
    contributions.
    """
    results = []
    for ticker, weight in weights.items():
        ticker_rets = load_returns([ticker], start, end)[ticker]
        combined = ticker_rets.to_frame("stock").join(factors, how="inner").dropna()
        excess = combined["stock"] - combined["rf"]
        reg = _run_ff7_ols(excess, combined)
        reg.update({
            "ticker":           ticker,
            "weight":           weight,
            "factor_basis":     _factor_basis_label(ticker),
            "wtd_beta_market":  round(weight * reg["beta_market"], 4),
            "wtd_beta_smb":     round(weight * reg["beta_smb"], 4),
            "wtd_beta_hml":     round(weight * reg["beta_hml"], 4),
            "wtd_beta_rmw":     round(weight * reg["beta_rmw"], 4),
            "wtd_beta_cma":     round(weight * reg["beta_cma"], 4),
            "wtd_beta_mom":     round(weight * reg["beta_mom"], 4),
            "wtd_beta_gp":      round(weight * reg["beta_gp"], 4),
        })
        results.append(reg)
    return results


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def analyze_portfolio(
    start: str = "2021-01-04",
    end: str = "2024-12-31",
    weights: dict[str, float] | None = None,
) -> dict:
    """
    Run the full portfolio factor analysis.

    Parameters
    ----------
    weights : optional dict of ticker -> weight
        Un-normalized (raw) portfolio weights. Defaults to the hardcoded
        example portfolio (module-level _RAW_WEIGHTS) when omitted. Weights
        need not sum to 1.0 — they are normalized internally before the
        regressions run.

    Returns
    -------
    dict with keys:
        weights         — normalized weight dict
        raw_weights     — the weights as passed in (un-normalized)
        factors         — the factor DataFrame used
        combined_rets   — portfolio combined return Series
        headline        — Tier 1 regression results dict
        per_holding     — Tier 2 list of per-ticker regression dicts
        summary_text    — plain-English interpretation string
        start, end      — analysis period
    """
    raw_weights = weights if weights is not None else _RAW_WEIGHTS
    total_raw = sum(raw_weights.values())
    norm_weights = {t: w / total_raw for t, w in raw_weights.items()}

    print(f"Fetching 7-factor data ({start} → {end})...")
    factors = get_ff7_daily(start, end)
    if factors.empty:
        raise ValueError(f"No factor data returned for {start}–{end}")

    print(f"Fetching price data for {list(norm_weights.keys())}...")
    all_returns = load_returns(list(norm_weights.keys()), start, end)

    print("Building combined portfolio return series...")
    combined_rets = build_combined_return_series(all_returns, norm_weights)

    print("Running headline (combined series) 7-factor regression...")
    headline = run_headline_regression(combined_rets, factors, start, end)

    print("Running per-holding 7-factor regressions...")
    per_holding = run_per_holding_regressions(factors, start, end, norm_weights)

    summary_text = generate_plain_english_summary(headline, per_holding)

    return {
        "weights":       norm_weights,
        "raw_weights":   raw_weights,
        "factors":       factors,
        "combined_rets": combined_rets,
        "headline":      headline,
        "per_holding":   per_holding,
        "summary_text":  summary_text,
        "start":         start,
        "end":           end,
    }
