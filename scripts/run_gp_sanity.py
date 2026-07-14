"""
Sanity checks for the GP (Gross Profitability) factor and its 5-factor loadings
(compute_factor_loadings() in factor_engine/factors/hml.py: mkt+smb+hml+mom+gp).

Unlike MOM (which has a Ken French official series to compare the ETF proxy
against), GP has no academic analog to correlate against — it's this
platform's own construction (Novy-Marx 2013 spec). So this script instead
verifies the specific directional sanity checks the factor was designed to
satisfy:

1. High-gross-margin, high-quality businesses (AAPL, MSFT, GOOGL) should show
   positive beta_gp.
2. Low-margin, commodity-economics businesses (XOM, KR — energy and grocery
   retail, both structurally thin-margin) should show negative beta_gp.
3. Heavy reinvestors in growth (AMZN, NVDA) should show neutral-to-positive
   beta_gp — this is the check that distinguishes GP from FCF yield, which
   would penalize these names for their capex/opex intensity even though
   their underlying gross margins are strong. See factor_engine/factors/gp.py
   module docstring for the full rationale.

First run of this script will trigger the full GP factor construction if it
hasn't been built yet (~1500-ticker fundamentals fetch, 45-90 minutes) —
subsequent runs read the disk cache (data/gp/) and are fast. See
factor_engine/gp_fundamentals.py for the resumable fetch design.

Run from the project root with the venv active:
    python scripts/run_gp_sanity.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import yfinance as yf

from factor_engine.factors.gp import build_gp_factor, get_gp_coverage_start
from factor_engine.factors.hml import compute_factor_loadings

# High-quality / high-margin candidates (expect beta_gp > 0)
QUALITY_CANDIDATES = ["AAPL", "MSFT", "GOOGL"]
# Low-margin commodity-economics candidates (expect beta_gp < 0)
COMMODITY_CANDIDATES = ["XOM", "KR"]
# Heavy reinvestors — expect neutral-to-positive, NOT penalized like FCF yield would
REINVESTOR_CANDIDATES = ["AMZN", "NVDA"]

ALL_CANDIDATES = QUALITY_CANDIDATES + COMMODITY_CANDIDATES + REINVESTOR_CANDIDATES

# Analysis window: GP now covers 2013-present, so this window is a deliberately
# recent slice for this sanity check, not a coverage constraint.
START = "2022-01-01"
END   = "2024-12-31"


def _ttm_gross_margin(ticker: str) -> "float | None":
    """Trailing gross margin = (Revenue - COGS) / Revenue, most recent quarter, for context."""
    try:
        inc = yf.Ticker(ticker).quarterly_income_stmt
        if inc.empty or "Total Revenue" not in inc.index or "Cost Of Revenue" not in inc.index:
            return None
        latest = inc.columns[0]
        revenue = inc.at["Total Revenue", latest]
        cogs = inc.at["Cost Of Revenue", latest]
        if not revenue:
            return None
        return float((revenue - cogs) / revenue)
    except Exception:
        return None


def main():
    coverage_start = get_gp_coverage_start()
    print(f"GP factor coverage starts: {coverage_start}  "
          f"(None means the factor hasn't been built yet — this run will build it)\n")

    print(f"Building GP factor ({START} to {END})...")
    gp_factor = build_gp_factor(START, END)
    if gp_factor.empty:
        print("ERROR: GP factor is empty for this window — check data/gp/ cache and "
              "factor_engine/gp_fundamentals.py fetch logs.")
        return

    gp = gp_factor["gp"]
    print(f"\n=== GP Factor Descriptives ===")
    print(f"  Trading days loaded : {len(gp_factor)}")
    print(f"  Mean daily GP       : {gp.mean()*100:+.4f}%  (annualised ≈ {gp.mean()*252*100:+.2f}%)")
    print(f"  Std dev daily GP    : {gp.std()*100:.4f}%  (annualised ≈ {gp.std()*(252**0.5)*100:.2f}%)")
    print(f"  Cumulative GP       : {gp.sum()*100:+.2f}%  (log-return sum)")

    print(f"\nComputing 5-factor loadings (mkt+smb+hml+mom+gp) for {ALL_CANDIDATES}...\n")
    results = []
    for ticker in ALL_CANDIDATES:
        print(f"  {ticker}...")
        r = compute_factor_loadings(ticker, START, END, gp_factor=gp_factor)
        r["ttm_gross_margin"] = _ttm_gross_margin(ticker)
        results.append(r)

    df = (
        pd.DataFrame(results)
        .set_index("ticker")
        [["ttm_gross_margin", "beta_gp", "t_stat_gp", "p_value_gp", "r_squared", "n_obs"]]
    )
    print("\n=== 5-Factor Loadings — Gross Profitability ===")
    print(df.to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\n=== Directional sanity checks ===")
    failures = []
    for t in QUALITY_CANDIDATES:
        b = df.at[t, "beta_gp"]
        ok = b > 0
        print(f"  {t:6s} beta_gp = {b:+.4f}  {'OK (positive, as expected)' if ok else 'FAIL — expected positive'}")
        if not ok:
            failures.append(t)
    for t in COMMODITY_CANDIDATES:
        b = df.at[t, "beta_gp"]
        ok = b < 0
        print(f"  {t:6s} beta_gp = {b:+.4f}  {'OK (negative, as expected)' if ok else 'FAIL — expected negative'}")
        if not ok:
            failures.append(t)
    for t in REINVESTOR_CANDIDATES:
        b = df.at[t, "beta_gp"]
        ok = b >= -0.05   # "neutral-to-positive" — small tolerance around zero
        print(f"  {t:6s} beta_gp = {b:+.4f}  "
              f"{'OK (neutral-to-positive, not penalized)' if ok else 'FAIL — expected neutral-to-positive'}")
        if not ok:
            failures.append(t)

    if failures:
        print(f"\n{len(failures)} candidate(s) failed the directional check: {failures}")
        print("This is a directional illustration over one sample window, not a strict")
        print("pass/fail gate. GP now has full 2013-present history, so short coverage")
        print("isn't the explanation for a failure here — the Novy-Marx GP/Assets ratio")
        print("rewards asset turnover as much as pure margin and doesn't control for size,")
        print("so it won't cleanly separate every intuitive case. The primary correctness")
        print("check is the synthetic OLS-recovery test")
        print("(tests/test_hml_factor.py::test_compute_factor_loadings_recovers_true_betas).")
    else:
        print("\nAll candidates matched their expected directional sign.")


if __name__ == "__main__":
    main()
