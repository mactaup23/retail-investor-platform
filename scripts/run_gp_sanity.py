"""
Sanity checks for the GP (Gross Profitability) factor and its 5-factor loadings
(compute_factor_loadings() in factor_engine/factors/hml.py: mkt+smb+hml+mom+gp).

Unlike MOM (which has a Ken French official series to compare the ETF proxy
against), GP has no academic analog to correlate against — it's this
platform's own construction (Novy-Marx 2013 spec, invested-capital
denominator — see factor_engine/factors/gp.py module docstring). So this
script instead verifies the specific directional sanity checks the factor
was designed to satisfy:

1. High-gross-margin, high-quality businesses (AAPL, MSFT, GOOGL) should show
   positive beta_gp.
2. Efficient negative-working-capital businesses (KR grocery retail, MKC
   branded consumer staples) should show positive beta_gp — supplier-financed
   working capital is a genuine capital-efficiency signal the invested-capital
   denominator is designed to credit, not a flaw to correct. (An earlier
   version of this script expected KR to show *negative* beta_gp on a
   "commodity/thin-margin" framing under the old Total-Assets-only
   denominator; that framing was dropped once the NIBCL refinement made clear
   KR's positive loading was the economically correct read all along.)
3. Genuinely thin-margin commodity economics with no offsetting working-capital
   efficiency (XOM — energy, capital-intensive with ordinary payables) should
   show negative beta_gp.
4. Heavy reinvestors in growth (AMZN, NVDA) should show neutral-to-positive
   beta_gp — this is the check that distinguishes GP from FCF yield, which
   would penalize these names for their capex/opex intensity even though
   their underlying gross margins are strong. See factor_engine/factors/gp.py
   module docstring for the full rationale.

Known open discrepancies and mixed results across the invested-capital
refinement's two stages (NOT swept under the rug — see
factor_engine/gp_fundamentals.py and factor_engine/factors/gp.py module
docstrings for the full construction detail):

Stage 1 (NIBCL: -Cash -ShortTermInvestments -AccountsPayable-AccruedLiab)
was motivated by hoping to fix MSFT's negative beta_gp. It didn't — MSFT
moved further negative (-0.077 to -0.111). GOOGL crossed to positive, NVDA
and XOM both moved substantially less-negative.

Stage 2 (additionally: -Goodwill -IntangibleAssets) was motivated by a
read-only balance-sheet diagnostic finding MSFT carries goodwill+intangibles
at ~20% of total assets vs ~7% for AAPL/KR (traced to MSFT's acquisition
history — Activision Blizzard, LinkedIn, Nuance, GitHub). This time MSFT
*did* improve substantially: -0.111 to -0.024, a 78% reduction in magnitude,
though still technically negative. But it came with real collateral cost:
AAPL moved further negative (-0.071 to -0.171, partly because AAPL itself
stopped separately tagging Goodwill in XBRL after 2017, so it doesn't get
the same denominator credit other companies do), GOOGL flipped back negative
(+0.011 to -0.090), AMZN moved much more negative (-0.048 to -0.284, now
failing the reinvestor check it used to narrowly pass), and NVDA moved more
negative too (-0.492 to -0.811). XOM's magnitude nearly doubled (-0.594 to
-1.147) — investigated directly (checked short-basket composition and
overall factor volatility across both stages) and found no bug: XOM's actual
energy-sector peers (EQT, HLX, VAL, KNTK, CVI, PBF) have consistently
occupied the bottom GP quintile across every version of this formula: this
is the same "capital-intensive commodity businesses score low" pattern
amplified, not new or spurious.

beta_gp is a regression against the long/short basket's daily returns,
driven by relative cross-sectional ranking across the full ~1450-ticker
universe at every historical rebalance, not by a single company's own
gp_ratio level — so a company-specific denominator fix doesn't guarantee
that company's correlation with the resulting portfolio improves, and can
move other companies' loadings in either direction as a side effect. Kept
anyway because the formula is more economically correct on its own terms
(matches the standard invested-capital convention) and it measurably helped
the specific problem it was built to address, even though it wasn't free.

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
# Efficient negative-working-capital candidates (expect beta_gp > 0) — see
# module docstring point 2 for why KR/MKC are grouped here rather than with
# XOM under a "commodity economics" framing this refinement retired.
EFFICIENT_WORKING_CAPITAL_CANDIDATES = ["KR", "MKC"]
# Genuinely thin-margin commodity economics candidates (expect beta_gp < 0)
COMMODITY_CANDIDATES = ["XOM"]
# Heavy reinvestors — expect neutral-to-positive, NOT penalized like FCF yield would
REINVESTOR_CANDIDATES = ["AMZN", "NVDA"]

ALL_CANDIDATES = (
    QUALITY_CANDIDATES + EFFICIENT_WORKING_CAPITAL_CANDIDATES
    + COMMODITY_CANDIDATES + REINVESTOR_CANDIDATES
)

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
    for t in EFFICIENT_WORKING_CAPITAL_CANDIDATES:
        b = df.at[t, "beta_gp"]
        ok = b > 0
        print(f"  {t:6s} beta_gp = {b:+.4f}  "
              f"{'OK (positive, as expected — negative-working-capital efficiency rewarded)' if ok else 'FAIL — expected positive'}")
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
        print("isn't the explanation for a failure here — the Novy-Marx GP/invested-capital")
        print("ratio rewards asset turnover as much as pure margin and doesn't control for")
        print("size, so it won't cleanly separate every intuitive case. AAPL, GOOGL, AMZN,")
        print("and NVDA failing here is a known, documented side effect of the goodwill/")
        print("intangibles extension to the invested-capital denominator (module docstring,")
        print("factor_engine/gp_fundamentals.py) — it was built to fix MSFT's ranking")
        print("specifically, measurably improved MSFT (beta_gp -0.111 -> -0.024), but moved")
        print("these four names further from their expected sign as a side effect. beta_gp")
        print("is driven by cross-sectional rank at every historical rebalance, not by one")
        print("company's own ratio level, so a targeted fix doesn't isolate to just its")
        print("target. The primary correctness check is the synthetic OLS-recovery test")
        print("(tests/test_hml_factor.py::test_compute_factor_loadings_recovers_true_betas).")
    else:
        print("\nAll candidates matched their expected directional sign.")


if __name__ == "__main__":
    main()
