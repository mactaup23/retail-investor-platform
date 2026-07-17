"""
Sanity checks for the DCF valuation engine (dcf/valuation.py).

Unlike the GP factor's directional checks (compare beta_gp sign across
economically distinct company archetypes), a DCF has no comparable "expected
sign" — a fair intrinsic value estimate can legitimately land above or below
the current price for any company. This script instead checks internal
consistency and plausibility across four archetypally different companies:

    AAPL  — mega-cap, high margin, moderate growth
    MSFT  — mega-cap, high margin, still-growing (cloud/AI capex-heavy)
    KO    — mature, slow-grower, high margin, stable — the "boring" case a
            DCF should handle cleanly (low growth, low WACC sensitivity)
    NVDA  — real portfolio holding (2.94% weight) AND watchlist entry
            ("AI infrastructure thesis") widely argued to trade at a
            stretched valuation — grounded in this platform's actual
            holdings, not an arbitrary market pick

Also spot-checks the business-model exclusion (dcf/exclusions.py) against
one real bank (JPM), one insurer (MET), and one REIT (O) — standard
unlevered-FCF DCF is a well-known poor methodological fit for these
business models even where the underlying XBRL data resolves cleanly (see
dcf/exclusions.py module docstring for why this is a different mechanism
from the GP factor's REIT/insurer exclusions).

What "sanity" means here (no ground truth to check against, so structural
plausibility is the bar):
  1. Every scenario runs without error and produces bear <= base <= bull
     per-share values (the growth-spread ordering should hold mechanically).
  2. WACC lands in a plausible range (~6%-14%) — a wildly negative or
     triple-digit WACC would indicate a beta/market-data bug, not a real
     result.
  3. % of enterprise value from terminal value is reported and is typically
     large (60-90%) for a 10-year DCF — flagged, not hidden, per the
     approved output-framing requirement.
  4. debt_source / tax_rate_source / n_annual_observations are surfaced so a
     thin-data ticker's result can be read with appropriate skepticism.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dcf.valuation import run_dcf

CANDIDATES = ["AAPL", "MSFT", "KO", "NVDA"]
EXCLUSION_SPOT_CHECKS = [("JPM", "bank"), ("MET", "insurer"), ("O", "reit")]

_WACC_PLAUSIBLE_RANGE = (0.04, 0.18)


def main():
    for ticker in CANDIDATES:
        print(f"\n{'=' * 70}\n{ticker}\n{'=' * 70}")
        result = run_dcf(ticker)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        b = result["baseline"]
        print(f"  Current price       : ${result['current_price']:.2f}")
        print(f"  Market cap          : ${result['market_cap'] / 1e9:.1f}B")
        print(f"  Beta (market)       : {result['beta']:+.3f}")
        print(f"  10yr Treasury (Rf)  : {result['risk_free_rate'] * 100:.2f}%")
        print(f"  Cost of equity      : {result['cost_of_equity'] * 100:.2f}%")
        cod = result["cost_of_debt"]
        print(f"  Cost of debt (a/t)  : {cod * 100:.2f}%" if cod is not None else "  Cost of debt (a/t)  : N/A (no debt)")
        print(f"  WACC                : {result['wacc'] * 100:.2f}%")
        wacc_ok = _WACC_PLAUSIBLE_RANGE[0] <= result["wacc"] <= _WACC_PLAUSIBLE_RANGE[1]
        print(f"    {'OK' if wacc_ok else 'FLAG'} — plausible range is "
              f"{_WACC_PLAUSIBLE_RANGE[0]*100:.0f}%-{_WACC_PLAUSIBLE_RANGE[1]*100:.0f}%")

        print(f"\n  Baseline inputs:")
        print(f"    EBIT margin (blended)  : {b['ebit_margin'] * 100:.1f}%")
        print(f"    D&A % of revenue       : {b['da_pct'] * 100:.1f}%")
        print(f"    Capex % of revenue     : {b['capex_pct'] * 100:.1f}%")
        print(f"    Effective tax rate     : {b['tax_rate'] * 100:.1f}%  (source: {b['tax_rate_source']})")
        print(f"    Year-1 growth (base)   : {b['start_growth'] * 100:.1f}%  "
              f"(clamped to [{-15:.0f}%, {30:.0f}%])")
        print(f"    Interest expense       : ${b['interest_expense'] / 1e9:.2f}B  (source: {b['interest_expense_source']})")
        print(f"    Total debt             : ${b['total_debt'] / 1e9:.2f}B  (source: {b['debt_source']})")
        print(f"    Cash                   : ${b['cash'] / 1e9:.2f}B")
        print(f"    Diluted shares         : {b['diluted_shares'] / 1e6:.0f}M")
        print(f"    Annual observations    : {b['n_annual_observations']}  (most recent: {b['most_recent_period']})")

        print(f"\n  Scenarios:")
        per_share_by_scenario = {}
        for name in ["bear", "base", "bull"]:
            sc = result["scenarios"][name]
            per_share_by_scenario[name] = sc["per_share"]
            upside = sc["upside_pct"]
            upside_str = f"{upside:+.1f}%" if upside is not None else "N/A"
            pct_tv = sc["pct_from_terminal_value"]
            pct_tv_str = f"{pct_tv * 100:.0f}%" if pct_tv is not None else "N/A"
            print(f"    {name.upper():5s}  start growth {sc['start_growth']*100:+.1f}%  "
                  f"-> ${sc['per_share']:.2f}/share  ({upside_str} vs current)  "
                  f"[{pct_tv_str} of value from terminal value]")

        ordered = (
            per_share_by_scenario["bear"] is not None
            and per_share_by_scenario["base"] is not None
            and per_share_by_scenario["bull"] is not None
            and per_share_by_scenario["bear"] <= per_share_by_scenario["base"] <= per_share_by_scenario["bull"]
        )
        print(f"    {'OK' if ordered else 'FLAG'} — bear <= base <= bull ordering "
              f"{'holds' if ordered else 'VIOLATED — investigate'}")

    print(f"\n{'=' * 70}\nBusiness-model exclusion spot-checks\n{'=' * 70}")
    for ticker, expected_reason in EXCLUSION_SPOT_CHECKS:
        result = run_dcf(ticker)
        actual = result.get("error"), result.get("reason")
        ok = actual == ("unsuitable_business_model", expected_reason)
        print(f"  {ticker:6s} expected '{expected_reason}' -> got {actual}  "
              f"{'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
