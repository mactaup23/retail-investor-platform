"""
Portfolio factor analysis and stress test report.

Usage (from project root with venv active):
    python scripts/run_portfolio_analysis.py

Optionally override the analysis window:
    python scripts/run_portfolio_analysis.py 2022-01-01 2024-12-31
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from factor_engine.portfolio import WEIGHTS, _RAW_WEIGHTS, FACTOR_BASIS_LABEL, analyze_portfolio
from factor_engine.stress_test import run_stress_tests

# ── Output helpers ─────────────────────────────────────────────────────────────

W = 80  # output width

def rule(char="─"):   print(char * W)
def thick():          print("═" * W)
def blank():          print()


def pct(x: float, decimals: int = 2) -> str:
    return f"{x * 100:+.{decimals}f}%"


def fmt_pct(x: float, decimals: int = 2) -> str:
    return f"{x * 100:.{decimals}f}%"


# ── Section printers ───────────────────────────────────────────────────────────

def print_header(start: str, end: str) -> None:
    thick()
    print(f"  PORTFOLIO FACTOR ANALYSIS  |  {start} → {end}  |  Fama-French 3-Factor Model")
    thick()


def print_composition() -> None:
    blank()
    rule()
    print("  PORTFOLIO COMPOSITION  (raw → normalized weights)")
    rule()
    total_raw = sum(_RAW_WEIGHTS.values())
    for ticker, raw_w in _RAW_WEIGHTS.items():
        norm_w = WEIGHTS[ticker]
        basis  = FACTOR_BASIS_LABEL[ticker]
        print(f"  {ticker:<6} {fmt_pct(raw_w):>7}  →  {fmt_pct(norm_w):>7}   {basis}")
    print(f"  {'TOTAL':<6} {fmt_pct(total_raw):>7}  →  {'100.00%':>7}")


def print_headline(headline: dict) -> None:
    blank()
    rule("═")
    print("  TIER 1 — HEADLINE FACTOR EXPOSURES  (Combined Portfolio Return Series)")
    rule("═")
    blank()

    def row(label, val, t, p):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {label:<28} {val:>8}   t = {t:>7.2f}  {sig}")

    row("Market Beta  (β_mkt)",  f"{headline['beta_market']:+.4f}", headline["t_stat_market"], headline["p_value_market"])
    row("Size Beta    (β_smb)",  f"{headline['beta_smb']:+.4f}",    headline["t_stat_smb"],    headline["p_value_smb"])
    row("Value Beta   (β_hml)", f"{headline['beta_hml']:+.4f}",    headline["t_stat_hml"],    headline["p_value_hml"])
    blank()
    print(f"  {'Alpha (annualised)':<28} {pct(headline['alpha_annualised']):>8}")
    print(f"  {'R²':<28} {headline['r_squared']:>8.4f}")
    print(f"  {'Observations':<28} {headline['n_obs']:>8d}")
    blank()
    print("  Significance: *** p<0.001  ** p<0.01  * p<0.05")


def print_summary(summary_text: str) -> None:
    blank()
    rule()
    print("  PLAIN-ENGLISH INTERPRETATION")
    rule()
    blank()
    for ln in summary_text.splitlines():
        print(f"  {ln}")


def print_attribution(per_holding: list[dict]) -> None:
    blank()
    rule("═")
    print("  TIER 2 — FACTOR ATTRIBUTION BY HOLDING")
    rule("═")
    blank()

    # Header
    hdr = (
        f"  {'Ticker':<6} {'Weight':>7}  {'Factor Basis':<24}"
        f"  {'β_mkt':>7}  {'β_smb':>7}  {'β_hml':>7}  {'R²':>6}"
        f"  {'Wtd_β_mkt':>10}  {'Wtd_β_smb':>10}  {'Wtd_β_hml':>10}"
    )
    print(hdr)
    rule("─")

    for r in per_holding:
        print(
            f"  {r['ticker']:<6} {fmt_pct(r['weight']):>7}  {r['factor_basis']:<24}"
            f"  {r['beta_market']:>+7.3f}  {r['beta_smb']:>+7.3f}  {r['beta_hml']:>+7.3f}"
            f"  {r['r_squared']:>6.3f}"
            f"  {r['wtd_beta_market']:>+10.4f}  {r['wtd_beta_smb']:>+10.4f}  {r['wtd_beta_hml']:>+10.4f}"
        )

    rule("─")

    # Attribution totals
    sum_wt   = sum(r["weight"]           for r in per_holding)
    sum_wmkt = sum(r["wtd_beta_market"]  for r in per_holding)
    sum_wsmb = sum(r["wtd_beta_smb"]     for r in per_holding)
    sum_whml = sum(r["wtd_beta_hml"]     for r in per_holding)

    print(
        f"  {'ATTRIBUTION SUM':<6} {fmt_pct(sum_wt):>7}  {'(weighted avg)':24}"
        f"  {'':>7}  {'':>7}  {'':>7}  {'':>6}"
        f"  {sum_wmkt:>+10.4f}  {sum_wsmb:>+10.4f}  {sum_whml:>+10.4f}"
    )
    blank()
    print("  Note: attribution sums approximate headline betas. Small differences")
    print("  arise because the combined-series regression captures diversification")
    print("  effects (cross-holding correlations) that per-holding regressions do not.")

    # VXUS note
    intl = [r["ticker"] for r in per_holding if "intl" in r["factor_basis"]]
    if intl:
        blank()
        for t in intl:
            print(f"  * {t}: factor basis is US FF3 (international approximation).")
            print(f"    Developed-market ex-US co-moves with US factors at r ≈ 0.70–0.85.")
            print(f"    R² will be lower than for domestic ETFs; loadings are real but understated.")


def print_stress_tests(results: list[dict]) -> None:
    blank()
    rule("═")
    print("  TIER 3 — STRESS TEST: ESTIMATED PORTFOLIO PERFORMANCE")
    rule("═")
    blank()

    hdr = (
        f"  {'Scenario':<30} {'Period':<24}"
        f"  {'Portfolio (Est.)':>16}  {'SPY (Actual)':>13}  {'Diff':>7}"
    )
    print(hdr)
    rule("─")

    for r in results:
        period = f"{r['start']} → {r['end']}"
        spy_str = f"{r['spy_return'] * 100:+.1f}%" if not pd.isna(r["spy_return"]) else "N/A"
        diff_str = f"{r['diff_vs_spy'] * 100:+.1f}%" if not pd.isna(r["spy_return"]) else ""
        print(
            f"  {r['label']:<30} {period:<24}"
            f"  {r['period_return'] * 100:>+14.1f}%  {spy_str:>13}  {diff_str:>7}"
        )

    blank()
    rule("─")
    print("  FACTOR DECOMPOSITION (additive log-return contributions, approximate)")
    rule("─")
    blank()

    col_w = 14
    print(
        f"  {'Scenario':<30}  {'Mkt contrib':>{col_w}}  {'SMB contrib':>{col_w}}"
        f"  {'HML contrib':>{col_w}}  {'Alpha contrib':>{col_w}}  {'RF contrib':>{col_w}}"
    )
    rule("─")

    for r in results:
        print(
            f"  {r['label']:<30}"
            f"  {r['mkt_contrib'] * 100:>{col_w}.1f}%"
            f"  {r['smb_contrib'] * 100:>{col_w}.1f}%"
            f"  {r['hml_contrib'] * 100:>{col_w}.1f}%"
            f"  {r['alpha_contrib'] * 100:>{col_w}.2f}%"
            f"  {r['rf_contrib'] * 100:>{col_w}.2f}%"
        )

    blank()
    print("  Decomposition is additive (sum of log-return components); headline return")
    print("  is compounded.  Small discrepancy is the compounding residual.")

    blank()
    rule("─")
    print("  SCENARIO NOTES")
    rule("─")
    for r in results:
        blank()
        print(f"  {r['label']}  ({r['start']} → {r['end']})")
        print(f"  {r['description']}")


def print_methodology() -> None:
    blank()
    rule()
    print("  METHODOLOGY")
    rule()
    print("  Factor source  : Official Fama-French daily series (Ken French data library)")
    print("  Model          : FF3  →  r_i − r_f = α + β_mkt·(Mkt-RF) + β_smb·SMB + β_hml·HML")
    print("  Headline betas : OLS on combined (weighted-sum) portfolio return series")
    print("  Attribution    : Independent OLS per holding; weighted contributions shown")
    print("  Stress tests   : Estimated daily portfolio returns via headline betas × scenario")
    print("                   factor realizations; SPY actual return shown as benchmark")
    print("  VXUS           : US FF3 factors used (see factor_engine/french_data.py for rationale)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else "2021-01-04"
    end   = sys.argv[2] if len(sys.argv) > 2 else "2024-12-31"

    print_header(start, end)
    print_composition()
    blank()

    results = analyze_portfolio(start, end)

    print_headline(results["headline"])
    print_summary(results["summary_text"])
    print_attribution(results["per_holding"])

    blank()
    print("Running stress tests...")
    h = results["headline"]
    stress_results = run_stress_tests(
        beta_market  = h["beta_market"],
        beta_smb     = h["beta_smb"],
        beta_hml     = h["beta_hml"],
        alpha_daily  = h["alpha_daily"],
    )
    print_stress_tests(stress_results)

    print_methodology()
    blank()
    thick()
    blank()


if __name__ == "__main__":
    main()
