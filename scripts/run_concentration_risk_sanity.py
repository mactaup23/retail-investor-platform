"""
Sanity checks for the new Concentration/Correlation (factor_engine/concentration.py)
and Volatility/Sharpe/Drawdown (factor_engine/risk_metrics.py) modules, run against
the real portfolio (factor_engine/portfolio.py's default 9-holding weights) over the
same default analysis window the dashboard uses (2021-01-04 to 2024-12-31).

What "sanity" means here: no single ground truth to check every number against
(these are new derived metrics, not a value with an independent source to
compare to), so plausibility is the bar —
  1. Every function runs end to end without error on real data.
  2. Top-N concentration values are monotonically non-decreasing and <= 100%.
  3. HHI's effective N is between 1 and the holding count.
  4. Trailing correlation clustering surfaces the expected real-world
     semiconductor/AI overlap (NVDA + QQQM + QTUM) — this is the concrete
     example the concentration analysis was motivated by.
  5. Stress-period correlation: gp_available should be False for 2008 only
     (predates GP's 2013-present coverage) and True for 2020/2022 — confirms
     the factor-implied reconstruction correctly reuses stress_test.py's own
     gating rather than silently including/excluding GP.
  6. Risk metrics land in a plausible range for a diversified US/international
     equity portfolio over a 2021-2024 window that includes the 2022 bear
     market: positive-but-moderate annualized vol, a real (negative) max
     drawdown, and a Sharpe/Sortino pair where Sortino >= Sharpe (expected
     whenever upside vol exceeds downside vol, the normal case for equities).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from factor_engine.portfolio import analyze_portfolio
from factor_engine.concentration import run_concentration_analysis
from factor_engine.risk_metrics import compute_risk_metrics

START = "2021-01-04"
END = "2024-12-31"


def main():
    print(f"Running portfolio factor analysis ({START} -> {END})...")
    data = analyze_portfolio(start=START, end=END)
    weights = data["weights"]
    all_returns = data["all_returns"]
    per_holding = data["per_holding"]
    combined_rets = data["combined_rets"]
    factors = data["factors"]

    print(f"\nHoldings ({len(weights)}): " + ", ".join(f"{t} {w*100:.1f}%" for t, w in
          sorted(weights.items(), key=lambda kv: -kv[1])))

    # -----------------------------------------------------------------
    # Concentration / Correlation
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONCENTRATION / CORRELATION")
    print("=" * 70)

    conc = run_concentration_analysis(weights, all_returns, per_holding)

    top_n = conc["top_n"]
    print(f"\nTop-N concentration (n_holdings={top_n['n_holdings']}):")
    for n in (3, 5, 10):
        print(f"  Top {n} (capped at {top_n[f'top_{n}_capped_at']}): {top_n[f'top_{n}']*100:.2f}%")
    monotonic = top_n["top_3"] <= top_n["top_5"] <= top_n["top_10"] <= 1.0001
    print(f"  Monotonic & <=100%: {'PASS' if monotonic else 'FAIL'}")

    hhi = conc["hhi"]
    print(f"\nHHI: {hhi['hhi']:.4f}  |  Effective N: {hhi['effective_n']:.2f} "
          f"(of {hhi['n_holdings']} nominal holdings)")
    print(f"  Meaningfully more concentrated than count suggests: "
          f"{hhi['meaningfully_more_concentrated_than_count']}")
    effective_n_valid = 1.0 <= hhi["effective_n"] <= hhi["n_holdings"] + 1e-6
    print(f"  Effective N in [1, n_holdings]: {'PASS' if effective_n_valid else 'FAIL'}")

    print(f"\nTrailing {conc['trailing_window_days']}-day correlation "
          f"(threshold={conc['correlation_threshold']}):")
    print(f"  Average pairwise correlation: {conc['trailing_avg_pairwise_correlation']:.3f}")
    if conc["trailing_clusters"]:
        for c in conc["trailing_clusters"]:
            print(f"  Cluster: {c['tickers']}  combined_weight={c['combined_weight']*100:.1f}%  "
                  f"avg_corr={c['avg_pairwise_correlation']:.3f}")
    else:
        print("  No clusters above threshold.")

    print(f"\n  Cliques (every pair mutually >{conc['correlation_threshold']}):")
    if conc["trailing_cliques"]:
        for c in conc["trailing_cliques"]:
            print(f"    Clique: {c['tickers']}  combined_weight={c['combined_weight']*100:.1f}%  "
                  f"avg_corr={c['avg_pairwise_correlation']:.3f}")
    else:
        print("    No cliques above threshold.")

    expected_cluster = {"NVDA", "QQQM", "QTUM"}
    found = any(expected_cluster.issubset(set(c["tickers"])) for c in conc["trailing_clusters"]) or \
        any(set(c["tickers"]).issubset(expected_cluster) and len(c["tickers"]) >= 2 for c in conc["trailing_clusters"])
    print(f"\n  NVDA/QQQM/QTUM semiconductor-AI overlap detected in clusters (fully or partially): "
          f"{'YES' if found else 'NO — see note below'}")

    print("\nStress-period factor-implied correlation:")
    for s in conc["stress_period_correlation"]:
        print(f"  {s['label']} ({s['start']} -> {s['end']}): "
              f"avg_corr={s['avg_pairwise_correlation']:.3f}  gp_available={s['gp_available']}")
        for c in s["clusters"]:
            print(f"    Cluster: {c['tickers']}  combined_weight={c['combined_weight']*100:.1f}%  "
                  f"avg_corr={c['avg_pairwise_correlation']:.3f}")
        for c in s["cliques"]:
            print(f"    Clique:  {c['tickers']}  combined_weight={c['combined_weight']*100:.1f}%  "
                  f"avg_corr={c['avg_pairwise_correlation']:.3f}")

    gp_gate_ok = True
    for s in conc["stress_period_correlation"]:
        expected_gp = s["key"] != "2008_financial_crisis"
        if s["gp_available"] != expected_gp:
            gp_gate_ok = False
    print(f"\n  GP availability gate matches expectation (False/True/True for 2008/2020/2022): "
          f"{'PASS' if gp_gate_ok else 'FAIL'}")

    # -----------------------------------------------------------------
    # Risk metrics
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VOLATILITY / SHARPE / SORTINO / DRAWDOWN")
    print("=" * 70)

    risk = compute_risk_metrics(combined_rets, factors)
    print(f"\n  Annualized return:      {risk['annualized_return']*100:+.2f}%")
    print(f"  Annualized volatility:  {risk['annualized_volatility']*100:.2f}%")
    print(f"  Annualized rf:          {risk['annualized_rf']*100:.2f}%")
    print(f"  Sharpe ratio:           {risk['sharpe_ratio']:.3f}")
    print(f"  Sortino ratio:          {risk['sortino_ratio']:.3f}")
    print(f"  Downside deviation:     {risk['downside_deviation']*100:.2f}%")
    print(f"  Max drawdown:           {risk['max_drawdown']*100:.2f}%  "
          f"({risk['max_drawdown_peak_date']} -> {risk['max_drawdown_trough_date']})")
    print(f"  n_obs:                  {risk['n_obs']}")

    vol_plausible = 0.05 < risk["annualized_volatility"] < 0.40
    dd_plausible = -0.60 < risk["max_drawdown"] < -0.02
    sortino_ge_sharpe = risk["sortino_ratio"] >= risk["sharpe_ratio"]
    print(f"\n  Volatility in plausible 5-40% range: {'PASS' if vol_plausible else 'FAIL'}")
    print(f"  Max drawdown in plausible -2% to -60% range: {'PASS' if dd_plausible else 'FAIL'}")
    print(f"  Sortino >= Sharpe (expected for equities, upside vol > downside vol): "
          f"{'PASS' if sortino_ge_sharpe else 'FAIL'}")


if __name__ == "__main__":
    main()
