"""
Verification: signal backtest against the real pipeline database.

Runs backtest.run_backtest() over every quarter with ConvergenceScore rows,
prints per-quarter coverage and IC for each (horizon, universe) combination,
the aggregated summary (mean IC / t-stat / hit rate / rolling averages), and
spot-checks a handful of individual observations so the entry/exit prices can
be eyeballed against knowledge_date() by hand.

Usage
-----
    .venv/bin/python scripts/verify_signal_backtest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_money.backtest import (
    HORIZONS_TRADING_DAYS,
    MIN_QUARTER_OBS,
    UNIVERSES,
    knowledge_date,
    run_backtest,
    summarize,
)
from smart_money.models import init_db


def _pct(v: float | None) -> str:
    return f"{v * 100:+.2f}%" if v is not None else "—"


def _ic(v: float | None) -> str:
    return f"{v:+.3f}" if v is not None else "n/a (< min obs)"


def main() -> None:
    init_db()

    print("[verify_signal_backtest] Running backtest across all available quarters…\n")
    quarter_ics = run_backtest()

    if not quarter_ics:
        print("  No ConvergenceScore rows found — run the pipeline before backtesting.")
        sys.exit(1)

    periods = sorted({q.period for q in quarter_ics})
    print(f"  Quarters found          : {len(periods)}  ({periods[0]} → {periods[-1]})")
    print(f"  Horizons (trading days) : {HORIZONS_TRADING_DAYS}")
    print(f"  Universes               : {UNIVERSES}")
    print(f"  Min obs for a quarter IC: {MIN_QUARTER_OBS}\n")

    # ── Per-quarter table, one horizon/universe combination at a time ────────
    by_key: dict[tuple[int, str], list] = {}
    for q in quarter_ics:
        by_key.setdefault((q.horizon_days, q.universe), []).append(q)

    for (horizon, uni), rows in sorted(by_key.items()):
        rows.sort(key=lambda r: r.period)
        print("=" * 78)
        print(f"  Horizon: {horizon}td   Universe: {uni}")
        print("=" * 78)
        print(f"{'Period':<12} {'Candidates':>10} {'Obs':>6} {'Coverage':>9} {'IC':>10}")
        print("─" * 78)
        for r in rows:
            print(
                f"{str(r.period):<12} {r.n_candidates:>10} {r.n_obs:>6} "
                f"{r.coverage_pct:>8.1%} {_ic(r.ic):>10}"
            )
        print()

    # ── Aggregate summary ─────────────────────────────────────────────────────
    summary = summarize(quarter_ics)
    print("=" * 78)
    print("  Aggregate summary")
    print("=" * 78)
    print(
        f"{'Horizon':>8} {'Universe':<10} {'N quarters':>11} {'Mean IC':>9} "
        f"{'Std IC':>8} {'t-stat':>8} {'Hit rate':>9}"
    )
    print("─" * 78)
    for h in summary.horizons:
        print(
            f"{h.horizon_days:>7}td {h.universe:<10} {h.n_quarters:>11} "
            f"{_ic(h.mean_ic):>9} "
            f"{(f'{h.std_ic:.3f}' if h.std_ic is not None else '—'):>8} "
            f"{(f'{h.t_stat:+.2f}' if h.t_stat is not None else '—'):>8} "
            f"{(f'{h.hit_rate:.0%}' if h.hit_rate is not None else '—'):>9}"
        )
    print()

    # ── Spot-check a handful of individual observations ──────────────────────
    print("=" * 78)
    print("  Spot-check: individual observations (watchlist universe, 63td horizon)")
    print("=" * 78)
    sample = next(
        (q for q in quarter_ics if q.universe == "watchlist" and q.horizon_days == 63 and q.n_obs > 0),
        None,
    )
    if sample is None:
        print("  No observations available to spot-check.")
    else:
        kd = knowledge_date(sample.period)
        print(f"  Period {sample.period}  →  knowledge_date {kd}  (period + 45 days)")
        print(f"  {'Ticker':<8} {'CUSIP':<11} {'Score':>7} {'Fwd Return':>11}")
        print("─" * 78)
        for obs in sample.observations[:8]:
            print(f"  {obs.ticker or '—':<8} {obs.cusip:<11} {obs.score:>7.3f} {_pct(obs.forward_return):>11}")
        print()

    # ── Structural checks ─────────────────────────────────────────────────────
    for q in quarter_ics:
        assert q.n_obs <= q.n_candidates, "BUG: n_obs > n_candidates"
        assert 0.0 <= q.coverage_pct <= 1.0, "BUG: coverage_pct out of [0, 1]"
        assert (q.ic is None) or (-1.0 <= q.ic <= 1.0), "BUG: IC out of [-1, 1]"
        assert (q.ic is None) == (q.n_obs < MIN_QUARTER_OBS), \
            "BUG: ic presence inconsistent with MIN_QUARTER_OBS gate"
        for obs in q.observations:
            assert obs.forward_return > -1.0, "BUG: forward_return <= -100%"
    print("  [check] n_obs ≤ n_candidates for all rows ✓")
    print("  [check] coverage_pct ∈ [0, 1] for all rows ✓")
    print("  [check] ic ∈ [-1, 1] (or None) for all rows ✓")
    print("  [check] ic presence matches MIN_QUARTER_OBS gate ✓")
    print()
    print("  Verification complete.")


if __name__ == "__main__":
    main()
