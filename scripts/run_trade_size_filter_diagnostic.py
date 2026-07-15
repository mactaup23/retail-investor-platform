"""
Diagnostic experiment: does hard-filtering out small/routine position changes
(vs. production's continuous tanh down-weighting) improve the 1/3-month IC?

Runs three variants at the 21td/63td horizons, full + watchlist universes:
    0%  — threshold_pct=0.0, i.e. no filter. Every INCREASED/DECREASED change
          passes, same as production. This is a SANITY CHECK: it should
          reproduce (near-exactly) the real backtest.run_backtest() numbers
          for 21td/63td, confirming convergence_diagnostic.py's reimplemented
          scoring path matches production before trusting the filtered runs.
    10% — exclude INCREASED/DECREASED changes with |shares_pct_change| < 10%.
    25% — exclude INCREASED/DECREASED changes with |shares_pct_change| < 25%.

Entirely in-memory — no DB writes, nothing persisted, safe to re-run anytime.

Run from the project root with the venv active:
    .venv/bin/python scripts/run_trade_size_filter_diagnostic.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_money.convergence_diagnostic import run_filtered_backtest_multi
from smart_money.models import init_db

_THRESHOLDS = [0.0, 10.0, 25.0]
_HORIZONS = (21, 63)


def _ic(v: float | None) -> str:
    return f"{v:+.3f}" if v is not None else "n/a"


def _label(threshold: float) -> str:
    return "none (0%)" if threshold == 0.0 else f"{threshold:.0f}%"


def main() -> None:
    init_db()

    start = time.time()
    summaries = run_filtered_backtest_multi(_THRESHOLDS, horizons=_HORIZONS)
    elapsed = time.time() - start

    print(f"{'Threshold':>10} {'Horizon':>8} {'Universe':<10} {'N quarters':>11} "
          f"{'Total sigs':>10} {'Total obs':>10} {'Avg obs/q':>10} "
          f"{'Mean IC':>9} {'t-stat':>8} {'Hit rate':>9}")
    print("-" * 108)

    for threshold in _THRESHOLDS:
        summary = summaries[threshold]
        for h in summary.horizons:
            rows = [
                q for q in summary.quarter_ics
                if q.horizon_days == h.horizon_days and q.universe == h.universe and q.ic is not None
            ]
            total_candidates = sum(q.n_candidates for q in rows)
            total_obs = sum(q.n_obs for q in rows)
            avg_obs = total_obs / len(rows) if rows else 0

            print(
                f"{_label(threshold):>10} {h.horizon_days:>7}td {h.universe:<10} {h.n_quarters:>11} "
                f"{total_candidates:>10} {total_obs:>10} {avg_obs:>10.0f} "
                f"{_ic(h.mean_ic):>9} "
                f"{(f'{h.t_stat:+.2f}' if h.t_stat is not None else '—'):>8} "
                f"{(f'{h.hit_rate:.0%}' if h.hit_rate is not None else '—'):>9}"
            )
        print()

    print(f"[total run time: {elapsed:.1f}s]\n")
    print("Sanity check: the 'none (0%)' row above should closely match the production")
    print("backtest.run_backtest() 21td/63td numbers (full: +0.008/+1.42/66% and")
    print("+0.008/+1.52/57%; watchlist: +0.006/+1.19/62% and +0.007/+1.52/53%).")


if __name__ == "__main__":
    main()
