"""
Diagnostic experiment: does restricting convergence scoring to only the
highest-skill-tier funds improve predictive power vs. the current inclusive
approach (all scored + unscored funds, skill-weighted 0.10-3.00x or bucket
default)?

Tiers tested (see smart_money/convergence_diagnostic.py's tier_filter):
    baseline           — no restriction (all 61 active funds), size_filter(0.0)
                          reproduces the unfiltered rule; sanity-checked against
                          the real backtest.run_backtest() numbers.
    high_confidence    — only funds with confidence_label starting "High"
                          (|alpha_t_stat| > 1.5 AND >= 12 quarters). This is a
                          STATISTICAL-PRECISION cut, not a skill-direction cut —
                          includes confidently-negative-alpha funds.
    positive_alpha     — only funds with alpha_annualized > 0, regardless of
                          confidence. A skill-DIRECTION cut, ignores precision.
    high_and_positive  — intersection of the two above. The strictest,
                          most-defensible "genuinely skilled" tier — only 3
                          funds, high risk of insufficient per-quarter coverage
                          given min_funds=2 requires 2 of those exact 3 funds
                          to move on the same cusip in the same quarter.

Unscored funds (no FundSkillResult row) are excluded from all three
restricted tiers — the whole premise of a tier test is "only include funds we
have some skill evidence for," which an unscored fund by definition lacks.
They remain included in baseline (matches current production behavior).

Tests only the 1-month (21td) horizon, both universes — the horizon that
showed the strongest (relative) signal in this session's earlier horizon-
extension test.

Entirely in-memory — no DB writes, nothing persisted, safe to re-run anytime.

Run from the project root with the venv active:
    .venv/bin/python scripts/run_fund_tier_diagnostic.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_money.convergence_diagnostic import run_variant_backtest_multi, size_filter, tier_filter
from smart_money.models import FundSkillResult, init_db

_HORIZONS = (21,)


def _build_tiers() -> dict[str, set[int]]:
    rows = list(FundSkillResult.select())
    high = {r.fund_id for r in rows if r.confidence_label.startswith("High")}
    positive = {r.fund_id for r in rows if r.alpha_annualized > 0}
    return {
        "high_confidence":   high,
        "positive_alpha":    positive,
        "high_and_positive": high & positive,
    }


def _ic(v: float | None) -> str:
    return f"{v:+.3f}" if v is not None else "n/a"


def main() -> None:
    init_db()

    tiers = _build_tiers()
    print("Tier fund counts:")
    print(f"  baseline (all active funds)      : 61")
    for label, ids in tiers.items():
        print(f"  {label:<32}: {len(ids)}")
    print()

    variants = {"baseline": size_filter(0.0)}
    for label, ids in tiers.items():
        variants[label] = tier_filter(ids)

    start = time.time()
    summaries = run_variant_backtest_multi(variants, horizons=_HORIZONS)
    elapsed = time.time() - start

    print(f"{'Tier':<20} {'Universe':<10} {'N quarters':>11} "
          f"{'Total sigs':>10} {'Total obs':>10} {'Avg obs/q':>10} "
          f"{'Mean IC':>9} {'t-stat':>8} {'Hit rate':>9}")
    print("-" * 110)

    for label in ["baseline"] + list(tiers.keys()):
        summary = summaries[label]
        for h in summary.horizons:
            rows = [
                q for q in summary.quarter_ics
                if q.horizon_days == h.horizon_days and q.universe == h.universe and q.ic is not None
            ]
            total_candidates = sum(q.n_candidates for q in rows)
            total_obs = sum(q.n_obs for q in rows)
            avg_obs = total_obs / len(rows) if rows else 0

            print(
                f"{label:<20} {h.universe:<10} {h.n_quarters:>11} "
                f"{total_candidates:>10} {total_obs:>10} {avg_obs:>10.0f} "
                f"{_ic(h.mean_ic):>9} "
                f"{(f'{h.t_stat:+.2f}' if h.t_stat is not None else '—'):>8} "
                f"{(f'{h.hit_rate:.0%}' if h.hit_rate is not None else '—'):>9}"
            )
        print()

    print(f"[total run time: {elapsed:.1f}s]\n")
    print("Sanity check: 'baseline' above should closely match the production")
    print("backtest.run_backtest() 21td numbers (full: +0.008/+1.42/66%; ")
    print("watchlist: +0.006/+1.19/62%).")


if __name__ == "__main__":
    main()
