"""
Backfill FinalSignal rows for every ConvergenceScore quarter that doesn't
have one yet.

signal.combine() reads the prior quarter's FinalSignal rows to compute
quarter-over-quarter delta and status (see _load_prior_signals in signal.py),
so missing quarters are processed oldest-first and persisted immediately —
each quarter's persisted output becomes the "prior" input for the next.

--force mode
------------
Gap-filling alone only covers quarters that have never been scored. It does
NOT pick up quarters whose FundSkillResult skill weights changed after the
FinalSignal row was already written (e.g. a factor model upgrade re-scores
every fund's alpha). Pass --force to wipe every FinalSignal row and
recompute all ConvergenceScore quarters from scratch, oldest-first, so the
whole history reflects the current skill scores.

Usage
-----
    .venv/bin/python scripts/backfill_signals.py
    .venv/bin/python scripts/backfill_signals.py --force
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_money import signal
from smart_money.models import ConvergenceScore, FinalSignal, init_db


def _distinct_periods(model) -> list:
    return [
        row.period
        for row in model.select(model.period).distinct().order_by(model.period)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe all FinalSignal rows and recompute every quarter from scratch "
             "(use after FundSkillResult / skill weights change).",
    )
    args = parser.parse_args()

    init_db()

    conv_periods = _distinct_periods(ConvergenceScore)

    if args.force:
        deleted = FinalSignal.delete().execute()
        print("[backfill_signals] --force: wiped FinalSignal")
        print(f"  Rows deleted              : {deleted}")
        print(f"  ConvergenceScore quarters : {len(conv_periods)}")
        print(f"  Recomputing all quarters from scratch (oldest → newest)…")
        print()
        missing = conv_periods
    else:
        final_periods = _distinct_periods(FinalSignal)
        final_set = set(final_periods)
        missing = [p for p in conv_periods if p not in final_set]

        print("[backfill_signals] Gap check")
        print(f"  ConvergenceScore quarters : {len(conv_periods)}")
        print(f"  FinalSignal quarters      : {len(final_periods)}")
        print(f"  Missing (to backfill)     : {len(missing)}")
        if missing:
            print(f"    {', '.join(str(p) for p in missing)}")
        print()

        if not missing:
            print("Nothing to backfill.")
            return

    backfilled = 0
    for period in missing:
        results = signal.combine(period)
        n_written = signal.persist(results)
        print(f"  {period}: {len(results)} signal rows computed, {n_written} persisted")
        backfilled += 1

    total_after = FinalSignal.select().count()
    print()
    print(f"{'Rebuilt' if args.force else 'Backfilled'} {backfilled} quarter(s).")
    print(f"Total FinalSignal rows after {'rebuild' if args.force else 'backfill'}: {total_after}")


if __name__ == "__main__":
    main()
