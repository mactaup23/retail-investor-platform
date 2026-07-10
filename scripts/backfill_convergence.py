"""
Backfill ConvergenceScore rows for every Filing quarter that doesn't have
one yet, then chain into scripts/backfill_signals.py to populate FinalSignal
for the newly added quarters.

convergence.scan_quarter()'s _convergence_trend() looks back at the prior two
quarters' persisted ConvergenceScore rows (new/accelerating/fading/stable), so
missing quarters are processed oldest-first and persisted immediately —
same pattern as backfill_signals.py.

--force mode
------------
Gap-filling alone only covers quarters that have never been scored. It does
NOT pick up quarters whose FundSkillResult skill weights changed after the
ConvergenceScore row was already written — e.g. a factor model upgrade
(FF3 → FF4 → FF7) re-scores every fund's alpha, which feeds directly into
_skill_weight() and therefore convergence_score. Pass --force to wipe every
ConvergenceScore row and recompute all Filing quarters from scratch,
oldest-first, then chain into backfill_signals.py --force so FinalSignal is
rebuilt too. Run this whenever skill scores change (new factors, new funds).

Usage
-----
    .venv/bin/python scripts/backfill_convergence.py
    .venv/bin/python scripts/backfill_convergence.py --force
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_money import convergence
from smart_money.models import ConvergenceScore, FinalSignal, Filing, init_db


def _distinct_periods(model, period_field) -> list:
    return [
        getattr(row, period_field.name)
        for row in model.select(period_field).distinct().order_by(period_field)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe all ConvergenceScore (and downstream FinalSignal) rows and "
             "recompute every quarter from scratch (use after FundSkillResult / "
             "skill weights change, e.g. a factor model upgrade).",
    )
    args = parser.parse_args()

    init_db()

    filing_periods = _distinct_periods(Filing, Filing.period_of_report)

    if args.force:
        deleted = ConvergenceScore.delete().execute()
        print("[backfill_convergence] --force: wiped ConvergenceScore")
        print(f"  Rows deleted              : {deleted}")
        print(f"  Filing quarters           : {len(filing_periods)}")
        print(f"  Recomputing all quarters from scratch (oldest → newest)…")
        print()
        missing = filing_periods
    else:
        conv_periods = _distinct_periods(ConvergenceScore, ConvergenceScore.period)
        conv_set = set(conv_periods)
        missing = [p for p in filing_periods if p not in conv_set]

        print("[backfill_convergence] Gap check")
        print(f"  Filing quarters           : {len(filing_periods)}")
        print(f"  ConvergenceScore quarters : {len(conv_periods)}")
        print(f"  Missing (to backfill)     : {len(missing)}")
        if missing:
            print(f"    {', '.join(str(p) for p in missing)}")
        print()

        if not missing:
            print("Nothing to backfill in ConvergenceScore.")

    if missing:
        backfilled = 0
        for period in missing:
            results = convergence.scan_quarter(period)
            n_written = convergence.persist_quarter(results)
            print(f"  {period}: {len(results)} convergence rows computed, {n_written} persisted")
            backfilled += 1

        total_conv_after = ConvergenceScore.select(ConvergenceScore.period).distinct().count()
        print()
        print(f"{'Rebuilt' if args.force else 'Backfilled'} {backfilled} quarter(s).")
        print(f"Total ConvergenceScore quarters after {'rebuild' if args.force else 'backfill'}: {total_conv_after}")

    print()
    verb = "rebuild" if args.force else "populate"
    print(f"[backfill_convergence] Chaining into backfill_signals.py to {verb} FinalSignal…")
    print("=" * 78)
    cmd = [sys.executable, str(Path(__file__).parent / "backfill_signals.py")]
    if args.force:
        cmd.append("--force")
    result = subprocess.run(cmd, check=False)
    print("=" * 78)
    if result.returncode != 0:
        print(f"backfill_signals.py exited with code {result.returncode}")
        sys.exit(result.returncode)

    final_conv_quarters = ConvergenceScore.select(ConvergenceScore.period).distinct().count()
    final_signal_quarters = FinalSignal.select(FinalSignal.period).distinct().count()

    print()
    print("[backfill_convergence] Final state")
    print(f"  ConvergenceScore quarters : {final_conv_quarters}")
    print(f"  FinalSignal quarters      : {final_signal_quarters}")


if __name__ == "__main__":
    main()
