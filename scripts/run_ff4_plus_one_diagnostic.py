"""
Diagnostic experiment: isolate which of FF7's three added factors (RMW, CMA,
GP) is responsible for the backtest IC degradation seen going from FF4
(+0.061 IC) to FF7 (+0.007, later +0.008 on the 60-fund universe).

Runs three variants — FF4+RMW, FF4+CMA, FF4+GP — each a single flat 5-factor
OLS (see smart_money/factor_apply_diagnostic.py). Each variant:
  1. Scores all active funds, writes FundSkillResult (.on_conflict_replace(),
     same table/semantics as the real pipeline's Phase 5).
  2. Rebuilds ConvergenceScore/FinalSignal from scratch (backfill_convergence.py
     --force, chains into backfill_signals.py --force) — required because
     skill weights changed, same reasoning as any factor-model change.
  3. Runs the full backtest (smart_money.backtest.run_backtest/summarize) and
     saves the aggregate IC/t-stat summary before the next variant overwrites
     FundSkillResult/ConvergenceScore/FinalSignal again.

Destructive to the live DB state across all three variants — FundSkillResult
etc. have no versioning dimension, so each variant's write clobbers the
current (already-validated) FF7/60-fund production state. This script
snapshots data/module3.db via the SQLite Online Backup API (WAL-safe, unlike
a raw file copy) BEFORE running anything, and restores it after all three
variants complete, with an explicit before/after signature check to confirm
the restore actually reproduced the original state bit-for-bit in the
tables that matter (FundSkillResult row count + a spot-check alpha value).

Run from the project root with the venv active:
    .venv/bin/python scripts/run_ff4_plus_one_diagnostic.py
"""

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from smart_money.backtest import run_backtest, summarize
from smart_money.factor_apply_diagnostic import run_diagnostic_variant
from smart_money.models import DB_PATH, FundSkillResult, db, init_db
from smart_money.pipeline import _setup

_BACKUP_PATH = str(DB_PATH) + ".pre_diagnostic_backup"
_VARIANTS = ["rmw", "cma", "gp"]
_SUMMARY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gp", "ff4_plus_one_diagnostic")


def _backup_db() -> None:
    print(f"[diagnostic] Backing up {DB_PATH} -> {_BACKUP_PATH} (SQLite Online Backup API, WAL-safe)...")
    if os.path.exists(_BACKUP_PATH):
        os.remove(_BACKUP_PATH)
    result = subprocess.run(
        ["sqlite3", str(DB_PATH), f".backup '{_BACKUP_PATH}'"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not os.path.exists(_BACKUP_PATH):
        print(f"[diagnostic] BACKUP FAILED: {result.stderr}")
        sys.exit(1)
    backup_size = os.path.getsize(_BACKUP_PATH)
    live_size = os.path.getsize(DB_PATH)
    print(f"[diagnostic] Backup confirmed: {_BACKUP_PATH} ({backup_size:,} bytes; "
          f"live db {live_size:,} bytes before WAL checkpoint, sizes may differ, that's expected)")


def _signature() -> dict:
    """A cheap before/after fingerprint of the tables the diagnostic touches."""
    n_skill = FundSkillResult.select().count()
    rows = list(
        FundSkillResult.select(FundSkillResult.fund, FundSkillResult.alpha_annualized)
        .order_by(FundSkillResult.fund)
    )
    alpha_checksum = round(sum(r.alpha_annualized for r in rows), 6)
    return {"n_fund_skill_result": n_skill, "alpha_checksum": alpha_checksum}


def _restore_db(pre_signature: dict) -> None:
    print(f"\n[diagnostic] Restoring {DB_PATH} from backup...")
    db.close()
    for suffix in ("", "-wal", "-shm"):
        path = str(DB_PATH) + suffix
        if os.path.exists(path):
            os.remove(path)
    subprocess.run(["cp", _BACKUP_PATH, str(DB_PATH)], check=True)

    init_db()
    post_signature = _signature()
    if post_signature == pre_signature:
        print(f"[diagnostic] RESTORE CONFIRMED — signature matches pre-experiment state: {post_signature}")
    else:
        print(f"[diagnostic] RESTORE MISMATCH — pre={pre_signature} post={post_signature}")
        print("[diagnostic] The backup file is still at", _BACKUP_PATH, "— do not delete it, investigate before retrying.")
        sys.exit(1)


def main() -> None:
    init_db()
    _backup_db()
    pre_signature = _signature()
    print(f"[diagnostic] Pre-experiment signature: {pre_signature}\n")

    os.makedirs(_SUMMARY_DIR, exist_ok=True)
    active_funds, _ = _setup()
    print(f"[diagnostic] {len(active_funds)} active funds loaded.\n")

    for variant in _VARIANTS:
        print("=" * 78)
        print(f"  VARIANT: FF4 + {variant.upper()}")
        print("=" * 78)

        start = time.time()
        n_scored, n_insufficient = run_diagnostic_variant(variant, active_funds)
        print(f"[diagnostic:{variant}] Scored {n_scored} funds, {n_insufficient} insufficient "
              f"({time.time()-start:.1f}s)")

        print(f"[diagnostic:{variant}] Rebuilding ConvergenceScore/FinalSignal (--force)...")
        conv_log = os.path.join(_SUMMARY_DIR, f"{variant}_backfill_convergence.log")
        with open(conv_log, "w") as f:
            result = subprocess.run(
                [sys.executable, "scripts/backfill_convergence.py", "--force"],
                stdout=f, stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            print(f"[diagnostic:{variant}] backfill_convergence.py FAILED (exit {result.returncode}) — see {conv_log}")
            sys.exit(1)
        print(f"[diagnostic:{variant}] Convergence/signal rebuild done — log at {conv_log}")

        print(f"[diagnostic:{variant}] Running backtest...")
        db.close()
        init_db()
        quarter_ics = run_backtest()
        summary = summarize(quarter_ics)

        summary_path = os.path.join(_SUMMARY_DIR, f"{variant}_backtest_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"FF4 + {variant.upper()} — aggregate backtest summary\n")
            f.write(f"{'Horizon':>8} {'Universe':<10} {'N quarters':>11} {'Mean IC':>9} "
                    f"{'Std IC':>8} {'t-stat':>8} {'Hit rate':>9}\n")
            for h in summary.horizons:
                line = (
                    f"{h.horizon_days:>7}td {h.universe:<10} {h.n_quarters:>11} "
                    f"{(f'{h.mean_ic:+.4f}' if h.mean_ic is not None else '—'):>9} "
                    f"{(f'{h.std_ic:.4f}' if h.std_ic is not None else '—'):>8} "
                    f"{(f'{h.t_stat:+.2f}' if h.t_stat is not None else '—'):>8} "
                    f"{(f'{h.hit_rate:.0%}' if h.hit_rate is not None else '—'):>9}\n"
                )
                f.write(line)

        print(f"\n[diagnostic:{variant}] Results (saved to {summary_path}):")
        with open(summary_path) as f:
            print(f.read())

    _restore_db(pre_signature)
    print("\n[diagnostic] All three variants complete. Live system restored to the "
          "current validated FF7/60-fund state.")


if __name__ == "__main__":
    main()
