"""
Diagnostic experiment: does restricting to the original 41-fund universe
(under the current production FF4 skill model) reproduce the historical
+0.061 IC (t=3.24) baseline?

Context: the isolated single-factor experiment (FF4+RMW, FF4+CMA, FF4+GP —
see scripts/run_ff4_plus_one_diagnostic.py) compared each variant against
+0.061, but that figure was measured on the OLD 41-fund universe before its
expansion to 60. A fresh FF4-only backtest on the CURRENT 60-fund universe
came back at +0.008 (t=1.52) — matching the "degraded" variants, not +0.061.
This script isolates the real variable: is it fund-universe composition (the
19 newly-added funds, several with alpha statistically indistinguishable
from zero — see app_pages/about.py's REIT/fund-universe-expansion
investigation) rather than factor count?

Method: temporarily mark the 19 funds added in the 41->60 expansion (see
git commit 37a0246 for the exact list) as Fund.excluded=True. This is the
same flag convergence.py's scan_quarter() already uses to determine its
active universe (fund_ids = {f.id for f in Fund.select().where(excluded ==
False)}), so no changes to convergence.py itself are needed — just a
temporary DB mutation, backup/restore-wrapped exactly like the prior
diagnostic. No re-scoring needed: the 41 remaining funds already have
FF4-based FundSkillResult rows from the current production state; excluding
the other 19 simply removes them from convergence's fund_ids set, so their
existing skill rows become irrelevant to this test without needing deletion.

Run from the project root with the venv active:
    .venv/bin/python scripts/run_41fund_ff4_diagnostic.py
"""

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from smart_money.backtest import run_backtest, summarize
from smart_money.models import DB_PATH, Fund, FundSkillResult, db, init_db

_BACKUP_PATH = str(DB_PATH) + ".pre_41fund_diagnostic_backup"
_SUMMARY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gp", "ff4_plus_one_diagnostic")

# The 19 funds added in the 41->60 expansion (commit 37a0246).
_NEW_FUND_NAMES = [
    "Norges Bank Investment Management", "Temasek Holdings", "Berkshire Hathaway",
    "Markel Group", "Fairfax Financial Holdings", "Loews Corporation",
    "Harvard Management Company", "UTIMCO", "Wellington Management Group",
    "Tweedy, Browne Company", "Baron Capital Group", "Harris Associates",
    "Davis Advisors", "Duquesne Family Office", "Royce Investment Partners",
    "Elliott Investment Management", "Ruane, Cunniff & Goldfarb",
    "Capital World Investors", "T. Rowe Price Investment Management",
]


def _backup_db() -> None:
    print(f"[41fund-diagnostic] Backing up {DB_PATH} -> {_BACKUP_PATH}...")
    if os.path.exists(_BACKUP_PATH):
        os.remove(_BACKUP_PATH)
    result = subprocess.run(
        ["sqlite3", str(DB_PATH), f".backup '{_BACKUP_PATH}'"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not os.path.exists(_BACKUP_PATH):
        print(f"[41fund-diagnostic] BACKUP FAILED: {result.stderr}")
        sys.exit(1)
    print(f"[41fund-diagnostic] Backup confirmed: {os.path.getsize(_BACKUP_PATH):,} bytes")


def _signature() -> dict:
    n_skill = FundSkillResult.select().count()
    n_excluded = Fund.select().where(Fund.excluded == True).count()  # noqa: E712
    return {"n_fund_skill_result": n_skill, "n_excluded": n_excluded}


def _restore_db(pre_signature: dict) -> None:
    print(f"\n[41fund-diagnostic] Restoring {DB_PATH} from backup...")
    db.close()
    for suffix in ("", "-wal", "-shm"):
        path = str(DB_PATH) + suffix
        if os.path.exists(path):
            os.remove(path)
    subprocess.run(["cp", _BACKUP_PATH, str(DB_PATH)], check=True)

    init_db()
    post_signature = _signature()
    if post_signature == pre_signature:
        print(f"[41fund-diagnostic] RESTORE CONFIRMED — signature matches pre-experiment state: {post_signature}")
    else:
        print(f"[41fund-diagnostic] RESTORE MISMATCH — pre={pre_signature} post={post_signature}")
        print("[41fund-diagnostic] Backup file preserved at", _BACKUP_PATH, "— investigate before retrying.")
        sys.exit(1)


def main() -> None:
    init_db()
    _backup_db()
    pre_signature = _signature()
    print(f"[41fund-diagnostic] Pre-experiment signature: {pre_signature}\n")

    os.makedirs(_SUMMARY_DIR, exist_ok=True)

    matched = list(Fund.select().where(Fund.name.in_(_NEW_FUND_NAMES)))
    print(f"[41fund-diagnostic] Matched {len(matched)}/{len(_NEW_FUND_NAMES)} of the 19 expansion funds by name:")
    for f in matched:
        print(f"    {f.name}")
    if len(matched) != len(_NEW_FUND_NAMES):
        missing = set(_NEW_FUND_NAMES) - {f.name for f in matched}
        print(f"[41fund-diagnostic] WARNING — {len(missing)} name(s) did not match any Fund row: {missing}")

    n_updated = (Fund
                 .update(excluded=True)
                 .where(Fund.name.in_(_NEW_FUND_NAMES))
                 .execute())
    print(f"\n[41fund-diagnostic] Marked {n_updated} fund(s) excluded=True "
          f"(temporary — restored at the end). Remaining active universe: "
          f"{Fund.select().where(Fund.excluded == False).count()} funds.\n")  # noqa: E712

    print("[41fund-diagnostic] Rebuilding ConvergenceScore/FinalSignal (--force) "
          "restricted to the 41-fund universe...")
    conv_log = os.path.join(_SUMMARY_DIR, "41fund_backfill_convergence.log")
    start = time.time()
    with open(conv_log, "w") as f:
        result = subprocess.run(
            [sys.executable, "scripts/backfill_convergence.py", "--force"],
            stdout=f, stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        print(f"[41fund-diagnostic] backfill_convergence.py FAILED (exit {result.returncode}) — see {conv_log}")
        _restore_db(pre_signature)
        sys.exit(1)
    print(f"[41fund-diagnostic] Convergence/signal rebuild done ({time.time()-start:.1f}s) — log at {conv_log}")

    print("[41fund-diagnostic] Running backtest...")
    db.close()
    init_db()
    quarter_ics = run_backtest()
    summary = summarize(quarter_ics)

    summary_path = os.path.join(_SUMMARY_DIR, "41fund_ff4_backtest_summary.txt")
    with open(summary_path, "w") as f:
        f.write("41-fund universe, FF4 skill model — aggregate backtest summary\n")
        f.write(f"{'Horizon':>8} {'Universe':<10} {'N quarters':>11} {'Mean IC':>9} "
                f"{'Std IC':>8} {'t-stat':>8} {'Hit rate':>9}\n")
        for h in summary.horizons:
            f.write(
                f"{h.horizon_days:>7}td {h.universe:<10} {h.n_quarters:>11} "
                f"{(f'{h.mean_ic:+.4f}' if h.mean_ic is not None else '—'):>9} "
                f"{(f'{h.std_ic:.4f}' if h.std_ic is not None else '—'):>8} "
                f"{(f'{h.t_stat:+.2f}' if h.t_stat is not None else '—'):>8} "
                f"{(f'{h.hit_rate:.0%}' if h.hit_rate is not None else '—'):>9}\n"
            )

    print(f"\n[41fund-diagnostic] Results (saved to {summary_path}):")
    with open(summary_path) as f:
        print(f.read())

    _restore_db(pre_signature)
    print("\n[41fund-diagnostic] Experiment complete. Live system restored to the "
          "current 60-fund FF4 production state.")


if __name__ == "__main__":
    main()
