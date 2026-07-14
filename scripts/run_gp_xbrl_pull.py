"""
Orchestrator for the GP factor's yfinance -> EDGAR XBRL migration.

Runs scripts/preflight_gp_xbrl.py's 30-ticker sample check first (fast,
under a minute). Only if that clears its gate does this proceed to the full
~1500-ticker fetch_universe_fundamentals(force=True) pull — a 4-6 hour first
run (resumable: interrupting and re-running picks up from both the raw XBRL
JSON cache in data/gp/xbrl_raw/ and the per-ticker CSV cache in
data/gp/fundamentals/, so nothing already fetched is repeated).

Intended to be launched as a long-running background process (the same
pattern used for the original yfinance fundamentals pull), e.g.:
    nohup .venv/bin/python scripts/run_gp_xbrl_pull.py > /tmp/gp_xbrl_pull.log 2>&1 &

Progress is checkpoint-logged every 50 tickers by
fetch_universe_fundamentals() (see factor_engine/gp_fundamentals.py) so the
log file shows visible progress without needing to actively watch the process.
"""

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_PREFLIGHT_SCRIPT = os.path.join(os.path.dirname(__file__), "preflight_gp_xbrl.py")


def main() -> None:
    print("[run_gp_xbrl_pull] Step 1/2 — preflight (30-ticker sample check)...\n", flush=True)
    preflight = subprocess.run([sys.executable, _PREFLIGHT_SCRIPT])

    if preflight.returncode != 0:
        print("\n[run_gp_xbrl_pull] Preflight did not clear its gate — halting. "
              "Full pull was NOT started. See the preflight report above.", flush=True)
        sys.exit(1)

    print("\n[run_gp_xbrl_pull] Step 2/2 — full universe pull (this is the 4-6 hour part)...\n", flush=True)

    from factor_engine.gp_fundamentals import fetch_universe_fundamentals
    from factor_engine.gp_universe import get_universe_tickers

    tickers = get_universe_tickers()
    print(f"[run_gp_xbrl_pull] Universe size: {len(tickers)} tickers.\n", flush=True)

    start = time.time()
    results = fetch_universe_fundamentals(tickers, force=True)
    elapsed_min = (time.time() - start) / 60.0

    n_usable = sum(1 for df in results.values() if not df.empty)
    n_empty = len(results) - n_usable
    print(f"\n[run_gp_xbrl_pull] Done in {elapsed_min:.1f} min. "
          f"{n_usable} tickers with usable data, {n_empty} empty.", flush=True)
    print("[run_gp_xbrl_pull] Next: run scripts/verify_gp_xbrl.py for the full-universe "
          "validation gate before trusting the pre-2021 history.", flush=True)


if __name__ == "__main__":
    main()
