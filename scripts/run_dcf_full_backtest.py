"""
DCF full-universe standalone backtest — chunked, resumable, checkpointed
scale-up of scripts/run_dcf_pilot_backtest.py's 100-ticker pilot (see
CLAUDE.md's "DCF Standalone Backtest — Pilot Result" section for the
pilot's own inconclusive-leaning-negative result and the decision to
re-test at full scale before drawing any final conclusion).

Why a separate script rather than just raising --n-tickers on the pilot
--------------------------------------------------------------------------
The pilot script does one thing per run: fetch, score everything, write a
single CSV at the very end. Fine at 100 tickers (throttled, but bounded)
— risky at ~1,500. The pilot's own module docstring documents a Yahoo
session-level soft-throttle that can silently degrade a long sustained run
regardless of per-call pacing, and a monolithic run with no checkpointing
would lose ALL progress to a single stall. This script adds three things on
top of the pilot's already-fixed fetch/throttle logic:

  1. Resumable state (data/dcf/full_backtest_state.json) — one entry per
     ticker: "scored", "no_scored_rows" (loaded fine but every quarter
     failed some other per-observation gate), or a load-time skip reason
     (unsuitable_business_model / no_xbrl_fundamentals / no_price_coverage
     / fetch_failure). A rerun with the same command only processes
     whatever ticker isn't already in this file.
  2. Incremental writes — each ticker's scored rows are appended to the
     panel CSV, and its state recorded, immediately after that ticker
     finishes — not buffered in memory until the whole run completes.
  3. Chunked execution (CHUNK_SIZE_DEFAULT tickers at a time, each chunk's
     price fetch batched in one call) with a cooldown between chunks, as
     extra insurance against the same session-level soft-throttle.

A companion business-model classification cache
(dcf/exclusions.py::check_business_model_fit) landed alongside this script
— see that module's docstring — cutting a previously-unnoticed source of
redundant per-quarter yfinance .info calls (compute_point_in_time_dcf calls
it fresh on every (ticker, as_of) pair, not just once per ticker).

Canary mode
-----------
    .venv/bin/python scripts/run_dcf_full_backtest.py --canary
Runs a small (default 15-ticker) timing check against tickers NOT in the
pilot's own 61-ticker scored sample (a genuine cold read on current Yahoo
responsiveness, not a warm on-disk cache hit) through the same two
per-ticker throttled call sites (business-model .info, splits) the full run
depends on. Reports avg seconds/ticker and whether yfinance_client's
backoff/retry logging fired. Does NOT write to the full run's state/panel
files. Healthy baseline (given yfinance_client._MIN_GAP=1.0s): roughly
1-3s/ticker with zero backoff warnings.

Usage
-----
    .venv/bin/python scripts/run_dcf_full_backtest.py --canary [--canary-n N]
    .venv/bin/python scripts/run_dcf_full_backtest.py                # full resumable run
    .venv/bin/python scripts/run_dcf_full_backtest.py --chunk-size 250
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from dcf.backtest import compute_point_in_time_dcf
from dcf.exclusions import check_business_model_fit
from dcf.wacc import fetch_risk_free_rate_as_of
from factor_engine.french_data import get_ff7_daily
from factor_engine.gp_universe import get_universe
from pead.backtest import summarize
from pead.prices import fetch_prices
from run_dcf_pilot_backtest import (
    HORIZONS_TRADING_DAYS,
    _FULL_HISTORY_START,
    _EVAL_START,
    _cohort_ic,
    _forward_return,
    _load_ticker_data,
    _quarter_ends,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

_DATA_DIR = Path(__file__).parent.parent / "data" / "dcf"
STATE_PATH = _DATA_DIR / "full_backtest_state.json"
PANEL_PATH = _DATA_DIR / "full_backtest_panel.csv"
PILOT_PANEL_PATH = _DATA_DIR / "pilot_backtest_panel.csv"

CHUNK_SIZE_DEFAULT = 250
CHUNK_COOLDOWN_SECONDS = 8
CANARY_N_DEFAULT = 15
RANDOM_SEED = 42
MIN_COHORT_OBS = 10   # mirrors pead.backtest.MIN_COHORT_OBS


# ---------------------------------------------------------------------------
# State / panel persistence
# ---------------------------------------------------------------------------

def load_state() -> dict[str, str]:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text())


def save_state(state: dict[str, str]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def append_panel_rows(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(PANEL_PATH, mode="a", header=not PANEL_PATH.exists(), index=False)


# ---------------------------------------------------------------------------
# Canary — small timing check before committing to the full run
# ---------------------------------------------------------------------------

def _pilot_tickers() -> set[str]:
    if not PILOT_PANEL_PATH.exists():
        return set()
    return set(pd.read_csv(PILOT_PANEL_PATH, usecols=["ticker"])["ticker"].unique())


class _WarningCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def run_canary(n: int) -> None:
    universe = get_universe()
    pilot = _pilot_tickers()
    candidates = [t for t in universe["ticker"].tolist() if t not in pilot]
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(candidates)
    sample = candidates[:n]
    log.info("Canary: %d tickers not in the pilot's 61-ticker sample: %s", len(sample), sample)

    handler = _WarningCapture()
    logging.getLogger("yfinance_client").addHandler(handler)

    per_ticker: list[tuple[str, float]] = []
    start = time.monotonic()
    for i, ticker in enumerate(sample, 1):
        t0 = time.monotonic()
        import yfinance as yf
        from yfinance_client import call_with_backoff
        check_business_model_fit(ticker)
        call_with_backoff(lambda t=ticker: yf.Ticker(t).splits)
        elapsed = time.monotonic() - t0
        per_ticker.append((ticker, elapsed))
        log.info("  [%d/%d] %s: %.2fs", i, len(sample), ticker, elapsed)
    total = time.monotonic() - start

    logging.getLogger("yfinance_client").removeHandler(handler)

    avg = total / len(sample) if sample else 0.0
    print("\n" + "=" * 60)
    print(f"CANARY RESULT — {len(sample)} fresh tickers")
    print("=" * 60)
    print(f"Total time: {total:.1f}s   Avg/ticker: {avg:.2f}s")
    print(f"Backoff/retry warnings fired: {len(handler.records)}")
    for w in handler.records[:10]:
        print(f"  - {w}")

    healthy = avg <= 3.0 and not handler.records
    verdict = "HEALTHY — safe to launch the full run" if healthy else "DEGRADED — hold off, do not launch the full run"
    print(f"\nVerdict: {verdict}")


# ---------------------------------------------------------------------------
# Summary printing (reads the full accumulated panel, across all runs)
# ---------------------------------------------------------------------------

def _print_summary() -> None:
    if not PANEL_PATH.exists():
        log.warning("No panel file yet — nothing to summarize.")
        return
    panel = pd.read_csv(PANEL_PATH, parse_dates=["as_of"])
    panel["as_of"] = panel["as_of"].dt.date

    print("\n" + "=" * 78)
    print(f"DCF FULL-UNIVERSE BACKTEST — {panel['ticker'].nunique()} tickers, "
          f"{panel['as_of'].nunique()} quarters, {len(panel)} rows")
    print("=" * 78)

    print(f"\n{'Horizon':<10} {'N quarters':>10} {'Mean IC':>9} {'Std IC':>8} {'t-stat':>8} {'Hit rate':>9}")
    for horizon in HORIZONS_TRADING_DAYS:
        label = {21: "1mo", 63: "3mo", 126: "6mo", 168: "8mo", 210: "10mo", 252: "12mo"}[horizon]
        fwd_col = f"fwd_{horizon}"
        if fwd_col not in panel.columns:
            continue
        quarter_ics = [
            _cohort_ic(group, fwd_col, as_of.isoformat(), horizon, MIN_COHORT_OBS)
            for as_of, group in panel.groupby("as_of")
        ]
        summary = summarize(quarter_ics)
        h = summary.horizons[0] if summary.horizons else None
        if h is None or h.mean_ic is None:
            print(f"{label:<10} {'n/a':>10} {'n/a':>9} {'n/a':>8} {'n/a':>8} {'n/a':>9}")
            continue
        print(f"{label:<10} {h.n_quarters:>10} {h.mean_ic:>+9.4f} {h.std_ic:>8.4f} "
              f"{h.t_stat:>+8.2f} {h.hit_rate:>9.1%}")

    print(f"\nPanel: {PANEL_PATH}")
    print(f"State: {STATE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary", action="store_true", help="Run a small rate-limit timing check and exit")
    parser.add_argument("--canary-n", type=int, default=CANARY_N_DEFAULT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT)
    args = parser.parse_args()

    if args.canary:
        run_canary(args.canary_n)
        return

    state = load_state()
    universe = get_universe()
    all_tickers = sorted(universe["ticker"].tolist())
    remaining = [t for t in all_tickers if t not in state]
    log.info("Full universe: %d tickers. Already processed: %d. Remaining: %d",
              len(all_tickers), len(state), len(remaining))

    if not remaining:
        log.info("Nothing left to process.")
        _print_summary()
        return

    eval_dates = _quarter_ends(_EVAL_START, datetime.date.today())
    log.info("Evaluation grid: %d quarter-ends (%s .. %s)", len(eval_dates), eval_dates[0], eval_dates[-1])

    log.info("Fetching FF7 factor panel (once, shared across all tickers)...")
    factors = get_ff7_daily(_FULL_HISTORY_START, datetime.date.today().isoformat())

    log.info("Fetching risk-free rate per quarter (once, shared across all tickers)...")
    rf_by_date = {d: fetch_risk_free_rate_as_of(d) for d in eval_dates}

    chunk_size = args.chunk_size
    chunks = [remaining[i:i + chunk_size] for i in range(0, len(remaining), chunk_size)]
    log.info("Processing %d tickers in %d chunk(s) of up to %d", len(remaining), len(chunks), chunk_size)

    for c_idx, chunk in enumerate(chunks, 1):
        log.info("=== Chunk %d/%d (%d tickers) ===", c_idx, len(chunks), len(chunk))
        chunk_prices = fetch_prices(chunk, datetime.date.fromisoformat(_FULL_HISTORY_START), datetime.date.today())
        log.info("Chunk price coverage: %d/%d tickers", len(chunk_prices), len(chunk))

        for i, ticker in enumerate(chunk, 1):
            data, skip_reason = _load_ticker_data(ticker, chunk_prices.get(ticker))
            if data is None:
                state[ticker] = skip_reason
                save_state(state)
                continue

            rows = []
            for as_of in eval_dates:
                rf = rf_by_date.get(as_of)
                if rf is None:
                    continue
                result = compute_point_in_time_dcf(
                    ticker, as_of,
                    fund_df=data["fund_df"], prices=data["prices"], splits=data["splits"],
                    returns=data["returns"], factors=factors, risk_free_rate=rf,
                )
                if "error" in result:
                    continue
                row = {"ticker": ticker, "as_of": as_of, "valuation_gap_pct": result["valuation_gap_pct"]}
                for horizon in HORIZONS_TRADING_DAYS:
                    row[f"fwd_{horizon}"] = _forward_return(data["prices"], as_of, horizon)
                rows.append(row)

            append_panel_rows(rows)
            state[ticker] = "scored" if rows else "no_scored_rows"
            save_state(state)

            if i % 25 == 0 or i == len(chunk):
                log.info("  chunk progress: %d/%d tickers processed", i, len(chunk))

        if c_idx < len(chunks):
            log.info("Chunk %d/%d done. Cooling down %ds before next chunk...", c_idx, len(chunks), CHUNK_COOLDOWN_SECONDS)
            time.sleep(CHUNK_COOLDOWN_SECONDS)

    _print_summary()


if __name__ == "__main__":
    main()
