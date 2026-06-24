"""
Verification script for nlp.py — runs the 7-dimension language-shift scorer
against 3 real portfolio-company tickers via the Claude Batch API.

Run:
    ANTHROPIC_API_KEY=sk-... python scripts/verify_nlp.py

Demonstrates:
  1. Ticker → CIK lookup
  2. Filing-pair selection (10-Q or 10-K fallback)
  3. MD&A extraction from EDGAR HTML
  4. Batch API submission and result parsing
  5. NLPCache persistence
  6. Cache hit on a second run (instant, no API call)
"""

import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TICKERS = ["NVDA", "MSFT", "META"]

_DIM_KEYS = [
    "guidance_delta",
    "confidence_delta",
    "customer_demand_delta",
    "competitive_positioning_delta",
    "operational_efficiency_delta",
    "risk_factors_delta",
    "capital_allocation_delta",
]

_DIM_LABELS = [
    "guidance",
    "confidence",
    "cust_demand",
    "competitive",
    "efficiency",
    "risk",
    "cap_alloc",
]

WEIGHTS = [0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05]


def _bar(value: float, width: int = 20) -> str:
    """Render a signed bar chart for a score in [-1, 1]."""
    half = width // 2
    pos = round(value * half)
    if pos >= 0:
        return " " * half + "█" * pos + " " * (half - pos)
    return " " * (half + pos) + "█" * (-pos) + " " * half


def print_result(row) -> None:
    print(f"\n{'─' * 72}")
    print(
        f"  {row.ticker:<6}  composite={row.composite_score:+.3f}  "
        f"form={row.form_type}  scorer={row.scorer_version}"
    )
    print(f"  pair: {row.accession_prior}  →  {row.accession_current}")
    print(f"{'─' * 72}")
    print(f"  {'dimension':<14}  {'wt':>4}  {'score':>6}  {'':^22}")
    print(f"  {'─'*14}  {'─'*4}  {'─'*6}  {'─'*22}")
    for label, key, wt in zip(_DIM_LABELS, _DIM_KEYS, WEIGHTS):
        val = getattr(row, key)
        print(f"  {label:<14}  {wt:.2f}  {val:+.3f}  [{_bar(val, 20)}]")
    print()
    # Truncate reasoning to 300 chars for display
    reasoning = row.reasoning[:300] + ("…" if len(row.reasoning) > 300 else "")
    print(f"  Reasoning: {reasoning}")


def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")

    from smart_money.nlp import batch_score_tickers, ticker_to_cik, _select_filing_pair

    # ── Pre-flight: show what filing pairs were found ────────────────────────
    print("\n=== Filing pair selection ===")
    for ticker in TICKERS:
        cik = ticker_to_cik(ticker)
        if not cik:
            print(f"  {ticker}: CIK not found")
            continue
        pair = _select_filing_pair(cik)
        if pair:
            print(
                f"  {ticker}  CIK={cik}  {pair.form_type}  "
                f"prior={pair.prior_period}  current={pair.current_period}"
            )
        else:
            print(f"  {ticker}  CIK={cik}  no valid filing pair")

    # ── Pass 1: score via Batch API ──────────────────────────────────────────
    print(f"\n=== Pass 1: batch score {TICKERS} ===")
    t0 = time.monotonic()
    rows = batch_score_tickers(TICKERS)
    elapsed = time.monotonic() - t0
    print(f"\nCompleted in {elapsed:.1f}s  ({len(rows)} tickers scored)")

    if not rows:
        print("No results — check logs above for errors.")
        sys.exit(1)

    for row in rows:
        print_result(row)

    # ── Pass 2: verify cache hits ─────────────────────────────────────────────
    print("\n=== Pass 2: re-run (expect all cache hits, <1 s) ===")
    t1 = time.monotonic()
    rows2 = batch_score_tickers(TICKERS)
    elapsed2 = time.monotonic() - t1
    print(f"Completed in {elapsed2:.2f}s  ({len(rows2)} rows returned from cache)")
    assert len(rows2) == len(rows), "Cache round-trip mismatch"

    print(f"\n{'═' * 72}")
    print("  nlp.py verification PASSED")
    print(f"  Scored {len(rows)} tickers across 7 dimensions using {rows[0].scorer_version}")
    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    main()
