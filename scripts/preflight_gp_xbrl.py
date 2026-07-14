"""
Preflight check for the GP factor's yfinance -> EDGAR XBRL migration: a fast,
small-sample sanity check that runs BEFORE committing to the full ~1500-ticker
pull (4-6 hours). Catches a systematic tag-mapping bug cheaply (30 fetches,
under a minute) rather than discovering it 4 hours into the real run.

Sampling
--------
30 tickers, deterministically and evenly spread across the alphabetically
sorted list of tickers that have non-empty cached data from the prior
yfinance-era pull (data/gp/fundamentals_yfinance_backup/) — a cheap proxy
for cross-sectional diversity (large/mid/small cap, multiple industries)
without needing sector metadata. Deterministic so re-running this script
samples the same 30 tickers.

Decision rule (per user instruction)
-------------------------------------
For each sampled ticker with overlapping (period_end) observations in both
the XBRL fetch (freshly forced) and the yfinance baseline, compute the
median absolute difference in gp_ratio across those overlapping periods.
A ticker "diverges" if that median exceeds _DIVERGENCE_THRESHOLD (10%).

  - If the diverging fraction (of tickers with any overlap to judge) exceeds
    _HALT_FRACTION (15%): HALT. Exit nonzero, print a report, do NOT write
    the low-confidence list or clear the way for the full pull — this
    pattern suggests a systematic tag-mapping issue worth fixing first, not
    a handful of noisy names.
  - Otherwise: PROCEED. Write the specific diverging tickers (however few)
    to data/gp/gp_preflight_divergent_tickers.txt, which
    factor_engine/gp_fundamentals.py reads to set low_confidence_vs_yfinance
    on every observation for those tickers during the full pull — flagged,
    not excluded, so a human can decide what to do with them later. Exit 0.

Run from the project root with the venv active:
    .venv/bin/python scripts/preflight_gp_xbrl.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from factor_engine.gp_fundamentals import _LOW_CONFIDENCE_LIST_PATH, fetch_ticker_fundamentals

_BASELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gp", "fundamentals_yfinance_backup")

_SAMPLE_SIZE = 30
_DIVERGENCE_THRESHOLD = 0.10   # median |gp_ratio diff| across a ticker's overlapping periods
_HALT_FRACTION = 0.15          # fraction of judged tickers exceeding the threshold that triggers a halt


def _pick_sample() -> list[str]:
    candidates = []
    for fname in sorted(os.listdir(_BASELINE_DIR)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(_BASELINE_DIR, fname)
        if os.path.getsize(path) > 100:   # cheap non-empty check before parsing
            candidates.append(fname[:-4])
    if len(candidates) <= _SAMPLE_SIZE:
        return candidates
    step = len(candidates) / _SAMPLE_SIZE
    return [candidates[int(i * step)] for i in range(_SAMPLE_SIZE)]


def _median_abs_diff(ticker: str, xbrl_df: pd.DataFrame) -> "float | None":
    baseline_path = os.path.join(_BASELINE_DIR, f"{ticker}.csv")
    baseline = pd.read_csv(baseline_path)
    if baseline.empty or xbrl_df.empty:
        return None
    merged = baseline[["period_end", "gp_ratio"]].merge(
        xbrl_df[["period_end", "gp_ratio"]], on="period_end", suffixes=("_yfinance", "_xbrl"),
    )
    if merged.empty:
        return None
    return float((merged["gp_ratio_xbrl"] - merged["gp_ratio_yfinance"]).abs().median())


def main() -> None:
    if not os.path.isdir(_BASELINE_DIR):
        print(f"  No yfinance baseline found at {_BASELINE_DIR} — nothing to preflight against.")
        sys.exit(1)

    sample = _pick_sample()
    print(f"[preflight_gp_xbrl] Sampled {len(sample)} tickers: {', '.join(sample)}\n")

    rows = []
    for i, ticker in enumerate(sample):
        print(f"  [preflight] fetching {ticker} ({i + 1}/{len(sample)})...", flush=True)
        xbrl_df = fetch_ticker_fundamentals(ticker, force=True)
        diff = _median_abs_diff(ticker, xbrl_df)
        rows.append({"ticker": ticker, "median_abs_diff": diff})
        if diff is None:
            print(f"    -> no overlapping periods to compare")
        else:
            flag = "  <-- DIVERGES" if diff > _DIVERGENCE_THRESHOLD else ""
            print(f"    -> median |diff| = {diff:.4f}{flag}")

    report = pd.DataFrame(rows)
    judged = report.dropna(subset=["median_abs_diff"])
    unjudged = report[report["median_abs_diff"].isna()]
    divergent = judged[judged["median_abs_diff"] > _DIVERGENCE_THRESHOLD]

    print()
    print("=" * 78)
    print("  Preflight summary")
    print("=" * 78)
    print(f"  Sampled          : {len(report)}")
    print(f"  Judged (overlap) : {len(judged)}")
    print(f"  No overlap data  : {len(unjudged)}  ({', '.join(unjudged['ticker']) if len(unjudged) else '-'})")
    if len(judged):
        frac = len(divergent) / len(judged)
        print(f"  Diverging (>{_DIVERGENCE_THRESHOLD:.0%} median diff) : {len(divergent)}/{len(judged)} ({frac:.1%})")
    else:
        frac = 0.0
        print("  No tickers could be judged — cannot evaluate the gate.")

    if len(divergent):
        print("\n  Diverging tickers:")
        print(divergent.sort_values("median_abs_diff", ascending=False).to_string(index=False))

    print()
    if len(judged) == 0:
        print("  HALT — no comparable data, cannot clear the gate.")
        sys.exit(1)
    elif frac > _HALT_FRACTION:
        print(f"  HALT — {frac:.1%} of judged tickers diverge, above the {_HALT_FRACTION:.0%} threshold.")
        print("  This pattern suggests a systematic tag-mapping issue — investigate before")
        print("  running the full pull. Not writing the low-confidence list; full pull not cleared.")
        sys.exit(1)
    else:
        os.makedirs(os.path.dirname(_LOW_CONFIDENCE_LIST_PATH), exist_ok=True)
        with open(_LOW_CONFIDENCE_LIST_PATH, "w") as f:
            for ticker in divergent["ticker"]:
                f.write(f"{ticker}\n")
        print(f"  PROCEED — {frac:.1%} of judged tickers diverge, within the {_HALT_FRACTION:.0%} threshold.")
        if len(divergent):
            print(f"  Wrote {len(divergent)} diverging ticker(s) to {_LOW_CONFIDENCE_LIST_PATH} "
                  "— the full pull will flag them as low_confidence_vs_yfinance.")
        print("  Full pull is cleared to run.")
        sys.exit(0)


if __name__ == "__main__":
    main()
