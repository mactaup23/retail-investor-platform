"""
Compute the high_confidence_pre_2021 ticker list for the GP factor's
XBRL-derived fundamentals.

Unlike scripts/preflight_gp_xbrl.py (a fast 30-ticker sample check run
BEFORE the full pull, using median |diff| as a halt/proceed gate), this
computes per-ticker Pearson correlation between XBRL-derived and
yfinance-derived gp_ratio across EVERY available overlapping observation
(2021-2025) for the FULL universe, run AFTER the pull completes. A ticker
needs at least 3 overlapping observations to compute a meaningful
correlation; fewer than that (or no overlap at all) leaves it unflagged
(conservative — absence means "not yet confirmed", not "confirmed good").

Excluded tickers (factor_engine/gp_exclusions.py) are skipped entirely —
they won't be in the factor regardless of this flag, so computing a
correlation for them is moot.

Threshold: ticker-level correlation >= 0.7 (deliberately lower than the
aggregate cross-sectional gate's 0.95 in scripts/verify_gp_xbrl.py) —
per-ticker time series here typically have only 4-6 overlapping annual
observations, a much smaller and noisier sample than the ~2900 pooled
observations the aggregate gate is computed over, so a per-ticker
correlation of 0.7 is a comparably meaningful bar given the sample size.

Writes data/gp/gp_high_confidence_pre2021_tickers.txt, read by
factor_engine/gp_fundamentals.py's _load_high_confidence_tickers().
A full re-fetch (force=True) is needed afterward for the flag to actually
appear in the per-ticker CSVs, since it's stamped at fetch time.

Run from the project root with the venv active:
    .venv/bin/python scripts/compute_gp_high_confidence.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from factor_engine.gp_exclusions import EXCLUDED_TICKERS, drop_implausible_observations

_NEW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gp", "fundamentals")
_BASELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gp", "fundamentals_yfinance_backup")
_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gp", "gp_high_confidence_pre2021_tickers.txt")

_MIN_OVERLAP_OBS = 3
_CORRELATION_THRESHOLD = 0.7


def main() -> None:
    if not os.path.isdir(_BASELINE_DIR) or not os.path.isdir(_NEW_DIR):
        print("  Missing baseline or new fundamentals directory — run the full pull first.")
        sys.exit(1)

    rows = []
    n_excluded = 0
    n_insufficient = 0
    for fname in sorted(os.listdir(_BASELINE_DIR)):
        if not fname.endswith(".csv"):
            continue
        ticker = fname[:-4]
        if ticker in EXCLUDED_TICKERS:
            n_excluded += 1
            continue

        base = pd.read_csv(os.path.join(_BASELINE_DIR, fname))
        newp = os.path.join(_NEW_DIR, fname)
        if base.empty or not os.path.exists(newp):
            continue
        new = drop_implausible_observations(pd.read_csv(newp))
        if new.empty:
            continue

        merged = base[["period_end", "gp_ratio"]].merge(
            new[["period_end", "gp_ratio"]], on="period_end", suffixes=("_yf", "_xbrl"),
        )
        if len(merged) < _MIN_OVERLAP_OBS:
            n_insufficient += 1
            continue

        corr = merged["gp_ratio_xbrl"].corr(merged["gp_ratio_yf"])
        if pd.isna(corr):
            n_insufficient += 1
            continue
        rows.append({"ticker": ticker, "n_obs": len(merged), "correlation": corr})

    report = pd.DataFrame(rows).sort_values("correlation")
    high_confidence = report[report["correlation"] >= _CORRELATION_THRESHOLD]

    print(f"  Tickers excluded (skipped)              : {n_excluded}")
    print(f"  Tickers with < {_MIN_OVERLAP_OBS} overlap obs (skipped) : {n_insufficient}")
    print(f"  Tickers judged                          : {len(report)}")
    print(f"  High confidence (corr >= {_CORRELATION_THRESHOLD})           : {len(high_confidence)}")
    print()
    print("  Lowest-correlation judged tickers (for spot-checking):")
    print(report.head(15).to_string(index=False))

    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
    with open(_OUTPUT_PATH, "w") as f:
        for ticker in sorted(high_confidence["ticker"]):
            f.write(f"{ticker}\n")
    print(f"\n  Wrote {len(high_confidence)} high-confidence ticker(s) to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
