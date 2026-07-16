"""
Verification: PEAD data pull + SUE signal construction on a stratified sample.

Pulls a stratified sample of the GP factor's cached S&P Composite 1500
universe (data/gp/universe.csv — reused as-is, no re-scrape) spanning large,
mid, and small cap, plus a handful of tickers already spot-checked manually
during design (AAPL, NVDA, KR, HLX, HELE, HAYW — covering AMC and BMO
reporters, deep and shallow history).

Reports:
  - fetch coverage (how many sample tickers returned usable data)
  - BMO/AMC/unknown session split
  - SUE vs. percentile-fallback split, and why (n_prior_quarters distribution)
  - top absolute-SUE rows as a sanity spot-check

This is a data/signal check only — no prices, no backtest. Those come after
this step is reviewed.

Usage
-----
    .venv/bin/python scripts/verify_pead.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from pead.signal import MIN_QUARTERS, WINDOW, compute_sue
from pead.surprises import fetch_surprises
from pead.universe import get_universe

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

SAMPLE_PER_INDEX = 25
MANUAL_SPOT_CHECK = ["AAPL", "NVDA", "KR", "HLX", "HELE", "HAYW"]


def build_sample() -> list[str]:
    universe = get_universe()
    tickers: list[str] = []
    for source in ("sp500", "sp400", "sp600"):
        subset = universe[universe["index_source"] == source]["ticker"].tolist()
        tickers.extend(subset[:SAMPLE_PER_INDEX])
    for t in MANUAL_SPOT_CHECK:
        if t not in tickers:
            tickers.append(t)
    return tickers


def main() -> None:
    sample = build_sample()
    print(f"Sample size: {len(sample)} tickers ({SAMPLE_PER_INDEX} each from sp500/sp400/sp600 + {len(MANUAL_SPOT_CHECK)} manual spot-checks)")

    print("\nFetching EPS surprise history (yfinance, limit=50/ticker)...")
    surprises = fetch_surprises(sample)

    n_hit = len(surprises)
    n_miss = len(sample) - n_hit
    print(f"Coverage: {n_hit}/{len(sample)} tickers returned usable data ({n_miss} missing/failed)")
    if n_miss:
        missing = [t for t in sample if t not in surprises]
        print(f"Missing: {missing}")

    total_rows = sum(len(df) for df in surprises.values())
    avg_rows = total_rows / n_hit if n_hit else 0
    print(f"Total announcement rows: {total_rows} (avg {avg_rows:.1f} quarters/ticker)")

    session_counts = pd.concat(surprises.values())["session"].value_counts()
    print(f"\nSession classification across all rows:\n{session_counts.to_string()}")

    print(f"\nComputing SUE (window={WINDOW}q, min_quarters={MIN_QUARTERS})...")
    panel = compute_sue(surprises)
    print(f"Panel: {len(panel)} rows across {panel['ticker'].nunique()} tickers")

    method_counts = panel["score_method"].value_counts()
    print(f"\nScore method split:\n{method_counts.to_string()}")

    print("\nWhy tickers fall back to percentile (n_prior_quarters at fallback rows):")
    fallback = panel[panel["score_method"] == "percentile"]
    if not fallback.empty:
        print(fallback.groupby("ticker")["n_prior_quarters"].max().sort_values().to_string())

    no_est = panel[panel["score_method"] == "no_estimate"]
    print(f"\nRows with no consensus estimate at all (excluded from scoring): {len(no_est)}")
    if not no_est.empty:
        print(no_est[["ticker", "announcement_date", "eps_estimate", "eps_actual", "eps_surprise_pct"]].to_string(index=False))

    print("\nManual spot-check tickers — most recent 5 rows each:")
    for t in MANUAL_SPOT_CHECK:
        sub = panel[panel["ticker"] == t].sort_values("announcement_date")
        if sub.empty:
            print(f"  {t}: no data")
            continue
        print(f"\n  {t}:")
        print(sub.tail(5)[["announcement_date", "session", "eps_surprise_pct", "eps_surprise_dollar", "n_prior_quarters", "score", "score_method"]].to_string(index=False))

    print("\nTop 15 |score| rows among SUE-scored observations (sanity check — should be recognizable large beats/misses):")
    sue_only = panel[panel["score_method"] == "sue"].copy()
    sue_only["abs_score"] = sue_only["score"].abs()
    top = sue_only.sort_values("abs_score", ascending=False).head(15)
    print(top[["ticker", "announcement_date", "eps_surprise_pct", "eps_surprise_dollar", "n_prior_quarters", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
