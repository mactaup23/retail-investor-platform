"""
Validation: EDGAR XBRL-derived GP fundamentals vs the prior yfinance-derived
cache, over their overlapping window (~2021-2025).

This does NOT re-derive trust in the XBRL pipeline by construction — it
checks the two independent data sources agree on the period both cover,
before the pre-2021 XBRL-only history (which yfinance can't corroborate at
all) is trusted for anything. See factor_engine/gp_fundamentals.py and
factor_engine/gp_xbrl_client.py for the fetch/derivation design this
validates.

Baseline: data/gp/fundamentals_yfinance_backup/{ticker}.csv — a copy of the
yfinance-era cache taken before the XBRL rebuild overwrote
data/gp/fundamentals/{ticker}.csv in place. Both share the same per-ticker
CSV schema (period_end, revenue, cogs, total_assets, gp_ratio, freq); the
new XBRL cache additionally carries a "source" column
(reported/estimated_from_margin) the yfinance-era files never had.

What this checks
-----------------
1. Per-(ticker, period_end) gp_ratio agreement in the overlap window:
   correlation and median absolute difference across all matched pairs.
2. Spot-check a handful of large, well-known names side by side
   (revenue/COGS/assets from both sources) — specifically watching for a
   units-scale bug (raw dollars vs thousands), since exactly that bug has
   bitten this project before in the 13F pipeline (see edgar.py /
   verify_value_units.py).
3. An explicit pass/fail gate — this script does not silently declare
   success; it prints a clear verdict and exits nonzero if the gate isn't
   cleared, so pre-2021 history should not be trusted without a human
   reading the result.

Run from the project root with the venv active, only after the XBRL rebuild
has populated data/gp/fundamentals/ (see scripts/run_gp_sanity.py, or a
direct fetch_universe_fundamentals(..., force=True) call):
    .venv/bin/python scripts/verify_gp_xbrl.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

_NEW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gp", "fundamentals")
_BASELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gp", "fundamentals_yfinance_backup")

# Gate thresholds — see module docstring point 3.
_MIN_CORRELATION = 0.95
_MAX_MEDIAN_ABS_DIFF = 0.10   # 10% of gp_ratio magnitude

_SPOT_CHECK_TICKERS = ["AAPL", "MSFT", "GOOGL", "XOM", "KR", "AMZN"]


def _load(dir_path: str, ticker: str) -> "pd.DataFrame | None":
    path = os.path.join(dir_path, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def _overlap_pairs(new_dir: str, baseline_dir: str) -> pd.DataFrame:
    """
    Match (ticker, period_end) rows present in both caches and return a
    DataFrame of [ticker, period_end, gp_ratio_xbrl, gp_ratio_yfinance].
    """
    tickers = sorted(
        f[:-4] for f in os.listdir(baseline_dir) if f.endswith(".csv")
    )
    rows = []
    for ticker in tickers:
        base = _load(baseline_dir, ticker)
        new = _load(new_dir, ticker)
        if base is None or new is None:
            continue
        merged = base[["period_end", "gp_ratio"]].merge(
            new[["period_end", "gp_ratio"]],
            on="period_end", suffixes=("_yfinance", "_xbrl"),
        )
        for _, r in merged.iterrows():
            rows.append({
                "ticker": ticker,
                "period_end": r["period_end"],
                "gp_ratio_yfinance": r["gp_ratio_yfinance"],
                "gp_ratio_xbrl": r["gp_ratio_xbrl"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    if not os.path.isdir(_BASELINE_DIR):
        print(f"  No yfinance baseline found at {_BASELINE_DIR} — nothing to validate against.")
        sys.exit(1)
    if not os.path.isdir(_NEW_DIR):
        print(f"  No XBRL cache found at {_NEW_DIR} — run the fundamentals rebuild first.")
        sys.exit(1)

    print("[verify_gp_xbrl] Matching overlapping (ticker, period_end) observations…\n")
    pairs = _overlap_pairs(_NEW_DIR, _BASELINE_DIR)

    if pairs.empty:
        print("  No overlapping observations found between the two caches.")
        sys.exit(1)

    diff = (pairs["gp_ratio_xbrl"] - pairs["gp_ratio_yfinance"]).abs()
    corr = pairs["gp_ratio_xbrl"].corr(pairs["gp_ratio_yfinance"])
    median_abs_diff = diff.median()

    print(f"  Overlapping observations : {len(pairs)}  ({pairs['ticker'].nunique()} tickers)")
    print(f"  Correlation (xbrl vs yfinance gp_ratio) : {corr:.4f}")
    print(f"  Median |diff|                            : {median_abs_diff:.4f}")
    print(f"  95th pct |diff|                          : {diff.quantile(0.95):.4f}\n")

    worst = pairs.assign(abs_diff=diff).sort_values("abs_diff", ascending=False).head(15)
    print("  Largest disagreements:")
    print(worst.to_string(index=False))
    print()

    print("=" * 78)
    print("  Spot-check: large well-known names, both sources side by side")
    print("=" * 78)
    for ticker in _SPOT_CHECK_TICKERS:
        base = _load(_BASELINE_DIR, ticker)
        new = _load(_NEW_DIR, ticker)
        print(f"\n  {ticker}")
        if base is None:
            print("    yfinance: no cached data")
        else:
            latest = base.sort_values("period_end").iloc[-1]
            print(f"    yfinance  {latest['period_end']}: revenue={latest['revenue']:,.0f} "
                  f"cogs={latest['cogs']:,.0f} assets={latest['total_assets']:,.0f} "
                  f"gp_ratio={latest['gp_ratio']:.4f}")
        if new is None:
            print("    xbrl:     no cached data")
        else:
            newest_overlap = new[new["period_end"].isin(base["period_end"])] if base is not None else new
            if newest_overlap is not None and not newest_overlap.empty:
                r = newest_overlap.sort_values("period_end").iloc[-1]
                print(f"    xbrl      {r['period_end']}: revenue={r['revenue']:,.0f} "
                      f"cogs={r['cogs']:,.0f} assets={r['total_assets']:,.0f} "
                      f"gp_ratio={r['gp_ratio']:.4f} source={r.get('source', '?')}")

    print()
    print("=" * 78)
    print("  Verdict")
    print("=" * 78)
    passed = corr >= _MIN_CORRELATION and median_abs_diff <= _MAX_MEDIAN_ABS_DIFF
    if passed:
        print(f"  PASS — correlation {corr:.4f} >= {_MIN_CORRELATION}, "
              f"median |diff| {median_abs_diff:.4f} <= {_MAX_MEDIAN_ABS_DIFF}")
        print("  Pre-2021 XBRL-only history can be trusted to extend the GP factor.")
    else:
        print(f"  FAIL — correlation {corr:.4f} (need >= {_MIN_CORRELATION}) or "
              f"median |diff| {median_abs_diff:.4f} (need <= {_MAX_MEDIAN_ABS_DIFF}) "
              "did not clear the gate.")
        print("  Do NOT trust the extended history yet — investigate the disagreements above first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
