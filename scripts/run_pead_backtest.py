"""
PEAD signal backtest — decision-gate run.

Pulls EPS surprise history + daily prices for the GP factor's cached S&P
Composite 1500 universe, computes the SUE panel, and reports Spearman IC
at 1-month (21 trading day) and 3-month (63 trading day) horizons — the
same output shape (IC, t-stat, hit rate, observation count) as the
existing 13F signal backtest (smart_money/backtest.py, surfaced via
scripts/verify_signal_backtest.py).

Decision gate (see CLAUDE.md): IC above ~0.02-0.03 with t-stat approaching
or exceeding 1.5-2.0 triggers investing in an EDGAR-sourced extension.
Flat or negative is a negative result, same as this project's other
diagnostic scripts (run_gp_sanity.py, run_ff4_plus_one_diagnostic.py).

Usage
-----
    .venv/bin/python scripts/run_pead_backtest.py            # full universe
    .venv/bin/python scripts/run_pead_backtest.py --sample 100
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from pead.backtest import run_backtest, summarize
from pead.prices import fetch_prices
from pead.signal import compute_sue
from pead.surprises import fetch_surprises
from pead.universe import get_universe_tickers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Limit to first N universe tickers (for a fast smoke test)")
    args = parser.parse_args()

    tickers = get_universe_tickers()
    if args.sample:
        tickers = tickers[: args.sample]
    log.info("Universe: %d tickers", len(tickers))

    log.info("Fetching EPS surprise history...")
    surprises = fetch_surprises(tickers)
    log.info("Surprise coverage: %d/%d tickers", len(surprises), len(tickers))

    log.info("Computing SUE panel...")
    panel = compute_sue(surprises)
    scored = panel[panel["score"].notna()]
    log.info(
        "Panel: %d rows (%d scored: %d sue, %d percentile, %d no_estimate excluded)",
        len(panel), len(scored),
        (panel["score_method"] == "sue").sum(),
        (panel["score_method"] == "percentile").sum(),
        (panel["score_method"] == "no_estimate").sum(),
    )

    start_date = panel["announcement_date"].min()
    end_date = datetime.date.today()
    log.info("Fetching prices from %s to %s...", start_date, end_date)
    prices = fetch_prices(sorted(panel["ticker"].unique()), start_date, end_date)
    log.info("Price coverage: %d/%d tickers", len(prices), panel["ticker"].nunique())

    log.info("Running backtest...")
    quarter_ics = run_backtest(panel, prices)
    summary = summarize(quarter_ics)

    print("\n" + "=" * 70)
    print("PEAD BACKTEST SUMMARY")
    print("=" * 70)
    for h in summary.horizons:
        label = "1-month" if h.horizon_days == 21 else "3-month" if h.horizon_days == 63 else f"{h.horizon_days}d"
        print(f"\nHorizon: {label} ({h.horizon_days} trading days)")
        print(f"  Quarters with computed IC : {h.n_quarters}")
        print(f"  Mean IC                   : {h.mean_ic:.4f}" if h.mean_ic is not None else "  Mean IC                   : n/a")
        print(f"  Std IC                    : {h.std_ic:.4f}" if h.std_ic is not None else "  Std IC                    : n/a")
        print(f"  t-stat                    : {h.t_stat:.2f}" if h.t_stat is not None else "  t-stat                    : n/a")
        print(f"  Hit rate                  : {h.hit_rate:.1%}" if h.hit_rate is not None else "  Hit rate                  : n/a")

    print("\nPer-cohort detail:")
    rows = [
        {
            "cohort": q.quarter_cohort,
            "horizon": q.horizon_days,
            "n_candidates": q.n_candidates,
            "n_obs": q.n_obs,
            "coverage_pct": round(q.coverage_pct, 3),
            "ic": round(q.ic, 4) if q.ic is not None else None,
        }
        for q in quarter_ics
    ]
    detail = pd.DataFrame(rows).sort_values(["horizon", "cohort"])
    print(detail.to_string(index=False))


if __name__ == "__main__":
    main()
