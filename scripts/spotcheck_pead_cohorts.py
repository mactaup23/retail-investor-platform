"""
Spot-check: pull real per-stock observations (score vs. forward return) for a
handful of PEAD backtest cohorts, to see the signal on specific cases rather
than only the aggregate IC.

Reuses fully cached data from scripts/run_pead_backtest.py's prior full run
(no network calls -- all surprises and prices already in data/pead/).

Usage
-----
    .venv/bin/python scripts/spotcheck_pead_cohorts.py
"""

import datetime
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from pead.backtest import compute_cohort_ic
from pead.prices import fetch_prices
from pead.signal import compute_sue
from pead.surprises import fetch_surprises
from pead.universe import get_universe_tickers

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

TARGET_COHORTS = ["2020Q1", "2020Q2", "2024Q3", "2025Q1"]
HORIZON = 21   # 1-month
N_EACH_SIDE = 5


def main() -> None:
    tickers = get_universe_tickers()
    surprises = fetch_surprises(tickers)   # served from cache
    panel = compute_sue(surprises)
    scored = panel[panel["score"].notna()]

    start_date = panel["announcement_date"].min()
    end_date = datetime.date.today()
    prices = fetch_prices(sorted(panel["ticker"].unique()), start_date, end_date)   # served from cache

    for cohort in TARGET_COHORTS:
        q = compute_cohort_ic(cohort, HORIZON, scored, prices)
        print("\n" + "=" * 70)
        print(f"Cohort {cohort} — {HORIZON}d (1-month) horizon")
        print(f"  n_candidates={q.n_candidates}  n_obs={q.n_obs}  coverage={q.coverage_pct:.1%}  IC={q.ic:.4f}" if q.ic is not None else f"  n_candidates={q.n_candidates}  n_obs={q.n_obs}  IC=n/a")

        obs = sorted(q.observations, key=lambda o: o.score, reverse=True)
        top = obs[:N_EACH_SIDE]
        bottom = obs[-N_EACH_SIDE:]

        rows = []
        for o in top + bottom:
            rows.append({
                "ticker": o.ticker,
                "announcement_date": o.announcement_date,
                "score": round(o.score, 3),
                "fwd_return_1mo": f"{o.forward_return:+.1%}",
            })
        df = pd.DataFrame(rows)
        print(f"\n  Top {N_EACH_SIDE} scores (strongest positive surprise):")
        print(df.iloc[:N_EACH_SIDE].to_string(index=False))
        print(f"\n  Bottom {N_EACH_SIDE} scores (strongest negative surprise):")
        print(df.iloc[N_EACH_SIDE:].to_string(index=False))


if __name__ == "__main__":
    main()
