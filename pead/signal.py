"""
PEAD signal construction — SUE (Standardized Unexpected Earnings).

Grounded in the specific academic construction (Bernard & Thomas, 1989 and
1990), not an ad hoc standardization — same rationale as grounding the GP
factor in Novy-Marx's gross-profitability construction rather than inventing
a bespoke ratio.

    SUE_t = (eps_surprise_dollar_t - mean(prior window)) / stddev(prior window)

Standardization is applied to the *dollar* EPS surprise (actual - estimate),
not the percentage surprise the fetch layer also reports. Percentage
surprise is unstable for near-zero consensus estimates — observed directly
in the yfinance pull (e.g. HLX quarters with a $0.02-0.03 estimate swing
between -157% and +256% surprise) — which is exactly the instability SUE is
designed to avoid by scaling against the stock's own historical surprise
distribution instead of dividing by a small, noisy denominator.

The mean is subtracted (not just a raw ratio to the standard deviation) to
net out a per-company systematic bias in analyst estimates — e.g. a company
that reliably beats by a couple of cents due to conservative guidance should
not register a "surprise" every quarter merely for being consistent.

Window and minimum history
---------------------------
WINDOW = 8 trailing quarters (2 years) — the standard rolling window in the
SUE literature; using an unbounded expanding window instead would blend a
young company's higher-volatility early quarters with its more mature,
lower-volatility recent ones.

MIN_QUARTERS = 4 (not the more conventional 8) — deliberately loosened for
this yfinance-based first pass, where shallow history is already an
accepted limitation. Requiring 8 would needlessly shrink universe coverage
before the signal's predictive value is even established. A future
EDGAR-sourced build (if the IC decision gate triggers it) can revisit this
back up toward 8 once deeper, more reliable history is available.

Percentile fallback
--------------------
Tickers with fewer than MIN_QUARTERS of *valid* prior surprises in the
trailing window (recent IPOs, sparse coverage) have no history to
standardize against and cannot get a SUE score. These fall back to a
cross-sectional percentile rank of eps_surprise_pct among all tickers with
a genuine estimate reporting in the same calendar-quarter cohort, so they
stay in the universe rather than being dropped outright — at the
deliberate cost of falling back to the noisier percentage-surprise measure
for exactly the names where it's least reliable (thin history). Not a
hidden tradeoff: score_method records which path each row took.

Rows with no consensus estimate at all ("no_estimate") get no score and
are excluded from the percentile cohort entirely — see data-quality note
below.

Two data-quality issues found empirically and handled explicitly
------------------------------------------------------------------
1. yfinance's displayed EPS Estimate / Reported EPS are rounded to 2
   decimals, while its own Surprise(%) is evidently computed from
   unrounded internal values. For low-EPS stocks this rounding can make
   eps_surprise_dollar read as exactly 0.00 across an entire trailing
   window even though Surprise(%) shows real, nonzero swings (observed
   directly: NVDA 2014-2016 shows eps_estimate == eps_actual == 0.01 for
   8 consecutive quarters while Surprise(%) ranges 10-67%). This produces
   a zero-variance window -> std guard below sends it to percentile
   fallback automatically, which is the correct behavior, but it means
   dollar-surprise SUE is systematically less reliable for low-priced /
   low-EPS names than for others.
2. A minority of rows have eps_estimate == null (no consensus existed)
   but yfinance's Surprise(%) is populated anyway with an implausible
   value (observed: AMD 2016-10-20 and 2017-07-25 show 855% and 7900%,
   clearly an artifact of dividing by a near-zero or missing implied
   estimate). These rows are excluded from scoring altogether
   ("no_estimate") rather than being allowed into the percentile fallback,
   where a bogus 7900% would otherwise misrepresent that observation as
   an extreme surprise.

No look-ahead
-------------
The trailing window for row i is strictly the announcements before it,
with any unscoreable (no-estimate) prior rows dropped before computing
the window's mean/std — n_prior_quarters counts only valid observations
that actually fed the statistic, not raw row count.
"""

from __future__ import annotations

import datetime
import logging

import pandas as pd

log = logging.getLogger(__name__)

WINDOW = 8         # trailing quarters used to estimate mean/stddev
MIN_QUARTERS = 4   # minimum prior quarters required to compute a SUE score


def _quarter_bucket(d: datetime.date) -> str:
    """Calendar-quarter cohort label for cross-sectional grouping, e.g. '2024Q3'."""
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"


def compute_sue(surprises: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute SUE (or percentile-rank fallback) for every observed earnings
    announcement across the universe.

    Parameters
    ----------
    surprises : dict[ticker, DataFrame]
        Output of pead.surprises.fetch_surprises() — one row per historical
        announcement per ticker, with columns eps_estimate, eps_actual,
        eps_surprise_pct, session, announcement_date.

    Returns
    -------
    pd.DataFrame, one row per (ticker, announcement_date), columns:
        ticker, announcement_date, session, eps_estimate, eps_actual,
        eps_surprise_pct, eps_surprise_dollar, quarter_cohort,
        n_prior_quarters, score, score_method ("sue" | "percentile")
    """
    rows = []
    for ticker, df in surprises.items():
        df = df.sort_values("announcement_date").reset_index(drop=True)
        has_estimate = df["eps_estimate"].notna() & df["eps_actual"].notna()
        surprise_dollar = (df["eps_actual"] - df["eps_estimate"]).where(has_estimate)

        for i in range(len(df)):
            row = df.iloc[i]
            window = surprise_dollar.iloc[max(0, i - WINDOW):i].dropna()   # strictly before this announcement
            n_prior = len(window)
            current = surprise_dollar.iloc[i]

            sue = None
            if pd.notna(current) and n_prior >= MIN_QUARTERS:
                std = window.std(ddof=1)
                if std and std > 0:
                    sue = (current - window.mean()) / std

            rows.append({
                "ticker": ticker,
                "announcement_date": row["announcement_date"],
                "session": row["session"],
                "eps_estimate": row["eps_estimate"],
                "eps_actual": row["eps_actual"],
                "eps_surprise_pct": row["eps_surprise_pct"],
                "eps_surprise_dollar": current,
                "has_estimate": bool(has_estimate.iloc[i]),
                "quarter_cohort": _quarter_bucket(row["announcement_date"]),
                "n_prior_quarters": n_prior,
                "sue": sue,
            })

    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel

    panel["score"] = panel["sue"]
    panel["score_method"] = "sue"

    no_estimate = ~panel["has_estimate"]
    panel.loc[no_estimate, "score_method"] = "no_estimate"

    needs_fallback = panel["sue"].isna() & panel["has_estimate"]
    if needs_fallback.any():
        # Rank only among rows with a genuine estimate, so no_estimate rows'
        # implausible Surprise(%) values never enter the cohort's ranking.
        valid = panel[panel["has_estimate"]]
        pct_rank = valid.groupby("quarter_cohort")["eps_surprise_pct"].rank(pct=True)
        panel.loc[needs_fallback, "score"] = pct_rank[needs_fallback]
        panel.loc[needs_fallback, "score_method"] = "percentile"

    return panel.drop(columns=["sue", "has_estimate"])
