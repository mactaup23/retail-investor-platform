"""
Unit tests for pead/signal.py's SUE (Standardized Unexpected Earnings)
construction — the rolling-window cap, the MIN_QUARTERS floor, the
no-look-ahead property, and the no_estimate / percentile-fallback paths.

All synthetic, deterministic, no network calls (mirrors the style of
tests/test_gp_factor.py).
"""

import datetime

import pandas as pd
import pytest

from pead.signal import MIN_QUARTERS, WINDOW, compute_sue


def _surprises_df(ticker: str, eps_estimates: list[float | None], eps_actuals: list[float | None]) -> pd.DataFrame:
    """Build a synthetic quarterly surprise history, one row per quarter starting 2015Q1."""
    n = len(eps_estimates)
    dates = [datetime.date(2015 + i // 4, 3 * (i % 4) + 1, 15) for i in range(n)]
    eps_surprise_pct = [
        ((a - e) / abs(e) * 100) if (e not in (None, 0) and a is not None) else None
        for e, a in zip(eps_estimates, eps_actuals)
    ]
    return pd.DataFrame({
        "ticker": ticker,
        "announcement_date": dates,
        "session": ["amc"] * n,
        "eps_estimate": eps_estimates,
        "eps_actual": eps_actuals,
        "eps_surprise_pct": eps_surprise_pct,
    })


# ── window cap (WINDOW=8, not expanding to full history) ────────────────────

def test_sue_window_is_capped_not_expanding():
    # 4 "regime A" quarters with dollar surprise ~0 (tight), then 8 "regime B"
    # quarters with dollar surprise ~0 but higher variance, then a 13th quarter
    # with surprise_dollar = 1.0. If the window were an unbounded expanding
    # window, all 12 prior quarters (mixed regimes) would feed the std. With
    # the WINDOW=8 cap, only the most recent 8 (regime B) should feed it.
    assert WINDOW == 8
    regime_a = [0.0, 0.01, -0.01, 0.0]                      # 4 quarters, tiny variance
    regime_b = [0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2]  # 8 quarters, larger variance
    estimates = [1.0] * 12
    actuals = [1.0 + s for s in regime_a + regime_b]
    df = {"T": _surprises_df("T", estimates + [1.0], actuals + [2.0])}   # 13th quarter: surprise_dollar=1.0

    panel = compute_sue(df)
    last_row = panel.iloc[-1]
    assert last_row["n_prior_quarters"] == 8   # capped at WINDOW, not 12

    # Manually compute SUE using only the trailing 8 (regime B) surprises.
    expected_mean = sum(regime_b) / 8
    expected_std = pd.Series(regime_b).std(ddof=1)
    expected_sue = (1.0 - expected_mean) / expected_std
    assert last_row["score"] == pytest.approx(expected_sue, abs=1e-9)


# ── MIN_QUARTERS floor ───────────────────────────────────────────────────────

def test_min_quarters_floor_blocks_scoring_below_threshold():
    assert MIN_QUARTERS == 4
    estimates = [1.0] * 4
    actuals = [1.1, 0.9, 1.2, 1.05]   # 3 prior quarters before the 4th (index 3) -> n_prior=3 < 4
    df = {"T": _surprises_df("T", estimates, actuals)}
    panel = compute_sue(df)

    row_at_index_3 = panel.iloc[3]
    assert row_at_index_3["n_prior_quarters"] == 3
    assert row_at_index_3["score_method"] == "percentile"   # not enough history for SUE


def test_min_quarters_floor_allows_scoring_at_threshold():
    estimates = [1.0] * 5
    actuals = [1.1, 0.9, 1.2, 1.05, 1.3]   # 4 prior quarters before index 4 -> n_prior=4 == MIN_QUARTERS
    df = {"T": _surprises_df("T", estimates, actuals)}
    panel = compute_sue(df)

    row_at_index_4 = panel.iloc[4]
    assert row_at_index_4["n_prior_quarters"] == 4
    assert row_at_index_4["score_method"] == "sue"


# ── no look-ahead ─────────────────────────────────────────────────────────────

def test_current_quarter_surprise_excluded_from_its_own_window():
    # If the current quarter's huge surprise leaked into its own baseline,
    # the SUE score would be artificially small (self-cancelling). Confirm
    # it isn't: an extreme final surprise should still produce a large |SUE|.
    estimates = [1.0] * 6
    actuals = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0]   # first 5 flat, 6th is a huge beat
    df = {"T": _surprises_df("T", estimates, actuals)}
    panel = compute_sue(df)

    last_row = panel.iloc[-1]
    # Prior window (all zeros) has std=0 -> SUE undefined -> falls to percentile,
    # which is itself proof the huge surprise didn't get folded into its own
    # baseline (a std computed *including* the 4.0 surprise would be nonzero).
    assert last_row["n_prior_quarters"] == 5
    assert last_row["score_method"] == "percentile"


# ── no_estimate exclusion ────────────────────────────────────────────────────

def test_no_estimate_rows_excluded_from_scoring_and_from_percentile_pool():
    # T1 has a null estimate in its most recent quarter (Yahoo sometimes
    # reports an implausible Surprise(%) here, e.g. 7900%) -- must not be
    # scored, and must not pollute T2's percentile rank in the same cohort.
    t1 = _surprises_df("T1", [1.0, 1.0, None], [1.0, 1.0, 5.0])
    t1["eps_surprise_pct"] = [0.0, 0.0, 9999.0]   # implausible value yfinance sometimes emits
    t2 = _surprises_df("T2", [1.0, 1.0, 1.0], [1.0, 1.0, 1.1])

    panel = compute_sue({"T1": t1, "T2": t2})

    t1_last = panel[(panel["ticker"] == "T1")].iloc[-1]
    assert t1_last["score_method"] == "no_estimate"
    assert pd.isna(t1_last["score"])

    # T2's percentile rank (also insufficient history -> fallback) must be
    # computed only against valid-estimate rows -- with T1's bogus 9999%
    # excluded, T2 should rank as the (only) valid observation in its cohort.
    t2_last = panel[(panel["ticker"] == "T2")].iloc[-1]
    assert t2_last["score_method"] == "percentile"
    assert t2_last["score"] == 1.0   # sole valid observation in its quarter_cohort -> 100th percentile


# ── std==0 degenerate window (e.g. rounding-precision artifact) ─────────────

def test_zero_variance_window_falls_back_to_percentile_not_divide_by_zero():
    estimates = [0.01] * 5
    actuals = [0.01, 0.01, 0.01, 0.01, 0.02]   # 4 identical prior quarters (std=0), then a real beat
    df = {"T": _surprises_df("T", estimates, actuals)}
    panel = compute_sue(df)

    last_row = panel.iloc[-1]
    assert last_row["n_prior_quarters"] == 4
    assert last_row["score_method"] == "percentile"   # std=0 guard, not a crash or inf/nan leak
