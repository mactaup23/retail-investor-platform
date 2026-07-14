"""
Unit tests for factor_engine/gp_fundamentals.py's XBRL extraction logic:
tag-priority resolution, duration/instant period matching, COGS gap-filling
via historical gross margin, and TTM source rollup. All deterministic given
synthetic XBRL fact dicts — no network calls.
"""

import pandas as pd
import pytest

from factor_engine.gp_fundamentals import (
    SOURCE_DERIVED,
    SOURCE_ESTIMATED,
    SOURCE_REPORTED,
    _build_observations,
    _build_quarterly_observations,
    _derive_q4_periods,
    _duration_periods,
    _instant_periods,
    _ttm_from_quarterly_rows,
)


def _fact(start=None, end=None, val=0.0, filed="2020-01-01"):
    d = {"end": end, "val": val, "filed": filed}
    if start is not None:
        d["start"] = start
    return d


# ── _duration_periods ───────────────────────────────────────────────────────

def test_duration_periods_higher_priority_tag_wins():
    # tag A (priority 0) and tag B (priority 1) both report the same period —
    # A's value must win regardless of filed date.
    tag_a = [_fact("2019-01-01", "2019-12-31", val=100.0, filed="2020-02-01")]
    tag_b = [_fact("2019-01-01", "2019-12-31", val=999.0, filed="2020-01-01")]
    resolved = _duration_periods([tag_a, tag_b], (350, 380))
    assert resolved[("2019-01-01", "2019-12-31")]["val"] == 100.0


def test_duration_periods_fills_gap_from_lower_priority_tag():
    tag_a = [_fact("2019-01-01", "2019-12-31", val=100.0)]
    tag_b = [_fact("2020-01-01", "2020-12-31", val=200.0)]   # different period, tag A silent here
    resolved = _duration_periods([tag_a, tag_b], (350, 380))
    assert resolved[("2019-01-01", "2019-12-31")]["val"] == 100.0
    assert resolved[("2020-01-01", "2020-12-31")]["val"] == 200.0


def test_duration_periods_earliest_filed_wins_within_same_tag():
    tag_a = [
        _fact("2019-01-01", "2019-12-31", val=100.0, filed="2020-03-01"),
        _fact("2019-01-01", "2019-12-31", val=105.0, filed="2020-02-01"),  # restatement filed earlier
    ]
    resolved = _duration_periods([tag_a], (350, 380))
    assert resolved[("2019-01-01", "2019-12-31")]["val"] == 105.0


def test_duration_periods_rejects_out_of_range_span():
    # 181-day span is neither a quarter nor a year — must be excluded.
    tag_a = [_fact("2019-01-01", "2019-07-01", val=100.0)]
    resolved = _duration_periods([tag_a], (350, 380))
    assert resolved == {}


def test_duration_periods_skips_entries_missing_start_or_val():
    tag_a = [{"end": "2019-12-31", "val": 100.0, "filed": "2020-01-01"}]  # no "start"
    resolved = _duration_periods([tag_a], (350, 380))
    assert resolved == {}


# ── _instant_periods ─────────────────────────────────────────────────────────

def test_instant_periods_higher_priority_tag_wins():
    tag_a = [_fact(end="2019-12-31", val=500.0)]
    tag_b = [_fact(end="2019-12-31", val=999.0)]
    resolved = _instant_periods([tag_a, tag_b])
    assert resolved["2019-12-31"]["val"] == 500.0


def test_instant_periods_excludes_duration_facts():
    # A duration-shaped fact (has "start") must never leak into instant resolution.
    tag_a = [_fact("2019-01-01", "2019-12-31", val=500.0)]
    resolved = _instant_periods([tag_a])
    assert resolved == {}


# ── _build_observations: reported vs estimated COGS ────────────────────────

def _us_gaap(revenue_entries, cogs_entries, assets_entries):
    return {
        "Revenues": {"units": {"USD": revenue_entries}},
        "CostOfRevenue": {"units": {"USD": cogs_entries}},
        "Assets": {"units": {"USD": assets_entries}},
    }


def test_build_observations_reported_when_cogs_tag_present():
    us_gaap = _us_gaap(
        revenue_entries=[_fact("2019-01-01", "2019-12-31", val=1000.0)],
        cogs_entries=[_fact("2019-01-01", "2019-12-31", val=600.0)],
        assets_entries=[_fact(end="2019-12-31", val=2000.0)],
    )
    obs = _build_observations(us_gaap, (350, 380), "A")
    assert len(obs) == 1
    row = obs[0]
    assert row["source"] == SOURCE_REPORTED
    assert row["revenue"] == 1000.0
    assert row["cogs"] == 600.0
    assert row["gp_ratio"] == pytest.approx((1000.0 - 600.0) / 2000.0)


def test_build_observations_estimates_cogs_from_median_margin():
    # Two reported periods at 40% margin (600/1000, 240/400), one gap period
    # with revenue=2000 and no COGS tag at all -> estimate cogs = 2000*0.6 = 1200.
    us_gaap = _us_gaap(
        revenue_entries=[
            _fact("2018-01-01", "2018-12-31", val=1000.0),
            _fact("2019-01-01", "2019-12-31", val=400.0),
            _fact("2020-01-01", "2020-12-31", val=2000.0),   # gap period
        ],
        cogs_entries=[
            _fact("2018-01-01", "2018-12-31", val=600.0),
            _fact("2019-01-01", "2019-12-31", val=240.0),
        ],
        assets_entries=[
            _fact(end="2018-12-31", val=5000.0),
            _fact(end="2019-12-31", val=5000.0),
            _fact(end="2020-12-31", val=5000.0),
        ],
    )
    obs = {o["period_end"]: o for o in _build_observations(us_gaap, (350, 380), "A")}
    assert obs["2018-12-31"]["source"] == SOURCE_REPORTED
    assert obs["2019-12-31"]["source"] == SOURCE_REPORTED
    gap = obs["2020-12-31"]
    assert gap["source"] == SOURCE_ESTIMATED
    assert gap["cogs"] == pytest.approx(2000.0 * 0.6)


def test_build_observations_skips_gap_period_with_fewer_than_two_reported_margins():
    # Only one reported margin observation exists -> not enough basis to
    # estimate (module requires >= 2) -> the gap period must be dropped, not fabricated.
    us_gaap = _us_gaap(
        revenue_entries=[
            _fact("2018-01-01", "2018-12-31", val=1000.0),
            _fact("2020-01-01", "2020-12-31", val=2000.0),   # gap period
        ],
        cogs_entries=[
            _fact("2018-01-01", "2018-12-31", val=600.0),
        ],
        assets_entries=[
            _fact(end="2018-12-31", val=5000.0),
            _fact(end="2020-12-31", val=5000.0),
        ],
    )
    obs = {o["period_end"]: o for o in _build_observations(us_gaap, (350, 380), "A")}
    assert "2018-12-31" in obs
    assert "2020-12-31" not in obs


def test_build_observations_skips_period_missing_assets():
    us_gaap = _us_gaap(
        revenue_entries=[_fact("2019-01-01", "2019-12-31", val=1000.0)],
        cogs_entries=[_fact("2019-01-01", "2019-12-31", val=600.0)],
        assets_entries=[],
    )
    assert _build_observations(us_gaap, (350, 380), "A") == []


# ── _ttm_from_quarterly_rows: source rollup ─────────────────────────────────

def _quarterly_row(period_end, source):
    return {
        "period_end": period_end, "revenue": 100.0, "cogs": 60.0,
        "total_assets": 1000.0, "gp_ratio": 0.04, "freq": "Q", "source": source,
    }


def test_ttm_source_is_reported_when_all_four_quarters_reported():
    rows = pd.DataFrame([
        _quarterly_row("2019-03-31", SOURCE_REPORTED),
        _quarterly_row("2019-06-30", SOURCE_REPORTED),
        _quarterly_row("2019-09-30", SOURCE_REPORTED),
        _quarterly_row("2019-12-31", SOURCE_REPORTED),
    ])
    ttm = _ttm_from_quarterly_rows(rows)
    assert len(ttm) == 1
    assert ttm[0]["source"] == SOURCE_REPORTED


def test_ttm_source_is_estimated_if_any_quarter_estimated():
    rows = pd.DataFrame([
        _quarterly_row("2019-03-31", SOURCE_REPORTED),
        _quarterly_row("2019-06-30", SOURCE_ESTIMATED),   # one estimated quarter taints the window
        _quarterly_row("2019-09-30", SOURCE_REPORTED),
        _quarterly_row("2019-12-31", SOURCE_REPORTED),
    ])
    ttm = _ttm_from_quarterly_rows(rows)
    assert len(ttm) == 1
    assert ttm[0]["source"] == SOURCE_ESTIMATED


def test_ttm_source_is_derived_if_worst_quarter_is_derived_not_estimated():
    # derived (exact YTD-subtraction) is worse than reported but better than
    # margin-estimated -- worst-tier rollup should land on DERIVED here, not
    # ESTIMATED, since no quarter in the window was actually margin-estimated.
    rows = pd.DataFrame([
        _quarterly_row("2019-03-31", SOURCE_REPORTED),
        _quarterly_row("2019-06-30", SOURCE_REPORTED),
        _quarterly_row("2019-09-30", SOURCE_REPORTED),
        _quarterly_row("2019-12-31", SOURCE_DERIVED),
    ])
    ttm = _ttm_from_quarterly_rows(rows)
    assert len(ttm) == 1
    assert ttm[0]["source"] == SOURCE_DERIVED


# ── _derive_q4_periods ───────────────────────────────────────────────────────

def test_derive_q4_periods_computes_fy_minus_ytd9():
    revenue_annual = {("2019-01-01", "2019-12-31"): {"val": 1000.0, "filed": "2020-02-01"}}
    cogs_annual = {("2019-01-01", "2019-12-31"): {"val": 600.0, "filed": "2020-02-01"}}
    revenue_ytd9 = {("2019-01-01", "2019-09-30"): {"val": 700.0, "filed": "2019-11-01"}}
    cogs_ytd9 = {("2019-01-01", "2019-09-30"): {"val": 420.0, "filed": "2019-11-01"}}

    derived_rev, derived_cogs = _derive_q4_periods(
        revenue_annual, cogs_annual, revenue_ytd9, cogs_ytd9, existing_quarter_ends=set(),
    )
    key = ("2019-09-30", "2019-12-31")
    assert derived_rev[key]["val"] == pytest.approx(300.0)   # 1000 - 700
    assert derived_cogs[key]["val"] == pytest.approx(180.0)  # 600 - 420


def test_derive_q4_periods_skips_when_quarter_already_directly_tagged():
    revenue_annual = {("2019-01-01", "2019-12-31"): {"val": 1000.0, "filed": "2020-02-01"}}
    cogs_annual = {("2019-01-01", "2019-12-31"): {"val": 600.0, "filed": "2020-02-01"}}
    revenue_ytd9 = {("2019-01-01", "2019-09-30"): {"val": 700.0, "filed": "2019-11-01"}}
    cogs_ytd9 = {("2019-01-01", "2019-09-30"): {"val": 420.0, "filed": "2019-11-01"}}

    derived_rev, _ = _derive_q4_periods(
        revenue_annual, cogs_annual, revenue_ytd9, cogs_ytd9,
        existing_quarter_ends={"2019-12-31"},   # a real quarter-span fact already covers this
    )
    assert derived_rev == {}


def test_derive_q4_periods_skips_when_ytd_fact_missing():
    revenue_annual = {("2019-01-01", "2019-12-31"): {"val": 1000.0, "filed": "2020-02-01"}}
    cogs_annual = {("2019-01-01", "2019-12-31"): {"val": 600.0, "filed": "2020-02-01"}}
    derived_rev, derived_cogs = _derive_q4_periods(
        revenue_annual, cogs_annual, revenue_ytd9={}, cogs_ytd9={}, existing_quarter_ends=set(),
    )
    assert derived_rev == {} and derived_cogs == {}


# ── _build_quarterly_observations: three-tier integration ──────────────────

def test_build_quarterly_observations_prefers_reported_over_derived():
    us_gaap = {
        "Revenues": {"units": {"USD": [
            _fact("2019-01-01", "2019-03-31", val=100.0),   # Q1, directly tagged
            _fact("2019-01-01", "2019-12-31", val=1000.0),  # FY
            _fact("2019-01-01", "2019-09-30", val=700.0),   # 9mo YTD
        ]}},
        "CostOfRevenue": {"units": {"USD": [
            _fact("2019-01-01", "2019-03-31", val=60.0),    # Q1, directly tagged
            _fact("2019-01-01", "2019-12-31", val=600.0),   # FY
            _fact("2019-01-01", "2019-09-30", val=420.0),   # 9mo YTD
        ]}},
        "Assets": {"units": {"USD": [
            _fact(end="2019-03-31", val=2000.0),
            _fact(end="2019-12-31", val=2000.0),
        ]}},
    }
    obs = {o["period_end"]: o for o in _build_quarterly_observations(us_gaap)}
    assert obs["2019-03-31"]["source"] == SOURCE_REPORTED
    assert obs["2019-12-31"]["source"] == SOURCE_DERIVED
    assert obs["2019-12-31"]["revenue"] == pytest.approx(300.0)
    assert obs["2019-12-31"]["cogs"] == pytest.approx(180.0)
