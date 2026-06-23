"""
Quarterly return reconstruction for Module 3 — 13F Smart-Money Positioning & Skill Tracker.

Public interface
----------------
    reconstruct_fund_quarter(fund, current_period, *, coverage_threshold)
        → FundQuarterReturn | None

    reconstruct_all_quarters(fund, *, coverage_threshold)
        → list[FundQuarterReturn]

Algorithm (Grinblatt-Titman buy-and-hold approximation)
-------------------------------------------------------
Uses prior-quarter (BOQ) 13F holdings as weights for the current quarter's return:

    R_fund(q) = Σ  w_i(q-1)  ×  r_i(q)

where:
    w_i(q-1)  = value_usd of holding i in the prior 13F filing (long equity only),
                 divided by total such value in that filing.

    r_i(q)    = adj_close at EOQ  /  adj_close at BOQ  −  1
                 (split- and dividend-adjusted closing prices from PriceCache)

Only is_price_eligible=True holdings with put_call=None are included.  Options
(put_call='Put' or 'Call') are excluded because PriceCache holds equity prices
only, not option premiums.

Missing prices
--------------
Holdings with no cached adj_close at either quarter-end boundary are excluded.
The remaining weights are renormalized to sum to 1 over the covered subset.
The pre-renorm covered weight fraction (by BOQ portfolio value) is recorded as
coverage_pct.

Price boundary lookup
---------------------
Quarter-end dates can fall on market holidays or weekends.  For each boundary,
_adj_close_near returns the adj_close of the last trading day on or before the
target date, searching within _PRICE_TOLERANCE calendar days.

Coverage gate
-------------
is_valid=True iff coverage_pct >= coverage_threshold (default COVERAGE_THRESHOLD=0.80).
The downstream skill decomposition must consume only is_valid=True rows.

First-filing rule
-----------------
A fund's first 13F establishes BOQ weights for the following quarter's return
but produces no return of its own — there is no prior quarter to weight against.
reconstruct_fund_quarter() returns None for a fund's earliest filing period.

DB contract
-----------
This module is pure computation.  It never calls init_db(); the caller is
responsible for initialising the database before invoking any function here.
"""

import datetime
from typing import TypedDict

from smart_money.models import Filing, Fund, Holding, PriceCache

COVERAGE_THRESHOLD = 0.80   # minimum covered fraction to mark is_valid
_PRICE_TOLERANCE   = 5      # calendar days to search for a nearest trading-day price


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

class FundQuarterReturn(TypedDict):
    fund_cik:              str
    fund_name:             str
    quarter:               str      # "2026Q1"
    period_start:          str      # "2025-12-31" — BOQ date (prior filing)
    period_end:            str      # "2026-03-31" — EOQ date (current filing)
    reconstructed_return:  float    # weighted return on the covered subset
    coverage_pct:          float    # fraction of BOQ portfolio value with prices, 0–1
    n_holdings_total:      int      # price-eligible long holdings in the prior filing
    n_holdings_with_price: int      # subset with valid adj_close at both boundaries
    is_valid:              bool     # coverage_pct >= coverage_threshold


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _quarter_label(d: datetime.date) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def _get_filing(fund: Fund, period: datetime.date) -> Filing | None:
    """Return the canonical (latest-amendment) Filing for (fund, period_of_report)."""
    return (
        Filing.select()
        .where(Filing.fund == fund, Filing.period_of_report == period)
        .order_by(Filing.filed_date.desc())
        .first()
    )


def _get_canonical_filings(fund: Fund) -> list[Filing]:
    """
    Return one Filing per period_of_report for the fund, keeping the latest
    amendment per period, sorted by period ascending.
    """
    rows = (
        Filing.select()
        .where(Filing.fund == fund)
        .order_by(Filing.period_of_report, Filing.filed_date)
    )
    by_period: dict[datetime.date, Filing] = {}
    for f in rows:
        by_period[f.period_of_report] = f   # later filed_date overwrites earlier
    return sorted(by_period.values(), key=lambda f: f.period_of_report)


def _eligible_holdings(filing: Filing) -> dict[str, float]:
    """
    Return {cusip: total_value_usd} for long equity holdings in a filing.

    Filtered to is_price_eligible=True and put_call=None.  Multiple rows
    sharing a CUSIP (sole/shared-discretion tranches) are summed.
    """
    agg: dict[str, float] = {}
    for h in (
        Holding.select()
        .where(
            Holding.filing == filing,
            Holding.is_price_eligible == True,
            Holding.put_call.is_null(),
        )
    ):
        agg[h.cusip] = agg.get(h.cusip, 0.0) + float(h.value_usd)
    return agg


def _adj_close_near(cusip: str, target: datetime.date) -> float | None:
    """
    Return the adj_close for the last trading day <= target within _PRICE_TOLERANCE days.

    Uses PriceCache.security_id directly (which is the CUSIP, the Security PK)
    to avoid a Security table join.  Returns None when no price row is found.
    """
    window_start = target - datetime.timedelta(days=_PRICE_TOLERANCE)
    row = (
        PriceCache.select(PriceCache.adj_close)
        .where(
            PriceCache.security_id == cusip,
            PriceCache.date.between(window_start, target),
        )
        .order_by(PriceCache.date.desc())
        .first()
    )
    return float(row.adj_close) if row else None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_quarter_return(
    boq_holdings: dict[str, float],    # {cusip: value_usd} from prior filing
    boq_date: datetime.date,
    eoq_date: datetime.date,
    *,
    coverage_threshold: float,
) -> tuple[float, float, int, int, bool]:
    """
    Buy-and-hold quarterly return over the covered subset of boq_holdings.

    Returns (reconstructed_return, coverage_pct, n_total, n_with_price, is_valid).

    reconstructed_return is the renormalized weighted average:
        Σ (w_i / covered_weight_sum) × r_i
    which equals Σ value_i × r_i / Σ value_i over the covered subset.

    Weights are BOQ value_usd (from 13F); returns are adj_close ratios.
    """
    total_boq_value = sum(boq_holdings.values())
    if total_boq_value == 0.0:
        return 0.0, 0.0, 0, 0, False

    covered_value    = 0.0
    weighted_return  = 0.0
    n_with_price     = 0

    for cusip, value in boq_holdings.items():
        p_boq = _adj_close_near(cusip, boq_date)
        p_eoq = _adj_close_near(cusip, eoq_date)
        if p_boq is None or p_eoq is None or p_boq <= 0.0:
            continue
        r = p_eoq / p_boq - 1.0
        covered_value   += value
        weighted_return += value * r
        n_with_price    += 1

    coverage_pct  = covered_value / total_boq_value
    reconstructed = weighted_return / covered_value if covered_value > 0.0 else 0.0
    is_valid      = coverage_pct >= coverage_threshold

    return reconstructed, coverage_pct, len(boq_holdings), n_with_price, is_valid


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def reconstruct_fund_quarter(
    fund: Fund,
    current_period: datetime.date,
    *,
    coverage_threshold: float = COVERAGE_THRESHOLD,
) -> "FundQuarterReturn | None":
    """
    Reconstruct the fund's return for the quarter ending on current_period.

    Returns None when current_period is the fund's first filing in the DB —
    that filing provides BOQ weights for the next quarter only.

    Parameters
    ----------
    fund : Fund
        Peewee Fund model instance.
    current_period : datetime.date
        Quarter-end date of the current (EOQ) 13F filing.
    coverage_threshold : float
        Minimum covered portfolio fraction to mark is_valid=True.

    Raises
    ------
    ValueError
        When no filing for (fund, current_period) exists in the DB.
    """
    current_filing = _get_filing(fund, current_period)
    if current_filing is None:
        raise ValueError(f"No filing for {fund.name!r} period {current_period}")

    prior_row = (
        Filing.select(Filing.period_of_report)
        .where(Filing.fund == fund, Filing.period_of_report < current_period)
        .order_by(Filing.period_of_report.desc())
        .first()
    )
    if prior_row is None:
        return None     # first filing — weights only, no prior quarter

    prior_period  = prior_row.period_of_report
    prior_filing  = _get_filing(fund, prior_period)
    boq_holdings  = _eligible_holdings(prior_filing)

    reconstructed, coverage_pct, n_total, n_with_price, is_valid = _compute_quarter_return(
        boq_holdings, prior_period, current_period,
        coverage_threshold=coverage_threshold,
    )

    return FundQuarterReturn(
        fund_cik              = fund.cik or "",
        fund_name             = fund.name,
        quarter               = _quarter_label(current_period),
        period_start          = prior_period.isoformat(),
        period_end            = current_period.isoformat(),
        reconstructed_return  = round(reconstructed, 6),
        coverage_pct          = round(coverage_pct, 6),
        n_holdings_total      = n_total,
        n_holdings_with_price = n_with_price,
        is_valid              = is_valid,
    )


def reconstruct_all_quarters(
    fund: Fund,
    *,
    coverage_threshold: float = COVERAGE_THRESHOLD,
) -> "list[FundQuarterReturn]":
    """
    Reconstruct returns for all quarters where the fund has at least two
    consecutive filings in the DB.

    The first filing (weights-only) produces no entry.  Quarters below the
    coverage threshold are included with is_valid=False.

    Returns results sorted by period_end ascending.
    """
    canonical = _get_canonical_filings(fund)
    if len(canonical) < 2:
        return []

    results: list[FundQuarterReturn] = []
    for i in range(1, len(canonical)):
        prior_filing   = canonical[i - 1]
        current_filing = canonical[i]
        boq_date       = prior_filing.period_of_report
        eoq_date       = current_filing.period_of_report
        boq_holdings   = _eligible_holdings(prior_filing)

        reconstructed, coverage_pct, n_total, n_with_price, is_valid = _compute_quarter_return(
            boq_holdings, boq_date, eoq_date,
            coverage_threshold=coverage_threshold,
        )

        results.append(FundQuarterReturn(
            fund_cik              = fund.cik or "",
            fund_name             = fund.name,
            quarter               = _quarter_label(eoq_date),
            period_start          = boq_date.isoformat(),
            period_end            = eoq_date.isoformat(),
            reconstructed_return  = round(reconstructed, 6),
            coverage_pct          = round(coverage_pct, 6),
            n_holdings_total      = n_total,
            n_holdings_with_price = n_with_price,
            is_valid              = is_valid,
        ))

    return results
