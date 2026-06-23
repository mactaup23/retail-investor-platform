"""
Quarter-over-quarter position change detector for Module 3.

Public interface
----------------
    detect_changes(fund, current_period, prior_period=None, include_unchanged=False)
        → list[PositionChange]

Change classification
---------------------
    NEW         — position present in current, absent in prior
    INCREASED   — present in both, current_shares > prior_shares
    DECREASED   — present in both, current_shares < prior_shares
    CLOSED      — present in prior, absent in current
    UNCHANGED   — present in both, shares equal (omitted by default)

Comparison key
--------------
    (cusip, put_call, investment_discretion)

Multiple Holding rows with the same key within a single filing (e.g. sole- and
shared-discretion tranches of the same position) are summed before comparison so
reporting splits do not produce false INCREASED/DECREASED signals.

Direction field (consumed by Module 4 convergence signal)
----------------------------------------------------------
    bullish_leaning : NEW, INCREASED
    bearish_leaning : CLOSED, DECREASED
    neutral         : UNCHANGED
"""

import datetime
from typing import TypedDict

from smart_money.models import Filing, Fund, Holding

# ---------------------------------------------------------------------------
# Direction mapping
# ---------------------------------------------------------------------------

_DIRECTION: dict[str, str] = {
    "NEW":       "bullish_leaning",
    "INCREASED": "bullish_leaning",
    "DECREASED": "bearish_leaning",
    "CLOSED":    "bearish_leaning",
    "UNCHANGED": "neutral",
}


# ---------------------------------------------------------------------------
# Public TypedDict
# ---------------------------------------------------------------------------

class PositionChange(TypedDict):
    fund_id:              int
    fund_name:            str
    cusip:                str
    issuer_name:          str
    put_call:             str | None
    investment_discretion: str
    change_type:          str        # NEW | INCREASED | DECREASED | CLOSED | UNCHANGED
    direction:            str        # bullish_leaning | bearish_leaning | neutral
    current_period:       str | None # "2024-12-31"
    prior_period:         str | None # "2024-09-30" | None when first_filing=True
    first_filing:         bool
    prior_shares:         int | None
    current_shares:       int | None
    shares_delta:         int | None        # current − prior; None for NEW / CLOSED
    shares_pct_change:    float | None      # None for NEW / CLOSED
    prior_value_usd:      int | None
    current_value_usd:    int | None
    value_delta_usd:      int | None        # None for NEW / CLOSED


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

_HoldingKey = tuple[str, str | None, str]  # (cusip, put_call, investment_discretion)


class _AggRow:
    """Aggregated shares and value for a single comparison key."""
    __slots__ = ("issuer_name", "shares", "value_usd")

    def __init__(self, issuer_name: str, shares: int, value_usd: int) -> None:
        self.issuer_name = issuer_name
        self.shares      = shares
        self.value_usd   = value_usd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_filing(fund: Fund, period: datetime.date) -> Filing | None:
    """Return the canonical Filing for (fund, period_of_report).

    When both an original and an amendment exist for the same period, the
    amendment (later filed_date) is returned — mirroring edgar.canonical_filings.
    """
    row = (
        Filing.select()
        .where(Filing.fund == fund, Filing.period_of_report == period)
        .order_by(Filing.filed_date.desc())
        .first()
    )
    return row  # None when not found


def _aggregate_holdings(filing: Filing) -> dict[_HoldingKey, _AggRow]:
    """Aggregate Holding rows for a filing by (cusip, put_call, investment_discretion).

    Rows sharing a key are summed so that reporting splits (sole/shared-discretion
    tranches, etc.) do not inflate share counts or produce spurious change signals.
    """
    agg: dict[_HoldingKey, _AggRow] = {}
    for h in Holding.select().where(Holding.filing == filing):
        key: _HoldingKey = (h.cusip, h.put_call, h.investment_discretion)
        if key in agg:
            agg[key].shares    += h.shares
            agg[key].value_usd += h.value_usd
        else:
            agg[key] = _AggRow(h.issuer_name, h.shares, h.value_usd)
    return agg


def _preceding_period(fund: Fund, before: datetime.date) -> datetime.date | None:
    """Return the period_of_report of the most recent filing strictly before *before*."""
    row = (
        Filing.select(Filing.period_of_report)
        .where(Filing.fund == fund, Filing.period_of_report < before)
        .order_by(Filing.period_of_report.desc())
        .first()
    )
    return row.period_of_report if row else None


def _make_entry(
    fund: Fund,
    key: _HoldingKey,
    change_type: str,
    *,
    current_period: str | None,
    prior_period: str | None,
    first_filing: bool,
    cur: "_AggRow | None",
    pri: "_AggRow | None",
) -> PositionChange:
    cusip, put_call, discretion = key
    issuer_name = (cur or pri).issuer_name  # type: ignore[union-attr]

    if cur is not None and pri is not None:
        shares_delta: int | None      = cur.shares - pri.shares
        pct: float | None             = (shares_delta / pri.shares * 100.0) if pri.shares else None
        shares_pct_change             = round(pct, 2) if pct is not None else None
        value_delta_usd: int | None   = cur.value_usd - pri.value_usd
    else:
        shares_delta      = None
        shares_pct_change = None
        value_delta_usd   = None

    return PositionChange(
        fund_id               = fund.id,
        fund_name             = fund.name,
        cusip                 = cusip,
        issuer_name           = issuer_name,
        put_call              = put_call,
        investment_discretion = discretion,
        change_type           = change_type,
        direction             = _DIRECTION[change_type],
        current_period        = current_period,
        prior_period          = prior_period,
        first_filing          = first_filing,
        prior_shares          = pri.shares   if pri else None,
        current_shares        = cur.shares   if cur else None,
        shares_delta          = shares_delta,
        shares_pct_change     = shares_pct_change,
        prior_value_usd       = pri.value_usd if pri else None,
        current_value_usd     = cur.value_usd if cur else None,
        value_delta_usd       = value_delta_usd,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def detect_changes(
    fund: Fund,
    current_period: datetime.date,
    prior_period: datetime.date | None = None,
    include_unchanged: bool = False,
) -> list[PositionChange]:
    """Detect quarter-over-quarter position changes for a fund.

    Parameters
    ----------
    fund : Fund
        Peewee Fund model instance.
    current_period : datetime.date
        Quarter-end date of the filing to analyse.
    prior_period : datetime.date | None
        Quarter-end date to compare against.  When None (default), the
        immediately preceding filing in the DB is used automatically.
    include_unchanged : bool
        When False (default), UNCHANGED positions are omitted from the result.

    Returns
    -------
    list[PositionChange]
        Sorted by descending current_value_usd; CLOSED positions (no current
        value) follow all open positions.

    Raises
    ------
    ValueError
        When no filing exists for (fund, current_period).
    """
    current_filing = _get_filing(fund, current_period)
    if current_filing is None:
        raise ValueError(
            f"No filing for {fund.name!r} period {current_period}"
        )

    if prior_period is None:
        prior_period = _preceding_period(fund, current_period)

    first_filing = prior_period is None
    prior_agg: dict[_HoldingKey, _AggRow] = {}
    if not first_filing:
        prior_filing = _get_filing(fund, prior_period)
        if prior_filing is not None:
            prior_agg = _aggregate_holdings(prior_filing)

    current_agg = _aggregate_holdings(current_filing)
    cur_str  = current_period.isoformat()
    pri_str  = prior_period.isoformat() if prior_period else None

    changes: list[PositionChange] = []

    for key in set(current_agg) | set(prior_agg):
        in_current = key in current_agg
        in_prior   = key in prior_agg

        if in_current and in_prior:
            cur, pri = current_agg[key], prior_agg[key]
            if cur.shares > pri.shares:
                ct = "INCREASED"
            elif cur.shares < pri.shares:
                ct = "DECREASED"
            else:
                ct = "UNCHANGED"
                if not include_unchanged:
                    continue
            changes.append(_make_entry(
                fund, key, ct,
                current_period=cur_str, prior_period=pri_str,
                first_filing=first_filing, cur=cur, pri=pri,
            ))
        elif in_current:
            changes.append(_make_entry(
                fund, key, "NEW",
                current_period=cur_str, prior_period=pri_str,
                first_filing=first_filing, cur=current_agg[key], pri=None,
            ))
        else:
            changes.append(_make_entry(
                fund, key, "CLOSED",
                current_period=cur_str, prior_period=pri_str,
                first_filing=first_filing, cur=None, pri=prior_agg[key],
            ))

    changes.sort(key=lambda c: (
        c["current_value_usd"] is None,  # open positions first (False < True)
        -(c["current_value_usd"] or 0),
    ))
    return changes
