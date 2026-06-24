"""
Module 4 — watchlist management.

Maintains a DB-backed watchlist of tickers/CUSIPs.  Positions are
soft-deleted (active=False) rather than hard-deleted so history is preserved.
score_watchlist() joins active entries against FinalSignal for a given quarter,
returning status/score/signal_drivers without recomputing any signals.

Public interface
----------------
    add(ticker_or_cusip, *, note, added_price) → Watchlist
    remove(ticker_or_cusip)                    → bool
    list_active()                              → list[Watchlist]
    score_watchlist(period)                    → list[WatchlistScore]

Resolution
----------
add() accepts either a ticker symbol or a 9-character CUSIP.  It resolves
the counterpart identifier and issuer_name via the Security table first,
then falls back to the most recent FinalSignal row.  If neither lookup
succeeds, the entry is stored with whichever identifier was provided.

Duplicate guard
---------------
add() is idempotent: if an active entry already exists for the same
CUSIP (or ticker when CUSIP is unknown), it returns the existing row
without modification.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass

from smart_money.models import FinalSignal, Security, Watchlist, init_db

log = logging.getLogger(__name__)

_CUSIP_LEN = 9


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class WatchlistScore:
    """Active watchlist entry paired with its FinalSignal row for a quarter."""

    entry:  Watchlist
    signal: FinalSignal | None

    @property
    def status(self) -> str | None:
        return self.signal.status if self.signal else None

    @property
    def final_score(self) -> float | None:
        return self.signal.final_score if self.signal else None

    @property
    def signal_drivers(self) -> str | None:
        return self.signal.signal_drivers if self.signal else None

    @property
    def display_name(self) -> str:
        """Ticker when available; issuer_name otherwise."""
        return self.entry.ticker if self.entry.ticker else self.entry.issuer_name


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_cusip(value: str) -> bool:
    return len(value) == _CUSIP_LEN and value.isalnum()


def _resolve(ticker_or_cusip: str) -> tuple[str | None, str | None, str]:
    """
    Return (ticker, cusip, issuer_name) for the given identifier.

    Resolution order:
      1. Security table (most reliable; populated by OpenFIGI resolver)
      2. Most recent FinalSignal row (covers tickers that FIGI couldn't resolve)
      3. Bare fallback — store whichever identifier was given; counterpart=None
    """
    val = ticker_or_cusip.strip().upper()

    if _is_cusip(val):
        sec = Security.get_or_none(Security.cusip == val)
        if sec:
            return sec.ticker, val, sec.security_name or val
        fs = (
            FinalSignal.select()
            .where(FinalSignal.cusip == val)
            .order_by(FinalSignal.period.desc())
            .first()
        )
        if fs:
            return fs.ticker, val, fs.issuer_name
        return None, val, val
    else:
        sec = Security.get_or_none(Security.ticker == val)
        if sec:
            return val, sec.cusip, sec.security_name or val
        fs = (
            FinalSignal.select()
            .where(FinalSignal.ticker == val)
            .order_by(FinalSignal.period.desc())
            .first()
        )
        if fs:
            return val, fs.cusip, fs.issuer_name
        return val, None, val


def _find_active(ticker: str | None, cusip: str | None) -> Watchlist | None:
    """Return the active Watchlist row matching either identifier, or None."""
    if cusip:
        row = Watchlist.get_or_none(
            (Watchlist.cusip == cusip) & (Watchlist.active == True)
        )
        if row:
            return row
    if ticker:
        row = Watchlist.get_or_none(
            (Watchlist.ticker == ticker) & (Watchlist.active == True)
        )
        if row:
            return row
    return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def add(
    ticker_or_cusip: str,
    *,
    note: str | None = None,
    added_price: float | None = None,
) -> Watchlist:
    """
    Add a ticker or CUSIP to the active watchlist.

    Parameters
    ----------
    ticker_or_cusip : str
        Ticker symbol (e.g. "NVDA") or 9-character CUSIP (e.g. "67066G104").
    note : str | None
        Optional free-text annotation displayed in the dashboard.
    added_price : float | None
        Optional entry price for return-since-addition tracking.

    Returns
    -------
    Watchlist
        The new row, or the existing active row if already on the watchlist.
    """
    init_db()
    ticker, cusip, issuer_name = _resolve(ticker_or_cusip)

    existing = _find_active(ticker, cusip)
    if existing:
        log.info("Already active on watchlist: %s (%s)", ticker, cusip)
        return existing

    row = Watchlist.create(
        ticker=ticker,
        cusip=cusip,
        issuer_name=issuer_name,
        date_added=datetime.date.today(),
        added_price=added_price,
        note=note,
        active=True,
    )
    log.info("Added to watchlist: %s (%s) — %s", ticker, cusip, issuer_name)
    return row


def remove(ticker_or_cusip: str) -> bool:
    """
    Deactivate a watchlist entry (soft delete — sets active=False).

    Parameters
    ----------
    ticker_or_cusip : str
        Ticker symbol or CUSIP matching an active watchlist entry.

    Returns
    -------
    bool
        True if an active entry was found and deactivated; False if not found.
    """
    init_db()
    val = ticker_or_cusip.strip().upper()

    if _is_cusip(val):
        n = (
            Watchlist.update(active=False)
            .where((Watchlist.cusip == val) & (Watchlist.active == True))
            .execute()
        )
    else:
        n = (
            Watchlist.update(active=False)
            .where((Watchlist.ticker == val) & (Watchlist.active == True))
            .execute()
        )

    if n:
        log.info("Removed from watchlist: %s (%d row(s) deactivated)", val, n)
    else:
        log.warning("remove() found no active entry for: %s", val)
    return n > 0


def list_active() -> list[Watchlist]:
    """Return all active watchlist entries, ordered by date_added ascending."""
    init_db()
    return list(
        Watchlist.select()
        .where(Watchlist.active == True)
        .order_by(Watchlist.date_added.asc())
    )


def score_watchlist(period: datetime.date) -> list[WatchlistScore]:
    """
    Join active watchlist entries against FinalSignal rows for period.

    Entries with no corresponding FinalSignal row (e.g. signal not yet computed
    for this quarter, or ticker below the discovery threshold) return a
    WatchlistScore with signal=None.

    Parameters
    ----------
    period : datetime.date
        Quarter-end date to look up signals for, e.g. datetime.date(2026, 3, 31).

    Returns
    -------
    list[WatchlistScore]
        One entry per active watchlist position, sorted by final_score
        descending (None scores sort last).
    """
    init_db()
    entries = list_active()
    if not entries:
        return []

    cusips  = {e.cusip   for e in entries if e.cusip}
    tickers = {e.ticker  for e in entries if e.ticker}

    # Build lookup maps from FinalSignal — two separate queries avoid empty-IN issues
    signal_by_cusip:   dict[str, FinalSignal] = {}
    signal_by_ticker:  dict[str, FinalSignal] = {}

    if cusips:
        for row in FinalSignal.select().where(
            (FinalSignal.period == period) & FinalSignal.cusip.in_(list(cusips))
        ):
            signal_by_cusip[row.cusip] = row

    if tickers:
        for row in FinalSignal.select().where(
            (FinalSignal.period == period) & FinalSignal.ticker.in_(list(tickers))
        ):
            signal_by_ticker[row.ticker] = row

    scores: list[WatchlistScore] = []
    for entry in entries:
        signal = (
            signal_by_cusip.get(entry.cusip)
            or (signal_by_ticker.get(entry.ticker) if entry.ticker else None)
        )
        scores.append(WatchlistScore(entry=entry, signal=signal))

    scores.sort(
        key=lambda ws: ws.final_score if ws.final_score is not None else float("-inf"),
        reverse=True,
    )
    return scores
