"""
Module 4 — cross-fund positioning convergence signal.

Public interface
---------------
    scan_quarter(period, *, min_funds, min_total_weight, fetch_sectors)
        → list[ConvergenceResult]

    persist_quarter(results) → int
        Upsert results into ConvergenceScore table; returns rows written.

    load_quarter(period, *, min_score) → list[ConvergenceScore]
        Read persisted rows for a period (used by downstream signal.py).

Algorithm
---------
1.  Load skill weights from FundSkillResult; unscored funds fall back to
    bucket-level defaults.
2.  Load Filing.total_value_usd per fund for the target period (portfolio totals
    needed for avg_position_pct_of_portfolio).
3.  Call detect_changes() for every non-excluded fund.  Skip first_filing entries
    — those lack a prior baseline and would inflate convergence.  Group remaining
    changes by CUSIP: {cusip → [(fund, [changes])]}.
4.  Filter to CUSIPs with ≥ min_funds distinct funds making any change.
5.  Build sector map: query Security.sector from DB; optionally fetch missing
    sectors from yfinance for convergence-candidate tickers (bounded by
    _SECTOR_FETCH_LIMIT to protect cold-start runs).  Peer CUSIPs (needed for
    sector_concentration) use cached data only.
6.  Build sector_bull_moves index: {(fund_id, sector) → {cusip}} for all
    bullish changes.  Enables O(1) sector_concentration lookup.
7.  Score each candidate CUSIP (see _score_cusip).
8.  Sort by convergence_score descending; return.

Enrichment fields
-----------------
avg_position_pct_of_portfolio
    Mean of (ticker_holding_usd / fund_total_portfolio_usd × 100) across
    bullish funds only.  Weights convergence by conviction size, not just fund
    count.  Aggregates all tranches (equity + options) for the same CUSIP.
    None when no bullish fund has a Filing.total_value_usd.

convergence_trend
    Requires prior ConvergenceScore rows in the DB (written by persist_quarter).
    "new"          — CUSIP absent from ConvergenceScore for both prior quarters
    "accelerating" — score improved in T-2Q→T-1Q and T-1Q→T (both deltas > 0)
    "stable"       — |score change vs prior quarter| < 0.10
    "fading"       — score declined in T-2Q→T-1Q and T-1Q→T (both deltas < 0)
    On the first full run no prior rows exist, so all scores are "new".

sector_concentration
    Fraction [0.0–1.0] of bullish funds that also made a bullish move on at
    least one other ticker in the same sector this quarter.
    None when this CUSIP's sector is unknown (no_match CUSIPs, foreign listings).
    Peer detection uses Security.sector cache; unresolved peers are excluded.
    A value near 1.0 signals a broad sector rotation; 0.0 is a pure name call.

Skill weight schedule
---------------------
Unscored funds (no FundSkillResult row):
    quant_systematic     → 0.40   (13F ≠ their actual book; heavy hedging)
    fundamental_value    → 0.90   (concentrated/high-conviction; coverage issue
                                   is the reason they're unscored, not skill)
    long_short_equity    → 0.85
    sector_specialist    → 0.85

Scored funds:
    unreliable (< 12 quarters)  → 0.80
    reliable, α ≥ 0             → clamp(1.0 + α_ann / 0.20, 0.80, 3.00)
    reliable, α < 0             → clamp(1.0 + α_ann / 0.20, 0.10, 0.79)
        Greenoaks  +38.5% → 3.00 (cap)   Viking +2.9% → 1.15
        Glenview  −21.8% → 0.10 (floor)  Coatue −0.1% → 0.99

Change multiplier
-----------------
NEW / CLOSED              → 1.00   (decisive; fresh or exited)
INCREASED / DECREASED     → 0.50 + 0.50 × tanh(|pct_change| / 50)
    +100% position size → 0.90;  +20% → 0.67;  +5% → 0.55
"""

import calendar
import datetime
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypedDict

import yfinance as yf

from smart_money.changes import PositionChange, detect_changes
from smart_money.models import (
    ConvergenceScore,
    Filing,
    Fund,
    FundSkillResult,
    Security,
    db,
    init_db,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BUCKET_WEIGHTS: dict[str, float] = {
    "long_short_equity": 0.85,
    "fundamental_value": 0.90,
    "sector_specialist": 0.85,
    "quant_systematic":  0.40,
}

_BREADTH_SATURATION = 4     # n_funds at which breadth saturates at 1.0
_SECTOR_FETCH_LIMIT = 300   # max yfinance fetches per scan (cold-start guard)
_SECTOR_WORKERS     = 8     # concurrent threads for sector fetch


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class FundMove(TypedDict):
    fund_id:           int
    fund_name:         str
    direction:         str           # bullish_leaning | bearish_leaning
    change_type:       str           # NEW | INCREASED | DECREASED | CLOSED
    skill_weight:      float
    change_mult:       float
    effective_weight:  float         # skill_weight × change_mult
    current_value_usd: int | None    # sum across all tranches (equity + options)
    portfolio_pct:     float | None  # current_value_usd / fund_portfolio × 100


@dataclass
class ConvergenceResult:
    cusip:          str
    issuer_name:    str
    ticker:         str | None
    period:         datetime.date
    # Core score
    convergence_score: float   # −1 to +1
    directional:       float   # (bull_w − bear_w) / (bull_w + bear_w)
    breadth:           float   # min(n_funds_total / _BREADTH_SATURATION, 1.0)
    n_funds_total:     int     # distinct funds with any change (bull + bear)
    n_funds_bullish:   int
    n_funds_bearish:   int
    bull_weight:       float
    bear_weight:       float
    # Enrichments
    avg_position_pct_of_portfolio: float | None
    convergence_trend:             str   | None  # new|accelerating|stable|fading
    sector:                        str   | None
    sector_concentration:          float | None  # 0.0–1.0
    fund_moves:                    list[FundMove]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prior_quarter_ends(period: datetime.date) -> tuple[datetime.date, datetime.date]:
    """Return the two quarter-end dates immediately preceding period."""
    def _prev(d: datetime.date) -> datetime.date:
        m, y = d.month - 3, d.year
        if m <= 0:
            m += 12
            y -= 1
        return datetime.date(y, m, calendar.monthrange(y, m)[1])
    q1 = _prev(period)
    return q1, _prev(q1)


def _skill_weight(fund: Fund, skill_map: dict[int, FundSkillResult]) -> float:
    result = skill_map.get(fund.id)
    if result is None:
        return _BUCKET_WEIGHTS.get(fund.bucket, 1.0)
    if not result.is_reliable:
        return 0.80
    raw = 1.0 + result.alpha_annualized / 0.20
    return max(0.10, min(3.00, raw))


def _change_mult(change: PositionChange) -> float:
    ct = change["change_type"]
    if ct in ("NEW", "CLOSED"):
        return 1.0
    pct = abs(change.get("shares_pct_change") or 0.0)
    return 0.5 + 0.5 * math.tanh(pct / 50.0)


def _primary_change(changes: list[PositionChange]) -> PositionChange:
    """Select the representative change for a (fund, CUSIP) group.

    Prefers the equity tranche (put_call=None) by descending current value.
    Falls back to the highest-value non-equity tranche when no equity row exists.
    """
    equity = [c for c in changes if c["put_call"] is None]
    pool = equity if equity else changes
    return max(pool, key=lambda c: (c["current_value_usd"] or 0))


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_skill_map() -> dict[int, FundSkillResult]:
    return {r.fund_id: r for r in FundSkillResult.select()}


def _load_portfolio_totals(
    period: datetime.date, fund_ids: set[int]
) -> dict[int, int]:
    """Return {fund_id → total_value_usd} for funds with a Filing in period."""
    totals: dict[int, int] = {}
    for f in Filing.select().where(
        Filing.period_of_report == period,
        Filing.fund_id.in_(fund_ids),
        Filing.total_value_usd.is_null(False),
    ):
        totals[f.fund_id] = f.total_value_usd
    return totals


def _collect_all_changes(
    period: datetime.date,
) -> dict[str, list[tuple[Fund, list[PositionChange]]]]:
    """
    Collect all position changes for the given quarter, grouped by CUSIP.

    First-filing entries are excluded because every position is "NEW" in a
    fund's first 13F — there is no prior-quarter baseline to infer genuine
    new conviction from.

    Returns
    -------
    {cusip → [(fund, [PositionChange, ...])]}
    The inner list contains one or more PositionChange entries per fund (e.g.
    equity tranche + put tranche on the same CUSIP).
    """
    grouped: dict[str, dict[int, tuple[Fund, list[PositionChange]]]] = {}

    for fund in Fund.select().where(Fund.excluded == False):
        has_filing = (
            Filing.select()
            .where(Filing.fund == fund, Filing.period_of_report == period)
            .exists()
        )
        if not has_filing:
            continue
        try:
            changes = detect_changes(fund, period)
        except ValueError:
            continue

        for c in changes:
            if c["first_filing"]:
                continue
            cusip = c["cusip"]
            grouped.setdefault(cusip, {})
            if fund.id not in grouped[cusip]:
                grouped[cusip][fund.id] = (fund, [])
            grouped[cusip][fund.id][1].append(c)

    return {
        cusip: list(fund_map.values())
        for cusip, fund_map in grouped.items()
    }


# ---------------------------------------------------------------------------
# Sector resolution
# ---------------------------------------------------------------------------

def _resolve_sectors(
    cusips: set[str],
    *,
    fetch_missing: bool,
) -> dict[str, str | None]:
    """
    Return {cusip → sector} for every CUSIP in cusips.

    Reads Security.sector from the DB first.  When fetch_missing=True,
    fetches missing sectors from yfinance for tickers with a resolved Security
    row, capping at _SECTOR_FETCH_LIMIT per call to protect cold-start runs.
    Results are persisted back to Security.sector so subsequent calls are fast.
    """
    sector_map: dict[str, str | None] = {}
    to_fetch: list[tuple[str, str]] = []

    if cusips:
        for row in (
            Security.select(Security.cusip, Security.ticker, Security.sector)
            .where(Security.cusip.in_(cusips))
        ):
            if row.sector:
                sector_map[row.cusip] = row.sector
            elif row.ticker and fetch_missing:
                to_fetch.append((row.cusip, row.ticker))
            else:
                sector_map[row.cusip] = None

    for cusip in cusips - set(sector_map) - {c for c, _ in to_fetch}:
        sector_map[cusip] = None

    if not to_fetch or not fetch_missing:
        return sector_map

    to_fetch = to_fetch[:_SECTOR_FETCH_LIMIT]

    def _fetch_one(pair: tuple[str, str]) -> tuple[str, str | None]:
        cusip, ticker = pair
        try:
            return cusip, yf.Ticker(ticker).info.get("sector")
        except Exception:
            return cusip, None

    with ThreadPoolExecutor(max_workers=_SECTOR_WORKERS) as ex:
        fetched: list[tuple[str, str | None]] = list(ex.map(_fetch_one, to_fetch))

    updates = [(cusip, sector) for cusip, sector in fetched if sector]
    if updates:
        with db.atomic():
            for cusip, sector in updates:
                Security.update(sector=sector).where(Security.cusip == cusip).execute()

    for cusip, sector in fetched:
        sector_map[cusip] = sector

    return sector_map


# ---------------------------------------------------------------------------
# Sector bull-move index
# ---------------------------------------------------------------------------

def _build_sector_bull_moves(
    all_changes: dict[str, list[tuple[Fund, list[PositionChange]]]],
    sector_map: dict[str, str | None],
) -> dict[tuple[int, str], set[str]]:
    """
    Build {(fund_id, sector) → {cusip}} for all bullish moves this quarter.

    Used for O(1) sector_concentration lookup: "which other CUSIPs in sector S
    did fund F make a bullish move on?"  CUSIPs without a known sector are
    excluded from peer sets (they cannot contribute to concentration).
    """
    index: dict[tuple[int, str], set[str]] = {}
    for cusip, fund_list in all_changes.items():
        sector = sector_map.get(cusip)
        if sector is None:
            continue
        for fund, changes in fund_list:
            primary = _primary_change(changes)
            if primary["direction"] != "bullish_leaning":
                continue
            key = (fund.id, sector)
            index.setdefault(key, set()).add(cusip)
    return index


# ---------------------------------------------------------------------------
# Convergence trend
# ---------------------------------------------------------------------------

def _convergence_trend(
    cusip: str,
    score_now: float,
    period: datetime.date,
) -> str:
    q1, q2 = _prior_quarter_ends(period)

    r1 = ConvergenceScore.get_or_none(
        (ConvergenceScore.cusip == cusip) & (ConvergenceScore.period == q1)
    )
    if r1 is None:
        return "new"

    delta1 = score_now - r1.convergence_score  # T-1 → T

    r2 = ConvergenceScore.get_or_none(
        (ConvergenceScore.cusip == cusip) & (ConvergenceScore.period == q2)
    )
    if r2 is None:
        if abs(delta1) < 0.10:
            return "stable"
        return "accelerating" if delta1 > 0 else "fading"

    delta2 = r1.convergence_score - r2.convergence_score  # T-2 → T-1

    if delta1 > 0.10 and delta2 > 0:
        return "accelerating"
    if delta1 < -0.10 and delta2 < 0:
        return "fading"
    return "stable"


# ---------------------------------------------------------------------------
# Per-CUSIP scorer
# ---------------------------------------------------------------------------

def _score_cusip(
    cusip: str,
    fund_changes: list[tuple[Fund, list[PositionChange]]],
    skill_map: dict[int, FundSkillResult],
    portfolio_totals: dict[int, int],
    sector_map: dict[str, str | None],
    sector_bull_moves: dict[tuple[int, str], set[str]],
    period: datetime.date,
    *,
    min_total_weight: float,
) -> ConvergenceResult | None:

    bull_w = bear_w = 0.0
    moves: list[FundMove] = []
    bull_pcts: list[float] = []

    for fund, changes in fund_changes:
        primary = _primary_change(changes)
        direction = primary["direction"]
        if direction == "neutral":
            continue

        sw = _skill_weight(fund, skill_map)
        cm = _change_mult(primary)
        ew = sw * cm

        # Sum holding value across all tranches for this (fund, CUSIP)
        total_holding_usd: int | None = None
        tranche_values = [c["current_value_usd"] for c in changes if c["current_value_usd"] is not None]
        if tranche_values:
            total_holding_usd = sum(tranche_values)

        port_total = portfolio_totals.get(fund.id)
        pct: float | None = (
            total_holding_usd / port_total * 100.0
            if total_holding_usd and port_total
            else None
        )

        moves.append(FundMove(
            fund_id=fund.id,
            fund_name=fund.name,
            direction=direction,
            change_type=primary["change_type"],
            skill_weight=round(sw, 4),
            change_mult=round(cm, 4),
            effective_weight=round(ew, 4),
            current_value_usd=total_holding_usd,
            portfolio_pct=round(pct, 4) if pct is not None else None,
        ))

        if direction == "bullish_leaning":
            bull_w += ew
            if pct is not None:
                bull_pcts.append(pct)
        else:
            bear_w += ew

    total_w = bull_w + bear_w
    if total_w < min_total_weight:
        return None

    n_bull  = sum(1 for m in moves if m["direction"] == "bullish_leaning")
    n_bear  = sum(1 for m in moves if m["direction"] == "bearish_leaning")
    n_total = n_bull + n_bear

    directional = (bull_w - bear_w) / total_w
    breadth     = min(n_total / _BREADTH_SATURATION, 1.0)
    score       = directional * breadth

    avg_pct = sum(bull_pcts) / len(bull_pcts) if bull_pcts else None

    trend = _convergence_trend(cusip, score, period)

    # sector_concentration
    sector = sector_map.get(cusip)
    sc: float | None = None
    if sector and n_bull > 0:
        bull_fund_ids = [m["fund_id"] for m in moves if m["direction"] == "bullish_leaning"]
        n_with_sector_peers = sum(
            1 for fid in bull_fund_ids
            if len(sector_bull_moves.get((fid, sector), set()) - {cusip}) > 0
        )
        sc = n_with_sector_peers / n_bull

    sec_row     = Security.get_or_none(Security.cusip == cusip)
    ticker      = sec_row.ticker if sec_row else None
    issuer_name = fund_changes[0][1][0]["issuer_name"]

    return ConvergenceResult(
        cusip=cusip,
        issuer_name=issuer_name,
        ticker=ticker,
        period=period,
        convergence_score=round(score, 4),
        directional=round(directional, 4),
        breadth=round(breadth, 4),
        n_funds_total=n_total,
        n_funds_bullish=n_bull,
        n_funds_bearish=n_bear,
        bull_weight=round(bull_w, 4),
        bear_weight=round(bear_w, 4),
        avg_position_pct_of_portfolio=round(avg_pct, 4) if avg_pct is not None else None,
        convergence_trend=trend,
        sector=sector,
        sector_concentration=round(sc, 4) if sc is not None else None,
        fund_moves=moves,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def scan_quarter(
    period: datetime.date,
    *,
    min_funds: int = 2,
    min_total_weight: float = 1.0,
    fetch_sectors: bool = True,
) -> list[ConvergenceResult]:
    """
    Compute convergence scores for all CUSIPs with ≥ min_funds changes in period.

    Parameters
    ----------
    period : datetime.date
        Quarter-end date, e.g. datetime.date(2026, 3, 31).
    min_funds : int
        Minimum distinct funds required to emit a score.  Default 2.
    min_total_weight : float
        Minimum sum of bull + bear effective weights.  Default 1.0.
        Filters out tickers where all movers are very low-weight funds.
    fetch_sectors : bool
        When True, populate missing Security.sector values from yfinance
        for convergence-candidate tickers (up to _SECTOR_FETCH_LIMIT).
        Peer CUSIPs always use the DB cache only.

    Returns
    -------
    list[ConvergenceResult]
        Sorted by convergence_score descending (most bullish first).
        Scores near +1.0 indicate many high-skilled funds moving bullish;
        scores near −1.0 indicate exit/bearish convergence.
    """
    init_db()
    skill_map   = _load_skill_map()
    fund_ids    = {f.id for f in Fund.select().where(Fund.excluded == False)}
    port_totals = _load_portfolio_totals(period, fund_ids)

    all_changes = _collect_all_changes(period)

    candidates: dict[str, list[tuple[Fund, list[PositionChange]]]] = {
        cusip: changes
        for cusip, changes in all_changes.items()
        if len(changes) >= min_funds
    }
    if not candidates:
        return []

    # Sector map: fetch for candidates; peers use DB cache only
    all_cusips       = set(all_changes)
    candidate_cusips = set(candidates)

    sector_map = _resolve_sectors(all_cusips, fetch_missing=False)
    if fetch_sectors:
        candidate_map = _resolve_sectors(candidate_cusips, fetch_missing=True)
        sector_map.update(candidate_map)

    sector_bull_moves = _build_sector_bull_moves(all_changes, sector_map)

    results: list[ConvergenceResult] = []
    for cusip, fund_changes in candidates.items():
        result = _score_cusip(
            cusip, fund_changes, skill_map, port_totals,
            sector_map, sector_bull_moves, period,
            min_total_weight=min_total_weight,
        )
        if result is not None:
            results.append(result)

    results.sort(key=lambda r: r.convergence_score, reverse=True)
    return results


def persist_quarter(results: list[ConvergenceResult]) -> int:
    """
    Upsert ConvergenceResult list into the ConvergenceScore table.

    Uses INSERT OR REPLACE semantics — safe to call repeatedly.
    Enables convergence_trend to look back on subsequent scan_quarter calls.

    Returns the number of rows written.
    """
    if not results:
        return 0

    rows = [
        {
            "cusip":               r.cusip,
            "issuer_name":         r.issuer_name,
            "ticker":              r.ticker,
            "period":              r.period,
            "convergence_score":   r.convergence_score,
            "directional":         r.directional,
            "breadth":             r.breadth,
            "n_funds_total":       r.n_funds_total,
            "n_funds_bullish":     r.n_funds_bullish,
            "n_funds_bearish":     r.n_funds_bearish,
            "bull_weight":         r.bull_weight,
            "bear_weight":         r.bear_weight,
            "avg_position_pct_of_portfolio": r.avg_position_pct_of_portfolio,
            "convergence_trend":   r.convergence_trend,
            "sector":              r.sector,
            "sector_concentration": r.sector_concentration,
            "fund_moves_json":     json.dumps(r.fund_moves),
            "computed_at":         datetime.datetime.utcnow(),
        }
        for r in results
    ]

    # SQLite limit: 32766 bound variables per statement.
    # ConvergenceScore has 19 fields → max ~1724 rows per batch.
    _CHUNK = 1700
    with db.atomic():
        for i in range(0, len(rows), _CHUNK):
            ConvergenceScore.replace_many(rows[i : i + _CHUNK]).execute()

    return len(rows)


def load_quarter(
    period: datetime.date,
    *,
    min_score: float | None = None,
) -> list[ConvergenceScore]:
    """
    Read persisted ConvergenceScore rows for the given quarter.

    Parameters
    ----------
    min_score : float | None
        When set, only rows with convergence_score >= min_score are returned.
    """
    init_db()
    q = ConvergenceScore.select().where(ConvergenceScore.period == period)
    if min_score is not None:
        q = q.where(ConvergenceScore.convergence_score >= min_score)
    return list(q.order_by(ConvergenceScore.convergence_score.desc()))
