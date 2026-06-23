"""
OpenFIGI CUSIP resolver for Module 3 — 13F Smart-Money Positioning & Skill Tracker.

Public interface
----------------
    resolve_cusips(cusips, *, skip_resolved=True) → dict[str, FIGIResult | None]

Writes results to the Security table (models.py) and returns the same data
in-memory so callers don't need a second DB read.

Rate limits (OpenFIGI v3)
--------------------------
Without API key : 10 IDs/request, 20 req/min  →  3 s gap enforced
With API key    : 100 IDs/request, 250 req/min →  0.25 s gap enforced

Set OPENFIGI_API_KEY env var before importing this module to enable the
higher tier.  The limits are read once at import time.

CUSIP → FIGI candidate selection
---------------------------------
OpenFIGI returns multiple listings per CUSIP (NYSE, NASDAQ, OTC, preferred,
warrant, ADR, etc.).  We score and pick one best candidate:

    marketSector == "Equity"     +10
    securityType "Common Stock"  +5
    securityType "ETP"           +4
    securityType "Preferred"     +3
    securityType "Depositary Receipt" +2
    securityType "Warrant"/"Right"    +1
    exchCode in US primaries (US/UN/UW) +3
    exchCode OTC (UR)            +2

The stored identifier is compositeFIGI — the cross-exchange Bloomberg
composite.  prices.py maps this composite FIGI to a yfinance ticker.
"""

import datetime
import os
import time
from typing import TypedDict

import requests

from smart_money.models import Security, init_db


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class FIGIResult(TypedDict):
    cusip: str
    composite_figi: str | None
    share_class_figi: str | None
    ticker: str | None
    exchange_code: str | None
    security_name: str | None
    security_type: str | None
    market_sector: str | None


# ---------------------------------------------------------------------------
# API configuration  (read once at import)
# ---------------------------------------------------------------------------

_API_KEY  = os.getenv("OPENFIGI_API_KEY", "")
_BASE_URL = "https://api.openfigi.com/v3/mapping"

# OpenFIGI v3 documented limits
_BATCH_SIZE = 100 if _API_KEY else 10     # CUSIPs per POST
_MIN_GAP    = 0.25 if _API_KEY else 3.0  # seconds between requests

_MAX_RETRIES = 3
_last_request_ts: float = 0.0


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _post(payload: list[dict]) -> requests.Response:
    """Throttled POST with exponential backoff on 429."""
    global _last_request_ts

    elapsed = time.monotonic() - _last_request_ts
    if elapsed < _MIN_GAP:
        time.sleep(_MIN_GAP - elapsed)

    headers = {"Content-Type": "application/json"}
    if _API_KEY:
        headers["X-OPENFIGI-APIKEY"] = _API_KEY

    delay = 2.0
    for attempt in range(_MAX_RETRIES + 1):
        _last_request_ts = time.monotonic()
        resp = requests.post(_BASE_URL, json=payload, headers=headers, timeout=30)
        if resp.status_code == 429 and attempt < _MAX_RETRIES:
            retry_after = float(resp.headers.get("Retry-After", delay))
            time.sleep(retry_after)
            delay = min(delay * 2, 60)
            continue
        resp.raise_for_status()
        return resp

    resp.raise_for_status()
    return resp  # unreachable; satisfies type checker


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

_SECTOR_SCORE: dict[str, int] = {"Equity": 10, "Corp": 5}
_TYPE_SCORE: dict[str, int] = {
    "Common Stock":       5,
    "ETP":                4,
    "Preferred Stock":    3,
    "Depositary Receipt": 2,
    "Warrant":            1,
    "Right":              1,
}
# Primary US venues and their scores
# US=NYSE, UN=NYSE ARCA, UW=NASDAQ Global Select, UQ=NASDAQ Capital Market,
# UR=OTC Markets, UP=Pink Sheets, UA=NYSE American, UC=FINRA OTC, UD=FINRA ADF
_EXCH_SCORE: dict[str, int] = {
    "US": 3, "UN": 3, "UW": 3, "UQ": 3,
    "UR": 2, "UA": 2, "UD": 2,
    "UP": 1, "UC": 1,
}
# Non-US exchange listings (Xetra, LSE, Euronext, etc.) are strongly penalised
# so the US composite always wins when a dual-listed security has both US and
# foreign candidates returned by OpenFIGI.
_US_EXCHANGES = frozenset(_EXCH_SCORE)
_NON_US_PENALTY = -20


def _score(candidate: dict) -> int:
    s = 0
    s += _SECTOR_SCORE.get(candidate.get("marketSector", ""), 0)
    s += _TYPE_SCORE.get(candidate.get("securityType", ""), 0)
    exch = candidate.get("exchCode", "")
    s += _EXCH_SCORE.get(exch, 0)
    if exch and exch not in _US_EXCHANGES:
        s += _NON_US_PENALTY
    return s


def _pick_best(candidates: list[dict]) -> dict | None:
    return max(candidates, key=_score) if candidates else None


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------

def _upsert_resolved(cusip: str, best: dict) -> FIGIResult:
    result = FIGIResult(
        cusip            = cusip,
        composite_figi   = best.get("compositeFIGI") or best.get("figi"),
        share_class_figi = best.get("shareClassFIGI"),
        ticker           = best.get("ticker"),
        exchange_code    = best.get("exchCode"),
        security_name    = best.get("name"),
        security_type    = best.get("securityType"),
        market_sector    = best.get("marketSector"),
    )
    (Security
        .insert(
            cusip             = cusip,
            composite_figi    = result["composite_figi"],
            share_class_figi  = result["share_class_figi"],
            ticker            = result["ticker"],
            exchange_code     = result["exchange_code"],
            security_name     = result["security_name"],
            security_type     = result["security_type"],
            market_sector     = result["market_sector"],
            resolution_status = "resolved",
            resolved_at       = datetime.datetime.utcnow(),
            resolution_error  = None,
        )
        .on_conflict_replace()
        .execute())
    return result


def _upsert_no_match(cusip: str) -> None:
    (Security
        .insert(
            cusip             = cusip,
            resolution_status = "no_match",
            resolved_at       = datetime.datetime.utcnow(),
        )
        .on_conflict_replace()
        .execute())


def _upsert_failed(cusip: str, error: str) -> None:
    (Security
        .insert(
            cusip             = cusip,
            resolution_status = "failed",
            resolved_at       = datetime.datetime.utcnow(),
            resolution_error  = error[:255],
        )
        .on_conflict_replace()
        .execute())


# ---------------------------------------------------------------------------
# Core batch resolver
# ---------------------------------------------------------------------------

def _resolve_batch(cusips: list[str]) -> dict[str, FIGIResult | None]:
    """POST one batch (≤ _BATCH_SIZE) to OpenFIGI; persist and return results."""
    payload = [{"idType": "ID_CUSIP", "idValue": c} for c in cusips]
    resp = _post(payload)
    items: list[dict] = resp.json()  # parallel to payload

    out: dict[str, FIGIResult | None] = {}
    for cusip, item in zip(cusips, items):
        if "error" in item:
            _upsert_failed(cusip, item["error"])
            out[cusip] = None
        elif not item.get("data"):
            _upsert_no_match(cusip)
            out[cusip] = None
        else:
            best = _pick_best(item["data"])
            if best is None:
                _upsert_no_match(cusip)
                out[cusip] = None
            else:
                out[cusip] = _upsert_resolved(cusip, best)

    return out


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def resolve_cusips(
    cusips: list[str],
    *,
    skip_resolved: bool = True,
) -> dict[str, FIGIResult | None]:
    """
    Resolve a list of CUSIPs to FIGI identifiers via OpenFIGI.

    When skip_resolved=True (default), CUSIPs already in the Security table
    with resolution_status="resolved" are returned from cache without an
    API call.  Set skip_resolved=False to force re-query of all inputs.

    Returns a dict mapping each input CUSIP to a FIGIResult (or None if
    OpenFIGI returned no match or the query failed).  Writes all new
    results to the Security table as a side effect.
    """
    if not cusips:
        return {}

    init_db()

    unique = list(dict.fromkeys(cusips))  # deduplicate, preserve first-seen order
    results: dict[str, FIGIResult | None] = {}
    to_query: list[str] = []

    if skip_resolved:
        cached = {
            row.cusip: row
            for row in Security.select().where(
                Security.cusip.in_(unique),
                Security.resolution_status == "resolved",
            )
        }
        for cusip in unique:
            if cusip in cached:
                row = cached[cusip]
                results[cusip] = FIGIResult(
                    cusip            = cusip,
                    composite_figi   = row.composite_figi,
                    share_class_figi = row.share_class_figi,
                    ticker           = row.ticker,
                    exchange_code    = row.exchange_code,
                    security_name    = row.security_name,
                    security_type    = row.security_type,
                    market_sector    = row.market_sector,
                )
            else:
                to_query.append(cusip)
    else:
        to_query = unique

    for i in range(0, len(to_query), _BATCH_SIZE):
        batch = to_query[i : i + _BATCH_SIZE]
        results.update(_resolve_batch(batch))

    return results
