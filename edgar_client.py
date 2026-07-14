"""
Shared SEC EDGAR HTTP client — throttled GET and ticker→CIK lookup.

This is the one place both independent subsystems (smart_money/ and
factor_engine/) are allowed to depend on. It holds nothing but stateless SEC
plumbing: no DB, no Peewee, no subsystem-specific parsing. smart_money/edgar.py
and smart_money/nlp.py re-export get()/ticker_to_cik() from here (their public
names — _get, ticker_to_cik — are unchanged) so existing Module 3 imports keep
working; factor_engine/gp_xbrl_client.py imports directly from here.

Rate limiting
-------------
SEC policy: <= 10 req/sec, User-Agent required. This module enforces a 0.12s
minimum gap between outbound requests (shared across every caller in the
process — smart_money's 13F fetches and factor_engine's XBRL fetches all
throttle against the same clock) and backs off exponentially on 429/503.

User-Agent
----------
Set EDGAR_USER_AGENT env var to override the default.
"""

import os
import time

import requests

_USER_AGENT = os.getenv(
    "EDGAR_USER_AGENT",
    "RetailInvestorPlatform mac.taupier@gmail.com",
)

_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_last_request_ts: float = 0.0
_MIN_GAP = 0.12          # seconds between requests (< 10/sec SEC limit)
_MAX_RETRIES = 3


def get(url: str, **kwargs) -> requests.Response:
    """Throttled GET with exponential backoff on 429/503."""
    global _last_request_ts

    elapsed = time.monotonic() - _last_request_ts
    if elapsed < _MIN_GAP:
        time.sleep(_MIN_GAP - elapsed)

    headers = {"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    headers.update(kwargs.pop("headers", {}))

    delay = 1.0
    for attempt in range(_MAX_RETRIES + 1):
        _last_request_ts = time.monotonic()
        resp = requests.get(url, headers=headers, timeout=30, **kwargs)
        if resp.status_code in (429, 503) and attempt < _MAX_RETRIES:
            time.sleep(delay)
            delay = min(delay * 2, 32)
            continue
        resp.raise_for_status()
        return resp

    resp.raise_for_status()  # will raise after exhausting retries
    return resp              # unreachable; satisfies type checker


# ---------------------------------------------------------------------------
# Ticker -> CIK lookup
# ---------------------------------------------------------------------------

_ticker_cik_map: dict[str, str] = {}


def _load_ticker_cik_map() -> dict[str, str]:
    """Fetch and cache the SEC company_tickers.json once per process."""
    if _ticker_cik_map:
        return _ticker_cik_map
    data = get(_COMPANY_TICKERS_URL).json()
    for entry in data.values():
        _ticker_cik_map[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)
    return _ticker_cik_map


def ticker_to_cik(ticker: str) -> str | None:
    """Return the zero-padded 10-digit CIK for a ticker, or None if not found."""
    return _load_ticker_cik_map().get(ticker.upper())
