"""
Shared yfinance call throttle — every direct yfinance call site
(Ticker.info, Ticker.splits, Ticker.history, yf.download) needs to share ONE
rate-limiting clock across the whole process, not throttle independently per
call site, the same reasoning edgar_client.py already applies to SEC EDGAR.

This module exists because no equivalent existed for yfinance until now: the
DCF pilot backtest (scripts/run_dcf_pilot_backtest.py) made ~300 untethered
per-ticker yfinance calls (business-model check, split history, return
series) with no shared throttle between them, triggered "Too Many Requests"
from Yahoo after roughly 50 tickers in under a minute, and silently dropped
233/300 tickers (78%) from that run — not a random subset, but whichever
tickers happened to process before the block kicked in, corrupting the
backtest's coverage in a way that wasn't obvious until the per-ticker error
log was actually read. Built as shared infrastructure (not inlined into the
DCF pilot script) because any future full-universe work over yfinance data
— scaling this same pilot, or anything else touching hundreds of tickers —
would hit the identical failure mode otherwise.

No official Yahoo Finance rate limit is published (unlike SEC's documented
<=10 req/sec, which edgar_client.py's _MIN_GAP cites directly) — _MIN_GAP
here is an empirically conservative choice, deliberately erring slow rather
than re-triggering the block that motivated building this in the first
place, not a cited policy number. Tighten it only after confirming (e.g. a
small canary run) that a faster rate doesn't reintroduce 429s.
"""

import logging
import time

log = logging.getLogger(__name__)

_last_request_ts: float = 0.0
_MIN_GAP = 1.0            # seconds between any two yfinance calls, process-wide
_MAX_RETRIES = 5
_BACKOFF_BASE_SECONDS = 5.0   # doubled on each successive retry


def throttle() -> None:
    """Block until at least _MIN_GAP seconds have passed since the last yfinance call anywhere in this process."""
    global _last_request_ts
    elapsed = time.monotonic() - _last_request_ts
    if elapsed < _MIN_GAP:
        time.sleep(_MIN_GAP - elapsed)
    _last_request_ts = time.monotonic()


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "too many requests" in msg or "rate limit" in msg


def call_with_backoff(fn, *args, **kwargs):
    """
    Run `fn(*args, **kwargs)` — any single yfinance-touching callable — behind
    the shared throttle, retrying with exponential backoff if it raises a
    rate-limit-shaped error. Only rate-limit errors (message contains "too
    many requests" or "rate limit") are retried; any other exception
    (bad ticker, network failure, etc.) propagates on the first attempt —
    silently retrying an unrelated failure 5 times would just be slow, not
    more correct.
    """
    last_exc: "Exception | None" = None
    for attempt in range(_MAX_RETRIES):
        throttle()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise
            last_exc = e
            wait = _BACKOFF_BASE_SECONDS * (2 ** attempt)
            log.warning(
                "yfinance rate-limited (attempt %d/%d), backing off %.0fs: %s",
                attempt + 1, _MAX_RETRIES, wait, e,
            )
            time.sleep(wait)
    raise last_exc
