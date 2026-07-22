"""
Portfolio analysis results cache persisted to data/portfolio_analysis_cache.json.

The factor analysis (FF4 regression + stress tests) downloads external data and
runs OLS — ~20s on first run. Results are cached to disk so the Portfolio page
loads instantly on subsequent opens. The user triggers a refresh explicitly.

Stored fields
-------------
cached_at       ISO timestamp of the run
start / end     Analysis date range
headline        Tier-1 regression results dict
per_holding     List of per-ticker regression dicts
summary_text    Plain-English interpretation string
weights         Normalised weight dict
raw_weights     Original (un-normalised) weight dict
stress_tests    List of scenario result dicts
concentration   factor_engine.concentration.run_concentration_analysis() output
risk_metrics    factor_engine.risk_metrics.compute_risk_metrics() output

concentration/risk_metrics are computed from all_returns/combined_rets/factors —
none of which are persisted here (they're large per-ticker/per-day arrays, cheap
to regenerate on an explicit refresh but not worth bloating this cache file).
A cache written before these two fields existed will simply lack them; the
Portfolio page shows an "unavailable, click Refresh" message rather than
attempting a partial recompute from data this cache never stored.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

CACHE_PATH = Path(__file__).parent.parent / "data" / "portfolio_analysis_cache.json"

_STORED_KEYS = {
    "cached_at", "start", "end", "headline", "per_holding",
    "summary_text", "weights", "raw_weights", "stress_tests",
    "concentration", "risk_metrics",
}


def load() -> dict | None:
    """Return cached results or None if absent / corrupt."""
    if not CACHE_PATH.exists():
        return None
    try:
        return json.loads(CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save(
    results: dict,
    stress_tests: list[dict],
    concentration: dict | None = None,
    risk_metrics: dict | None = None,
) -> str:
    """
    Persist analysis results with a current timestamp. Returns that timestamp
    (ISO string) so callers — e.g. app_pages/portfolio.py's _run_analysis() —
    can fold the exact same cached_at into the in-memory results dict they
    hand back to the page, rather than that dict lacking the field entirely
    until the next reload re-reads it from disk.
    """
    cached_at = datetime.datetime.now().isoformat()
    to_store = {
        "cached_at":      cached_at,
        "start":          results["start"],
        "end":            results["end"],
        "headline":       results["headline"],
        "per_holding":    results["per_holding"],
        "summary_text":   results["summary_text"],
        "weights":        results["weights"],
        "raw_weights":    results["raw_weights"],
        "stress_tests":   stress_tests,
        "concentration":  concentration,
        "risk_metrics":   risk_metrics,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(to_store, indent=2, default=str))
    return cached_at


def age_str(data: dict) -> str:
    """Human-readable age string, e.g. '3d ago', '2h ago', 'just now'."""
    try:
        cached_at = datetime.datetime.fromisoformat(data["cached_at"])
    except (KeyError, ValueError):
        return "unknown"
    age = datetime.datetime.now() - cached_at
    if age.days >= 1:
        return f"{age.days}d ago"
    hours = age.seconds // 3600
    if hours >= 1:
        return f"{hours}h ago"
    minutes = age.seconds // 60
    return f"{minutes}m ago" if minutes > 0 else "just now"
