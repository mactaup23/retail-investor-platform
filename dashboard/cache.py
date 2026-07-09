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
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

CACHE_PATH = Path(__file__).parent.parent / "data" / "portfolio_analysis_cache.json"

_STORED_KEYS = {
    "cached_at", "start", "end", "headline", "per_holding",
    "summary_text", "weights", "raw_weights", "stress_tests",
}


def load() -> dict | None:
    """Return cached results or None if absent / corrupt."""
    if not CACHE_PATH.exists():
        return None
    try:
        return json.loads(CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save(results: dict, stress_tests: list[dict]) -> None:
    """Persist analysis results with a current timestamp."""
    to_store = {
        "cached_at":    datetime.datetime.now().isoformat(),
        "start":        results["start"],
        "end":          results["end"],
        "headline":     results["headline"],
        "per_holding":  results["per_holding"],
        "summary_text": results["summary_text"],
        "weights":      results["weights"],
        "raw_weights":  results["raw_weights"],
        "stress_tests": stress_tests,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(to_store, indent=2, default=str))


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
