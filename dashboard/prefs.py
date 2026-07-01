"""
User preferences persisted to data/user_prefs.json.

Stores tax rates, account selection, and portfolio holdings so they survive
browser refreshes. All fields have sensible defaults and are validated on load.
"""
from __future__ import annotations

import json
from pathlib import Path

from factor_engine.portfolio import _RAW_WEIGHTS

PREFS_PATH = Path(__file__).parent.parent / "data" / "user_prefs.json"

# Default portfolio shown to new users — mirrors factor_engine/portfolio.py's
# hardcoded example portfolio so the dashboard has something meaningful to
# show before a user edits their own holdings.
_DEFAULT_PORTFOLIO: list = [
    {"ticker": t, "weight": w} for t, w in _RAW_WEIGHTS.items()
]

DEFAULTS: dict = {
    "st_rate":    0.37,   # federal short-term / ordinary income rate
    "lt_rate":    0.20,   # federal long-term capital gains rate
    "state_rate": 0.0,    # state rate (additive to both ST and LT)
    "niit":       False,  # add 3.8% Net Investment Income Tax surcharge
    "account_id": "default",
    "portfolio":  _DEFAULT_PORTFOLIO,  # list of {"ticker": str, "weight": float}
}

_FLOAT_KEYS = {"st_rate", "lt_rate", "state_rate"}
_BOOL_KEYS  = {"niit"}
_STR_KEYS   = {"account_id"}
_LIST_KEYS  = {"portfolio"}

_MAX_PORTFOLIO_POSITIONS = 20


def _valid_portfolio(v) -> bool:
    """A portfolio is a list of 1-20 {ticker: str, weight: float} dicts."""
    if not isinstance(v, list) or not (1 <= len(v) <= _MAX_PORTFOLIO_POSITIONS):
        return False
    for row in v:
        if not isinstance(row, dict):
            return False
        if not isinstance(row.get("ticker"), str) or not row["ticker"].strip():
            return False
        if not isinstance(row.get("weight"), (int, float)):
            return False
    return True


def load() -> dict:
    """Return current preferences, falling back to defaults for any missing/invalid key."""
    prefs = dict(DEFAULTS)
    if PREFS_PATH.exists():
        try:
            saved = json.loads(PREFS_PATH.read_text())
            for k, v in saved.items():
                if k in _FLOAT_KEYS and isinstance(v, (int, float)):
                    prefs[k] = float(v)
                elif k in _BOOL_KEYS and isinstance(v, bool):
                    prefs[k] = v
                elif k in _STR_KEYS and isinstance(v, str):
                    prefs[k] = v
                elif k in _LIST_KEYS and k == "portfolio" and _valid_portfolio(v):
                    prefs[k] = [
                        {"ticker": row["ticker"].strip().upper(), "weight": float(row["weight"])}
                        for row in v
                    ]
        except (json.JSONDecodeError, OSError):
            pass
    return prefs


def save(prefs: dict) -> None:
    """Persist updated preferences."""
    PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    to_save = {k: prefs.get(k, DEFAULTS[k]) for k in DEFAULTS}
    PREFS_PATH.write_text(json.dumps(to_save, indent=2))
