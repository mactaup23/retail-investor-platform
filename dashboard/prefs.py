"""
User preferences persisted to data/user_prefs.json.

Stores tax rates and account selection so they survive browser refreshes.
All fields have sensible defaults and are validated on load.
"""
from __future__ import annotations

import json
from pathlib import Path

PREFS_PATH = Path(__file__).parent.parent / "data" / "user_prefs.json"

DEFAULTS: dict = {
    "st_rate":    0.37,   # federal short-term / ordinary income rate
    "lt_rate":    0.20,   # federal long-term capital gains rate
    "state_rate": 0.0,    # state rate (additive to both ST and LT)
    "niit":       False,  # add 3.8% Net Investment Income Tax surcharge
    "account_id": "default",
}

_FLOAT_KEYS = {"st_rate", "lt_rate", "state_rate"}
_BOOL_KEYS  = {"niit"}
_STR_KEYS   = {"account_id"}


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
        except (json.JSONDecodeError, OSError):
            pass
    return prefs


def save(prefs: dict) -> None:
    """Persist updated preferences."""
    PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    to_save = {k: prefs.get(k, DEFAULTS[k]) for k in DEFAULTS}
    PREFS_PATH.write_text(json.dumps(to_save, indent=2))
