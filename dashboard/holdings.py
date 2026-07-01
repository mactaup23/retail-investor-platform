"""
Portfolio holdings editor support — ticker validation, CSV import, and
weight normalization for the Portfolio page's holdings table.

Ticker validation and company-name lookups hit yfinance, so results are
cached (24h) to keep the editor responsive on repeat renders — Streamlit
reruns the whole script on every widget interaction.
"""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

MIN_POSITIONS = 1
MAX_POSITIONS = 20
MIN_WEIGHT_PCT = 0.1
MAX_WEIGHT_PCT = 100.0

CSV_TEMPLATE = "ticker,weight\nVTI,0.2437\nQQQM,0.1140\nSCHD,0.1178\n"


@st.cache_data(ttl=86400, show_spinner=False)
def lookup_ticker(ticker: str) -> dict:
    """
    Validate a ticker against yfinance and fetch its display name.

    Returns {"valid": bool, "name": str | None, "error": str | None}.
    """
    import yfinance as yf

    try:
        hist = yf.Ticker(ticker).history(period="5d")
    except Exception:
        hist = None

    if hist is None or hist.empty:
        return {
            "valid": False,
            "name": None,
            "error": f"Ticker {ticker} not found — please check the symbol and try again.",
        }

    name = ticker
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ticker
    except Exception:
        pass

    return {"valid": True, "name": name, "error": None}


def total_weight_pct(holdings: list[dict]) -> float:
    return sum(h["weight"] for h in holdings) * 100


def csv_template_bytes() -> bytes:
    return CSV_TEMPLATE.encode("utf-8")


def parse_csv(file_bytes: bytes) -> tuple[list[dict], list[str]]:
    """
    Parse an uploaded ticker/weight CSV.

    Weight values are auto-detected as percentages (e.g. 24.37) or decimal
    fractions (e.g. 0.2437) based on magnitude — if any value exceeds 1.5 the
    whole column is treated as percentages.

    Returns (rows, errors). rows is a list of {"ticker": str, "weight": float}
    with weight always normalized to a decimal fraction. On any structural
    error, rows is empty and errors explains why.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        return [], [f"Could not read CSV: {e}"]

    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns or "weight" not in df.columns:
        return [], ["CSV must have 'ticker' and 'weight' columns."]

    df = df[["ticker", "weight"]].dropna()
    if df.empty:
        return [], ["CSV has no valid rows."]
    if len(df) > MAX_POSITIONS:
        return [], [f"CSV has {len(df)} rows — maximum is {MAX_POSITIONS} positions."]

    try:
        df["weight"] = df["weight"].astype(float)
    except (ValueError, TypeError):
        return [], ["Weight column must be numeric."]

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    if df["ticker"].duplicated().any():
        dupes = sorted(set(df["ticker"][df["ticker"].duplicated()]))
        return [], [f"Duplicate tickers in CSV: {', '.join(dupes)}"]

    is_percentage = (df["weight"] > 1.5).any()
    weights = df["weight"] / 100 if is_percentage else df["weight"]

    errors = []
    rows = []
    for ticker, weight in zip(df["ticker"], weights):
        if not (MIN_WEIGHT_PCT / 100 <= weight <= MAX_WEIGHT_PCT / 100):
            errors.append(
                f"{ticker}: weight {weight * 100:.2f}% is outside the "
                f"{MIN_WEIGHT_PCT}%–{MAX_WEIGHT_PCT}% range."
            )
            continue
        rows.append({"ticker": ticker, "weight": float(weight)})

    return rows, errors


def weights_dict(holdings: list[dict]) -> dict[str, float]:
    """Convert a holdings list ([{ticker, weight}, ...]) to a ticker -> weight dict."""
    return {h["ticker"]: h["weight"] for h in holdings}


def normalize_weights_dict(weights: dict[str, float]) -> dict[str, float]:
    """
    Scale a ticker -> weight dict proportionally so it sums to 1.0.

    Used as the single normalization point shared by the Portfolio page and
    the Signals-tab prewarm, so both compare/cache against the same
    canonical (normalized) representation of a given set of holdings.
    """
    total = sum(weights.values())
    if total <= 0:
        return dict(weights)
    return {t: w / total for t, w in weights.items()}


def weights_match(a: dict[str, float] | None, b: dict[str, float]) -> bool:
    """True if two ticker -> weight dicts have the same tickers and near-equal weights."""
    if not a:
        return False
    if set(a) != set(b):
        return False
    return all(abs(float(a[t]) - b[t]) < 1e-6 for t in b)
