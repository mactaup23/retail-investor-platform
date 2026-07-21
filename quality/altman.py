"""
Altman Z''-Score (Altman, Hartzell & Peck, 1995) — the industry-neutral
variant of the Altman family, chosen over the original 1968 Z-Score
specifically to avoid needing a manufacturing-vs-non-manufacturing
classifier this codebase has no existing taxonomy for (only a coarse
bank/insurer/REIT flag exists — see dcf/exclusions.py — not a full
GICS/SIC taxonomy). Z'' drops the original's X5 (Sales/Total Assets) term
entirely, since asset turnover isn't comparable across industries, which is
exactly why this variant doesn't need a manufacturing/non-manufacturing
split in the first place.

    Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

    X1 = Working Capital / Total Assets = (Current Assets - Current Liabilities) / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Book Value of Equity / Total Liabilities   (book, not market — this
         variant was built to work for private/emerging-market firms too,
         so it deliberately never needs a market cap)

Zones (Altman et al. 1995):
    Z'' > 2.6         -> "Safe" zone (low bankruptcy risk)
    1.1 < Z'' <= 2.6   -> "Grey" zone (some risk, ambiguous)
    Z'' <= 1.1         -> "Distress" zone (elevated bankruptcy risk)

Business-model exclusion: reuses dcf/exclusions.py's bank/insurer/REIT
classification (see quality/inputs.py's module docstring) — Altman's own
work never applies this family of models to financial-services companies;
current-asset/liability tags are also frequently unresolvable for these
filers regardless of methodology stance.
"""

from quality.inputs import build_quality_panel

ZONE_SAFE = "safe"
ZONE_GREY = "grey"
ZONE_DISTRESS = "distress"

_SAFE_THRESHOLD = 2.6
_DISTRESS_THRESHOLD = 1.1


def _zone(z: float) -> str:
    if z > _SAFE_THRESHOLD:
        return ZONE_SAFE
    if z <= _DISTRESS_THRESHOLD:
        return ZONE_DISTRESS
    return ZONE_GREY


def compute_altman_z(ticker: str) -> dict:
    """
    Returns a dict with the Z'' score, zone, the 4 component ratios,
    period_end, and "status" of "ok" / "excluded" / "insufficient_data".
    "excluded" means dcf/exclusions.py flagged this ticker as a
    bank/insurer/REIT (business_model_flag carries the reason).
    """
    panel, business_model_flag = build_quality_panel(ticker)
    if business_model_flag is not None:
        return {"status": "excluded", "business_model_flag": business_model_flag}
    if panel.empty:
        return {"status": "insufficient_data", "business_model_flag": None}

    row = panel.iloc[-1]
    ca, cl = row["current_assets"], row["current_liabilities"]
    re, ta = row["retained_earnings"], row["total_assets"]
    ebit, equity, tl = row["ebit"], row["equity"], row["total_liabilities"]

    if any(v is None or (isinstance(v, float) and v != v) for v in (ca, cl, re, ta, ebit, equity, tl)):
        return {"status": "insufficient_data", "business_model_flag": None, "period_end": row["period_end"]}
    if not ta or not tl:
        return {"status": "insufficient_data", "business_model_flag": None, "period_end": row["period_end"]}

    x1 = (ca - cl) / ta
    x2 = re / ta
    x3 = ebit / ta
    x4 = equity / tl
    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    return {
        "status": "ok",
        "period_end": row["period_end"],
        "business_model_flag": None,
        "z_score": z,
        "zone": _zone(z),
        "x1_working_capital_to_assets": x1,
        "x2_retained_earnings_to_assets": x2,
        "x3_ebit_to_assets": x3,
        "x4_equity_to_liabilities": x4,
    }
