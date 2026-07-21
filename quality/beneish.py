"""
Beneish M-Score (Beneish, 1999) — 8-variable probability model for earnings
manipulation risk.

    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI
             - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

All 8 variables compare fiscal year t to fiscal year t-1:

    DSRI = (AR_t/Sales_t) / (AR_t-1/Sales_t-1)
           Days-sales-in-receivables index; receivables growing faster than
           sales suggests aggressive revenue recognition.
    GMI  = GrossMargin_t-1 / GrossMargin_t, GrossMargin = (Sales-COGS)/Sales
           >1 means margin deteriorated — firms with worsening margins have
           more incentive to manipulate.
    AQI  = [1-(CA_t+PPE_t)/TA_t] / [1-(CA_t-1+PPE_t-1)/TA_t-1]
           Rising share of "other" (non-current, non-PPE) assets suggests
           cost capitalization that should have been expensed. Beneish's
           original formula adds a third "Securities" term (long-term
           investment securities); dropped here (see note below), so this
           is the standard simplification used when that term isn't
           separately resolvable.
    SGI  = Sales_t / Sales_t-1
           Growth alone isn't manipulation, but growth firms face pressure
           to sustain the growth story.
    DEPI = DepreciationRate_t-1 / DepreciationRate_t, rate = Depreciation/(Depreciation+PPE_net)
           >1 means the depreciation rate slowed — assets depreciated more
           slowly, understating expense.
    SGAI = (SGA_t/Sales_t) / (SGA_t-1/Sales_t-1)
           Disproportionate SG&A growth relative to sales.
    TATA = (NetIncome_t - CFO_t) / TA_t
           Total accruals to total assets — Beneish's original variable is
           Income from Continuing Operations minus CFO; approximated here
           with Net Income (standard practitioner substitution — flagged,
           not silently substituted, same discipline as every other
           approximation in this codebase).
    LVGI = [(CL_t+LTD_t)/TA_t] / [(CL_t-1+LTD_t-1)/TA_t-1]
           LTD is long-term debt only (quality/fundamentals.py's
           long_term_debt field) — matches Beneish's own "total long-term
           debt" definition, distinct from dcf/fundamentals.py's total_debt
           which also includes short-term borrowings.

Securities (AQI's third term in Beneish's original formula) is dropped
rather than approximated. It was first approximated with
short_term_investments (GP's cached field) during development, but that
double-counts: short-term/marketable securities are already a component of
AssetsCurrent for every filer checked, so adding STI on top of CA inflates
the "core assets" sum above Total Assets for companies with large
short-term investment balances — confirmed empirically on NVDA, whose
short-term-investments-heavy balance sheet (t-1: CA=$80.1B, STI=$34.6B,
CA+PPE+STI=$121.0B > TA=$111.6B) drove AQI to -4.05, an impossible ratio
for this index. No long-term-investments XBRL tag is resolved anywhere in
this codebase, so the term is simply omitted (AQI = [1-(CA+PPE)/TA]_t /
[1-(CA+PPE)/TA]_t-1) rather than approximated with a component proven to
double-count — the standard simplification used in practitioner
implementations when a clean long-term-securities breakout isn't available.
Flagged via aqi_excludes_securities=True on every result, not silently
narrowed.

Threshold is genuinely ambiguous across sources and reported both ways:
Beneish's own paper reports -1.78 as the cost-minimizing cutoff (under a
20:1 assumed cost ratio of missing a manipulator vs. a false positive);
-2.22 is the more conservative threshold commonly cited in practitioner
literature. Both are surfaced rather than picking one silently.

All 8 inputs are required for both years — unlike Piotroski's 9
independent binary signals, Beneish's terms are multiplicatively combined
with fitted coefficients, so a missing term has no safe placeholder value
(there's no "neutral" ratio to default to without biasing the weighted
sum). A ticker missing any single component returns insufficient_data
rather than a partially-computed M-Score.

Business-model exclusion: reuses dcf/exclusions.py's bank/insurer/REIT
classification (see quality/inputs.py) — same rationale as Altman/Piotroski.
"""

from datetime import date

from quality.inputs import build_quality_panel

ORIGINAL_THRESHOLD = -1.78
PRACTITIONER_THRESHOLD = -2.22

_ADJACENT_YEAR_SPAN_DAYS = (350, 380)


def compute_beneish_m(ticker: str) -> dict:
    """
    Returns a dict with the M-Score, the 8 component indices, both
    threshold flags, period_end, prior_period_end, and "status" of
    "ok" / "excluded" / "insufficient_data".
    """
    panel, business_model_flag = build_quality_panel(ticker)
    if business_model_flag is not None:
        return {"status": "excluded", "business_model_flag": business_model_flag}
    if len(panel) < 2:
        return {"status": "insufficient_data", "business_model_flag": None}

    t, t1 = panel.iloc[-1], panel.iloc[-2]
    span = (date.fromisoformat(t["period_end"]) - date.fromisoformat(t1["period_end"])).days
    if not (_ADJACENT_YEAR_SPAN_DAYS[0] <= span <= _ADJACENT_YEAR_SPAN_DAYS[1]):
        return {"status": "insufficient_data", "business_model_flag": None}

    required = [
        t["accounts_receivable"], t["revenue"], t["cogs"], t["current_assets"],
        t["ppe_net"], t["total_assets"], t["depreciation"],
        t["sga"], t["net_income"], t["cfo"], t["current_liabilities"], t["long_term_debt"],
        t1["accounts_receivable"], t1["revenue"], t1["cogs"], t1["current_assets"],
        t1["ppe_net"], t1["total_assets"], t1["depreciation"],
        t1["sga"], t1["current_liabilities"], t1["long_term_debt"],
    ]
    if any(v is None or (isinstance(v, float) and v != v) for v in required):
        return {"status": "insufficient_data", "business_model_flag": None, "period_end": t["period_end"]}
    if not t["revenue"] or not t1["revenue"] or not t["total_assets"] or not t1["total_assets"]:
        return {"status": "insufficient_data", "business_model_flag": None, "period_end": t["period_end"]}

    gm_t = (t["revenue"] - t["cogs"]) / t["revenue"]
    gm_t1 = (t1["revenue"] - t1["cogs"]) / t1["revenue"]
    if not gm_t or not gm_t1:
        return {"status": "insufficient_data", "business_model_flag": None, "period_end": t["period_end"]}

    dep_rate_t = t["depreciation"] / (t["depreciation"] + t["ppe_net"]) if (t["depreciation"] + t["ppe_net"]) else None
    dep_rate_t1 = t1["depreciation"] / (t1["depreciation"] + t1["ppe_net"]) if (t1["depreciation"] + t1["ppe_net"]) else None
    if not dep_rate_t or not dep_rate_t1:
        return {"status": "insufficient_data", "business_model_flag": None, "period_end": t["period_end"]}

    sga_ratio_t = t["sga"] / t["revenue"]
    sga_ratio_t1 = t1["sga"] / t1["revenue"]
    if not sga_ratio_t1:
        return {"status": "insufficient_data", "business_model_flag": None, "period_end": t["period_end"]}

    other_assets_share_t = 1 - (t["current_assets"] + t["ppe_net"]) / t["total_assets"]
    other_assets_share_t1 = 1 - (t1["current_assets"] + t1["ppe_net"]) / t1["total_assets"]
    if not other_assets_share_t1:
        return {"status": "insufficient_data", "business_model_flag": None, "period_end": t["period_end"]}

    lvgi_t = (t["current_liabilities"] + t["long_term_debt"]) / t["total_assets"]
    lvgi_t1 = (t1["current_liabilities"] + t1["long_term_debt"]) / t1["total_assets"]
    if not lvgi_t1:
        return {"status": "insufficient_data", "business_model_flag": None, "period_end": t["period_end"]}

    dsri = (t["accounts_receivable"] / t["revenue"]) / (t1["accounts_receivable"] / t1["revenue"])
    gmi = gm_t1 / gm_t
    aqi = other_assets_share_t / other_assets_share_t1
    sgi = t["revenue"] / t1["revenue"]
    depi = dep_rate_t1 / dep_rate_t
    sgai = sga_ratio_t / sga_ratio_t1
    tata = (t["net_income"] - t["cfo"]) / t["total_assets"]
    lvgi = lvgi_t / lvgi_t1

    m_score = (
        -4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
        + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi
    )

    return {
        "status": "ok",
        "period_end": t["period_end"],
        "prior_period_end": t1["period_end"],
        "business_model_flag": None,
        "m_score": m_score,
        "flagged_original_threshold": m_score > ORIGINAL_THRESHOLD,
        "flagged_practitioner_threshold": m_score > PRACTITIONER_THRESHOLD,
        "dsri": dsri, "gmi": gmi, "aqi": aqi, "sgi": sgi,
        "depi": depi, "sgai": sgai, "tata": tata, "lvgi": lvgi,
        "aqi_excludes_securities": True,
        "tata_income_source": "net_income_proxy",
    }
