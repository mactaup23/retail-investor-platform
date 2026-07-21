"""
Piotroski F-Score (Piotroski, 2000) — 9 binary signals across profitability,
leverage/liquidity, and operating efficiency, summed to a 0-9 score.

Profitability (4):
    1. ROA > 0                          (Net Income_t / Total Assets_t > 0)
    2. CFO > 0                          (Operating Cash Flow_t > 0)
    3. Delta ROA > 0                    (ROA_t > ROA_t-1)
    4. Accruals: CFO/TA > ROA           (quality of earnings — cash profitability exceeds accrual profitability)

Leverage / Liquidity / Source of funds (3):
    5. Delta Leverage < 0               (LongTermDebt_t/TA_t < LongTermDebt_t-1/TA_t-1 — lower leverage is better)
    6. Delta Current Ratio > 0          (CurrentAssets_t/CurrentLiabilities_t > the prior year's ratio)
    7. No new shares issued             (shares outstanding_t <= shares outstanding_t-1 — no dilution)

Operating efficiency (2):
    8. Delta Gross Margin > 0           ((Revenue-COGS)/Revenue improved YoY)
    9. Delta Asset Turnover > 0         (Revenue_t/TA_t improved YoY)

Signal 5 uses long-term debt specifically (quality/fundamentals.py's
long_term_debt field, excluding short-term borrowings) — Piotroski's
original leverage signal is about long-term debt reduction, a different
form of deleveraging than paying down short-term revolving credit, and
conflating the two would misclassify a company that termed out short-term
debt into long-term debt as "more leveraged" when it isn't.

Piotroski's own paper validated this specifically on a high book-to-market
(value stock) universe — high F-Scores predicting outperformance was shown
in that context, not as a universal claim across all stocks. Surfaced as a
caveat wherever this score is displayed, not treated as a standalone buy
signal.

Business-model exclusion: reuses dcf/exclusions.py's bank/insurer/REIT
classification (see quality/inputs.py). This is doubly justified here
specifically (not just methodological preference): AssetsCurrent /
LiabilitiesCurrent — needed for signal 6 — are frequently unresolvable for
these filers' balance-sheet templates regardless of any methodology
choice, a genuine data-availability gap on top of the standard
methodological-fit argument.

Requires two consecutive fiscal years (roughly 350-380 days apart) — a
wider or narrower gap means the two most recent rows in the panel aren't
actually adjacent fiscal years (a data gap), and the score is not computed
rather than silently comparing non-adjacent periods.
"""

from datetime import date

from quality.inputs import build_quality_panel

_ADJACENT_YEAR_SPAN_DAYS = (350, 380)


def compute_piotroski_f(ticker: str) -> dict:
    """
    Returns a dict with the F-Score (0-9), each of the 9 individual signal
    booleans, period_end (current year) and prior_period_end, and "status"
    of "ok" / "excluded" / "insufficient_data".
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
        t["net_income"], t["total_assets"], t["cfo"], t["long_term_debt"],
        t["current_assets"], t["current_liabilities"], t["shares_outstanding_instant"],
        t["revenue"], t["cogs"],
        t1["net_income"], t1["total_assets"], t1["long_term_debt"],
        t1["current_assets"], t1["current_liabilities"], t1["shares_outstanding_instant"],
        t1["revenue"], t1["cogs"],
    ]
    if any(v is None or (isinstance(v, float) and v != v) for v in required):
        return {"status": "insufficient_data", "business_model_flag": None, "period_end": t["period_end"]}
    if not t["total_assets"] or not t1["total_assets"] or not t["current_liabilities"] or not t1["current_liabilities"] or not t["revenue"] or not t1["revenue"]:
        return {"status": "insufficient_data", "business_model_flag": None, "period_end": t["period_end"]}

    roa_t = t["net_income"] / t["total_assets"]
    roa_t1 = t1["net_income"] / t1["total_assets"]
    cfo_to_ta_t = t["cfo"] / t["total_assets"]
    leverage_t = t["long_term_debt"] / t["total_assets"]
    leverage_t1 = t1["long_term_debt"] / t1["total_assets"]
    current_ratio_t = t["current_assets"] / t["current_liabilities"]
    current_ratio_t1 = t1["current_assets"] / t1["current_liabilities"]
    gross_margin_t = (t["revenue"] - t["cogs"]) / t["revenue"]
    gross_margin_t1 = (t1["revenue"] - t1["cogs"]) / t1["revenue"]
    asset_turnover_t = t["revenue"] / t["total_assets"]
    asset_turnover_t1 = t1["revenue"] / t1["total_assets"]

    signals = {
        "f1_positive_roa":        roa_t > 0,
        "f2_positive_cfo":        t["cfo"] > 0,
        "f3_improving_roa":       roa_t > roa_t1,
        "f4_cfo_exceeds_roa":     cfo_to_ta_t > roa_t,
        "f5_decreasing_leverage": leverage_t < leverage_t1,
        "f6_improving_current_ratio": current_ratio_t > current_ratio_t1,
        "f7_no_dilution":         t["shares_outstanding_instant"] <= t1["shares_outstanding_instant"],
        "f8_improving_gross_margin": gross_margin_t > gross_margin_t1,
        "f9_improving_asset_turnover": asset_turnover_t > asset_turnover_t1,
    }
    f_score = sum(signals.values())

    return {
        "status": "ok",
        "period_end": t["period_end"],
        "prior_period_end": t1["period_end"],
        "business_model_flag": None,
        "f_score": f_score,
        **signals,
    }
