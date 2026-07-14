"""
Documented exclusions and observation-level filtering for the GP factor's
XBRL-derived fundamentals, discovered during the yfinance -> EDGAR XBRL
migration's full-universe validation (scripts/verify_gp_xbrl.py).

This module does NOT touch the raw fetch cache (data/gp/fundamentals/) —
that stays as "what we actually extracted" for auditability. Exclusion and
filtering are applied downstream, at factor-construction time
(factor_engine/factors/gp.py::_build_full_history()), the same way the
CUSIP non-US penalty is applied in convergence scoring rather than baked
into ingestion, or Baupost's filing-completeness caveat is a documented flag
rather than a silent data change.

Two distinct problems, two distinct fixes
-------------------------------------------
1. EXCLUDED_TICKERS — whole-ticker exclusion. For tickers where the
   implausibility is PERVASIVE (>=50% of periods) or structural (the
   business model has no COGS-equivalent XBRL concept at all), excluding
   individual bad observations wouldn't help because most/all of the
   ticker's history is affected. Three sub-categories, all excluded the
   same way but reasoned about differently:

   - "cogs_tag_mismatch_pervasive": the standard COGS-family tags
     (CostOfRevenue, CostOfGoodsAndServicesSold, CostOfGoodsSold,
     CostOfServices) don't capture this business's dominant cost driver.
     Confirmed examples: health insurers (UNH, ELV, CNC), whose real cost
     is claims/medical benefits paid, tagged under entirely different
     insurance-specific XBRL concepts; C.H. Robinson (CHRW), a freight
     broker whose dominant cost is "purchased transportation," not COGS;
     Murphy USA (MUSA), a fuel retailer. The remaining tickers in this
     category show the same >=50%-of-history implausibility pattern but
     the specific misrouted cost concept per ticker wasn't individually
     confirmed — documented as "pervasive, mechanism unconfirmed" rather
     than overclaiming a specific cause per name.

   - "reit_no_cogs_concept": confirmed via direct inspection of raw XBRL
     facts — zero entries across all four COGS-family tags for VNO, O, and
     SPG (spot-checked), and the same empty-data signature across the rest
     of this list. REITs' income statement (rental revenue minus
     depreciation/property-operating-expense/interest) has no
     production-cost analog for Novy-Marx GP to measure — this is scoping
     the factor to business models it was designed for, the same way
     financials are conventionally excluded from academic profitability
     factors, not a workaround for a fixable tag gap.

   - "units_mismatch_unexplained": raw revenue magnitude vs the prior
     yfinance-derived data differs by >2x or <0.5x with no identified
     common cause (checked: not a REIT, not in the pervasive COGS-mismatch
     set). Documented as unexplained rather than silently included.

2. drop_implausible_observations() — row-level filtering for everything
   NOT in EXCLUDED_TICKERS. Many tickers had exactly one (occasionally a
   few) severely corrupted observation — e.g. Smurfit WestRock (SW) had one
   period where a shell-company placeholder Assets value of $111 (from an
   early post-merger holding-company filing, superseded by the real
   consolidated $14.05B figure in a later filing) produced a gp_ratio of
   27.5 million — alongside dozens of other perfectly good quarters.
   Excluding the whole ticker over one bad filing would throw away good
   data (Kimberly-Clark: 66 good quarters, 1 corrupted one). Instead, only
   the specific implausible observation(s) are dropped, keeping the rest.
   A small number of tickers also show a MILD single-period outlier close
   to 1.0 (e.g. Netflix 2010: gp_ratio 1.02, a genuinely thin balance-sheet
   quarter early in its history) — these are filtered by the same
   mechanism for consistency, on the reasoning that Novy-Marx GP_ratio
   should not economically exceed 1.0 (Revenue - COGS > Total Assets),
   whether the cause is data corruption or a real extreme outlier quarter
   neither is representative of the factor's ongoing signal for that name.

Discovery method: full-universe validation comparing every XBRL-derived
gp_ratio against the yfinance-derived baseline on their 2021-2025 overlap
window (scripts/verify_gp_xbrl.py), which found the aggregate correlation
gate failing (-0.007) purely because of these tickers — excluding them
brought correlation on the "clean" 94% of the universe to 0.93.
"""

import pandas as pd

# ── Whole-ticker exclusions ─────────────────────────────────────────────────

_INSURANCE_COGS_MISMATCH = {
    "UNH": "Health insurer; dominant cost is medical claims/benefits, tagged under "
           "insurance-specific XBRL concepts, not any COGS-family tag.",
    "ELV": "Health insurer; same mechanism as UNH.",
    "CNC": "Health insurer; same mechanism as UNH.",
}

_LOGISTICS_FRANCHISE_FUEL_COGS_MISMATCH = {
    "CHRW": "Freight brokerage; dominant cost is purchased transportation, not COGS.",
    "MUSA": "Fuel retailer; COGS tag captured a much narrower cost sub-line than total "
            "cost of fuel + merchandise sold.",
}

_PERVASIVE_MECHANISM_UNCONFIRMED = {
    ticker: "Pervasive implausible gp_ratio (>=50% of periods) matching the same "
            "COGS-tag-mismatch signature as the confirmed cases above, but the "
            "specific misrouted cost concept for this ticker wasn't individually verified."
    for ticker in ["PRG", "DY", "WINA", "CHWY", "TTEK", "CAKE", "EAT", "CBRL",
                   "AGNT", "DPZ", "PRSU", "ENS", "CARG"]
}

_REIT_NO_COGS_CONCEPT = {
    ticker: "REIT; zero entries across all four COGS-family XBRL tags (confirmed by "
            "direct inspection for VNO/O/SPG) — no production-cost concept exists for "
            "a rental-income business model."
    for ticker in [
        "O", "FR", "HR", "UE", "AAT", "ABR", "ADC", "AKR", "ARR", "DLR", "EGP", "ELS",
        "EPR", "ESS", "FRT", "GTY", "HIW", "KIM", "KRC", "KRG", "NNN", "RWT", "SLG",
        "VNO", "BXMT", "EPRT", "FBRT", "FCPT", "IIPR", "NTST", "NXRT", "REXR", "SBRA",
        "STWD", "TRNO",
    ]
}

_UNITS_MISMATCH_UNEXPLAINED = {
    ticker: "Raw revenue magnitude vs yfinance differs by >2x or <0.5x with no "
            "identified common cause (not a REIT, not in the confirmed COGS-mismatch set)."
    for ticker in ["CPT", "FLS", "CALY", "TDS", "WOR"]   # PRSU already in the pervasive set above
}

EXCLUDED_TICKERS: dict[str, str] = {
    **_INSURANCE_COGS_MISMATCH,
    **_LOGISTICS_FRANCHISE_FUEL_COGS_MISMATCH,
    **_PERVASIVE_MECHANISM_UNCONFIRMED,
    **_REIT_NO_COGS_CONCEPT,
    **_UNITS_MISMATCH_UNEXPLAINED,
}


# ── Observation-level filter ────────────────────────────────────────────────

_MAX_PLAUSIBLE_GP_RATIO = 1.0   # Revenue - COGS should not exceed Total Assets in a
                                # legitimate observation; see module docstring point 2.


def drop_implausible_observations(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Remove individual rows with |gp_ratio| > 1.0 from a ticker's fundamentals
    DataFrame, keeping the rest of its history. Only meant to be called for
    tickers NOT in EXCLUDED_TICKERS — see module docstring for why these are
    two different mechanisms.
    """
    if df is None or df.empty:
        return df
    return df[df["gp_ratio"].abs() <= _MAX_PLAUSIBLE_GP_RATIO].reset_index(drop=True)
