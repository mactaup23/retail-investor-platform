"""
Per-ticker fundamental data fetcher for the Quality & Health metrics
(DuPont, Altman Z''-Score, Piotroski F-Score, Beneish M-Score).

Pulls only the XBRL concepts none of the existing modules already extract.
Revenue, COGS, Total Assets, Cash come from factor_engine/gp_fundamentals.py's
cache; EBIT, effective tax rate, total debt, diluted shares come from
dcf/fundamentals.py's fetch. This module adds: Net Income, Stockholders'
Equity, Current Assets, Current Liabilities, Total Liabilities, Retained
Earnings, Operating Cash Flow (CFO), Accounts Receivable, PP&E (net), SG&A
expense, isolated Depreciation (not combined D&A), long-term debt only
(excluding short-term borrowings), and instant common shares outstanding.

All of these concepts were confirmed present in the already-cached raw XBRL
companyfacts JSON (data/gp/xbrl_raw/{cik}.json) during scoping — this module
costs zero new EDGAR requests, only new tag-parsing logic, same as DCF's own
fundamentals module before it. Annual observations only (no TTM-quarterly
rollup): these four metrics are academically specified as annual-frequency
comparisons (Piotroski/Beneish both compare fiscal year t to fiscal year
t-1), and the dashboard use case here is a live snapshot, not a backtest —
unlike GP, which needs quarterly refresh cadence for its rolling factor
construction.

Not persisted to its own CSV cache, same reasoning as dcf/fundamentals.py:
the underlying raw XBRL JSON is already cached by gp_xbrl_client, so
re-deriving here on each call is a cheap local parse, not a network round
trip.

Two fields get a source-tracked approximation rather than a clean tag,
flagged explicitly rather than silently substituted (same discipline as
DCF's tax-rate/debt/interest-expense source columns):

  - SG&A: SellingGeneralAndAdministrativeExpense resolves for many filers,
    but some (e.g. grocery/retail, per GP's own COGS-tag docstring) never
    tag a combined SG&A concept and instead separately tag
    GeneralAndAdministrativeExpense and SellingAndMarketingExpense. Falls
    back to summing those two when both resolve; sga_source distinguishes
    "reported" (combined tag) / "derived_sum" (G&A + S&M summed) / "none".
  - Depreciation (isolated from Amortization): Beneish's DEPI needs pure
    Depreciation, not the combined D&A figure dcf/fundamentals.py already
    resolves for FCF purposes. The standalone "Depreciation" tag doesn't
    resolve for every filer; when it doesn't, this falls back to DCF's own
    combined D&A value as an approximation, flagged
    depreciation_source="approximated_from_combined_da" rather than
    treated as equivalent — Amortization mixed in will understate the true
    depreciation rate's period-over-period movement for a filer with
    material intangible amortization, a real (documented, not hidden)
    limitation of using this proxy.

Long-term debt is deliberately NOT the same figure as dcf/fundamentals.py's
total_debt (which sums long-term debt AND short-term borrowings). Piotroski's
original leverage signal and Beneish's LVGI are both defined against
long-term debt specifically — short-term revolving/commercial-paper
borrowings are a different form of leverage than what these two papers'
"change in long-term debt" and "current liabilities + long-term debt"
constructions were built around. Reuses the same LT-debt tag lists
dcf/fundamentals.py already defined (imported, not duplicated) but resolves
only the long-term-debt subset, excluding dcf/fundamentals.py's separate
short-term-borrowings tags.
"""

import pandas as pd

from edgar_client import ticker_to_cik
from factor_engine import gp_xbrl_client
from factor_engine.gp_fundamentals import (
    _ANNUAL_SPAN_DAYS,
    _duration_periods,
    _facts_by_tag,
    _instant_periods,
)
from dcf.fundamentals import _LT_DEBT_CURRENT_TAGS, _LT_DEBT_NONCURRENT_TAGS

_NET_INCOME_TAGS = ["NetIncomeLoss", "ProfitLoss"]
_EQUITY_TAGS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]
_CURRENT_ASSETS_TAGS = ["AssetsCurrent"]
_CURRENT_LIABILITIES_TAGS = ["LiabilitiesCurrent"]
_TOTAL_LIABILITIES_TAGS = ["Liabilities"]
_RETAINED_EARNINGS_TAGS = ["RetainedEarningsAccumulatedDeficit"]
_CFO_TAGS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]
_AR_TAGS = ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"]
_PPE_TAGS = ["PropertyPlantAndEquipmentNet"]
_SGA_COMBINED_TAGS = ["SellingGeneralAndAdministrativeExpense"]
_GA_TAGS = ["GeneralAndAdministrativeExpense"]
_SM_TAGS = ["SellingAndMarketingExpense"]
_DEPRECIATION_ONLY_TAGS = ["Depreciation"]
_SHARES_OUTSTANDING_INSTANT_TAGS = ["CommonStockSharesOutstanding", "CommonStockSharesIssued"]

SGA_REPORTED = "reported"
SGA_DERIVED_SUM = "derived_sum"
SGA_NONE = "none"

DEPR_REPORTED = "reported"
DEPR_APPROXIMATED = "approximated_from_combined_da"
DEPR_NONE = "none"

_EMPTY_SENTINEL = pd.DataFrame({
    "period_end": [], "net_income": [], "equity": [], "current_assets": [],
    "current_liabilities": [], "total_liabilities": [], "retained_earnings": [],
    "cfo": [], "accounts_receivable": [], "ppe_net": [], "sga": [], "sga_source": [],
    "depreciation": [], "depreciation_source": [], "long_term_debt": [],
    "shares_outstanding_instant": [],
})


def _shares_facts_by_tag(us_gaap: dict, tags: list[str]) -> list[list[dict]]:
    """Instant share-count facts live under units.shares, not units.USD — see dcf/fundamentals.py's identical helper."""
    return [us_gaap.get(tag, {}).get("units", {}).get("shares", []) for tag in tags]


def _resolve_sga(us_gaap: dict) -> dict[str, dict]:
    combined_by_end = {
        end: float(e["val"])
        for (_s, end), e in _duration_periods(_facts_by_tag(us_gaap, _SGA_COMBINED_TAGS), _ANNUAL_SPAN_DAYS).items()
    }
    ga_by_end = {
        end: float(e["val"])
        for (_s, end), e in _duration_periods(_facts_by_tag(us_gaap, _GA_TAGS), _ANNUAL_SPAN_DAYS).items()
    }
    sm_by_end = {
        end: float(e["val"])
        for (_s, end), e in _duration_periods(_facts_by_tag(us_gaap, _SM_TAGS), _ANNUAL_SPAN_DAYS).items()
    }

    result: dict[str, dict] = {}
    for end in set(combined_by_end) | set(ga_by_end) | set(sm_by_end):
        if end in combined_by_end:
            result[end] = {"sga": combined_by_end[end], "sga_source": SGA_REPORTED}
        elif end in ga_by_end and end in sm_by_end:
            result[end] = {"sga": ga_by_end[end] + sm_by_end[end], "sga_source": SGA_DERIVED_SUM}
    return result


def _resolve_depreciation(us_gaap: dict, da_by_end: dict[str, float]) -> dict[str, dict]:
    """Isolated Depreciation, falling back to DCF's combined D&A when the standalone tag doesn't resolve — see module docstring."""
    depr_by_end = {
        end: float(e["val"])
        for (_s, end), e in _duration_periods(_facts_by_tag(us_gaap, _DEPRECIATION_ONLY_TAGS), _ANNUAL_SPAN_DAYS).items()
    }
    result: dict[str, dict] = {}
    for end in set(depr_by_end) | set(da_by_end):
        if end in depr_by_end:
            result[end] = {"depreciation": depr_by_end[end], "depreciation_source": DEPR_REPORTED}
        elif end in da_by_end:
            result[end] = {"depreciation": da_by_end[end], "depreciation_source": DEPR_APPROXIMATED}
    return result


def _resolve_long_term_debt(us_gaap: dict) -> dict[str, float]:
    """LT-noncurrent + LT-current only — excludes short-term borrowings, unlike dcf/fundamentals.py's total_debt. See module docstring."""
    lt_noncurrent_by_end = _instant_periods(_facts_by_tag(us_gaap, _LT_DEBT_NONCURRENT_TAGS))
    lt_current_by_end = _instant_periods(_facts_by_tag(us_gaap, _LT_DEBT_CURRENT_TAGS))
    result: dict[str, float] = {}
    for end in set(lt_noncurrent_by_end) | set(lt_current_by_end):
        result[end] = (
            (float(lt_noncurrent_by_end[end]["val"]) if end in lt_noncurrent_by_end else 0.0)
            + (float(lt_current_by_end[end]["val"]) if end in lt_current_by_end else 0.0)
        )
    return result


def _build_annual_observations(us_gaap: dict) -> list[dict]:
    ni_by_end = {
        end: float(e["val"])
        for (_s, end), e in _duration_periods(_facts_by_tag(us_gaap, _NET_INCOME_TAGS), _ANNUAL_SPAN_DAYS).items()
    }
    equity_by_end = _instant_periods(_facts_by_tag(us_gaap, _EQUITY_TAGS))
    ca_by_end = _instant_periods(_facts_by_tag(us_gaap, _CURRENT_ASSETS_TAGS))
    cl_by_end = _instant_periods(_facts_by_tag(us_gaap, _CURRENT_LIABILITIES_TAGS))
    total_liab_by_end = _instant_periods(_facts_by_tag(us_gaap, _TOTAL_LIABILITIES_TAGS))
    re_by_end = _instant_periods(_facts_by_tag(us_gaap, _RETAINED_EARNINGS_TAGS))
    cfo_by_end = {
        end: float(e["val"])
        for (_s, end), e in _duration_periods(_facts_by_tag(us_gaap, _CFO_TAGS), _ANNUAL_SPAN_DAYS).items()
    }
    ar_by_end = _instant_periods(_facts_by_tag(us_gaap, _AR_TAGS))
    ppe_by_end = _instant_periods(_facts_by_tag(us_gaap, _PPE_TAGS))
    sga_by_end = _resolve_sga(us_gaap)
    da_by_end = {
        end: float(e["val"])
        for (_s, end), e in _duration_periods(_facts_by_tag(us_gaap, ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet"]), _ANNUAL_SPAN_DAYS).items()
    }
    depr_by_end = _resolve_depreciation(us_gaap, da_by_end)
    ltd_by_end = _resolve_long_term_debt(us_gaap)
    shares_instant_by_end = _instant_periods(_shares_facts_by_tag(us_gaap, _SHARES_OUTSTANDING_INSTANT_TAGS))

    obs = []
    for end, ni_val in ni_by_end.items():
        if end not in equity_by_end:
            continue   # Equity is load-bearing for every one of these four metrics — skip, don't fabricate

        sga_info = sga_by_end.get(end, {"sga": None, "sga_source": SGA_NONE})
        depr_info = depr_by_end.get(end, {"depreciation": None, "depreciation_source": DEPR_NONE})

        obs.append({
            "period_end":              end,
            "net_income":              ni_val,
            "equity":                  float(equity_by_end[end]["val"]),
            "current_assets":          float(ca_by_end[end]["val"]) if end in ca_by_end else None,
            "current_liabilities":     float(cl_by_end[end]["val"]) if end in cl_by_end else None,
            "total_liabilities":       float(total_liab_by_end[end]["val"]) if end in total_liab_by_end else None,
            "retained_earnings":       float(re_by_end[end]["val"]) if end in re_by_end else None,
            "cfo":                     cfo_by_end.get(end),
            "accounts_receivable":     float(ar_by_end[end]["val"]) if end in ar_by_end else None,
            "ppe_net":                 float(ppe_by_end[end]["val"]) if end in ppe_by_end else None,
            "sga":                     sga_info["sga"],
            "sga_source":              sga_info["sga_source"],
            "depreciation":            depr_info["depreciation"],
            "depreciation_source":     depr_info["depreciation_source"],
            "long_term_debt":          ltd_by_end.get(end),
            "shares_outstanding_instant": shares_instant_by_end[end]["val"] if end in shares_instant_by_end else None,
        })
    return obs


def fetch_ticker_quality_fundamentals(ticker: str) -> pd.DataFrame:
    """
    Annual observations of the new fields needed by the Quality & Health
    metrics: period_end, net_income, equity, current_assets,
    current_liabilities, total_liabilities, retained_earnings, cfo,
    accounts_receivable, ppe_net, sga, sga_source, depreciation,
    depreciation_source, long_term_debt, shares_outstanding_instant.

    Returns an empty DataFrame (never raises) if the ticker has no
    resolvable CIK, no XBRL companyfacts, or no period with both Net Income
    and Equity resolvable (the two facts every one of these four metrics
    needs). Fields other metrics don't universally need (current
    assets/liabilities, retained earnings, CFO, AR, PP&E, SG&A,
    depreciation, long-term debt, instant shares) are left as None per
    period rather than dropping the whole observation — each metric module
    validates exactly which of these it needs and flags insufficient data
    itself, since different metrics need different subsets.
    """
    cik = ticker_to_cik(ticker)
    if cik is None:
        return _EMPTY_SENTINEL.copy()

    facts = gp_xbrl_client.fetch_company_facts(cik)
    if facts is None:
        return _EMPTY_SENTINEL.copy()

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return _EMPTY_SENTINEL.copy()

    observations = _build_annual_observations(us_gaap)
    if not observations:
        return _EMPTY_SENTINEL.copy()

    return (
        pd.DataFrame(observations)
        .drop_duplicates(subset="period_end")
        .sort_values("period_end")
        .reset_index(drop=True)
    )
