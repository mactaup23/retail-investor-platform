"""
Per-ticker fundamental data fetcher for the DCF valuation engine.

Pulls the additional XBRL concepts a DCF needs that the GP factor's
gp_fundamentals.py doesn't already extract: EBIT, D&A, capex, interest
expense, total debt, effective tax rate, diluted shares outstanding. Revenue
and cash are NOT re-derived here — they're read straight from GP's own
cached fundamentals CSV (data/gp/fundamentals/{ticker}.csv), same underlying
XBRL facts GP already resolved, with a same-module fallback (re-deriving
Revenue from XBRL directly) for a ticker GP hasn't fetched yet.

Fact-resolution machinery (_facts_by_tag / _duration_periods /
_instant_periods / _ANNUAL_SPAN_DAYS / _REVENUE_TAGS) is imported directly
from factor_engine.gp_fundamentals rather than reimplemented — it already
handles tag-priority fallback, restatement tie-breaking (earliest-filed
wins), and duration-span filtering correctly; duplicating it here would just
be a second copy of the same edge cases to maintain.

Raw JSON source: the same data/gp/xbrl_raw/{cik}.json cache GP's own fetch
populated (factor_engine/gp_xbrl_client.py) — for any ticker already in the
GP universe (~1500 S&P Composite 1500 names), pulling these new concepts
costs zero new EDGAR requests, only new tag-parsing logic. A ticker outside
that universe still works: gp_xbrl_client.fetch_company_facts() fetches (and
caches) it fresh on first call.

Effective tax rate
-------------------
tax_expense / pretax_income, clamped to [0.10, 0.35] — a one-off year (NOL
carryforward, repatriation charge, valuation-allowance release) can otherwise
swing a single year's effective rate to something implausible to hold
constant across a 10-year forward projection, the same discipline as PEAD's
growth-rate capping and GP's _MAX_PLAUSIBLE_GP_RATIO recalibration. If no
historical period has a resolvable pretax-income tag at all, falls back to
the US statutory federal rate (0.21) with tax_rate_source="assumed_statutory"
— excluding the ticker entirely over one unresolvable ratio (when revenue,
EBIT, and everything else needed is otherwise fine) would be a worse trade
than a documented, conservative default.

Total debt
----------
Sums whichever of {combined long+short debt tag, long-term debt (noncurrent),
long-term debt due within one year, short-term borrowings} resolve for a
period, defaulting any that don't resolve to 0. Unlike NIBCL's AP/accrued
split (where "AP but no accrued" vs "neither" are meaningfully different
gaps), a company legitimately having zero current debt or zero short-term
borrowings is common and unremarkable — so debt_source here is a 2-tier
signal (DEBT_RESOLVED / DEBT_NONE), not a 3-tier confidence gradient like
NIBCL/goodwill: a fake "partial" tier would imply a confidence distinction
this concept doesn't actually support.
"""

import os
import statistics

import pandas as pd

from edgar_client import ticker_to_cik
from factor_engine import gp_xbrl_client
from factor_engine.gp_fundamentals import (
    _ANNUAL_SPAN_DAYS,
    _REVENUE_TAGS,
    _duration_periods,
    _facts_by_tag,
    _instant_periods,
)

_GP_FUNDAMENTALS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gp", "fundamentals")

MIN_TAX_RATE = 0.10
MAX_TAX_RATE = 0.35
_STATUTORY_TAX_RATE = 0.21

DEBT_RESOLVED = "resolved"   # at least one debt-shaped tag resolved for this period
DEBT_NONE = "none"           # no debt tag resolved at all — total_debt defaults to 0

_EBIT_TAGS = ["OperatingIncomeLoss"]
_DA_TAGS = [
    "DepreciationDepletionAndAmortization",
    "DepreciationAmortizationAndAccretionNet",
    "Depreciation",
]
_CAPEX_TAGS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "PaymentsForCapitalImprovements",
]
_INTEREST_EXPENSE_TAGS = ["InterestExpense", "InterestExpenseDebt", "InterestAndDebtExpense"]
_TAX_EXPENSE_TAGS = ["IncomeTaxExpenseBenefit"]
_PRETAX_INCOME_TAGS = [
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
]
_DILUTED_SHARES_TAGS = ["WeightedAverageNumberOfDilutedSharesOutstanding"]

_DEBT_COMBINED_TAGS = ["DebtLongtermAndShorttermCombinedAmount"]
# Tried in priority order *per period* (see _instant_periods) — some filers
# (confirmed: KO) switched from the plain LongTermDebt*/ShortTermBorrowings
# tags to lease-inclusive *AndCapitalLeaseObligations* variants starting in
# 2025, the same per-period tag-switch pattern GP's module docstring
# documents for Revenue around the ASC 606 adoption wave (~2018). The
# lease-inclusive tags mix in capitalized operating leases, which is
# standard practice for a post-ASC-842 total-obligations figure, not a
# distinct/lesser concept.
_LT_DEBT_NONCURRENT_TAGS = ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"]
_LT_DEBT_CURRENT_TAGS = ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"]
_ST_BORROWINGS_TAGS = ["ShortTermBorrowings", "DebtCurrent", "OtherShortTermBorrowings"]

_EMPTY_SENTINEL = pd.DataFrame({
    "period_end": [], "revenue": [], "cash": [], "ebit": [], "da": [], "capex": [],
    "interest_expense": [], "interest_expense_source": [], "total_debt": [], "debt_source": [],
    "tax_expense": [], "pretax_income": [], "effective_tax_rate": [], "tax_rate_source": [],
    "diluted_shares": [],
})


def _shares_facts_by_tag(us_gaap: dict, tags: list[str]) -> list[list[dict]]:
    """
    Like gp_fundamentals._facts_by_tag, but for the "shares" unit rather than
    "USD" — diluted-share-count facts are reported under units.shares, not
    units.USD, so the shared USD-only helper silently returns an empty list
    for this concept (confirmed empirically: AAPL's
    WeightedAverageNumberOfDilutedSharesOutstanding fact only exists under
    units.shares). Kept local to this module rather than generalizing the
    shared helper, since every other tag list here (and every GP tag) is
    genuinely USD-denominated.
    """
    return [us_gaap.get(tag, {}).get("units", {}).get("shares", []) for tag in tags]


def _revenue_and_cash_from_gp_cache(ticker: str) -> "pd.DataFrame | None":
    """
    Annual (period_end, revenue, cash) rows from GP's already-cached
    fundamentals CSV, if present. Returns None if no cache file exists for
    this ticker (caller falls back to re-deriving Revenue directly; Cash has
    no fallback here — see fetch_ticker_dcf_fundamentals).
    """
    path = os.path.join(_GP_FUNDAMENTALS_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    annual = df[df["freq"] == "A"][["period_end", "revenue", "cash"]].copy()
    return annual if not annual.empty else None


def _revenue_from_xbrl(us_gaap: dict) -> dict[str, float]:
    """Fallback Revenue derivation for a ticker GP hasn't cached — same tag list, annual span only."""
    revenue_periods = _duration_periods(_facts_by_tag(us_gaap, _REVENUE_TAGS), _ANNUAL_SPAN_DAYS)
    return {end: float(e["val"]) for (_start, end), e in revenue_periods.items()}


def _resolve_debt(us_gaap: dict) -> dict[str, dict]:
    """(total_debt, debt_source) per balance-sheet date — see module docstring."""
    combined_by_end = _instant_periods(_facts_by_tag(us_gaap, _DEBT_COMBINED_TAGS))
    lt_noncurrent_by_end = _instant_periods(_facts_by_tag(us_gaap, _LT_DEBT_NONCURRENT_TAGS))
    lt_current_by_end = _instant_periods(_facts_by_tag(us_gaap, _LT_DEBT_CURRENT_TAGS))
    st_borrowings_by_end = _instant_periods(_facts_by_tag(us_gaap, _ST_BORROWINGS_TAGS))

    all_ends = (
        set(combined_by_end) | set(lt_noncurrent_by_end)
        | set(lt_current_by_end) | set(st_borrowings_by_end)
    )
    result: dict[str, dict] = {}
    for end in all_ends:
        if end in combined_by_end:
            debt_val = float(combined_by_end[end]["val"])
        else:
            debt_val = (
                float(lt_noncurrent_by_end[end]["val"]) if end in lt_noncurrent_by_end else 0.0
            ) + (
                float(lt_current_by_end[end]["val"]) if end in lt_current_by_end else 0.0
            ) + (
                float(st_borrowings_by_end[end]["val"]) if end in st_borrowings_by_end else 0.0
            )
        result[end] = {"total_debt": debt_val, "debt_source": DEBT_RESOLVED}
    return result


INTEREST_REPORTED = "reported"
INTEREST_CARRIED_FORWARD = "carried_forward"   # most recent period this tag DID resolve, reused as-is
INTEREST_NONE = "none"                          # tag never resolved in this company's history


def _carry_forward(values_by_end: dict[str, float], target_end: str) -> tuple[float, str]:
    """
    Interest expense specifically has a real, structural gap this codebase
    has already seen once before (AAPL's Goodwill tag stopping in 2017):
    AAPL's InterestExpense/InterestExpenseDebt tags have no entries after FY
    2023 (confirmed empirically — Apple folded interest expense into "Other
    income/expense, net" starting FY2024, the same kind of disclosure change
    as the earlier Goodwill drop). Silently defaulting a missing period to 0
    would be indistinguishable from a genuinely debt-free company paying no
    interest — a materially wrong signal for a company like AAPL carrying
    $90B+ of debt. Instead: if the target period itself has no resolved
    value, carry forward the most recent *earlier* period that did (ISO
    date strings sort correctly as strings), flagged as such rather than
    silently substituted. Only falls to "none" (0.0) if the tag never
    resolved anywhere in the company's history.
    """
    if target_end in values_by_end:
        return values_by_end[target_end], INTEREST_REPORTED
    earlier = [e for e in values_by_end if e <= target_end]
    if earlier:
        latest = max(earlier)
        return values_by_end[latest], INTEREST_CARRIED_FORWARD
    return 0.0, INTEREST_NONE


def _build_annual_observations(ticker: str, us_gaap: dict) -> list[dict]:
    revenue_cash = _revenue_and_cash_from_gp_cache(ticker)
    if revenue_cash is not None:
        revenue_by_end = dict(zip(revenue_cash["period_end"], revenue_cash["revenue"]))
        cash_by_end = dict(zip(revenue_cash["period_end"], revenue_cash["cash"]))
    else:
        revenue_by_end = _revenue_from_xbrl(us_gaap)
        cash_by_end = {}   # no fallback for Cash — periods without it just carry cash=0 (see below)

    ebit_by_end = {
        end: float(e["val"])
        for (_s, end), e in _duration_periods(_facts_by_tag(us_gaap, _EBIT_TAGS), _ANNUAL_SPAN_DAYS).items()
    }
    da_by_end = {
        end: float(e["val"])
        for (_s, end), e in _duration_periods(_facts_by_tag(us_gaap, _DA_TAGS), _ANNUAL_SPAN_DAYS).items()
    }
    capex_by_end = {
        end: float(e["val"])
        for (_s, end), e in _duration_periods(_facts_by_tag(us_gaap, _CAPEX_TAGS), _ANNUAL_SPAN_DAYS).items()
    }
    interest_by_end = {
        end: float(e["val"])
        for (_s, end), e in _duration_periods(_facts_by_tag(us_gaap, _INTEREST_EXPENSE_TAGS), _ANNUAL_SPAN_DAYS).items()
    }
    tax_expense_by_end = {
        end: float(e["val"])
        for (_s, end), e in _duration_periods(_facts_by_tag(us_gaap, _TAX_EXPENSE_TAGS), _ANNUAL_SPAN_DAYS).items()
    }
    pretax_by_end = {
        end: float(e["val"])
        for (_s, end), e in _duration_periods(_facts_by_tag(us_gaap, _PRETAX_INCOME_TAGS), _ANNUAL_SPAN_DAYS).items()
    }
    shares_by_end = {
        end: float(e["val"])
        for (_s, end), e in _duration_periods(_shares_facts_by_tag(us_gaap, _DILUTED_SHARES_TAGS), _ANNUAL_SPAN_DAYS).items()
    }
    debt_by_end = _resolve_debt(us_gaap)

    # Historical median effective tax rate — computed once across all periods
    # with a resolvable pretax income, used as the assumed_statutory fallback
    # basis check (see below) and to detect the "never resolvable" case.
    resolvable_rates = [
        tax_expense_by_end[end] / pretax_by_end[end]
        for end in pretax_by_end
        if end in tax_expense_by_end and pretax_by_end[end]
    ]

    obs = []
    for end, revenue_val in revenue_by_end.items():
        if end not in ebit_by_end:
            continue   # EBIT is load-bearing for FCF — skip periods without it, don't fabricate

        if end in tax_expense_by_end and end in pretax_by_end and pretax_by_end[end]:
            raw_rate = tax_expense_by_end[end] / pretax_by_end[end]
            effective_tax_rate = min(max(raw_rate, MIN_TAX_RATE), MAX_TAX_RATE)
            tax_rate_source = "reported"
        elif resolvable_rates:
            effective_tax_rate = min(max(statistics.median(resolvable_rates), MIN_TAX_RATE), MAX_TAX_RATE)
            tax_rate_source = "company_median"
        else:
            effective_tax_rate = _STATUTORY_TAX_RATE
            tax_rate_source = "assumed_statutory"

        debt_info = debt_by_end.get(end, {"total_debt": 0.0, "debt_source": DEBT_NONE})
        interest_val, interest_source = _carry_forward(interest_by_end, end)

        obs.append({
            "period_end":           end,
            "revenue":              revenue_val,
            "cash":                 cash_by_end.get(end, 0.0),
            "ebit":                 ebit_by_end[end],
            "da":                   da_by_end.get(end, 0.0),
            "capex":                capex_by_end.get(end, 0.0),
            "interest_expense":     interest_val,
            "interest_expense_source": interest_source,
            "total_debt":           debt_info["total_debt"],
            "debt_source":          debt_info["debt_source"],
            "tax_expense":          tax_expense_by_end.get(end),
            "pretax_income":        pretax_by_end.get(end),
            "effective_tax_rate":   effective_tax_rate,
            "tax_rate_source":      tax_rate_source,
            "diluted_shares":       shares_by_end.get(end),
        })
    return obs


def fetch_ticker_dcf_fundamentals(ticker: str) -> pd.DataFrame:
    """
    Annual DCF-input observations for one ticker: period_end, revenue, cash,
    ebit, da, capex, interest_expense, total_debt, debt_source, tax_expense,
    pretax_income, effective_tax_rate, tax_rate_source, diluted_shares.

    Returns an empty DataFrame (never raises) if the ticker has no
    resolvable CIK, no XBRL companyfacts, or no period with a resolvable
    Revenue + EBIT pair (the two facts every downstream computation needs).
    Not persisted to its own CSV cache — the underlying raw XBRL JSON is
    already cached by gp_xbrl_client, so re-deriving here is a cheap local
    parse, not a network round trip.
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

    observations = _build_annual_observations(ticker, us_gaap)
    if not observations:
        return _EMPTY_SENTINEL.copy()

    return (
        pd.DataFrame(observations)
        .drop_duplicates(subset="period_end")
        .sort_values("period_end")
        .reset_index(drop=True)
    )
