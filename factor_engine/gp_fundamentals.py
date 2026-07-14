"""
Per-ticker fundamental data fetcher for the Gross Profitability (GP) factor.

Novy-Marx (2013) gross profitability: GP_ratio = (Revenue - COGS) / Total Assets

Data source: SEC EDGAR XBRL companyfacts (not yfinance)
---------------------------------------------------------
yfinance's free endpoint exposes at most ~5 years of annual financial
statements and ~5 quarters of quarterly statements per ticker — a hard wall
of that data source that bounded the GP factor to roughly 2021-present (see
git history / CLAUDE.md for that era's design notes). This module now pulls
Revenue, COGS, and Total Assets directly from SEC EDGAR's XBRL companyfacts
API (data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json), which carries a
company's full tagged filing history back to its XBRL adoption (~2009-2011
depending on filer size) — long enough to reach the platform's other 2013
history floors (13F XML cutoff, MTUM ETF inception). The raw fetch/cache
layer lives in gp_xbrl_client.py; this module owns tag selection, duration
filtering, and TTM/annual assembly. See get_gp_coverage_start() in
factor_engine/factors/gp.py for the actual realized coverage floor once data
is pulled — it's derived from the data, not hardcoded here.

XBRL taxonomy tag inconsistency
--------------------------------
Companies don't all tag the same GAAP concept with the same XBRL element —
most visibly, a widespread switch from "SalesRevenueNet"-style tags to
"RevenueFromContractWithCustomerExcludingAssessedTax" around the ASC 606
adoption wave (~2018), and less commonly "CostOfGoodsSold" vs
"CostOfGoodsAndServicesSold" vs "CostOfRevenue" vs "CostOfServices" for cost
of revenue depending on the filer's industry template. _REVENUE_TAGS /
_COGS_TAGS / _ASSETS_TAGS below are tried in priority order *per period*
(not once per company) — a company that changed tags mid-history has both
tags checked for every period, so pre- and post-switch quarters both resolve
correctly rather than one era going missing.

The Q4 gap: almost no filer tags a discrete fourth quarter
-------------------------------------------------------------
This is the single largest source of "missing" quarterly data, and it's
structural, not a bug: 10-Ks report full-year figures, and the vast majority
of filers never separately tag "three months ended [fiscal year end]" —
verified empirically across the sampled universe (e.g. AAPL: 12/63 quarterly
periods missing a direct quarterly tag, and every single one is its fiscal
Q4). A naive "trailing-twelve-month window tainted if any component is
estimated" rollup would therefore mark *100% of TTM observations* as
estimated forever, since every valid 4-quarter window necessarily contains
exactly one Q4 — making the source column constant and useless as an audit
trail.

Fix: wherever a fiscal year's full-year (FY) duration fact AND its matching
nine-month year-to-date (YTD) duration fact both resolve for revenue AND
cogs (the Q3 10-Q almost always tags both the discrete quarter and the
YTD-cumulative figure), Q4 is derived as FY − 9mo-YTD — exact arithmetic
from two directly-tagged facts, not an estimate. This is a distinct,
higher-confidence source tier ("derived_from_ytd_subtraction") from the
margin-based fallback below, which only fires when even the YTD fact is
unavailable.

COGS gap-filling via historical gross margin (last resort)
-------------------------------------------------------------
Some filers (services-heavy businesses especially) never tag a COGS-shaped
concept for some or all periods even though Revenue and Assets are present,
and — for Q4 specifically — the YTD-subtraction path above also isn't
available. Rather than dropping those periods outright, this module
estimates COGS as Revenue x (1 - median historical gross margin), using the
company's own median gross margin across whichever of its periods (same
frequency bucket) have reported or YTD-derived COGS — never a cross-company
or industry average, and never periods that were themselves estimated.
Estimation only fires when at least 2 such margin observations exist to
compute a median from; otherwise the period is skipped exactly as it would
be with no fallback at all.

Every observation carries a "source" column, one of three tiers in
confidence order — "reported" (both revenue and COGS resolved from an
actual directly-tagged XBRL fact), "derived_from_ytd_subtraction" (Q4 only —
exact arithmetic from FY and 9mo-YTD facts, see above), or
"estimated_from_margin" (COGS backed out from this company's own historical
margin, last resort). This is the same audit-trail principle as the
Baupost/PDT filing-completeness flags (config/fund_universe.yaml
filing_completeness_note) and the per-filer value-unit scaling overrides
(_VALUE_UNIT_SCALE in smart_money/edgar.py): if GP factor results ever look
off for a subset of tickers, this column lets that be isolated to a specific
tier rather than guessed at. A quarterly TTM observation (see
_ttm_from_quarterly_rows) inherits the *worst* (lowest-confidence) tier
among its 4 contributing quarters — one weak quarter is enough to set the
trailing sum's confidence, but with YTD-subtraction in place this is no
longer constant across every TTM row the way a "reported vs estimated"
binary would have been (see the Q4 section above for why that mattered).

A separate "low_confidence_vs_yfinance" boolean column carries a *different*
axis of doubt: whether this ticker's XBRL-derived gp_ratio disagreed with
the prior yfinance-derived value on their overlapping (2021-2025) window,
per scripts/preflight_gp_xbrl.py's 30-ticker sample check. This is
per-ticker (every row for a flagged ticker is True), not per-observation
like "source" — a ticker either showed a divergence pattern worth
distrusting or it didn't; mirroring config/fund_universe.yaml's per-fund
flags being separate from the per-filing value-scale override rather than
folded into one field. Deliberately kept as a *separate* column from
"source" rather than a third source value, since a "reported" observation
can still disagree with yfinance (e.g. yfinance's own data has a units bug)
— conflating "how was this derived" with "does it independently corroborate"
would lose exactly the isolation this audit trail exists for. Populated from
data/gp/gp_preflight_divergent_tickers.txt (see _load_low_confidence_tickers
below); defaults to False for every ticker until that file exists.

Fetch strategy
--------------
One companyfacts request per company (CIK) returns that company's entire
tagged history in one round trip — unlike yfinance, which needed 4 separate
calls per ticker (annual/quarterly income statement + balance sheet) each
bounded to a narrow window. gp_xbrl_client.py caches the raw JSON per CIK;
this module derives (period_end, revenue, cogs, total_assets, gp_ratio,
freq, source, low_confidence_vs_yfinance) rows from that cache and writes
its own per-ticker CSV to data/gp/fundamentals/{ticker}.csv — unchanged
cache location and base schema (plus the two new provenance columns) from
the yfinance era, so factor_engine/factors/gp.py needs no changes to
consume it.

Resumability: each ticker's result (or None-marker for tickers with no
usable data) is written to data/gp/fundamentals/{ticker}.csv immediately on
fetch. fetch_universe_fundamentals() skips any ticker that already has a
cache file, so a full ~1500-ticker run only ever pays its network cost once
— interrupting and re-running picks up where it left off. Because the raw
XBRL JSON is *also* cached (gp_xbrl_client.py), even a forced re-derivation
after a tag-list or estimation-logic fix costs zero network calls.
"""

import os
import statistics
import time
from datetime import date

import pandas as pd

from edgar_client import ticker_to_cik
from factor_engine import gp_xbrl_client

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gp", "fundamentals")
_LOW_CONFIDENCE_LIST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gp", "gp_preflight_divergent_tickers.txt")
_HIGH_CONFIDENCE_LIST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gp", "gp_high_confidence_pre2021_tickers.txt")

MAX_RETRIES = 2
RETRY_SLEEP_SECONDS = 2.0   # edgar_client.get() already backs off on 429/503;
                            # this is only a safety net for non-HTTP errors
                            # (JSON decode issues, transient connection resets).

SOURCE_REPORTED = "reported"
SOURCE_DERIVED = "derived_from_ytd_subtraction"
SOURCE_ESTIMATED = "estimated_from_margin"
_SOURCE_RANK = {SOURCE_REPORTED: 0, SOURCE_DERIVED: 1, SOURCE_ESTIMATED: 2}   # confidence order, best first

# Tried in priority order *per period* — see module docstring.
_REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
]
_COGS_TAGS = [
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "CostOfGoodsSold",
    "CostOfServices",
]
_ASSETS_TAGS = ["Assets"]   # standardized enough that a fallback list isn't needed

_ANNUAL_SPAN_DAYS = (350, 380)     # a single fiscal year's duration fact
_QUARTER_SPAN_DAYS = (75, 100)     # a single fiscal quarter's duration fact (covers 4-4-5 calendars)
_NINE_MONTH_SPAN_DAYS = (250, 290) # a "nine months ended" YTD-cumulative duration fact

# A ticker with this exact single-row sentinel cached means "fetched, but no
# usable data" — distinguishes a permanently-empty result (don't retry it on
# every resumed run) from "never attempted" (no cache file at all).
_EMPTY_SENTINEL = pd.DataFrame({
    "period_end": [], "revenue": [], "cogs": [], "total_assets": [],
    "gp_ratio": [], "freq": [], "source": [],
    "low_confidence_vs_yfinance": [], "high_confidence_pre_2021": [],
})

_low_confidence_tickers: "set[str] | None" = None


def _load_low_confidence_tickers() -> set[str]:
    """
    Tickers flagged by scripts/preflight_gp_xbrl.py's 30-ticker sample check
    as diverging from their yfinance-derived gp_ratio on the overlap window.
    Cached per process (mirrors edgar_client's ticker->CIK map caching).
    Returns an empty set if the file doesn't exist yet (e.g. preflight
    hasn't run) — fetches proceed normally, just unflagged.
    """
    global _low_confidence_tickers
    if _low_confidence_tickers is not None:
        return _low_confidence_tickers
    if not os.path.exists(_LOW_CONFIDENCE_LIST_PATH):
        _low_confidence_tickers = set()
        return _low_confidence_tickers
    with open(_LOW_CONFIDENCE_LIST_PATH) as f:
        _low_confidence_tickers = {line.strip() for line in f if line.strip()}
    return _low_confidence_tickers


_high_confidence_tickers: "set[str] | None" = None


def _load_high_confidence_tickers() -> set[str]:
    """
    Tickers whose full-universe overlap-window (2021-2025) correlation
    between XBRL-derived and yfinance-derived gp_ratio is >= 0.7 — see
    scripts/compute_gp_high_confidence.py, which computes this from every
    available overlapping observation (not just the 30-ticker preflight
    sample low_confidence_vs_yfinance is based on). A ticker is only ever
    added here after the real pull completes; before that script has run,
    every ticker is unflagged (conservative default — absence means
    "not yet confirmed", not "confirmed low confidence").
    """
    global _high_confidence_tickers
    if _high_confidence_tickers is not None:
        return _high_confidence_tickers
    if not os.path.exists(_HIGH_CONFIDENCE_LIST_PATH):
        _high_confidence_tickers = set()
        return _high_confidence_tickers
    with open(_HIGH_CONFIDENCE_LIST_PATH) as f:
        _high_confidence_tickers = {line.strip() for line in f if line.strip()}
    return _high_confidence_tickers


def _cache_path(ticker: str) -> str:
    return os.path.join(_CACHE_DIR, f"{ticker}.csv")


# ---------------------------------------------------------------------------
# XBRL fact extraction
# ---------------------------------------------------------------------------

def _facts_by_tag(us_gaap: dict, tags: list[str]) -> list[list[dict]]:
    """USD-unit fact entries for each tag, tag list order preserved (one list per tag)."""
    return [us_gaap.get(tag, {}).get("units", {}).get("USD", []) for tag in tags]


def _duration_periods(entries_by_tag: list[list[dict]], span_days: tuple[int, int]) -> dict[tuple[str, str], dict]:
    """
    Resolve one fact per (start, end) period across a tag priority list.

    A higher-priority tag's match for a given period always wins over a
    lower-priority tag's match for that same period. Within a single tag,
    multiple filings can report the same period (restatements in a later
    quarter's comparative column) — the earliest-filed entry wins, i.e. the
    value as originally disclosed.
    """
    resolved: dict[tuple[str, str], dict] = {}
    for tag_entries in entries_by_tag:
        candidates: dict[tuple[str, str], dict] = {}
        for e in tag_entries:
            start, end, val = e.get("start"), e.get("end"), e.get("val")
            if not start or not end or val is None:
                continue
            span = (date.fromisoformat(end) - date.fromisoformat(start)).days
            if not (span_days[0] <= span <= span_days[1]):
                continue
            key = (start, end)
            filed = e.get("filed", "")
            if key not in candidates or filed < candidates[key]["filed"]:
                candidates[key] = e
        for key, e in candidates.items():
            if key not in resolved:
                resolved[key] = e
    return resolved


def _instant_periods(entries_by_tag: list[list[dict]]) -> dict[str, dict]:
    """Resolve one fact per balance-sheet-date across a tag priority list (see _duration_periods)."""
    resolved: dict[str, dict] = {}
    for tag_entries in entries_by_tag:
        candidates: dict[str, dict] = {}
        for e in tag_entries:
            end, val = e.get("end"), e.get("val")
            if not end or val is None or e.get("start"):   # instant facts carry no "start"
                continue
            filed = e.get("filed", "")
            if end not in candidates or filed < candidates[end]["filed"]:
                candidates[end] = e
        for end, e in candidates.items():
            if end not in resolved:
                resolved[end] = e
    return resolved


def _build_observations(us_gaap: dict, span_days: tuple[int, int], freq: str) -> list[dict]:
    """
    Assemble (period_end, revenue, cogs, total_assets, gp_ratio, freq, source)
    rows for one frequency bucket (annual duration or quarterly duration).

    COGS gap-filling: periods with Revenue + Assets but no resolvable COGS
    tag get an estimated COGS from this company's own median historical
    gross margin (computed only from this company's periods where both
    Revenue and COGS actually resolved) — see module docstring.
    """
    revenue_periods = _duration_periods(_facts_by_tag(us_gaap, _REVENUE_TAGS), span_days)
    cogs_periods = _duration_periods(_facts_by_tag(us_gaap, _COGS_TAGS), span_days)
    assets_by_end = _instant_periods(_facts_by_tag(us_gaap, _ASSETS_TAGS))

    reported_margins = [
        1.0 - cogs_periods[key]["val"] / rev_e["val"]
        for key, rev_e in revenue_periods.items()
        if key in cogs_periods and rev_e["val"]
    ]
    median_margin = statistics.median(reported_margins) if len(reported_margins) >= 2 else None

    obs = []
    for (start, end), rev_e in revenue_periods.items():
        assets_e = assets_by_end.get(end)
        if assets_e is None or not assets_e["val"]:
            continue

        cogs_e = cogs_periods.get((start, end))
        if cogs_e is not None:
            cogs_val, source = float(cogs_e["val"]), SOURCE_REPORTED
        elif median_margin is not None:
            cogs_val, source = float(rev_e["val"]) * (1.0 - median_margin), SOURCE_ESTIMATED
        else:
            continue   # no reported COGS and no basis to estimate — skip, don't fabricate

        revenue_val, assets_val = float(rev_e["val"]), float(assets_e["val"])
        obs.append({
            "period_end":   end,
            "revenue":      revenue_val,
            "cogs":         cogs_val,
            "total_assets": assets_val,
            "gp_ratio":     (revenue_val - cogs_val) / assets_val,
            "freq":         freq,
            "source":       source,
        })
    return obs


def _derive_q4_periods(
    revenue_annual: dict[tuple[str, str], dict],
    cogs_annual: dict[tuple[str, str], dict],
    revenue_ytd9: dict[tuple[str, str], dict],
    cogs_ytd9: dict[tuple[str, str], dict],
    existing_quarter_ends: set[str],
) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], dict]]:
    """
    Derive Q4 revenue/cogs as FY - 9mo_YTD wherever a fiscal year's FY and
    matching 9-month-YTD facts both resolve for revenue AND cogs, and a real
    quarter-span fact doesn't already cover that fiscal year end (the
    minority of filers who do tag Q4 directly take priority — this only
    fills genuine gaps). Keyed like the other duration-period dicts (start,
    end), using the YTD fact's end as the derived quarter's start.

    Assumes at most one 9-month-YTD fact per fiscal-year start date, which
    holds for standard calendar/fiscal-quarter reporters; an edge case (e.g.
    a fiscal-year-change stub period) could pick the wrong YTD fact, but the
    result is still explicitly labeled SOURCE_DERIVED for auditability
    rather than silently blended into "reported".
    """
    derived_rev: dict[tuple[str, str], dict] = {}
    derived_cogs: dict[tuple[str, str], dict] = {}
    for (fy_start, fy_end), fy_rev in revenue_annual.items():
        if fy_end in existing_quarter_ends:
            continue   # a real quarter-span fact already covers this fiscal year end
        if (fy_start, fy_end) not in cogs_annual:
            continue
        ytd_key = next((k for k in revenue_ytd9 if k[0] == fy_start), None)
        if ytd_key is None or ytd_key not in cogs_ytd9:
            continue

        fy_cogs = cogs_annual[(fy_start, fy_end)]
        ytd_rev, ytd_cogs = revenue_ytd9[ytd_key], cogs_ytd9[ytd_key]
        q4_key = (ytd_key[1], fy_end)
        derived_rev[q4_key] = {"val": fy_rev["val"] - ytd_rev["val"], "filed": fy_rev["filed"]}
        derived_cogs[q4_key] = {"val": fy_cogs["val"] - ytd_cogs["val"], "filed": fy_cogs["filed"]}
    return derived_rev, derived_cogs


def _build_quarterly_observations(us_gaap: dict) -> list[dict]:
    """
    Assemble single-quarter (freq="Q") observations combining three source
    tiers, in confidence order — see the module docstring's "Q4 gap" and
    "COGS gap-filling" sections for the rationale behind each:

      1. reported                     - revenue AND COGS both resolve from a
                                         directly-tagged quarter-span fact.
      2. derived_from_ytd_subtraction - Q4 only: FY - 9mo-YTD, exact
                                         arithmetic from two directly-tagged
                                         facts, used when no direct quarterly
                                         tag exists for Q4 (the common case).
      3. estimated_from_margin        - COGS still missing after (1) and
                                         (2): backed out from this company's
                                         own median historical gross margin.
    """
    revenue_q = _duration_periods(_facts_by_tag(us_gaap, _REVENUE_TAGS), _QUARTER_SPAN_DAYS)
    cogs_q = _duration_periods(_facts_by_tag(us_gaap, _COGS_TAGS), _QUARTER_SPAN_DAYS)
    revenue_annual = _duration_periods(_facts_by_tag(us_gaap, _REVENUE_TAGS), _ANNUAL_SPAN_DAYS)
    cogs_annual = _duration_periods(_facts_by_tag(us_gaap, _COGS_TAGS), _ANNUAL_SPAN_DAYS)
    revenue_ytd9 = _duration_periods(_facts_by_tag(us_gaap, _REVENUE_TAGS), _NINE_MONTH_SPAN_DAYS)
    cogs_ytd9 = _duration_periods(_facts_by_tag(us_gaap, _COGS_TAGS), _NINE_MONTH_SPAN_DAYS)
    assets_by_end = _instant_periods(_facts_by_tag(us_gaap, _ASSETS_TAGS))

    existing_quarter_ends = {end for (_start, end) in revenue_q.keys()}
    derived_rev, derived_cogs = _derive_q4_periods(
        revenue_annual, cogs_annual, revenue_ytd9, cogs_ytd9, existing_quarter_ends,
    )
    # Directly-tagged quarters take priority; derived Q4s only fill gaps
    # (guarded above via existing_quarter_ends, but keep revenue_q second so
    # a key collision — which shouldn't happen by construction — still
    # favors the directly-tagged fact).
    all_revenue = {**derived_rev, **revenue_q}

    # Margin basis: reported or YTD-derived COGS only — never an already-
    # estimated period, so the estimate never compounds off a guess.
    high_confidence_cogs = {**derived_cogs, **cogs_q}
    reported_margins = [
        1.0 - high_confidence_cogs[key]["val"] / rev_e["val"]
        for key, rev_e in all_revenue.items()
        if key in high_confidence_cogs and rev_e["val"]
    ]
    median_margin = statistics.median(reported_margins) if len(reported_margins) >= 2 else None

    obs = []
    for (start, end), rev_e in all_revenue.items():
        assets_e = assets_by_end.get(end)
        if assets_e is None or not assets_e["val"]:
            continue

        if (start, end) in cogs_q:
            cogs_val, source = float(cogs_q[(start, end)]["val"]), SOURCE_REPORTED
        elif (start, end) in derived_cogs:
            cogs_val, source = float(derived_cogs[(start, end)]["val"]), SOURCE_DERIVED
        elif median_margin is not None:
            cogs_val, source = float(rev_e["val"]) * (1.0 - median_margin), SOURCE_ESTIMATED
        else:
            continue

        revenue_val, assets_val = float(rev_e["val"]), float(assets_e["val"])
        obs.append({
            "period_end":   end,
            "revenue":      revenue_val,
            "cogs":         cogs_val,
            "total_assets": assets_val,
            "gp_ratio":     (revenue_val - cogs_val) / assets_val,
            "freq":         "Q",
            "source":       source,
        })
    return obs


# A single quarter's Revenue/COGS is a ~3-month flow; total_assets is a
# point-in-time balance. Dividing a 1-quarter flow by an annual-scale balance
# understates gp_ratio by roughly 4x relative to a company reporting on an
# annual observation — a units mismatch, not a real profitability
# difference. Fixed by summing 4 consecutive quarters' Revenue/COGS into a
# trailing-twelve-month flow before dividing by total_assets — the
# Novy-Marx (2013) annual specification applied at quarterly refresh
# cadence, scale-consistent with the annual observations.
_TTM_MIN_SPAN_DAYS = 250
_TTM_MAX_SPAN_DAYS = 320


def _ttm_from_quarterly_rows(quarterly_obs: pd.DataFrame) -> list[dict]:
    """
    Convert single-quarter (revenue, cogs, total_assets, source) rows into
    trailing-twelve-month gp_ratio observations.

    For each run of 4 consecutive quarters (by period_end), sums revenue and
    cogs across the window and divides by total_assets as of the most recent
    quarter in that window. Windows must span 250-320 days end-to-end (3
    quarter-gaps of ~90 days each ≈ 273 days) — a wider or narrower span
    means the "4 rows" aren't actually 4 consecutive fiscal quarters (a data
    gap), and that window is skipped rather than silently mixing periods.

    A TTM observation's source is the *worst* (lowest-confidence) tier among
    its 4 contributing quarters, per _SOURCE_RANK — one weak quarter is
    enough to set the trailing sum's confidence, so this rolls up
    conservatively rather than averaging it away.
    """
    if quarterly_obs is None or quarterly_obs.empty:
        return []

    q = quarterly_obs.sort_values("period_end").reset_index(drop=True)
    dates = pd.to_datetime(q["period_end"])

    obs = []
    for i in range(3, len(q)):
        window = q.iloc[i - 3:i + 1]
        span_days = (dates.iloc[i] - dates.iloc[i - 3]).days
        if not (_TTM_MIN_SPAN_DAYS <= span_days <= _TTM_MAX_SPAN_DAYS):
            continue

        rev_ttm = float(window["revenue"].sum())
        cogs_ttm = float(window["cogs"].sum())
        assets = float(window["total_assets"].iloc[-1])
        if assets == 0:
            continue

        source = max(window["source"], key=lambda s: _SOURCE_RANK[s])
        obs.append({
            "period_end":   q.at[i, "period_end"],
            "revenue":      rev_ttm,
            "cogs":         cogs_ttm,
            "total_assets": assets,
            "gp_ratio":     (rev_ttm - cogs_ttm) / assets,
            "freq":         "Q",
            "source":       source,
        })
    return obs


def _fetch_one(ticker: str) -> pd.DataFrame:
    """Fetch and combine annual + TTM-quarterly XBRL observations for a single ticker."""
    cik = ticker_to_cik(ticker)
    if cik is None:
        return _EMPTY_SENTINEL.copy()

    facts = gp_xbrl_client.fetch_company_facts(cik)
    if facts is None:
        return _EMPTY_SENTINEL.copy()

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return _EMPTY_SENTINEL.copy()

    annual_obs = _build_observations(us_gaap, _ANNUAL_SPAN_DAYS, "A")
    quarterly_raw = _build_quarterly_observations(us_gaap)
    quarterly_ttm_obs = _ttm_from_quarterly_rows(pd.DataFrame(quarterly_raw)) if quarterly_raw else []
    observations = annual_obs + quarterly_ttm_obs

    if not observations:
        return _EMPTY_SENTINEL.copy()

    df = pd.DataFrame(observations).drop_duplicates(subset="period_end").sort_values("period_end")
    df["low_confidence_vs_yfinance"] = ticker in _load_low_confidence_tickers()
    df["high_confidence_pre_2021"] = ticker in _load_high_confidence_tickers()
    return df.reset_index(drop=True)


def fetch_ticker_fundamentals(ticker: str, force: bool = False) -> "pd.DataFrame | None":
    """
    Fetch (or load cached) fundamental observations for one ticker.

    Returns a DataFrame with columns [period_end, revenue, cogs, total_assets,
    gp_ratio, freq, source, low_confidence_vs_yfinance], or an empty
    DataFrame if the ticker has no usable data (no resolvable CIK, no XBRL
    companyfacts, or no period with enough tagged data to build an
    observation) — never raises for a single bad ticker; retries transient
    (non-HTTP) failures before giving up.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache = _cache_path(ticker)
    if not force and os.path.exists(cache):
        return pd.read_csv(cache)

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            df = _fetch_one(ticker)
            df.to_csv(cache, index=False)
            return df
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP_SECONDS)

    print(f"  [gp_fundamentals] {ticker}: failed after {MAX_RETRIES} attempts ({last_error!r}); caching as empty")
    _EMPTY_SENTINEL.to_csv(cache, index=False)
    return _EMPTY_SENTINEL.copy()


def fetch_universe_fundamentals(tickers: list[str], force: bool = False) -> dict[str, pd.DataFrame]:
    """
    Fetch fundamentals for a full ticker universe, resumable.

    Skips tickers that already have a cache file unless force=True. Rate
    limiting is enforced centrally by edgar_client.get() (shared with
    smart_money's 13F/NLP fetches), so no additional per-ticker sleep is
    needed here. Prints progress verbosely for tickers actually fetched over
    the network (not loaded from cache): a running counter, remaining count,
    and a rolling ETA once enough samples exist to estimate a rate — so a
    long run is visibly progressing rather than looking stalled. Tickers
    already cached (a resumed run) are counted but not logged individually,
    to avoid flooding output on a mostly-complete resume.

    Returns dict[ticker -> observations DataFrame] (empty DataFrame for
    tickers with no usable data).
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    n_fetched = 0
    n_cached = 0
    n_empty = 0
    start_time = time.time()
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        already_cached = os.path.exists(_cache_path(ticker)) and not force

        if already_cached:
            n_cached += 1
            results[ticker] = fetch_ticker_fundamentals(ticker, force=force)
            continue

        remaining = total - i - 1
        elapsed = time.time() - start_time
        eta_str = ""
        if n_fetched >= 5:
            rate = elapsed / n_fetched  # seconds per freshly-fetched ticker
            eta_min = (rate * remaining) / 60.0
            eta_str = f", ETA ~{eta_min:.0f} min"
        print(f"  [gp_fundamentals] fetching {ticker:<8s} "
              f"({i + 1}/{total}, {remaining} remaining{eta_str})...", flush=True)

        df = fetch_ticker_fundamentals(ticker, force=force)
        results[ticker] = df
        n_fetched += 1
        if df.empty:
            n_empty += 1
            print(f"  [gp_fundamentals]   -> {ticker}: no usable data", flush=True)
        else:
            n_derived = int((df["source"] == SOURCE_DERIVED).sum())
            n_estimated = int((df["source"] == SOURCE_ESTIMATED).sum())
            est_str = ""
            if n_derived:
                est_str += f", {n_derived} derived_from_ytd"
            if n_estimated:
                est_str += f", {n_estimated} estimated_from_margin"
            flag_str = " [LOW CONFIDENCE vs yfinance]" if bool(df["low_confidence_vs_yfinance"].iloc[0]) else ""
            print(f"  [gp_fundamentals]   -> {ticker}: {len(df)} observation(s) "
                  f"({df['period_end'].min()} to {df['period_end'].max()}{est_str}){flag_str}", flush=True)

        if n_fetched % 50 == 0:
            elapsed_min = elapsed / 60.0
            print(f"  [gp_fundamentals] === checkpoint: {i + 1}/{total} processed, "
                  f"{n_fetched} freshly fetched ({n_empty} empty), "
                  f"{elapsed_min:.1f} min elapsed ===", flush=True)

    total_elapsed_min = (time.time() - start_time) / 60.0
    print(f"  [gp_fundamentals] Done in {total_elapsed_min:.1f} min. "
          f"{n_fetched} tickers freshly fetched ({n_empty} with no usable data), "
          f"{n_cached} loaded from cache.")
    return results
