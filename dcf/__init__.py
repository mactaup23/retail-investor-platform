"""
DCF — discounted cash flow intrinsic valuation engine.

A fourth independent research pillar alongside factor_engine/ (Modules 1-2),
smart_money/ (Modules 3-5), and pead/ — projects unlevered free cash flow
forward and compares the resulting per-share intrinsic value range to current
market price. Feeds a valuation badge into the existing Discovery/Watchlist
signal cards and the Valuation tab (app_pages/signals.py) alongside the
already-present trading multiples / FCF-yield / analyst-consensus content.

Data source: SEC EDGAR XBRL companyfacts, reusing the GP factor's already-
cached raw JSON (data/gp/xbrl_raw/{cik}.json) rather than yfinance — for any
ticker already in the GP universe this costs zero new network calls, only new
tag-parsing logic (fundamentals.py). This mirrors the reasoning that closed
out the GP factor's own yfinance -> XBRL migration: yfinance's financial-
statement endpoints cap at ~5yr annual / ~5 quarters, XBRL carries full
history back to a filer's XBRL adoption (~2009-2013+). Revenue and cash are
reused directly from the GP factor's own cached fundamentals CSV
(data/gp/fundamentals/{ticker}.csv) rather than re-derived — same underlying
XBRL facts, zero duplicated tag logic — with a same-module fallback if the
ticker isn't already in that cache.

Stated simplifications (approved, documented rather than silently applied —
see fundamentals.py / valuation.py docstrings for the specific reasoning):
  - Free cash flow = EBIT x (1 - tax rate) + D&A - Capex. Change in net
    working capital is NOT modeled (assumed zero) — a granular NWC build-up
    would inherit the same AP/accrued-liabilities XBRL tag-coverage gaps the
    GP factor's NIBCL work already found (only 76%/65% standalone coverage).
  - EBIT margin, D&A-as-%-of-revenue, and capex-as-%-of-revenue are all held
    flat across the 10-year projection at a blended 60% TTM / 40% trailing-
    3-year-average baseline — no separate margin-fade curve on top of the
    growth-fade curve.
  - Growth fades linearly from a company's own (clamped) trailing 5-year
    revenue CAGR to a terminal growth rate (the current 10-year Treasury
    yield — see wacc.py) by year 10.
  - Three explicit Bull/Base/Bear scenarios, not a Monte Carlo simulation —
    sampling from an assumed distribution would imply statistical rigor this
    model doesn't have; three labeled, explainable scenarios are the more
    honest framing.

Modules
-------
    fundamentals.py — per-ticker XBRL fundamentals fetch (EBIT, D&A, capex,
                       interest expense, total debt, effective tax rate,
                       diluted shares; revenue/cash reused from GP's cache)
    wacc.py         — CAPM cost of equity (reusing Module 1's beta), cost of
                       debt, blended WACC, 10-year Treasury risk-free rate
    valuation.py    — growth fade, FCF projection, Gordon Growth terminal
                       value, Bull/Base/Bear scenario assembly
"""
