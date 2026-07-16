"""
PEAD — post-earnings-announcement drift signal.

A third, independent research pillar alongside factor_engine/ (Modules 1-2,
portfolio factor exposure) and smart_money/ (Modules 3-5, 13F positioning).
Motivated by the Module 3/4 signal-improvement investigation finding that
tuning the 13F positioning signal alone had hit its practical ceiling (see
CLAUDE.md and app_pages/about.py) — this is a genuinely different data
domain (earnings surprises vs. institutional holdings), not a variant of
the existing signal.

Same universe as the GP factor (~1,500 S&P Composite 1500 tickers, see
factor_engine/gp_universe.py) and the same CSV-cache-over-DB pattern (no
Peewee/SQLite involvement) — there is no CUSIP or 13F relationship here,
so bolting this onto smart_money's schema would be a poor fit.

Modules
-------
    universe.py   — ticker universe (thin wrapper over gp_universe)
    surprises.py  — yfinance EPS actual-vs-estimate pull, BMO/AMC session
                     classification, CSV cache
    signal.py     — SUE (Standardized Unexpected Earnings) construction

First-pass scope: yfinance-sourced, EPS surprise only (revenue surprise is
a placeholder column, not yet computed). See CLAUDE.md for the IC decision
threshold gating a future EDGAR-sourced extension.
"""
