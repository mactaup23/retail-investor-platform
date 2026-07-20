"""
Business-model exclusions for the DCF valuation engine.

Different mechanism from factor_engine/gp_exclusions.py's REIT/insurer
exclusions — worth stating explicitly since the overlap in affected
business models (banks, insurers, REITs) could otherwise read as "the same
problem twice":

  - GP's exclusions are a DATA-AVAILABILITY problem: no COGS-equivalent
    XBRL concept exists for these business models at all (confirmed by
    direct inspection — zero entries across every COGS-family tag), so the
    factor construction can't even compute a ratio. That required a
    hand-verified, per-ticker list, because the failure had to be confirmed
    ticker by ticker.

  - This module's exclusions are a METHODOLOGICAL-VALIDITY problem: EBIT
    resolves just fine for a bank, insurer, or REIT — the engine would
    happily produce a per-share number. That number would be conceptually
    invalid, not missing, which is worse than an error: a wrong-shaped
    answer presented with the same confidence as a valid one. Standard
    equity-research practice doesn't apply enterprise DCF to these business
    models at all:

    - Banks: interest expense/income IS the core operating business (net
      interest margin), not a financing cost EBIT should strip out; capital
      structure is regulatory-driven (Basel capital ratios), not the
      WACC-minimizing choice the CAPM/WACC framing here assumes. Standard
      practice: dividend discount or excess-return models on equity
      directly, not enterprise DCF.
    - Insurers: float and claims reserves function as operating and
      financing leverage simultaneously; standard practice is DDM or
      embedded-value/P-to-book approaches.
    - REITs: required to distribute ~90% of taxable income as dividends, so
      the "reinvest FCF for growth" framing barely applies; D&A is large
      but doesn't track real economic depreciation of (often appreciating)
      property. Industry-standard metric is FFO/AFFO, not EBIT-based FCF.

Because this is a category-level methodological question rather than a
per-ticker data quirk, it's checked via live GICS sector/industry
classification (yfinance's info dict) rather than a hand-maintained ticker
list — confirmed empirically to cleanly separate the intended cases (JPM ->
Financial Services/Banks, MET -> Financial Services/Insurance, O/SPG ->
Real Estate/REIT) from a superficially similar but structurally different
case (UNH, a health insurer, classified Healthcare/Healthcare Plans, not
Financial Services — correctly NOT excluded here, since a health insurer's
economics don't share banks/insurers' capital-structure-is-regulatory
problem the way a life/P&C carrier's do).
"""

_BANK_SECTOR = "Financial Services"
_BANK_INDUSTRY_MARKER = "bank"
_INSURANCE_INDUSTRY_MARKER = "insurance"
_REIT_SECTOR = "Real Estate"
_REIT_INDUSTRY_MARKER = "reit"


def check_business_model_fit(ticker: str) -> "str | None":
    """
    Returns None if standard unlevered-FCF DCF is a methodologically
    reasonable fit for this ticker, or a short reason string (one of
    "bank", "insurer", "reit") if it isn't. Never raises — a yfinance
    lookup failure just means the check can't be performed, which is
    treated as "no flag" rather than blocking the run over an unrelated
    data-availability hiccup.
    """
    try:
        import yfinance as yf
        from yfinance_client import call_with_backoff
        info = call_with_backoff(lambda: yf.Ticker(ticker).info)
    except Exception:
        return None

    sector = (info.get("sector") or "").strip()
    industry = (info.get("industry") or "").strip().lower()

    if sector == _BANK_SECTOR and _BANK_INDUSTRY_MARKER in industry:
        return "bank"
    if sector == _BANK_SECTOR and _INSURANCE_INDUSTRY_MARKER in industry:
        return "insurer"
    if sector == _REIT_SECTOR and _REIT_INDUSTRY_MARKER in industry:
        return "reit"
    return None
