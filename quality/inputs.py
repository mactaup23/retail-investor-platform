"""
Assembles one merged annual fundamentals panel per ticker for the Quality &
Health metrics (DuPont, Altman Z'', Piotroski F, Beneish M), joining three
already-separate sources rather than re-deriving anything:

  - factor_engine/gp_fundamentals.py's cache: revenue, cogs, total_assets, cash
  - dcf/fundamentals.py's fetch: ebit, effective_tax_rate, total_debt, diluted_shares
  - quality/fundamentals.py's fetch (new in this package): net_income, equity,
    current_assets, current_liabilities, total_liabilities, retained_earnings,
    cfo, accounts_receivable, ppe_net, sga, depreciation, long_term_debt,
    shares_outstanding_instant

Joined on period_end, inner join across all three — a fiscal year missing
any one source's data can't feed any of the four metrics anyway, so a
partial row would just be a bigger, differently-shaped "insufficient data"
case than an outright missing one. Same "skip rather than fabricate"
discipline as GP/DCF.

total_liabilities falls back to Total Assets - Equity when the direct
`Liabilities` XBRL tag doesn't resolve for a period (flagged
liabilities_source="derived_from_assets_minus_equity" vs "reported") —
algebraically exact given both components are already required elsewhere
in this panel, not an approximation.

Business-model applicability: reuses dcf/exclusions.py's bank/insurer/REIT
classification directly (same live-GICS-sector cache, ~1500 tickers already
populated) rather than a separate list. Altman, Piotroski, and Beneish all
assume a classified balance sheet and non-financial accrual structure —
Piotroski's current-ratio signal and Beneish's AQI are literally
uncomputable for banks/REITs regardless of methodology stance, since
AssetsCurrent/LiabilitiesCurrent don't resolve for filers using an
unclassified financial-services balance sheet template (confirmed during
scoping: JPM/O lack these tags entirely). DuPont's own formula has no such
precondition, but in practice it's blocked for these tickers too today
since this panel gates on GP's cache being non-empty, and GP's cache
requires resolvable COGS to produce any row — an inherited plumbing
constraint, not a deliberate exclusion; see quality/dupont.py's module
docstring for why this wasn't fixed as part of this feature.
business_model_flag is still returned in all cases (even when the panel
itself is empty) so callers can label *why* a ticker has no result.
"""

import pandas as pd

from dcf.exclusions import check_business_model_fit
from dcf.fundamentals import fetch_ticker_dcf_fundamentals
from factor_engine.gp_fundamentals import fetch_ticker_fundamentals
from quality.fundamentals import fetch_ticker_quality_fundamentals

LIABILITIES_REPORTED = "reported"
LIABILITIES_DERIVED = "derived_from_assets_minus_equity"


def build_quality_panel(ticker: str) -> "tuple[pd.DataFrame, str | None]":
    """
    Returns (panel, business_model_flag).

    panel columns: period_end, revenue, cogs, total_assets, cash,
    short_term_investments, ebit,
    effective_tax_rate, tax_rate_source, total_debt, debt_source,
    diluted_shares, net_income, equity, current_assets, current_liabilities,
    total_liabilities, liabilities_source, retained_earnings, cfo,
    accounts_receivable, ppe_net, sga, sga_source, depreciation,
    depreciation_source, long_term_debt, shares_outstanding_instant —
    sorted ascending by period_end (oldest first), so callers can index
    from the end for "most recent" / "most recent two" years.

    business_model_flag is None if standard DuPont/Altman/Piotroski/Beneish
    application is a reasonable fit, else "bank" / "insurer" / "reit" (see
    module docstring). Never raises — an empty panel (fewer than the
    caller's required rows) is the "insufficient data" signal, same as
    every other module here.
    """
    gp_df = fetch_ticker_fundamentals(ticker)
    gp_annual = gp_df[gp_df["freq"] == "A"] if not gp_df.empty else gp_df
    if gp_annual.empty:
        return pd.DataFrame(), check_business_model_fit(ticker)
    gp_annual = gp_annual[["period_end", "revenue", "cogs", "total_assets", "cash", "short_term_investments"]]

    dcf_df = fetch_ticker_dcf_fundamentals(ticker)
    if dcf_df.empty:
        return pd.DataFrame(), check_business_model_fit(ticker)
    dcf_cols = dcf_df[[
        "period_end", "ebit", "effective_tax_rate", "tax_rate_source",
        "total_debt", "debt_source", "diluted_shares",
    ]]

    quality_df = fetch_ticker_quality_fundamentals(ticker)
    if quality_df.empty:
        return pd.DataFrame(), check_business_model_fit(ticker)

    panel = gp_annual.merge(dcf_cols, on="period_end", how="inner")
    panel = panel.merge(quality_df, on="period_end", how="inner")
    if panel.empty:
        return pd.DataFrame(), check_business_model_fit(ticker)

    def _liabilities_row(row):
        if pd.notna(row["total_liabilities"]):
            return row["total_liabilities"], LIABILITIES_REPORTED
        return row["total_assets"] - row["equity"], LIABILITIES_DERIVED

    derived = panel.apply(_liabilities_row, axis=1, result_type="expand")
    panel["total_liabilities"] = derived[0]
    panel["liabilities_source"] = derived[1]

    panel = panel.sort_values("period_end").reset_index(drop=True)
    return panel, check_business_model_fit(ticker)
