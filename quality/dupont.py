"""
5-step DuPont ROE decomposition.

    ROE = Tax Burden x Interest Burden x Operating Margin x Asset Turnover x Equity Multiplier
        = (NI/Pretax Income) x (Pretax Income/EBIT) x (EBIT/Revenue) x (Revenue/Total Assets) x (Total Assets/Equity)

Standard 5-step extension of the textbook 3-step identity (Net Margin x
Asset Turnover x Equity Multiplier) — chosen over 3-step because Pretax
Income and EBIT are already cached from the DCF work, so the richer
decomposition costs nothing extra and separates financing/tax effects
(interest burden, tax burden) from the underlying operating margin, which
3-step conflates into a single "net margin" term.

Pretax Income isn't in quality/inputs.py's merged panel (dcf/fundamentals.py
resolves it internally for its own tax-rate calculation but doesn't expose
it in the columns quality/inputs.py selects) — refetched directly from
dcf/fundamentals.py's own output here rather than widening the shared panel
for a field only this one metric needs.

Unlike Altman/Piotroski/Beneish, this is a pure algebraic identity with no
business-model precondition of its own — DuPont's 5 components don't need
COGS, current assets/liabilities, or anything else that's structurally
missing for banks/insurers/REITs. In practice it still returns
insufficient_data for these tickers today, because quality/inputs.py's
panel gates on GP's cached fundamentals being non-empty, and GP's cache
requires a resolvable COGS (or a same-company historical-margin estimate)
to produce any row at all — no COGS-equivalent XBRL concept exists for
these business models (see factor_engine/gp_fundamentals.py). This is an
inherited plumbing constraint, not a considered business-model-validity
judgment the way Altman/Piotroski/Beneish's exclusions are: a lighter
Revenue/EBIT/Assets-only path bypassing GP's cache (mirroring
dcf/fundamentals.py's own direct-XBRL Revenue fallback) could unblock this,
but wasn't built since DuPont for financials isn't this feature's core ask
and the other three metrics are excluded for these tickers regardless.
Left as a known limitation, not fixed unilaterally.
"""

from dcf.fundamentals import fetch_ticker_dcf_fundamentals
from quality.inputs import build_quality_panel


def compute_dupont(ticker: str) -> dict:
    """
    Returns a dict with the 5 decomposition components, the recomposed ROE
    (product of all 5 — a cross-check against net_income/equity computed
    directly), period_end, business_model_flag, and a "status" of "ok" or
    "insufficient_data".
    """
    panel, business_model_flag = build_quality_panel(ticker)
    if panel.empty:
        return {"status": "insufficient_data", "business_model_flag": business_model_flag}

    row = panel.iloc[-1]
    dcf_df = fetch_ticker_dcf_fundamentals(ticker)
    pretax_row = dcf_df[dcf_df["period_end"] == row["period_end"]]
    pretax_income = pretax_row["pretax_income"].iloc[0] if not pretax_row.empty else None

    revenue, ebit, equity = row["revenue"], row["ebit"], row["equity"]
    net_income, total_assets = row["net_income"], row["total_assets"]

    if not pretax_income or not ebit or not revenue or not total_assets or not equity:
        return {
            "status": "insufficient_data",
            "business_model_flag": business_model_flag,
            "period_end": row["period_end"],
        }

    tax_burden = net_income / pretax_income
    interest_burden = pretax_income / ebit
    operating_margin = ebit / revenue
    asset_turnover = revenue / total_assets
    equity_multiplier = total_assets / equity
    roe_recomposed = tax_burden * interest_burden * operating_margin * asset_turnover * equity_multiplier

    return {
        "status": "ok",
        "period_end": row["period_end"],
        "business_model_flag": business_model_flag,
        "tax_burden": tax_burden,
        "interest_burden": interest_burden,
        "operating_margin": operating_margin,
        "asset_turnover": asset_turnover,
        "equity_multiplier": equity_multiplier,
        "roe": roe_recomposed,
        "roe_direct": net_income / equity,
    }
