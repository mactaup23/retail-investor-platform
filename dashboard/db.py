"""
Cached database query helpers for the dashboard.

All public functions return plain dicts / lists-of-dicts so Streamlit's
@st.cache_data can pickle the results safely (Peewee model instances are
not serialisable).

Call get_db() once at app startup (streamlit_app.py) to ensure the DB is
initialised before any page renders.
"""
from __future__ import annotations

import datetime

import streamlit as st


@st.cache_resource
def get_db():
    """Initialise SQLite connection (once per server process)."""
    from smart_money.models import init_db
    return init_db()


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl="5m", max_entries=5)
def get_available_periods() -> list[str]:
    """Return ISO date strings for all quarters that have FinalSignal rows, newest first."""
    from smart_money.models import FinalSignal
    rows = (
        FinalSignal.select(FinalSignal.period)
        .distinct()
        .order_by(FinalSignal.period.desc())
    )
    return [str(r.period) for r in rows]


@st.cache_data(ttl="5m", max_entries=5)
def get_pipeline_status() -> dict:
    """Return the most recent computed_at timestamp across all FinalSignal rows."""
    from smart_money.models import FinalSignal
    row = FinalSignal.select(FinalSignal.computed_at).order_by(FinalSignal.computed_at.desc()).first()
    return {"last_computed": str(row.computed_at) if row else None}


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

@st.cache_data(ttl="5m", max_entries=20)
def load_signals(period_str: str) -> list[dict]:
    """All FinalSignal rows for a quarter, sorted by final_score desc."""
    from smart_money.models import FinalSignal
    period = datetime.date.fromisoformat(period_str)
    rows = (
        FinalSignal.select()
        .where(FinalSignal.period == period)
        .order_by(FinalSignal.final_score.desc())
    )
    return [_fs_to_dict(r) for r in rows]


def _fs_to_dict(r) -> dict:
    return {
        "cusip":                         r.cusip,
        "ticker":                        r.ticker,
        "issuer_name":                   r.issuer_name,
        "period":                        str(r.period),
        "convergence_score":             r.convergence_score,
        "nlp_composite_score":           r.nlp_composite_score,
        "final_score":                   r.final_score,
        "nlp_available":                 r.nlp_available,
        "contradicted":                  r.contradicted,
        "status":                        r.status,
        "signal_drivers":                r.signal_drivers,
        "n_funds_bullish":               r.n_funds_bullish,
        "n_funds_bearish":               r.n_funds_bearish,
        "convergence_trend":             r.convergence_trend,
        "sector":                        r.sector,
        "avg_position_pct_of_portfolio": r.avg_position_pct_of_portfolio,
        "computed_at":                   str(r.computed_at),
        "display_name":                  r.ticker if r.ticker else r.issuer_name,
    }


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

@st.cache_data(ttl="30s", max_entries=5)
def load_watchlist_scored(period_str: str) -> list[dict]:
    """Active watchlist entries joined to FinalSignal for period."""
    from smart_money import watchlist as wl_module
    period = datetime.date.fromisoformat(period_str)
    scores = wl_module.score_watchlist(period)
    result = []
    for ws in scores:
        sig = ws.signal
        result.append({
            "ticker":              ws.entry.ticker,
            "cusip":               ws.entry.cusip,
            "issuer_name":         ws.entry.issuer_name,
            "display_name":        ws.display_name,
            "date_added":          str(ws.entry.date_added),
            "added_price":         ws.entry.added_price,
            "note":                ws.entry.note,
            "status":              ws.status,
            "final_score":         ws.final_score,
            "signal_drivers":      ws.signal_drivers,
            "convergence_score":   sig.convergence_score if sig else None,
            "nlp_composite_score": sig.nlp_composite_score if sig else None,
            "n_funds_bullish":     sig.n_funds_bullish if sig else None,
            "n_funds_bearish":     sig.n_funds_bearish if sig else None,
            "convergence_trend":   sig.convergence_trend if sig else None,
            "sector":              sig.sector if sig else None,
            "contradicted":        sig.contradicted if sig else None,
        })
    return result


@st.cache_data(ttl="30s", max_entries=5)
def load_watchlist_tickers() -> set[str]:
    """Set of active watchlist tickers (for highlight in discovery list)."""
    from smart_money.models import Watchlist
    rows = Watchlist.select(Watchlist.ticker).where(Watchlist.active == True)
    return {r.ticker for r in rows if r.ticker}


def watchlist_add(ticker_or_cusip: str) -> bool:
    """Add to watchlist; invalidate cached watchlist data. Returns True if newly added."""
    from smart_money import watchlist as wl_module
    from smart_money.models import Watchlist
    existing_count = Watchlist.select().where(
        (Watchlist.ticker == ticker_or_cusip.upper()) & (Watchlist.active == True)
    ).count()
    wl_module.add(ticker_or_cusip)
    _bust_watchlist_cache()
    return existing_count == 0


def watchlist_remove(ticker_or_cusip: str) -> bool:
    """Remove from watchlist; invalidate cached watchlist data."""
    from smart_money import watchlist as wl_module
    result = wl_module.remove(ticker_or_cusip)
    _bust_watchlist_cache()
    return result


def _bust_watchlist_cache():
    load_watchlist_scored.clear()
    load_watchlist_tickers.clear()


# ---------------------------------------------------------------------------
# Tax Lots
# ---------------------------------------------------------------------------

@st.cache_data(ttl="30s", max_entries=10)
def load_tax_lots(account_id: str = "default") -> list[dict]:
    """All TaxLot rows for an account."""
    from smart_money.models import TaxLot
    rows = TaxLot.select().where(TaxLot.account_id == account_id).order_by(
        TaxLot.ticker, TaxLot.acquisition_date
    )
    return [_lot_to_dict(r) for r in rows]


def _lot_to_dict(r) -> dict:
    return {
        "lot_id":               r.lot_id,
        "account_id":           r.account_id,
        "brokerage":            r.brokerage,
        "ticker":               r.ticker,
        "description":          r.description,
        "quantity":             r.quantity,
        "cost_basis_per_share": r.cost_basis_per_share,
        "total_cost_basis":     r.total_cost_basis,
        "acquisition_date":     str(r.acquisition_date),
        "valuation_date":       str(r.valuation_date),
        "current_price":        r.current_price,
        "current_value":        r.current_value,
        "unrealized_gl":        r.unrealized_gl,
        "unrealized_gl_pct":    r.unrealized_gl_pct,
        "holding_days":         r.holding_days,
        "is_long_term":         r.is_long_term,
        "days_to_lt":           r.days_to_lt,
        "near_lt":              r.near_lt,
        "wash_sale_risk":       r.wash_sale_risk,
    }


def bust_tax_lot_cache():
    load_tax_lots.clear()


# ---------------------------------------------------------------------------
# Fund skill scores
# ---------------------------------------------------------------------------

@st.cache_data(ttl="10m", max_entries=5)
def load_fund_skills() -> list[dict]:
    """FundSkillResult rows joined to Fund name."""
    from smart_money.models import FundSkillResult, Fund
    rows = (
        FundSkillResult.select(FundSkillResult, Fund)
        .join(Fund)
        .order_by(FundSkillResult.alpha_annualized.desc())
    )
    result = []
    for r in rows:
        d = dict(r.__data__)
        d["fund_name"] = r.fund.name
        d["fund_bucket"] = r.fund.bucket
        result.append(d)
    return result
