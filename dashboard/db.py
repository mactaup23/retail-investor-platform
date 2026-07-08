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
            "nlp_available":       sig.nlp_available if sig else False,
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
# Convergence detail (Signal tab)
# ---------------------------------------------------------------------------

_UNIVERSE_SIZE = 38  # total tracked funds in fund_universe.yaml


@st.cache_data(ttl="5m", max_entries=100)
def load_convergence_detail(cusip: str, period_str: str) -> dict | None:
    """
    Full ConvergenceScore row for (cusip, period) with fund_moves_json parsed
    and a count of how many universe funds currently hold this position.
    """
    import json, datetime
    from smart_money.models import ConvergenceScore, db

    period = datetime.date.fromisoformat(period_str)
    row = ConvergenceScore.get_or_none(
        (ConvergenceScore.cusip == cusip) & (ConvergenceScore.period == period)
    )
    if row is None:
        return None

    # Count distinct funds that filed a holding for this CUSIP this period
    try:
        result = db.execute_sql(
            "SELECT COUNT(DISTINCT f.fund_id) FROM filing f "
            "JOIN holding h ON h.filing_id = f.id "
            "WHERE h.cusip = ? AND f.period_of_report = ?",
            (cusip, str(period)),
        ).fetchone()
        n_holding = result[0] if result else 0
    except Exception:
        n_holding = None

    return {
        "convergence_score":    row.convergence_score,
        "directional":          row.directional,
        "breadth":              row.breadth,
        "n_funds_total":        row.n_funds_total,
        "n_funds_bullish":      row.n_funds_bullish,
        "n_funds_bearish":      row.n_funds_bearish,
        "bull_weight":          row.bull_weight,
        "bear_weight":          row.bear_weight,
        "avg_position_pct":     row.avg_position_pct_of_portfolio,
        "sector_concentration": row.sector_concentration,
        "convergence_trend":    row.convergence_trend,
        "n_holding":            n_holding,
        "universe_size":        _UNIVERSE_SIZE,
        "fund_moves":           json.loads(row.fund_moves_json),
    }


@st.cache_data(ttl="10m", max_entries=5)
def load_fund_skill_map() -> dict[str, dict]:
    """
    {fund_name → {alpha_annualized, is_reliable, confidence_label, n_quarters}}
    Built from FundSkillResult for quick lookup in the Signal tab fund moves table.
    """
    from smart_money.models import FundSkillResult, Fund
    rows = FundSkillResult.select(FundSkillResult, Fund).join(Fund)
    return {
        r.fund.name: {
            "alpha_annualized":  r.alpha_annualized,
            "is_reliable":       r.is_reliable,
            "confidence_label":  r.confidence_label,
            "n_quarters":        r.n_quarters,
        }
        for r in rows
    }


# ---------------------------------------------------------------------------
# NLP detail
# ---------------------------------------------------------------------------

@st.cache_data(ttl="5m", max_entries=100)
def load_nlp_detail(ticker: str) -> dict | None:
    """Most recent NLPCache row for a ticker: composite score, reasoning, dimension deltas."""
    from smart_money.models import NLPCache
    row = (
        NLPCache.select()
        .where(NLPCache.ticker == ticker)
        .order_by(NLPCache.scored_at.desc())
        .first()
    )
    if row is None:
        return None
    return {
        "composite_score":               row.composite_score,
        "reasoning":                     row.reasoning,
        "scored_at":                     str(row.scored_at),
        "guidance_delta":                row.guidance_delta,
        "confidence_delta":              row.confidence_delta,
        "customer_demand_delta":         row.customer_demand_delta,
        "competitive_positioning_delta": row.competitive_positioning_delta,
        "operational_efficiency_delta":  row.operational_efficiency_delta,
        "risk_factors_delta":            row.risk_factors_delta,
        "capital_allocation_delta":      row.capital_allocation_delta,
    }


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


# ---------------------------------------------------------------------------
# Signal backtest (Module 4 extension)
# ---------------------------------------------------------------------------

@st.cache_data(ttl="10m", max_entries=5)
def load_backtest() -> dict:
    """
    Run the full signal backtest and return plain-dict summary + per-quarter rows.

    Structure:
        {
            "quarter_ics": [{period, horizon_days, universe, n_candidates,
                              n_obs, coverage_pct, ic}, ...],
            "horizons": [{horizon_days, universe, n_quarters, mean_ic, std_ic,
                          t_stat, hit_rate, rolling_4q, rolling_8q}, ...],
        }
    """
    from smart_money.backtest import run_backtest, summarize

    quarter_ics = run_backtest()
    summary = summarize(quarter_ics)

    return {
        "quarter_ics": [
            {
                "period":        str(q.period),
                "horizon_days":  q.horizon_days,
                "universe":      q.universe,
                "n_candidates":  q.n_candidates,
                "n_obs":         q.n_obs,
                "coverage_pct":  q.coverage_pct,
                "ic":            q.ic,
            }
            for q in quarter_ics
        ],
        "horizons": [
            {
                "horizon_days": h.horizon_days,
                "universe":     h.universe,
                "n_quarters":   h.n_quarters,
                "mean_ic":      h.mean_ic,
                "std_ic":       h.std_ic,
                "t_stat":       h.t_stat,
                "hit_rate":     h.hit_rate,
                "rolling_4q":   [(str(p), v) for p, v in h.rolling_4q],
                "rolling_8q":   [(str(p), v) for p, v in h.rolling_8q],
            }
            for h in summary.horizons
        ],
    }


@st.cache_data(ttl="10m", max_entries=20)
def load_backtest_observations(period_str: str, horizon_days: int, universe: str) -> list[dict]:
    """Per-cusip score/forward_return pairs for one (period, horizon, universe) cell."""
    from smart_money.backtest import compute_quarter_ic

    period = datetime.date.fromisoformat(period_str)
    q = compute_quarter_ic(period, horizon_days, universe)
    return [
        {"cusip": o.cusip, "ticker": o.ticker, "score": o.score, "forward_return": o.forward_return}
        for o in q.observations
    ]
