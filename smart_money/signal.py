"""
Module 4 — final signal combiner.

Blends ConvergenceScore (70 %) and NLPCache composite_score (30 %) into a
single final_score, applies a contradiction override, determines watchlist
status, and constructs a plain-English signal_drivers string.

Public interface
----------------
    combine(period, *, min_score)           → list[SignalResult]
    persist(results)                        → int
    load(period, *, min_score, status)      → list[FinalSignal]

Weighting
---------
Normal blend:    final = 0.70 × conv + 0.30 × nlp
Contradiction:   final = (0.85 × conv + 0.15 × nlp) × 0.80

Contradiction fires when |conv| ≥ 0.25 AND |nlp| ≥ 0.25 AND opposite signs.
The 0.80 damp reflects elevated uncertainty, not a preference reversal.

NLP unavailable: final = conv (100 % convergence weight), nlp_available=False.
No penalty — convergence is self-contained; NLP is a corroborating signal.

Discovery threshold
-------------------
DISCOVERY_THRESHOLD = 0.30  Minimum final_score to appear in output.
STRONG_THRESHOLD    = 0.55  Minimum for STRENGTHENING on first appearance.

Watchlist status
----------------
Status is computed from final_score level and quarter-over-quarter delta
against the prior FinalSignal row for the same CUSIP.

EXIT SIGNAL is assigned only when a prior FinalSignal row exists and the
current final_score has dropped below DISCOVERY_THRESHOLD.  Tickers with no
prior row and final_score < DISCOVERY_THRESHOLD are filtered silently — you
cannot exit a position that was never signalled.  Once EXIT SIGNAL is assigned
it is always emitted (never filtered by min_score).

Status              Condition (evaluated top-to-bottom)
-----------         -------------------------------------------
EXIT SIGNAL         prior row exists and final_score < DISCOVERY_THRESHOLD
STRENGTHENING       new + score ≥ STRONG_THRESHOLD
                    OR delta ≥ +0.10
                    OR delta ≥ +0.05 AND trend == "accelerating"
WEAKENING           delta ≤ −0.10
                    OR delta ≤ −0.05 AND trend == "fading"
HOLDING             all other above-threshold cases (including first appearance
                    with score in [DISCOVERY_THRESHOLD, STRONG_THRESHOLD))
"""

from __future__ import annotations

import calendar
import datetime
import json
import logging
from dataclasses import dataclass

from smart_money.convergence import load_quarter
from smart_money.models import ConvergenceScore, FinalSignal, NLPCache, db, init_db
from smart_money.nlp import load_scores

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONV_WEIGHT  = 0.70
NLP_WEIGHT   = 0.30

CONTRADICTION_THRESHOLD   = 0.25   # |conv| and |nlp| must both exceed this
CONTRADICTION_CONV_WEIGHT = 0.85
CONTRADICTION_NLP_WEIGHT  = 0.15
CONTRADICTION_DAMP        = 0.80   # magnitude multiplier when contradicted

DISCOVERY_THRESHOLD = 0.30   # minimum final_score to appear on the watchlist
STRONG_THRESHOLD    = 0.55   # minimum for STRENGTHENING on first appearance

DELTA_STRONG = 0.10    # quarter-over-quarter delta for STRENGTHENING
DELTA_NUDGE  = 0.05    # weaker delta that triggers status with trend confirmation
DELTA_WEAK   = -0.10   # quarter-over-quarter delta for WEAKENING


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    cusip:                         str
    ticker:                        str | None
    issuer_name:                   str
    period:                        datetime.date
    convergence_score:             float
    nlp_composite_score:           float | None
    final_score:                   float
    nlp_available:                 bool
    contradicted:                  bool
    status:                        str   # STRENGTHENING | HOLDING | WEAKENING | EXIT SIGNAL
    signal_drivers:                str
    n_funds_bullish:               int
    n_funds_bearish:               int
    convergence_trend:             str | None
    sector:                        str | None
    avg_position_pct_of_portfolio: float | None

    @property
    def display_name(self) -> str:
        """Ticker symbol when resolved; issuer_name otherwise. Use for all UI display."""
        return self.ticker if self.ticker else self.issuer_name


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prior_quarter_end(period: datetime.date) -> datetime.date:
    m, y = period.month - 3, period.year
    if m <= 0:
        m += 12
        y -= 1
    return datetime.date(y, m, calendar.monthrange(y, m)[1])


def _blend(conv: float, nlp: float) -> tuple[float, bool]:
    """Return (final_score, contradicted)."""
    if (
        abs(conv) >= CONTRADICTION_THRESHOLD
        and abs(nlp) >= CONTRADICTION_THRESHOLD
        and conv * nlp < 0
    ):
        raw = CONTRADICTION_CONV_WEIGHT * conv + CONTRADICTION_NLP_WEIGHT * nlp
        return raw * CONTRADICTION_DAMP, True
    return CONV_WEIGHT * conv + NLP_WEIGHT * nlp, False


def _status(
    final_score: float,
    prior_score: float | None,
    convergence_trend: str | None,
) -> str | None:
    """
    Return the watchlist status string, or None if the ticker should be filtered.

    None is returned when the ticker has never appeared on the watchlist
    (prior_score is None) and the current final_score is below DISCOVERY_THRESHOLD.
    """
    if prior_score is not None and final_score < DISCOVERY_THRESHOLD:
        return "EXIT SIGNAL"

    if final_score < DISCOVERY_THRESHOLD:
        return None   # never discovered; suppress

    if prior_score is None:
        return "STRENGTHENING" if final_score >= STRONG_THRESHOLD else "HOLDING"

    delta = final_score - prior_score

    if delta >= DELTA_STRONG:
        return "STRENGTHENING"
    if delta >= DELTA_NUDGE and convergence_trend == "accelerating":
        return "STRENGTHENING"
    if delta <= DELTA_WEAK:
        return "WEAKENING"
    if delta <= -DELTA_NUDGE and convergence_trend == "fading":
        return "WEAKENING"
    return "HOLDING"


# Fund names where the first-word rule produces an ambiguous or unreadable result.
# Keys are prefix matches against the full fund name (lowercased, startswith).
_SHORT_NAMES: dict[str, str] = {
    "two sigma":    "TwoSigma",
    "d.e. shaw":    "DEShaw",
    "d. e. shaw":   "DEShaw",
    "d1 capital":   "D1",
    "light street": "LightSt",
}


def _short_name(fund_name: str) -> str:
    """
    Compact identifier for a fund name used in signal_drivers strings.

    Checks _SHORT_NAMES prefix table first; falls back to first word.
    """
    if not fund_name:
        return fund_name
    lower = fund_name.lower()
    for prefix, short in _SHORT_NAMES.items():
        if lower.startswith(prefix):
            return short
    return fund_name.split()[0]


def _signal_drivers(
    conv: ConvergenceScore,
    nlp: NLPCache | None,
) -> str:
    """
    Build a plain-English tooltip string from existing structured data.

    Format (semicolons separate clauses):
        "{n} fund(s) bullish ({Name1} {TYPE1}, {Name2} {TYPE2}[, +N more])[; {n} bearish];
         NLP {direction} ({top dimension} {±val}); {trend} trend."

    No additional API calls — pure string formatting from fund_moves_json and
    NLP dimension deltas already stored in the database.
    """
    parts: list[str] = []

    # --- Fund moves ---
    try:
        moves: list[dict] = json.loads(conv.fund_moves_json or "[]")
    except (json.JSONDecodeError, TypeError):
        moves = []

    bull_moves = sorted(
        [m for m in moves if m.get("direction") == "bullish_leaning"],
        key=lambda m: m.get("effective_weight", 0.0),
        reverse=True,
    )
    bear_moves = [m for m in moves if m.get("direction") == "bearish_leaning"]

    if bull_moves:
        top3 = bull_moves[:3]
        names = ", ".join(
            f"{_short_name(m['fund_name'])} {m['change_type']}" for m in top3
        )
        extra = len(bull_moves) - 3
        suffix = f", +{extra} more" if extra > 0 else ""
        n = conv.n_funds_bullish
        parts.append(f"{n} fund{'s' if n != 1 else ''} bullish ({names}{suffix})")

    if bear_moves:
        n = conv.n_funds_bearish
        parts.append(f"{n} fund{'s' if n != 1 else ''} bearish")

    # --- NLP ---
    if nlp is not None:
        composite = nlp.composite_score
        direction = (
            "positive"  if composite >  0.10 else
            "negative"  if composite < -0.10 else
            "neutral"
        )
        dim_scores = {
            "guidance":                nlp.guidance_delta,
            "confidence":              nlp.confidence_delta,
            "customer demand":         nlp.customer_demand_delta,
            "competitive positioning": nlp.competitive_positioning_delta,
            "operational efficiency":  nlp.operational_efficiency_delta,
            "risk factors":            nlp.risk_factors_delta,
            "capital allocation":      nlp.capital_allocation_delta,
        }
        top_dim = max(dim_scores, key=lambda d: abs(dim_scores[d]))
        top_val = dim_scores[top_dim]
        sign = "+" if top_val >= 0 else ""
        parts.append(f"NLP {direction} ({top_dim} {sign}{top_val:.1f})")

    # --- Trend ---
    trend = conv.convergence_trend
    if trend == "new":
        parts.append("new position")
    elif trend in ("accelerating", "fading"):
        parts.append(f"{trend} trend")
    # "stable" is omitted — no signal value

    return ("; ".join(parts) + ".") if parts else "insufficient data."


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_prior_signals(period: datetime.date) -> dict[str, float]:
    """Return {cusip → final_score} from the immediately preceding quarter."""
    prior_period = _prior_quarter_end(period)
    return {
        row.cusip: row.final_score
        for row in FinalSignal.select(FinalSignal.cusip, FinalSignal.final_score)
        .where(FinalSignal.period == prior_period)
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def combine(
    period: datetime.date,
    *,
    min_score: float | None = None,
) -> list[SignalResult]:
    """
    Compute final signals for all ConvergenceScore rows in period.

    Parameters
    ----------
    period : datetime.date
        Quarter-end date, e.g. datetime.date(2026, 3, 31).
    min_score : float | None
        When set, only rows with final_score >= min_score are returned.
        EXIT SIGNAL rows are exempt from this filter and always included.

    Returns
    -------
    list[SignalResult]
        Sorted by final_score descending.  Tickers with no prior FinalSignal
        row and final_score below DISCOVERY_THRESHOLD are excluded entirely.
    """
    init_db()

    conv_rows: list[ConvergenceScore] = load_quarter(period)
    if not conv_rows:
        log.warning("No ConvergenceScore rows found for %s", period)
        return []

    tickers = [r.ticker for r in conv_rows if r.ticker]
    nlp_map: dict[str, NLPCache] = load_scores(tickers)
    prior_map: dict[str, float]  = _load_prior_signals(period)

    nlp_hits  = sum(1 for t in tickers if t in nlp_map)
    log.info(
        "signal.combine: %d convergence rows, %d NLP hits, %d prior signals",
        len(conv_rows), nlp_hits, len(prior_map),
    )

    results: list[SignalResult] = []

    for conv in conv_rows:
        nlp = nlp_map.get(conv.ticker) if conv.ticker else None

        if nlp is not None:
            final_score, contradicted = _blend(conv.convergence_score, nlp.composite_score)
            nlp_composite = nlp.composite_score
            nlp_available = True
        else:
            final_score   = conv.convergence_score
            contradicted  = False
            nlp_composite = None
            nlp_available = False

        prior_score = prior_map.get(conv.cusip)
        status = _status(final_score, prior_score, conv.convergence_trend)

        if status is None:
            continue   # below threshold, never on watchlist → suppress

        drivers = _signal_drivers(conv, nlp)

        if min_score is not None and final_score < min_score and status != "EXIT SIGNAL":
            continue

        results.append(SignalResult(
            cusip                         = conv.cusip,
            ticker                        = conv.ticker,
            issuer_name                   = conv.issuer_name,
            period                        = period,
            convergence_score             = conv.convergence_score,
            nlp_composite_score           = nlp_composite,
            final_score                   = round(final_score, 4),
            nlp_available                 = nlp_available,
            contradicted                  = contradicted,
            status                        = status,
            signal_drivers                = drivers,
            n_funds_bullish               = conv.n_funds_bullish,
            n_funds_bearish               = conv.n_funds_bearish,
            convergence_trend             = conv.convergence_trend,
            sector                        = conv.sector,
            avg_position_pct_of_portfolio = conv.avg_position_pct_of_portfolio,
        ))

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results


def persist(results: list[SignalResult]) -> int:
    """
    Upsert SignalResult list into the FinalSignal table.

    Uses INSERT OR REPLACE semantics — safe to call repeatedly.
    Returns the number of rows written.
    """
    if not results:
        return 0

    rows = [
        {
            "cusip":                         r.cusip,
            "ticker":                        r.ticker,
            "issuer_name":                   r.issuer_name,
            "period":                        r.period,
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
            "computed_at":                   datetime.datetime.utcnow(),
        }
        for r in results
    ]

    # SQLite limit: 32766 bound variables per statement.
    # FinalSignal has 17 fields → max ~1927 rows per batch.
    _CHUNK = 1700
    with db.atomic():
        for i in range(0, len(rows), _CHUNK):
            FinalSignal.replace_many(rows[i : i + _CHUNK]).execute()

    return len(rows)


def load(
    period: datetime.date,
    *,
    min_score: float | None = None,
    status: str | None = None,
) -> list[FinalSignal]:
    """
    Read persisted FinalSignal rows for a period.

    Parameters
    ----------
    min_score : float | None
        When set, only rows with final_score >= min_score are returned.
        EXIT SIGNAL rows are always included regardless of this filter.
    status : str | None
        When set, filter to rows matching this status exactly.
    """
    init_db()
    q = FinalSignal.select().where(FinalSignal.period == period)
    if status is not None:
        q = q.where(FinalSignal.status == status)
    elif min_score is not None:
        q = q.where(
            (FinalSignal.final_score >= min_score)
            | (FinalSignal.status == "EXIT SIGNAL")
        )
    return list(q.order_by(FinalSignal.final_score.desc()))
