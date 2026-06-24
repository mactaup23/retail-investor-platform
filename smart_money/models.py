"""
Peewee ORM schema for Module 3 — 13F Smart-Money Positioning & Skill Tracker.

Database: SQLite at data/module3.db (WAL mode, foreign keys enforced).

Table hierarchy:
    Fund → Filing → Holding
    Holding references Security (via cusip)
    Security → PriceCache

Quant-fund treatment
--------------------
For funds with bucket == "quant_systematic", all holdings are stored so that
crowding queries have the full position list.  Only the top 200 positions by
USD value (rank_by_value <= 200) are marked is_price_eligible=True; the price
fetcher and return reconstructor filter to those rows.  For all other buckets
every holding is price-eligible.
"""

import datetime
from pathlib import Path

from peewee import (
    BigIntegerField,
    BooleanField,
    CharField,
    DateField,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
)

DB_PATH = Path(__file__).parent.parent / "data" / "module3.db"

db = SqliteDatabase(
    None,  # initialised lazily by init_db()
    pragmas={"journal_mode": "wal", "foreign_keys": 1},
)


class BaseModel(Model):
    class Meta:
        database = db


# ---------------------------------------------------------------------------
# Fund
# ---------------------------------------------------------------------------

class Fund(BaseModel):
    """
    One row per fund in fund_universe.yaml (including excluded / conditional).

    The is_quant property drives the top-200 price-eligibility gate; it is
    derived from bucket at runtime rather than stored to avoid drift.
    """

    name = CharField(unique=True)
    manager = CharField()
    # long_short_equity | fundamental_value | quant_systematic | sector_specialist
    bucket = CharField()
    sector = CharField(null=True)       # populated for sector_specialist funds
    aum_tier = CharField()
    cik = CharField(null=True)
    # confirmed | not_found | verify | unknown
    cik_status = CharField(default="unknown")
    excluded = BooleanField(default=False)      # True = no valid CIK, skip pipeline
    conditional = BooleanField(default=False)   # True = threshold check required
    created_at = DateTimeField(default=datetime.datetime.utcnow)

    @property
    def is_quant(self) -> bool:
        return self.bucket == "quant_systematic"

    class Meta:
        table_name = "fund"


# ---------------------------------------------------------------------------
# Filing
# ---------------------------------------------------------------------------

class Filing(BaseModel):
    """
    One row per 13F-HR (or 13F-HR/A) filing retrieved from EDGAR.

    total_value_usd is in raw dollars (sum of Holding.value_usd for this filing).
    A Filing with form_type == "13F-HR/A" is an amendment; the pipeline should
    prefer the amendment over the original for the same (fund, period_of_report).
    """

    fund = ForeignKeyField(Fund, backref="filings")
    period_of_report = DateField()          # quarter-end date, e.g. 2024-12-31
    filed_date = DateField(null=True)
    accession_number = CharField(unique=True)   # e.g. "0001103804-25-000012"
    form_type = CharField(default="13F-HR")     # "13F-HR" | "13F-HR/A"
    total_value_usd = BigIntegerField(null=True)    # raw dollars
    total_holdings_count = IntegerField(null=True)
    fetched_at = DateTimeField(default=datetime.datetime.utcnow)

    class Meta:
        table_name = "filing"
        indexes = (
            # non-unique index for common query pattern
            (("fund_id", "period_of_report"), False),
        )


# ---------------------------------------------------------------------------
# Holding
# ---------------------------------------------------------------------------

class Holding(BaseModel):
    """
    One row per position line in a 13F filing.

    The same CUSIP can appear more than once in a single filing when a fund
    reports both a long equity position and a put/call on the same security, or
    when sole- and shared-discretion tranches are reported separately.  For that
    reason there is no unique constraint on (filing, cusip) — the combination of
    (filing, cusip, put_call, investment_discretion) is what uniquely identifies
    a row in practice.

    rank_by_value is 1-based descending by value_usd within the filing and is
    assigned at ingest time after all rows for that filing are sorted.

    is_price_eligible controls whether the price fetcher and return reconstructor
    include this position:
        - non-quant fund  →  always True
        - quant fund      →  True iff rank_by_value <= QUANT_PRICE_GATE (200)
    """

    QUANT_PRICE_GATE = 200

    filing = ForeignKeyField(Filing, backref="holdings")
    cusip = CharField()
    issuer_name = CharField()
    value_usd = BigIntegerField()           # raw dollars, direct from 13F XML (empirically verified)
    shares = BigIntegerField()
    # Sole | Shared | Other
    investment_discretion = CharField(default="Sole")
    put_call = CharField(null=True)         # "Put" | "Call" | None
    other_manager = CharField(null=True)    # populated when discretion == "Shared"
    rank_by_value = IntegerField()          # 1 = largest position in this filing
    is_price_eligible = BooleanField(default=True)

    class Meta:
        table_name = "holding"
        indexes = (
            (("filing_id", "cusip"), False),        # non-unique (see docstring)
            (("cusip", "is_price_eligible"), False), # crowding + price queries
        )


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

class Security(BaseModel):
    """
    CUSIP → FIGI resolution cache populated by the OpenFIGI resolver.

    composite_figi is the Bloomberg Composite FIGI — one identifier per
    security regardless of which exchange it trades on.  It is the canonical
    cross-exchange identifier stored here.

    resolution_status lifecycle:
        pending   →  not yet queried
        resolved  →  composite_figi and ticker populated
        no_match  →  OpenFIGI returned no results (e.g. private placement)
        failed    →  API error; resolution_error contains the detail
    """

    cusip = CharField(primary_key=True, max_length=9)
    composite_figi = CharField(null=True)
    share_class_figi = CharField(null=True)
    ticker = CharField(null=True)
    exchange_code = CharField(null=True)    # primary listing exchange code
    security_name = CharField(null=True)
    # "Common Stock" | "Warrant" | "Depositary Receipt" | etc.
    security_type = CharField(null=True)
    market_sector = CharField(null=True)    # "Equity" | "Corp" | etc.
    sector = CharField(null=True)           # yfinance GICS sector, e.g. "Technology"
    # pending | resolved | no_match | failed
    resolution_status = CharField(default="pending")
    resolved_at = DateTimeField(null=True)
    resolution_error = CharField(null=True)

    class Meta:
        table_name = "security"
        indexes = (
            (("ticker",), False),
            (("composite_figi",), False),
        )


# ---------------------------------------------------------------------------
# PriceCache
# ---------------------------------------------------------------------------

class PriceCache(BaseModel):
    """
    Daily closing prices for securities fetched from yfinance.

    Only populated for Securities that appear in at least one Holding with
    is_price_eligible=True.  adj_close is split- and dividend-adjusted.
    """

    security = ForeignKeyField(Security, backref="prices")
    date = DateField()
    close = FloatField()
    adj_close = FloatField()
    source = CharField(default="yfinance")
    fetched_at = DateTimeField(default=datetime.datetime.utcnow)

    class Meta:
        table_name = "price_cache"
        indexes = (
            (("security_id", "date"), True),    # one price per security per day
        )


# ---------------------------------------------------------------------------
# FundSkillResult
# ---------------------------------------------------------------------------

class FundSkillResult(BaseModel):
    """
    FF3 skill decomposition scores written by the pipeline (Phase 5).

    One row per fund; INSERT OR REPLACE semantics mean each pipeline run
    overwrites the previous score.  Module 4 reads this table directly
    for signal weighting rather than recomputing scores on every request.

    quarters_used is JSON-encoded, e.g. '["2023Q1","2023Q2","2024Q1"]'.
    """

    fund             = ForeignKeyField(Fund, backref="skill_result", unique=True)
    scored_at        = DateTimeField()
    n_quarters       = IntegerField()
    is_reliable      = BooleanField()
    confidence_label = CharField()
    quarters_used    = TextField()          # JSON list[str]
    alpha_quarterly  = FloatField()
    alpha_annualized = FloatField()
    alpha_t_stat     = FloatField()
    alpha_p_value    = FloatField()
    beta_market      = FloatField()
    beta_smb         = FloatField()
    beta_hml         = FloatField()
    t_stat_market    = FloatField()
    t_stat_smb       = FloatField()
    t_stat_hml       = FloatField()
    r_squared        = FloatField()
    avg_excess_return_q = FloatField()
    return_from_market  = FloatField()
    return_from_smb     = FloatField()
    return_from_hml     = FloatField()

    class Meta:
        table_name = "fund_skill_result"


# ---------------------------------------------------------------------------
# ConvergenceScore
# ---------------------------------------------------------------------------

class ConvergenceScore(BaseModel):
    """
    Per-CUSIP per-quarter convergence signal written by convergence.scan_quarter().

    Persisted so that convergence_trend can compare current scores to the prior
    two quarters without recomputing them.  INSERT OR REPLACE semantics (via
    replace_many) mean each scan_quarter run overwrites prior results for the
    same period.

    fund_moves_json is a JSON-encoded list[FundMove] (see convergence.py).
    """

    cusip           = CharField()
    issuer_name     = CharField()
    ticker          = CharField(null=True)
    period          = DateField()
    convergence_score = FloatField()    # −1 to +1
    directional     = FloatField()
    breadth         = FloatField()
    n_funds_total   = IntegerField()
    n_funds_bullish = IntegerField()
    n_funds_bearish = IntegerField()
    bull_weight     = FloatField()
    bear_weight     = FloatField()
    # Enrichments
    avg_position_pct_of_portfolio = FloatField(null=True)
    convergence_trend             = CharField(null=True)  # new|accelerating|stable|fading
    sector                        = CharField(null=True)
    sector_concentration          = FloatField(null=True)  # 0.0–1.0
    fund_moves_json               = TextField()
    computed_at     = DateTimeField(default=datetime.datetime.utcnow)

    class Meta:
        table_name = "convergence_score"
        indexes = (
            (("cusip", "period"), True),   # one row per CUSIP per quarter
        )


# ---------------------------------------------------------------------------
# NLPCache
# ---------------------------------------------------------------------------

class NLPCache(BaseModel):
    """
    Language-shift scores for portfolio company 10-Q/10-K MD&A sections.

    Keyed by (ticker, accession_current, accession_prior, scorer_version).
    Written by nlp.py via the Claude Batch API; consumed by convergence.py as
    an optional enrichment signal.  composite_score is the weighted sum of all
    7 dimension deltas.
    """

    ticker                        = CharField()
    cik_company                   = CharField()
    accession_current             = CharField()
    accession_prior               = CharField()
    form_type                     = CharField()      # "10-Q" | "10-K"
    scorer_version                = CharField()      # e.g. "v1"
    confidence_delta              = FloatField()
    guidance_delta                = FloatField()
    risk_factors_delta            = FloatField()
    capital_allocation_delta      = FloatField()
    competitive_positioning_delta = FloatField()
    customer_demand_delta         = FloatField()
    operational_efficiency_delta  = FloatField()
    composite_score               = FloatField()
    reasoning                     = TextField()
    scored_at                     = DateTimeField()

    class Meta:
        table_name = "nlp_cache"
        indexes = (
            (("ticker", "accession_current", "accession_prior", "scorer_version"), True),
            (("ticker", "scorer_version"), False),
        )


# ---------------------------------------------------------------------------
# Registry and init
# ---------------------------------------------------------------------------

TABLES = [Fund, Filing, Holding, Security, PriceCache, FundSkillResult, ConvergenceScore, NLPCache]

QUANT_PRICE_GATE = Holding.QUANT_PRICE_GATE  # re-export for callers


def init_db(path: Path | None = None) -> SqliteDatabase:
    """
    Initialise the database, create tables if absent, and return the db handle.

    Call once at application startup.  Safe to call multiple times (idempotent).
    Pass path to override the default DB_PATH (useful in tests).
    """
    target = path if path is not None else DB_PATH
    db.init(str(target))
    db.connect(reuse_if_open=True)
    db.execute_sql("PRAGMA journal_mode=WAL")
    db.execute_sql("PRAGMA foreign_keys=ON")
    db.create_tables(TABLES, safe=True)
    # Additive column migrations for existing databases
    try:
        db.execute_sql("ALTER TABLE security ADD COLUMN sector TEXT")
    except Exception:
        pass  # column already present
    return db
