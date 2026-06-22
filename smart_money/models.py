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
# Registry and init
# ---------------------------------------------------------------------------

TABLES = [Fund, Filing, Holding, Security, PriceCache]

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
    db.create_tables(TABLES, safe=True)
    return db
