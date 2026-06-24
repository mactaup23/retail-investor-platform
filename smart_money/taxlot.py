"""
Module 5 — Tax-Lot Engine.

Ingests lot-level cost basis data from brokerage CSV exports, computes
unrealized gain/loss and holding period per lot, models the tax cost of
proposed sell decisions across four lot-selection methods, flags wash-sale
risk, and identifies tax-loss harvesting candidates.

Supported brokerage formats (auto-detected from CSV headers):
    fidelity  Lot-Level Detail export (Account > Positions > Download)
    schwab    Portfolio Lot Detail export (Accounts > Holdings > Export)
    ibkr      Open Lots report (Reports > Tax Documents)
    generic   User-specified column mapping via col_map parameter

Public interface
----------------
    ingest(source, *, account_id, brokerage, col_map, valuation_date,
           fetch_prices)                           → list[TaxLot]

    load_lots(account_id, *, ticker)               → list[TaxLot]

    model_sell(ticker, quantity, *, account_id, lot_ids, sell_price,
               rates, sell_date)          → dict[str, SellDecision]

    harvest_candidates(account_id, *, rates, valuation_date)
                                                   → list[HarvestCandidate]

Lot-selection methods
---------------------
FIFO      Oldest lots first (default IRS method when SpecID not elected).
LIFO      Newest lots first.
MIN_TAX   Engine minimises tax drag using this priority order:
              1. LT losses, largest first (harvest — saves at lt_rate)
              2. ST losses, largest first (harvest — saves at st_rate)
              3. LT gains, smallest per-share first (lowest effective rate)
              4. ST gains, smallest per-share first (minimise ordinary gain)
          Lots with confirmed wash-sale disallowances are excluded.
SPEC_ID   Caller specifies lot_ids; engine fills in order provided.

model_sell returns all applicable methods in one call so the dashboard can
render a side-by-side comparison without multiple round-trips.

Wash-sale rule
--------------
A capital loss is disallowed when a substantially identical security is
purchased within the 61-day window centred on the sale date (30 days before
+ sale date + 30 days after).  This engine checks same-ticker purchases only;
options on the same underlying are outside scope (noted in WashSaleFlag).

WashSaleFlag.kind values:
    disallowed   A same-ticker lot was acquired within 30 days before the
                 proposed sell date — the loss is already disallowed.  The
                 disallowed amount is added to the replacement lot's basis
                 (reported but not automatically adjusted here).
    warning      No disqualifying prior purchase found; prospective risk only.
                 Buying back the same ticker within 30 days of the sale will
                 retroactively disallow the loss.

Tax rates
---------
TaxRates accepts st_rate (federal ordinary), lt_rate (federal LT cap gains),
state_rate (optional, default 0, applied additively to both ST and LT), and
niit (bool, adds 3.8% NIIT surcharge).  The engine does not compute AGI or
determine which bracket applies — caller provides rates.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yfinance as yf

from smart_money.models import PriceCache, Security, TaxLot, db, init_db

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LONG_TERM_DAYS = 365   # IRS: "more than one year" → holding_days > 365
NEAR_LT_DAYS   = 30    # flag lots within this many days of LT flip
NIIT_RATE      = 0.038


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaxRates:
    st_rate:    float          # federal marginal ordinary income rate
    lt_rate:    float          # federal LT capital gains rate (0/0.15/0.20)
    state_rate: float = 0.0   # state rate, applied to both ST and LT
    niit:       bool  = False  # add 3.8% NIIT surcharge

    @property
    def effective_st_rate(self) -> float:
        return self.st_rate + self.state_rate + (NIIT_RATE if self.niit else 0.0)

    @property
    def effective_lt_rate(self) -> float:
        return self.lt_rate + self.state_rate + (NIIT_RATE if self.niit else 0.0)


@dataclass(frozen=True)
class WashSaleFlag:
    lot_id:               str
    ticker:               str
    kind:                 str          # "disallowed" | "warning"
    disqualifying_lot_id: str | None   # None for "warning" kind
    disallowed_amount:    float | None # abs(loss); None for "warning" kind
    explanation:          str


@dataclass(frozen=True)
class SellLot:
    lot_id:               str
    ticker:               str
    acquisition_date:     datetime.date
    is_long_term:         bool
    quantity_sold:        float
    cost_basis_per_share: float
    proceeds:             float   # quantity_sold × sell_price
    gain_loss:            float   # proceeds − (quantity_sold × cost_basis_per_share)
    gain_type:            str     # "LONG_TERM" | "SHORT_TERM"
    tax_owed:             float   # 0 when wash-sale disallows the loss
    after_tax_proceeds:   float


@dataclass(frozen=True)
class SellDecision:
    ticker:              str
    quantity:            float
    sell_price:          float
    lot_selection:       str          # "FIFO" | "LIFO" | "MIN_TAX" | "SPEC_ID"
    lots_sold:           tuple[SellLot, ...]
    total_proceeds:      float
    total_st_gain:       float
    total_lt_gain:       float
    total_st_loss:       float        # positive number; represents losses
    total_lt_loss:       float        # positive number
    net_tax_owed:        float
    effective_tax_rate:  float        # net_tax_owed / total_proceeds
    after_tax_proceeds:  float
    wash_sale_flags:     tuple[WashSaleFlag, ...]
    rates:               TaxRates


@dataclass(frozen=True)
class HarvestCandidate:
    lot_id:           str
    ticker:           str
    acquisition_date: datetime.date
    quantity:         float
    unrealized_gl:    float           # negative
    gain_type:        str             # "LONG_TERM" | "SHORT_TERM"
    near_lt:          bool
    days_to_lt:       int
    tax_savings:      float | None    # abs(unrealized_gl) × rate; None if no rates given
    wash_sale_flag:   WashSaleFlag | None
    recommendation:   str             # "HARVEST NOW" | "WAIT Nd" | "WASH SALE — DISALLOWED"


# ---------------------------------------------------------------------------
# Brokerage format definitions
# ---------------------------------------------------------------------------

# Each entry: canonical_key → CSV column name, plus "date_fmt" for strptime.
_BROKERAGE_MAPS: dict[str, dict[str, str]] = {
    "fidelity": {
        "ticker":               "Symbol",
        "description":          "Description",
        "quantity":             "Quantity",
        "cost_basis_per_share": "Cost Basis Per Share",
        "acquisition_date":     "Acquisition Date",
        "date_fmt":             "%m/%d/%Y",
    },
    "schwab": {
        "ticker":               "Symbol",
        "description":          "Description",
        "quantity":             "Quantity",
        "cost_basis_per_share": "Cost Basis Per Share",
        "acquisition_date":     "Date Acquired",
        "date_fmt":             "%m/%d/%Y",
    },
    "ibkr": {
        "ticker":               "Symbol",
        "description":          "Description",
        "quantity":             "Quantity",
        "cost_basis_per_share": "CostBasisPrice",
        "acquisition_date":     "OpenDateTime",   # datetime string; truncated to date
        "date_fmt":             "%Y-%m-%d",
    },
}

_SKIP_TICKERS = {"symbol", "ticker", "total", "totals", "subtotal", ""}


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def _clean_number(s: str) -> str:
    """Strip $, commas, spaces; convert (n) parenthetical notation to -n."""
    s = s.strip().replace("$", "").replace(",", "").replace(" ", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    return s


def _detect_brokerage(headers: list[str]) -> str:
    h = {col.strip().lower() for col in headers}
    if {"opendatetime", "costbasisprice"}.issubset(h):
        return "ibkr"
    if "date acquired" in h and "cost basis per share" in h:
        return "schwab"
    if "acquisition date" in h and "cost basis per share" in h:
        return "fidelity"
    return "generic"


def _read_csv(
    source: Path | str | bytes | io.IOBase,
    brokerage: str | None,
) -> tuple[list[dict], str]:
    """
    Read the source into a list of raw row dicts and return (rows, brokerage).

    Handles:
      - BOM stripping (utf-8-sig)
      - Schwab / Fidelity metadata lines before the real header row
      - IBKR short positions (negative quantity → take absolute value)
    """
    if isinstance(source, Path):
        raw = source.read_text(encoding="utf-8-sig")
    elif isinstance(source, bytes):
        raw = source.decode("utf-8-sig")
    elif isinstance(source, str) and "\n" in source:
        raw = source  # already text content
    elif isinstance(source, str):
        raw = Path(source).read_text(encoding="utf-8-sig")
    else:
        content = source.read()
        raw = content.decode("utf-8-sig") if isinstance(content, bytes) else content

    lines = [ln for ln in raw.splitlines() if ln.strip()]

    # Find the real header row: first row whose first field is a known header token.
    _HEADER_STARTS = {"symbol", "ticker", "clientaccountid"}
    header_idx = 0
    for i, line in enumerate(lines):
        first = line.split(",")[0].strip().strip('"').lower()
        if first in _HEADER_STARTS:
            header_idx = i
            break

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    headers = list(reader.fieldnames or [])
    detected = brokerage or _detect_brokerage(headers)

    rows: list[dict] = []
    for row in reader:
        ticker_val = (row.get("Symbol") or row.get("Ticker") or "").strip().upper()
        if ticker_val.lower() in _SKIP_TICKERS:
            continue
        rows.append(dict(row))

    return rows, detected


def _normalize_row(
    row: dict,
    col_map: dict[str, str],
    date_fmt: str,
) -> dict | None:
    """
    Apply col_map to a raw CSV row and return normalized canonical fields.
    Returns None when required fields are missing or unparseable.
    """
    def get(key: str) -> str:
        return str(row.get(col_map.get(key, ""), "")).strip()

    ticker = get("ticker").upper()
    if not ticker or ticker.lower() in _SKIP_TICKERS:
        return None

    qty_s    = _clean_number(get("quantity"))
    cost_s   = _clean_number(get("cost_basis_per_share"))
    date_s   = get("acquisition_date").split()[0]   # truncate time for IBKR

    if not qty_s or not cost_s or not date_s:
        return None

    try:
        quantity             = abs(float(qty_s))   # IBKR short positions are negative
        cost_basis_per_share = float(cost_s)
    except ValueError:
        return None

    if quantity <= 0 or cost_basis_per_share <= 0:
        return None

    try:
        acquisition_date = datetime.datetime.strptime(date_s, date_fmt).date()
    except ValueError:
        return None

    return {
        "ticker":               ticker,
        "description":          get("description") or None,
        "quantity":             quantity,
        "cost_basis_per_share": cost_basis_per_share,
        "acquisition_date":     acquisition_date,
    }


# ---------------------------------------------------------------------------
# Price resolution
# ---------------------------------------------------------------------------

def _resolve_prices(
    tickers: set[str],
    valuation_date: datetime.date,
    *,
    fetch_missing: bool = True,
) -> dict[str, float]:
    """
    Return {ticker → adj_close} from PriceCache (most recent ≤ valuation_date).
    Falls back to yfinance for tickers absent from PriceCache.
    """
    prices: dict[str, float] = {}

    for sec in Security.select().where(Security.ticker.in_(list(tickers))):
        row = (
            PriceCache.select()
            .where(
                (PriceCache.security_id == sec.get_id())
                & (PriceCache.date <= valuation_date)
            )
            .order_by(PriceCache.date.desc())
            .first()
        )
        if row:
            prices[sec.ticker] = row.adj_close

    if fetch_missing:
        for ticker in tickers - set(prices):
            try:
                info  = yf.Ticker(ticker).fast_info
                price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
                if price:
                    prices[ticker] = float(price)
                    log.info("yfinance fallback price: %s = %.2f", ticker, float(price))
            except Exception as exc:
                log.warning("Cannot fetch price for %s: %s", ticker, exc)

    return prices


# ---------------------------------------------------------------------------
# Lot computation helpers
# ---------------------------------------------------------------------------

def _lot_id(
    account_id: str,
    ticker: str,
    acquisition_date: datetime.date,
    quantity: float,
    cost_basis_per_share: float,
) -> str:
    """Stable synthetic lot identifier (12-char hex). Idempotent across re-ingests."""
    raw = f"{account_id}:{ticker}:{acquisition_date}:{quantity:.6f}:{cost_basis_per_share:.6f}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _compute_derived(
    quantity: float,
    cost_basis_per_share: float,
    acquisition_date: datetime.date,
    current_price: float | None,
    valuation_date: datetime.date,
) -> dict:
    """Compute all derived fields for a lot row."""
    holding_days = (valuation_date - acquisition_date).days
    is_long_term = holding_days > LONG_TERM_DAYS
    days_to_lt   = max(0, LONG_TERM_DAYS + 1 - holding_days)  # 0 once LT
    near_lt      = 0 < days_to_lt <= NEAR_LT_DAYS

    total_cost = round(quantity * cost_basis_per_share, 2)

    if current_price is not None:
        current_value   = round(quantity * current_price, 2)
        unrealized_gl   = round(current_value - total_cost, 2)
        unrealized_pct  = round(unrealized_gl / total_cost * 100, 4) if total_cost else None
    else:
        current_value = unrealized_gl = unrealized_pct = None

    return {
        "total_cost_basis":  total_cost,
        "current_price":     current_price,
        "current_value":     current_value,
        "unrealized_gl":     unrealized_gl,
        "unrealized_gl_pct": unrealized_pct,
        "holding_days":      holding_days,
        "is_long_term":      is_long_term,
        "days_to_lt":        days_to_lt,
        "near_lt":           near_lt,
    }


# ---------------------------------------------------------------------------
# Wash-sale detection
# ---------------------------------------------------------------------------

def _check_wash_sale(
    lots: list[TaxLot],
    sell_date: datetime.date,
) -> dict[str, WashSaleFlag]:
    """
    Return {lot_id → WashSaleFlag} for every loss lot in lots.

    Retrospective check: if another lot of the same ticker was acquired within
    30 days before sell_date, the loss is disallowed.
    Prospective warning: emitted for all remaining loss lots (buying back within
    30 days after the sale will retroactively disallow the loss).
    """
    by_ticker: dict[str, list[TaxLot]] = {}
    for lot in lots:
        by_ticker.setdefault(lot.ticker, []).append(lot)

    window_open = sell_date - datetime.timedelta(days=30)
    flags: dict[str, WashSaleFlag] = {}

    for lot in lots:
        if lot.unrealized_gl is None or lot.unrealized_gl >= 0:
            continue   # no loss → wash-sale irrelevant

        # Retrospective: same-ticker purchase within 30 days before sell_date
        disqualifying = [
            other for other in by_ticker.get(lot.ticker, [])
            if other.lot_id != lot.lot_id
            and window_open <= other.acquisition_date <= sell_date
        ]

        if disqualifying:
            disq = sorted(disqualifying, key=lambda o: o.acquisition_date)[-1]
            gap  = (sell_date - disq.acquisition_date).days
            flags[lot.lot_id] = WashSaleFlag(
                lot_id               = lot.lot_id,
                ticker               = lot.ticker,
                kind                 = "disallowed",
                disqualifying_lot_id = disq.lot_id,
                disallowed_amount    = abs(lot.unrealized_gl),
                explanation          = (
                    f"Loss of ${abs(lot.unrealized_gl):,.2f} disallowed: "
                    f"{lot.ticker} purchased on {disq.acquisition_date} "
                    f"({gap}d before proposed sale) falls within the 30-day "
                    f"wash-sale window. Disallowed amount added to replacement "
                    f"lot's cost basis."
                ),
            )
        else:
            flags[lot.lot_id] = WashSaleFlag(
                lot_id               = lot.lot_id,
                ticker               = lot.ticker,
                kind                 = "warning",
                disqualifying_lot_id = None,
                disallowed_amount    = None,
                explanation          = (
                    f"Realising a loss of ${abs(lot.unrealized_gl):,.2f} on "
                    f"{lot.ticker}. Do not repurchase {lot.ticker} (or substantially "
                    f"identical securities) within 30 days of the sale date or "
                    f"the loss will be retroactively disallowed."
                ),
            )

    return flags


# ---------------------------------------------------------------------------
# Lot selection
# ---------------------------------------------------------------------------

def _fill_quantity(
    ordered_lots: list[TaxLot],
    quantity: float,
) -> list[tuple[TaxLot, float]]:
    """Consume lots in order until quantity is filled. Supports fractional lots."""
    result: list[tuple[TaxLot, float]] = []
    remaining = quantity
    for lot in ordered_lots:
        if remaining <= 1e-9:
            break
        qty = min(lot.quantity, remaining)
        result.append((lot, qty))
        remaining -= qty
    return result


def _select_fifo(lots: list[TaxLot], quantity: float) -> list[tuple[TaxLot, float]]:
    return _fill_quantity(sorted(lots, key=lambda l: l.acquisition_date), quantity)


def _select_lifo(lots: list[TaxLot], quantity: float) -> list[tuple[TaxLot, float]]:
    return _fill_quantity(sorted(lots, key=lambda l: l.acquisition_date, reverse=True), quantity)


def _select_min_tax(
    lots: list[TaxLot],
    quantity: float,
    ws_flags: dict[str, WashSaleFlag],
) -> list[tuple[TaxLot, float]]:
    """
    Select lots in tax-minimising order (see module docstring).
    Lots with confirmed disallowed losses are excluded from selection.
    """
    disallowed = {lid for lid, f in ws_flags.items() if f.kind == "disallowed"}

    def sort_key(lot: TaxLot) -> tuple[int, float]:
        gl  = lot.unrealized_gl or 0.0
        psg = gl / lot.quantity if lot.quantity else 0.0   # per-share GL

        if gl < 0 and lot.is_long_term and lot.lot_id not in disallowed:
            return (0, psg)   # LT loss: most negative first
        if gl < 0 and not lot.is_long_term and lot.lot_id not in disallowed:
            return (1, psg)   # ST loss: most negative first
        if gl >= 0 and lot.is_long_term:
            return (2, psg)   # LT gain: smallest per-share first
        return (3, psg)       # ST gain: smallest per-share first (highest cost)

    eligible = [l for l in lots if l.lot_id not in disallowed or (l.unrealized_gl or 0) >= 0]
    return _fill_quantity(sorted(eligible, key=sort_key), quantity)


def _select_spec_id(
    lots: list[TaxLot],
    lot_ids: list[str],
    quantity: float,
) -> list[tuple[TaxLot, float]]:
    """Sell the caller-specified lots in the order provided."""
    by_id = {l.lot_id: l for l in lots}
    ordered = [by_id[lid] for lid in lot_ids if lid in by_id]
    return _fill_quantity(ordered, quantity)


# ---------------------------------------------------------------------------
# Tax computation
# ---------------------------------------------------------------------------

def _build_sell_decision(
    ticker: str,
    sell_price: float,
    method: str,
    pairs: list[tuple[TaxLot, float]],
    rates: TaxRates,
    ws_flags: dict[str, WashSaleFlag],
) -> SellDecision:
    sell_lots: list[SellLot] = []
    total_proceeds = total_st_gain = total_lt_gain = 0.0
    total_st_loss  = total_lt_loss = net_tax = 0.0
    active_flags: list[WashSaleFlag] = []

    for lot, qty_sold in pairs:
        proceeds   = round(qty_sold * sell_price, 2)
        cost       = round(qty_sold * lot.cost_basis_per_share, 2)
        gain       = round(proceeds - cost, 2)
        gain_type  = "LONG_TERM" if lot.is_long_term else "SHORT_TERM"
        rate       = rates.effective_lt_rate if lot.is_long_term else rates.effective_st_rate

        flag = ws_flags.get(lot.lot_id)
        taxable = 0.0 if (flag and flag.kind == "disallowed" and gain < 0) else gain

        if flag:
            active_flags.append(flag)

        tax = round(taxable * rate, 2)

        if gain >= 0:
            if lot.is_long_term: total_lt_gain += gain
            else:                total_st_gain += gain
        else:
            if lot.is_long_term: total_lt_loss += abs(gain)
            else:                total_st_loss += abs(gain)

        total_proceeds += proceeds
        net_tax        += tax

        sell_lots.append(SellLot(
            lot_id               = lot.lot_id,
            ticker               = ticker,
            acquisition_date     = lot.acquisition_date,
            is_long_term         = lot.is_long_term,
            quantity_sold        = round(qty_sold, 6),
            cost_basis_per_share = lot.cost_basis_per_share,
            proceeds             = proceeds,
            gain_loss            = gain,
            gain_type            = gain_type,
            tax_owed             = tax,
            after_tax_proceeds   = round(proceeds - tax, 2),
        ))

    tp  = round(total_proceeds, 2)
    eff = round(net_tax / tp, 4) if tp else 0.0

    return SellDecision(
        ticker             = ticker,
        quantity           = round(sum(q for _, q in pairs), 6),
        sell_price         = sell_price,
        lot_selection      = method,
        lots_sold          = tuple(sell_lots),
        total_proceeds     = tp,
        total_st_gain      = round(total_st_gain, 2),
        total_lt_gain      = round(total_lt_gain, 2),
        total_st_loss      = round(total_st_loss, 2),
        total_lt_loss      = round(total_lt_loss, 2),
        net_tax_owed       = round(net_tax, 2),
        effective_tax_rate = eff,
        after_tax_proceeds = round(tp - net_tax, 2),
        wash_sale_flags    = tuple(active_flags),
        rates              = rates,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def ingest(
    source: Path | str | bytes | io.IOBase,
    *,
    account_id: str = "default",
    brokerage: str | None = None,
    col_map: dict[str, str] | None = None,
    valuation_date: datetime.date | None = None,
    fetch_prices: bool = True,
) -> list[TaxLot]:
    """
    Parse a brokerage CSV export and persist lot records to the DB.

    Replaces all existing TaxLot rows for account_id inside a single
    transaction — each export is a complete point-in-time snapshot.

    Parameters
    ----------
    source : Path | str | bytes | IO
        File path, raw CSV text, bytes, or file-like object.
    account_id : str
        User-facing account label (e.g. "fidelity-taxable"). Default "default".
    brokerage : str | None
        Override auto-detection. One of "fidelity", "schwab", "ibkr", "generic".
    col_map : dict | None
        Required when brokerage=="generic". Maps canonical keys to CSV columns.
        Keys: ticker, quantity, cost_basis_per_share, acquisition_date.
        Optionally include "date_fmt" (strptime format, default "%m/%d/%Y").
    valuation_date : date | None
        Defaults to today.  Used to compute holding_days and resolve prices.
    fetch_prices : bool
        When True, fetch missing prices from yfinance after PriceCache lookup.

    Returns
    -------
    list[TaxLot]
        Newly created TaxLot rows, in acquisition_date order.
    """
    init_db()
    val_date = valuation_date or datetime.date.today()

    raw_rows, detected = _read_csv(source, brokerage)
    log.info("ingest: %d raw rows detected, brokerage=%s", len(raw_rows), detected)

    mapping: dict[str, str]
    date_fmt: str

    if detected == "generic":
        if not col_map:
            raise ValueError(
                "col_map is required for generic format. "
                "Provide a dict mapping canonical keys to your CSV columns."
            )
        mapping  = col_map
        date_fmt = col_map.get("date_fmt", "%m/%d/%Y")
    else:
        fmt_entry = _BROKERAGE_MAPS[detected]
        mapping   = fmt_entry
        date_fmt  = fmt_entry["date_fmt"]

    normalized: list[dict] = []
    for row in raw_rows:
        n = _normalize_row(row, mapping, date_fmt)
        if n:
            normalized.append(n)

    if not normalized:
        log.warning("ingest: no valid rows after normalization")
        return []

    # Resolve current prices
    tickers = {r["ticker"] for r in normalized}
    prices  = _resolve_prices(tickers, val_date, fetch_missing=fetch_prices)
    log.info("ingest: %d tickers, %d prices resolved", len(tickers), len(prices))

    # Build lot rows with derived fields
    lot_rows: list[dict] = []
    for r in normalized:
        lid     = _lot_id(account_id, r["ticker"], r["acquisition_date"],
                          r["quantity"], r["cost_basis_per_share"])
        derived = _compute_derived(
            r["quantity"], r["cost_basis_per_share"],
            r["acquisition_date"], prices.get(r["ticker"]), val_date,
        )
        lot_rows.append({
            "lot_id":               lid,
            "account_id":           account_id,
            "brokerage":            detected,
            "ticker":               r["ticker"],
            "description":          r.get("description"),
            "quantity":             r["quantity"],
            "cost_basis_per_share": r["cost_basis_per_share"],
            "valuation_date":       val_date,
            "acquisition_date":     r["acquisition_date"],
            "wash_sale_risk":       False,  # back-filled below
            "ingested_at":          datetime.datetime.utcnow(),
            **derived,
        })

    # Back-fill wash_sale_risk using valuation_date as the proxy sell_date
    # (conservative: any loss lot with a same-ticker purchase in the prior 30 days)
    temp_lots = [_dict_to_namedlot(r) for r in lot_rows]
    ws_flags  = _check_wash_sale(temp_lots, val_date)
    for r in lot_rows:
        flag = ws_flags.get(r["lot_id"])
        r["wash_sale_risk"] = flag is not None and flag.kind == "disallowed"

    with db.atomic():
        TaxLot.delete().where(TaxLot.account_id == account_id).execute()
        TaxLot.insert_many(lot_rows).execute()

    return list(
        TaxLot.select()
        .where(TaxLot.account_id == account_id)
        .order_by(TaxLot.acquisition_date)
    )


def load_lots(
    account_id: str = "default",
    *,
    ticker: str | None = None,
) -> list[TaxLot]:
    """Return TaxLot rows for an account, optionally filtered to one ticker."""
    init_db()
    q = TaxLot.select().where(TaxLot.account_id == account_id)
    if ticker:
        q = q.where(TaxLot.ticker == ticker.upper())
    return list(q.order_by(TaxLot.acquisition_date))


def model_sell(
    ticker: str,
    quantity: float,
    *,
    account_id: str = "default",
    lot_ids: list[str] | None = None,
    sell_price: float | None = None,
    rates: TaxRates,
    sell_date: datetime.date | None = None,
) -> dict[str, SellDecision]:
    """
    Model the tax cost of selling quantity shares of ticker.

    Returns a dict keyed by method name containing SellDecision for each
    applicable lot-selection method.  SPEC_ID is included only when lot_ids
    is provided.

    Parameters
    ----------
    ticker : str
    quantity : float
        Number of shares to sell.
    account_id : str
    lot_ids : list[str] | None
        Required for SPEC_ID.  Lots are consumed in the order provided.
    sell_price : float | None
        Defaults to the most recent PriceCache price for the ticker.
    rates : TaxRates
    sell_date : date | None
        Defaults to today.  Used for the wash-sale 30-day window check.
    """
    init_db()
    lots = load_lots(account_id, ticker=ticker)
    if not lots:
        return {}

    sd = sell_date or datetime.date.today()

    if sell_price is None:
        prices     = _resolve_prices({ticker.upper()}, sd, fetch_missing=True)
        sell_price = prices.get(ticker.upper())
        if sell_price is None:
            raise ValueError(f"Cannot determine sell price for {ticker}")

    ws_flags = _check_wash_sale(lots, sd)

    decisions: dict[str, SellDecision] = {}
    for method, pairs in [
        ("FIFO",    _select_fifo(lots, quantity)),
        ("LIFO",    _select_lifo(lots, quantity)),
        ("MIN_TAX", _select_min_tax(lots, quantity, ws_flags)),
    ]:
        if pairs:
            decisions[method] = _build_sell_decision(
                ticker, sell_price, method, pairs, rates, ws_flags
            )

    if lot_ids:
        spec_pairs = _select_spec_id(lots, lot_ids, quantity)
        if spec_pairs:
            decisions["SPEC_ID"] = _build_sell_decision(
                ticker, sell_price, "SPEC_ID", spec_pairs, rates, ws_flags
            )

    return decisions


def harvest_candidates(
    account_id: str = "default",
    *,
    rates: TaxRates | None = None,
    valuation_date: datetime.date | None = None,
) -> list[HarvestCandidate]:
    """
    Return all loss lots for account_id, sorted by tax-savings potential.

    Sort order: LT losses first (immediately deductible at lt_rate), then ST
    losses; within each group, largest absolute loss first.

    Parameters
    ----------
    rates : TaxRates | None
        When provided, tax_savings is computed.  Otherwise tax_savings=None.
    valuation_date : date | None
        Defaults to today; used as the proxy sell_date for wash-sale checks.
    """
    init_db()
    val_date = valuation_date or datetime.date.today()
    lots     = load_lots(account_id)

    loss_lots = [l for l in lots if l.unrealized_gl is not None and l.unrealized_gl < 0]
    if not loss_lots:
        return []

    ws_flags = _check_wash_sale(loss_lots, val_date)

    def sort_key(lot: TaxLot) -> tuple[int, float]:
        # LT losses first (tier 0), then ST (tier 1); within tier largest loss first
        return (0 if lot.is_long_term else 1, lot.unrealized_gl or 0.0)

    candidates: list[HarvestCandidate] = []
    for lot in sorted(loss_lots, key=sort_key):
        flag = ws_flags.get(lot.lot_id)
        loss = abs(lot.unrealized_gl)

        if rates:
            rate     = rates.effective_lt_rate if lot.is_long_term else rates.effective_st_rate
            savings  = round(loss * rate, 2)
        else:
            savings  = None

        if flag and flag.kind == "disallowed":
            rec = "WASH SALE — LOSS DISALLOWED"
        elif lot.near_lt:
            rec = f"WAIT {lot.days_to_lt}d (LT flip)"
        elif flag and flag.kind == "warning":
            rec = "HARVEST — WASH SALE RISK (see note)"
        else:
            rec = "HARVEST NOW"

        candidates.append(HarvestCandidate(
            lot_id           = lot.lot_id,
            ticker           = lot.ticker,
            acquisition_date = lot.acquisition_date,
            quantity         = lot.quantity,
            unrealized_gl    = lot.unrealized_gl,
            gain_type        = "LONG_TERM" if lot.is_long_term else "SHORT_TERM",
            near_lt          = lot.near_lt,
            days_to_lt       = lot.days_to_lt,
            tax_savings      = savings,
            wash_sale_flag   = flag,
            recommendation   = rec,
        ))

    return candidates


# ---------------------------------------------------------------------------
# Internal shim: dict → lightweight object for wash-sale pre-pass
# ---------------------------------------------------------------------------

class _DictLot:
    """Minimal duck-typed TaxLot for the ingest-time wash-sale pre-pass."""
    __slots__ = (
        "lot_id", "ticker", "acquisition_date",
        "unrealized_gl", "is_long_term", "quantity",
    )

    def __init__(self, d: dict) -> None:
        self.lot_id          = d["lot_id"]
        self.ticker          = d["ticker"]
        self.acquisition_date = d["acquisition_date"]
        self.unrealized_gl   = d.get("unrealized_gl")
        self.is_long_term    = d.get("is_long_term", False)
        self.quantity        = d.get("quantity", 0.0)


def _dict_to_namedlot(d: dict) -> Any:
    return _DictLot(d)
