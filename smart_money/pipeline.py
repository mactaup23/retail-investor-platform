"""
Module 3 pipeline orchestrator — EDGAR → CUSIP → Prices → Returns → Skill.

Phases
------
    1  edgar       Fetch 13F filings and holdings from EDGAR; persist to DB
    2  cusip       Resolve CUSIPs to FIGI/ticker via OpenFIGI
    3  prices      Fetch daily prices for resolved tickers via yfinance
    4  returns     Reconstruct quarterly returns (diagnostic log; no DB write)
    5  skill       FF3 skill decomposition; persist scores to FundSkillResult

Usage
-----
    python -m smart_money.pipeline [options]

    --refresh         Re-check EDGAR for new filings even for already-ingested funds.
                      Default: skip funds that already have ≥1 filing in the DB.
    --fund NAME       Run all phases for one fund only (exact name from fund_universe.yaml).
    --from-phase N    Start at phase N (1–5).  Default: 1.
    --to-phase N      Stop after phase N (1–5).  Default: 5.
    --db PATH         Override default DB path (data/module3.db).
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import yaml

from smart_money import cusip as cusip_mod
from smart_money import edgar
from smart_money import prices as prices_mod
from smart_money.factor_apply import MIN_QUARTERS_REG, FundSkillScore, score_fund
from smart_money.models import (
    DB_PATH,
    Filing,
    Fund,
    FundSkillResult,
    Holding,
    QUANT_PRICE_GATE,
    init_db,
)
from smart_money.returns import reconstruct_all_quarters

FUND_UNIVERSE_PATH = Path(__file__).parent.parent / "config" / "fund_universe.yaml"

_t_start: float = 0.0
_INGEST_CHUNK   = 500    # Holding rows per INSERT batch
_XML_CUTOFF     = "2013-06-01"  # EDGAR mandated structured XML for 13F around this date


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _elapsed() -> str:
    return f"{time.monotonic() - _t_start:.1f}s"


def _log(tag: str, msg: str) -> None:
    print(f"[{tag:<10}] {msg}", flush=True)


def _phase_banner(n: int, label: str) -> None:
    bar = "─" * max(0, 54 - len(label))
    _log("pipeline", f"Phase {n}: {label} {bar} ({_elapsed()})")


# ---------------------------------------------------------------------------
# Setup — fund universe loading (always runs before phases 1–5)
# ---------------------------------------------------------------------------

def _upsert_fund(entry: dict) -> Fund:
    """Insert-or-update a Fund row from a fund_universe.yaml entry."""
    excluded = (
        entry.get("cik") is None
        or "excluded_no_edgar_cik" in entry.get("filing_flags", [])
    )
    cik = str(entry["cik"]) if entry.get("cik") is not None else None

    (Fund
     .insert(
         name=entry["name"],
         manager=entry.get("manager", ""),
         bucket=entry.get("bucket", ""),
         sector=entry.get("sector"),
         aum_tier=entry.get("aum_tier", "unknown"),
         cik=cik,
         cik_status=entry.get("cik_status", "unknown"),
         excluded=excluded,
         conditional=entry.get("conditional", False),
     )
     .on_conflict(
         conflict_target=[Fund.name],
         update={
             Fund.manager:     entry.get("manager", ""),
             Fund.bucket:      entry.get("bucket", ""),
             Fund.sector:      entry.get("sector"),
             Fund.aum_tier:    entry.get("aum_tier", "unknown"),
             Fund.cik:         cik,
             Fund.cik_status:  entry.get("cik_status", "unknown"),
             Fund.excluded:    excluded,
             Fund.conditional: entry.get("conditional", False),
         },
     )
     .execute())
    return Fund.get(Fund.name == entry["name"])


def _setup(path: Path = FUND_UNIVERSE_PATH) -> tuple[list[Fund], set[str]]:
    """
    Load fund_universe.yaml, upsert Fund rows, promote any conditionals that
    have live 13F filings on EDGAR.

    Returns
    -------
    active_funds : list[Fund]
        Funds eligible for phases 1–5 (has CIK, not excluded, not conditional
        or conditional that was just promoted).
    completeness_warn : set[str]
        Fund names carrying the verify_filing_completeness flag.
    """
    with path.open() as f:
        data = yaml.safe_load(f)
    entries: list[dict] = data["funds"]

    active:      list[Fund] = []
    conditional: list[Fund] = []
    excluded:    list[Fund] = []
    completeness_warn: set[str] = set()

    for entry in entries:
        fund = _upsert_fund(entry)
        if "verify_filing_completeness" in entry.get("filing_flags", []):
            completeness_warn.add(fund.name)
        if fund.excluded:
            excluded.append(fund)
        elif fund.conditional:
            conditional.append(fund)
        else:
            active.append(fund)

    _log("init", (
        f"{len(active)} active  |  {len(conditional)} conditional"
        f"  |  {len(excluded)} excluded"
    ))
    if completeness_warn:
        _log("init", (
            f"  NOTE: verify_filing_completeness flag on: "
            + ", ".join(sorted(completeness_warn))
        ))

    # Try to promote conditional funds by checking EDGAR
    for fund in conditional:
        try:
            filings = edgar.list_13f_filings(fund.cik)
            if filings:
                Fund.update(conditional=False).where(Fund.id == fund.id).execute()
                promoted = Fund.get_by_id(fund.id)
                active.append(promoted)
                _log("init", f"  conditional PROMOTED  {fund.name}  ({len(filings)} 13F filings)")
            else:
                _log("init", f"  conditional SKIPPED   {fund.name}  (0 filings on EDGAR)")
        except Exception as e:
            _log("init", f"  conditional ERROR     {fund.name}  {e}")

    return active, completeness_warn


# ---------------------------------------------------------------------------
# Phase 1 — EDGAR ingest
# ---------------------------------------------------------------------------

def _ingest_filing(
    fund: Fund,
    meta: edgar.FilingMeta,
    holdings: list[edgar.HoldingRow],
) -> None:
    """Persist one filing and its holdings.  Assigns rank_by_value and is_price_eligible."""
    filing = Filing.create(
        fund=fund,
        period_of_report=datetime.date.fromisoformat(meta["period_of_report"]),
        filed_date=(
            datetime.date.fromisoformat(meta["filed_date"])
            if meta.get("filed_date") else None
        ),
        accession_number=meta["accession_number"],
        form_type=meta["form_type"],
    )

    # Sort descending by value to assign rank
    sorted_h = sorted(holdings, key=lambda h: h["value_usd"], reverse=True)
    is_quant = fund.is_quant

    rows = [
        {
            "filing":                filing,
            "cusip":                 h["cusip"],
            "issuer_name":           h["issuer_name"],
            "value_usd":             h["value_usd"],
            "shares":                h["shares"],
            "investment_discretion": h["investment_discretion"],
            "put_call":              h["put_call"],
            "other_manager":         h["other_manager"],
            "rank_by_value":         rank,
            "is_price_eligible":     (not is_quant) or (rank <= QUANT_PRICE_GATE),
        }
        for rank, h in enumerate(sorted_h, start=1)
    ]
    for i in range(0, len(rows), _INGEST_CHUNK):
        Holding.insert_many(rows[i : i + _INGEST_CHUNK]).execute()

    Filing.update(
        total_value_usd=sum(h["value_usd"] for h in holdings),
        total_holdings_count=len(holdings),
    ).where(Filing.id == filing.id).execute()


def run_phase1_edgar(
    active_funds: list[Fund],
    *,
    refresh: bool,
    completeness_warn: set[str],
) -> None:
    _phase_banner(1, "EDGAR ingest")
    total_new_f = total_new_h = 0

    for fund in active_funds:
        existing: set[str] = {
            row.accession_number
            for row in Filing.select(Filing.accession_number).where(Filing.fund == fund)
        }

        # Default (no --refresh): skip any fund already in the DB
        if not refresh and existing:
            _log("edgar", f"  SKIP  {fund.name}  ({len(existing)} filings in DB)")
            continue

        if fund.name in completeness_warn:
            _log("edgar", f"  NOTE  {fund.name}  verify_filing_completeness — holdings may be partial")

        try:
            all_filings = edgar.list_13f_filings(fund.cik)
        except Exception as e:
            _log("edgar", f"  ERROR {fund.name}  list_13f_filings: {e}")
            continue

        pre_xml = [f for f in all_filings if f.get("filed_date", "") < _XML_CUTOFF]
        xml_era = [f for f in all_filings if f.get("filed_date", "") >= _XML_CUTOFF]
        if pre_xml:
            _log("edgar", (
                f"  {fund.name}: skipped {len(pre_xml)} pre-XML filings"
                f" (filed before {_XML_CUTOFF})"
            ))

        canonical = edgar.canonical_filings(xml_era)
        new_f = new_h = 0

        for meta in canonical:
            if meta["accession_number"] in existing:
                continue
            try:
                holdings = edgar.fetch_holdings(fund.cik, meta["accession_number"])
            except Exception as e:
                _log("edgar", (
                    f"  ERROR {fund.name}  {meta['period_of_report']}"
                    f"  fetch_holdings: {e}"
                ))
                continue
            _ingest_filing(fund, meta, holdings)
            new_f += 1
            new_h += len(holdings)

        label = "NEW " if new_f else "OK  "
        _log("edgar", (
            f"  {label} {fund.name}"
            f"  {len(canonical)}q canonical"
            f"  +{new_f} new  (+{new_h} holdings)"
        ))
        total_new_f += new_f
        total_new_h += new_h

    _log("pipeline", f"Phase 1 done  +{total_new_f} filings  +{total_new_h} holdings  ({_elapsed()})")


# ---------------------------------------------------------------------------
# Phase 2 — CUSIP resolution
# ---------------------------------------------------------------------------

def run_phase2_cusip(active_funds: list[Fund]) -> None:
    _phase_banner(2, "CUSIP resolution")
    fund_ids = [f.id for f in active_funds]

    cusips = [
        row.cusip
        for row in (
            Holding.select(Holding.cusip)
            .join(Filing)
            .where(Filing.fund_id.in_(fund_ids), Holding.is_price_eligible == True)
            .distinct()
        )
    ]
    _log("pipeline", f"  {len(cusips)} unique price-eligible CUSIPs")

    results = cusip_mod.resolve_cusips(cusips, skip_resolved=True)

    resolved  = sum(1 for v in results.values() if v is not None)
    no_result = len(results) - resolved
    _log("pipeline", (
        f"Phase 2 done  {resolved} resolved  |  {no_result} no_match/failed  ({_elapsed()})"
    ))


# ---------------------------------------------------------------------------
# Phase 3 — Price fetch
# ---------------------------------------------------------------------------

def run_phase3_prices(active_funds: list[Fund]) -> None:
    _phase_banner(3, "price fetch")
    fund_ids = [f.id for f in active_funds]

    periods = sorted(
        row.period_of_report
        for row in Filing.select(Filing.period_of_report).where(Filing.fund_id.in_(fund_ids))
    )
    if not periods:
        _log("pipeline", "Phase 3: no filings in DB — skipping")
        return

    start_date, end_date = periods[0], periods[-1]
    _log("pipeline", f"  date range {start_date} → {end_date}")

    cusips = [
        row.cusip
        for row in (
            Holding.select(Holding.cusip)
            .join(Filing)
            .where(Filing.fund_id.in_(fund_ids), Holding.is_price_eligible == True)
            .distinct()
        )
    ]
    price_map = prices_mod.fetch_prices_for_cusips(
        cusips, start_date, end_date, skip_cached=True
    )
    _log("pipeline", f"Phase 3 done  {len(price_map)} tickers with price data  ({_elapsed()})")


# ---------------------------------------------------------------------------
# Phase 4 — Return reconstruction (diagnostic; no DB write)
# ---------------------------------------------------------------------------

def run_phase4_returns(active_funds: list[Fund]) -> None:
    _phase_banner(4, "return reconstruction")
    total_q = valid_q = 0

    for fund in active_funds:
        n_filings = Filing.select().where(Filing.fund == fund).count()
        if n_filings < 2:
            _log("returns", f"  SKIP  {fund.name}  ({n_filings} filing in DB)")
            continue

        try:
            quarters = reconstruct_all_quarters(fund)
        except Exception as e:
            _log("returns", f"  ERROR {fund.name}  {e}")
            continue

        valid = [q for q in quarters if q["is_valid"]]
        mean_cov = (
            sum(q["coverage_pct"] for q in quarters) / len(quarters)
            if quarters else 0.0
        )
        _log("returns", (
            f"  {fund.name}"
            f"  {len(quarters)}q  {len(valid)} valid"
            f"  mean_cov={mean_cov:.1%}"
        ))
        total_q += len(quarters)
        valid_q += len(valid)

    _log("pipeline", f"Phase 4 done  {total_q} quarters  {valid_q} valid  ({_elapsed()})")


# ---------------------------------------------------------------------------
# Phase 5 — Skill scoring + persistence
# ---------------------------------------------------------------------------

def run_phase5_skill(active_funds: list[Fund]) -> None:
    _phase_banner(5, "skill scoring")
    now = datetime.datetime.utcnow()
    scored:       list[tuple[Fund, FundSkillScore]] = []
    insufficient: list[str]                         = []

    for fund in active_funds:
        try:
            result = score_fund(fund)
        except Exception as e:
            _log("skill", f"  ERROR  {fund.name}  {e}")
            insufficient.append(fund.name)
            continue

        if result is None:
            insufficient.append(fund.name)
            continue

        (FundSkillResult
         .insert(
             fund=fund,
             scored_at=now,
             n_quarters=result["n_quarters"],
             is_reliable=result["is_reliable"],
             confidence_label=result["confidence_label"],
             quarters_used=json.dumps(result["quarters_used"]),
             alpha_quarterly=result["alpha_quarterly"],
             alpha_annualized=result["alpha_annualized"],
             alpha_t_stat=result["alpha_t_stat"],
             alpha_p_value=result["alpha_p_value"],
             beta_market=result["beta_market"],
             beta_smb=result["beta_smb"],
             beta_hml=result["beta_hml"],
             t_stat_market=result["t_stat_market"],
             t_stat_smb=result["t_stat_smb"],
             t_stat_hml=result["t_stat_hml"],
             r_squared=result["r_squared"],
             avg_excess_return_q=result["avg_excess_return_q"],
             return_from_market=result["return_from_market"],
             return_from_smb=result["return_from_smb"],
             return_from_hml=result["return_from_hml"],
         )
         .on_conflict_replace()
         .execute())

        scored.append((fund, result))

    scored.sort(key=lambda x: x[1]["alpha_annualized"], reverse=True)

    # Ranked output table
    print()
    hdr = f"  {'#':>3}  {'Fund':<36}  {'α/yr':>6}  {'n_q':>4}  {'rel':>3}  confidence"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for i, (_, s) in enumerate(scored, start=1):
        rel = "✓" if s["is_reliable"] else ""
        print(
            f"  {i:>3}  {s['fund_name']:<36}  {s['alpha_annualized']:>+6.1%}"
            f"  {s['n_quarters']:>4}  {rel:>3}  {s['confidence_label']}"
        )

    if insufficient:
        print(f"\n  insufficient data (< {MIN_QUARTERS_REG} valid quarters):")
        for name in insufficient:
            print(f"    - {name}")
    print()

    _log("pipeline", (
        f"Phase 5 done  {len(scored)} scored"
        f"  |  {len(insufficient)} insufficient  ({_elapsed()})"
    ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _t_start
    _t_start = time.monotonic()

    parser = argparse.ArgumentParser(
        description="Module 3 pipeline: EDGAR → CUSIP → Prices → Returns → Skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Re-check EDGAR for new filings even for already-ingested funds. "
            "Default: skip funds that already have ≥1 filing in the DB."
        ),
    )
    parser.add_argument(
        "--fund",
        metavar="NAME",
        help="Run all phases for a single fund (exact name from fund_universe.yaml).",
    )
    parser.add_argument(
        "--from-phase",
        type=int,
        default=1,
        choices=range(1, 6),
        metavar="N",
        help="Start at phase N (1–5).  Default: 1.",
    )
    parser.add_argument(
        "--to-phase",
        type=int,
        default=5,
        choices=range(1, 6),
        metavar="N",
        help="Stop after phase N (1–5).  Default: 5.",
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        type=Path,
        help=f"Override DB path (default: {DB_PATH}).",
    )
    args = parser.parse_args()

    if args.from_phase > args.to_phase:
        parser.error(f"--from-phase ({args.from_phase}) must be ≤ --to-phase ({args.to_phase})")

    _log("pipeline", (
        f"Module 3 pipeline  "
        f"{'--refresh' if args.refresh else 'default (skip ingested)'}"
        f"  phases {args.from_phase}–{args.to_phase}"
    ))

    init_db(args.db)

    # Setup always runs — loads YAML, upserts Fund rows, promotes conditionals
    active_funds, completeness_warn = _setup()

    if args.fund:
        matched = [f for f in active_funds if f.name == args.fund]
        if not matched:
            names = "\n    ".join(f.name for f in active_funds)
            print(
                f"ERROR: --fund '{args.fund}' not found in active funds.\n"
                f"Active funds:\n    {names}",
                file=sys.stderr,
            )
            sys.exit(1)
        active_funds = matched
        _log("pipeline", f"  --fund filter: {args.fund}")

    _log("pipeline", f"  {len(active_funds)} fund(s) entering phases {args.from_phase}–{args.to_phase}")

    phase_dispatch = {
        1: lambda: run_phase1_edgar(active_funds, refresh=args.refresh, completeness_warn=completeness_warn),
        2: lambda: run_phase2_cusip(active_funds),
        3: lambda: run_phase3_prices(active_funds),
        4: lambda: run_phase4_returns(active_funds),
        5: lambda: run_phase5_skill(active_funds),
    }

    for n in range(args.from_phase, args.to_phase + 1):
        phase_dispatch[n]()

    _log("pipeline", f"all done  ({_elapsed()})")


if __name__ == "__main__":
    main()
