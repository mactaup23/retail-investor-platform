"""
Verification: Q1 2026 return reconstruction for Viking Global Investors.

Uses a temporary SQLite DB populated with:
  - Viking Q4 2025 and Q1 2026 13F-HR holdings fetched live from EDGAR
  - Security rows for the 5 CUSIPs resolved in prior sessions
    (V, TSM, SCHW, DIS, FTV)
  - PriceCache rows for Dec 2025 → Mar 2026 fetched live from yfinance

Reconstructs Viking's Q1 2026 return via returns.reconstruct_fund_quarter,
reports the per-holding breakdown and coverage %, then compares against
SPY's actual Q1 2026 return as a directional sanity check.

Usage
-----
    python scripts/verify_returns.py
"""

import datetime
import sys
import tempfile
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_money import edgar
from smart_money.models import Filing, Fund, Holding, PriceCache, Security, init_db
from smart_money.returns import (
    COVERAGE_THRESHOLD,
    _adj_close_near,
    _eligible_holdings,
    _get_filing,
    reconstruct_fund_quarter,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VIKING_CIK  = "1103804"
VIKING_NAME = "Viking Global Investors"

# The 5 CUSIPs resolved and price-verified in prior Module 3 sessions.
# These are the only securities for which PriceCache will be populated here;
# all other Viking holdings will be counted in n_holdings_total but excluded
# from the return computation (contributing to the coverage gap).
KNOWN_SECURITIES: dict[str, dict] = {
    "92826C839": {"ticker": "V",    "security_name": "VISA INC-CLASS A SHARES"},
    "874039100": {"ticker": "TSM",  "security_name": "TAIWAN SEMICONDUCTOR-SP ADR"},
    "808513105": {"ticker": "SCHW", "security_name": "SCHWAB (CHARLES) CORP"},
    "254687106": {"ticker": "DIS",  "security_name": "WALT DISNEY CO/THE"},
    "34959J108": {"ticker": "FTV",  "security_name": "FORTIVE CORP"},
}

# Price window: covers Dec 31, 2025 (BOQ) and Mar 31, 2026 (EOQ) with margin
PRICE_START = datetime.date(2025, 12, 1)
PRICE_END   = datetime.date(2026, 3, 31)

Q4_2025 = datetime.date(2025, 12, 31)
Q1_2026 = datetime.date(2026, 3, 31)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ingest_filing(
    fund: Fund,
    meta: edgar.FilingMeta,
    holdings: list[edgar.HoldingRow],
) -> Filing:
    """Insert a Filing and its Holdings into the DB; return the Filing row."""
    total_value = sum(h["value_usd"] for h in holdings)
    filing = Filing.create(
        fund                 = fund,
        period_of_report     = datetime.date.fromisoformat(meta["period_of_report"]),
        filed_date           = datetime.date.fromisoformat(meta["filed_date"]),
        accession_number     = meta["accession_number"],
        form_type            = meta["form_type"],
        total_value_usd      = total_value,
        total_holdings_count = len(holdings),
    )
    ranked = sorted(holdings, key=lambda h: h["value_usd"], reverse=True)
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
            "rank_by_value":         rank + 1,
            "is_price_eligible":     True,
        }
        for rank, h in enumerate(ranked)
    ]
    Holding.insert_many(rows).execute()
    return filing


def _seed_securities(db_path: Path) -> None:
    """Insert the 5 known Security rows into the temp DB."""
    now = datetime.datetime.utcnow()
    for cusip, info in KNOWN_SECURITIES.items():
        Security.get_or_create(
            cusip=cusip,
            defaults={
                "ticker":             info["ticker"],
                "security_name":      info["security_name"],
                "resolution_status":  "resolved",
                "resolved_at":        now,
            },
        )


def _fetch_and_cache_prices() -> None:
    """
    Fetch adj_close prices for the 5 known tickers over the Q1 2026 window
    and upsert into PriceCache in the current (temp) DB.
    """
    tickers = [info["ticker"] for info in KNOWN_SECURITIES.values()]
    cusip_by_ticker = {info["ticker"]: cusip for cusip, info in KNOWN_SECURITIES.items()}
    sec_by_cusip: dict[str, Security] = {}
    for cusip in KNOWN_SECURITIES:
        sec_by_cusip[cusip] = Security.get(Security.cusip == cusip)

    end_excl = PRICE_END + datetime.timedelta(days=1)
    print(f"  [prices] Fetching from yfinance: {tickers}  {PRICE_START} → {PRICE_END}")
    raw = yf.download(
        tickers,
        start=PRICE_START.isoformat(),
        end=end_excl.isoformat(),
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if raw.empty:
        print("  [prices] ERROR: yfinance returned no data")
        return

    now = datetime.datetime.utcnow()
    for ticker in tickers:
        cusip = cusip_by_ticker[ticker]
        sec   = sec_by_cusip[cusip]
        try:
            close_s     = raw["Close"][ticker]
            adj_close_s = raw["Adj Close"][ticker]
        except (KeyError, TypeError):
            print(f"  [prices] WARNING: no data for {ticker}")
            continue

        rows = []
        for ts, close_val, adj_val in zip(close_s.index, close_s.values, adj_close_s.values):
            import math
            if math.isnan(float(close_val)) or math.isnan(float(adj_val)):
                continue
            row_date = ts.date() if hasattr(ts, "date") else ts
            rows.append({
                "security":   sec,
                "date":       row_date,
                "close":      float(close_val),
                "adj_close":  float(adj_val),
                "source":     "yfinance",
                "fetched_at": now,
            })

        if rows:
            for i in range(0, len(rows), 500):
                PriceCache.insert_many(rows[i:i+500]).on_conflict_replace().execute()
            print(f"  [prices] {ticker}: {len(rows)} rows cached ({rows[0]['date']} → {rows[-1]['date']})")


def _spy_q1_2026_return() -> float | None:
    """Fetch SPY adj_close on Dec 31, 2025 and Mar 31, 2026 from yfinance."""
    end_excl = (Q1_2026 + datetime.timedelta(days=1)).isoformat()
    raw = yf.download(
        ["SPY"],
        start="2025-12-29",
        end=end_excl,
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        return None
    try:
        adj = raw["Adj Close"]["SPY"] if "SPY" in raw["Adj Close"].columns else raw["Adj Close"]
        adj = adj.dropna()
    except (KeyError, AttributeError):
        return None
    if len(adj) < 2:
        return None
    # First row ≈ Dec 31, last row ≈ Mar 31
    return float(adj.iloc[-1]) / float(adj.iloc[0]) - 1.0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:+.2f}%"


def _usd_b(v: float) -> str:
    return f"${v / 1e9:.3f}B"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── 1. Temp DB ───────────────────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    init_db(db_path)
    print(f"[verify_returns] Temp DB: {db_path}\n")

    # ── 2. Seed Securities ───────────────────────────────────────────────────
    print("[verify_returns] Seeding Security rows…")
    _seed_securities(db_path)
    print(f"  {len(KNOWN_SECURITIES)} securities seeded\n")

    # ── 3. Fetch and cache prices ────────────────────────────────────────────
    print("[verify_returns] Fetching prices…")
    _fetch_and_cache_prices()
    print()

    # ── 4. Fetch Viking 13F filings ──────────────────────────────────────────
    print(f"[verify_returns] Fetching 13F filing list for Viking CIK {VIKING_CIK}…")
    all_filings = edgar.list_13f_filings(VIKING_CIK)
    canonical   = edgar.canonical_filings(all_filings)

    # Locate Q4 2025 and Q1 2026
    by_period = {f["period_of_report"]: f for f in canonical}
    q4_meta = by_period.get("2025-12-31")
    q1_meta = by_period.get("2026-03-31")

    if q4_meta is None or q1_meta is None:
        found = sorted(by_period.keys())[-4:]
        print(f"  ERROR: Q4 2025 or Q1 2026 filing not found.  Most recent 4: {found}")
        sys.exit(1)

    print(f"  Q4 2025 : {q4_meta['form_type']}  filed {q4_meta['filed_date']}")
    print(f"  Q1 2026 : {q1_meta['form_type']}  filed {q1_meta['filed_date']}\n")

    # ── 5. Fetch holdings ────────────────────────────────────────────────────
    print("[verify_returns] Fetching Q4 2025 holdings…")
    q4_holdings = edgar.fetch_holdings(VIKING_CIK, q4_meta["accession_number"])
    print(f"  {len(q4_holdings)} holding rows")

    print("[verify_returns] Fetching Q1 2026 holdings…")
    q1_holdings = edgar.fetch_holdings(VIKING_CIK, q1_meta["accession_number"])
    print(f"  {len(q1_holdings)} holding rows\n")

    # ── 6. Ingest into temp DB ───────────────────────────────────────────────
    viking = Fund.create(
        name       = VIKING_NAME,
        manager    = "Andreas Halvorsen",
        bucket     = "long_short_equity",
        aum_tier   = "large",
        cik        = VIKING_CIK,
        cik_status = "confirmed",
    )
    print("[verify_returns] Ingesting Q4 2025 filing (BOQ weights)…")
    _ingest_filing(viking, q4_meta, q4_holdings)
    print("[verify_returns] Ingesting Q1 2026 filing…")
    _ingest_filing(viking, q1_meta, q1_holdings)
    print()

    # ── 7. Reconstruct Q1 2026 return ────────────────────────────────────────
    print("[verify_returns] Running return reconstruction…\n")
    result = reconstruct_fund_quarter(viking, Q1_2026)

    if result is None:
        print("  ERROR: reconstruct_fund_quarter returned None (first-filing logic?)")
        sys.exit(1)

    # ── 8. Per-holding breakdown ─────────────────────────────────────────────
    prior_filing  = _get_filing(viking, Q4_2025)
    boq_holdings  = _eligible_holdings(prior_filing)
    total_boq_val = sum(boq_holdings.values())

    print(f"{'Ticker':<8} {'CUSIP':<10} {'BOQ Value':>13} {'BOQ Wt':>8} "
          f"{'BOQ Px':>9} {'EOQ Px':>9} {'Return':>8}  {'In Coverage?'}")
    print("─" * 84)

    for cusip, info in KNOWN_SECURITIES.items():
        ticker = info["ticker"]
        val    = boq_holdings.get(cusip, 0.0)
        p_boq  = _adj_close_near(cusip, Q4_2025)
        p_eoq  = _adj_close_near(cusip, Q1_2026)

        if val == 0.0:
            in_cov = "not in prior filing"
        elif p_boq is None or p_eoq is None or p_boq <= 0.0:
            in_cov = "no price → excluded"
        else:
            ret    = p_eoq / p_boq - 1.0
            in_cov = f"yes  r={_pct(ret)}"

        wt  = val / total_boq_val if total_boq_val > 0 else 0.0
        px_boq_str = f"${p_boq:.2f}" if p_boq else "—"
        px_eoq_str = f"${p_eoq:.2f}" if p_eoq else "—"

        print(
            f"{ticker:<8} {cusip:<10} {_usd_b(val):>13} {wt:>7.2%} "
            f"{px_boq_str:>9} {px_eoq_str:>9}          {in_cov}"
        )
    print()

    # ── 9. Summary ───────────────────────────────────────────────────────────
    print("=" * 68)
    print(f"  {VIKING_NAME}  —  Q1 2026 Return Reconstruction")
    print("=" * 68)
    print(f"  Period              : {result['period_start']}  →  {result['period_end']}")
    print(f"  Holdings (prior)    : {result['n_holdings_total']}  long equity, price-eligible")
    print(f"  Holdings with price : {result['n_holdings_with_price']}  (only 5 CUSIPs resolved)")
    print(f"  Coverage            : {result['coverage_pct']:.1%}  (by BOQ portfolio value)")
    print(f"  Coverage gate (80%) : {'PASS ✓' if result['is_valid'] else 'FAIL — insufficient for skill decomp'}")
    print(f"  Reconstructed return: {_pct(result['reconstructed_return'])}")
    print()

    # ── 10. SPY comparison ───────────────────────────────────────────────────
    print("[verify_returns] Fetching SPY Q1 2026 return…")
    spy_ret = _spy_q1_2026_return()
    if spy_ret is not None:
        diff = result["reconstructed_return"] - spy_ret
        print(f"\n  SPY Q1 2026 actual     : {_pct(spy_ret)}")
        print(f"  Viking reconstructed   : {_pct(result['reconstructed_return'])}")
        print(f"  Difference             : {_pct(diff)}  (Viking vs SPY)")
        if not result["is_valid"]:
            print(
                f"\n  NOTE: coverage is {result['coverage_pct']:.1%} — well below the 80% gate.\n"
                f"  The reconstructed return reflects only {result['n_holdings_with_price']} of "
                f"{result['n_holdings_total']} holdings and should not be compared\n"
                f"  to SPY as a reliable fund-level estimate.  Full pipeline with all\n"
                f"  CUSIPs resolved would be needed for a valid comparison."
            )
    else:
        print("  SPY: could not fetch Q1 2026 prices from yfinance")
    print()

    # ── 11. Structural checks ─────────────────────────────────────────────────
    assert result["n_holdings_with_price"] <= result["n_holdings_total"], \
        "BUG: n_holdings_with_price > n_holdings_total"
    assert 0.0 <= result["coverage_pct"] <= 1.0, \
        "BUG: coverage_pct out of [0, 1]"
    assert result["is_valid"] == (result["coverage_pct"] >= COVERAGE_THRESHOLD), \
        "BUG: is_valid flag inconsistent with coverage_pct"
    print("  [check] n_holdings_with_price ≤ n_holdings_total ✓")
    print("  [check] coverage_pct ∈ [0, 1] ✓")
    print("  [check] is_valid consistent with coverage_pct ✓")

    # First-filing guard: Q4 2025 should return None
    q4_result = reconstruct_fund_quarter(viking, Q4_2025)
    assert q4_result is None, "BUG: first filing should return None"
    print("  [check] first filing (Q4 2025) returns None ✓")
    print()
    print(f"  Verification complete.")

    # Cleanup
    db_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
