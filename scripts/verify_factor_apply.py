"""
Verification: FF4 skill decomposition (factor_apply.py) for Module 3.

Two checks are run in sequence:

Check 1 — Synthetic regression math
    Constructs synthetic quarterly fund returns from known alpha/beta parameters
    using actual FF4 quarterly factors (2020Q1–2024Q4, 20 quarters).  The fund
    returns are deterministic (no noise), so OLS must recover the exact input
    parameters.  Confirms: factor aggregation, excess-return construction, regression
    setup, and decomposition identity.

Check 2 — Viking EDGAR pipeline
    Fetches the 12 most recent canonical Viking 13F filings from EDGAR, seeds the
    5 known CUSIPs with prices over the full window, and runs the full pipeline via
    score_from_returns.  coverage_threshold=0.0 and min_quarters=4 are passed to
    bypass production gates — this produces a real but low-coverage estimate suitable
    only for verifying the end-to-end wiring, not for drawing conclusions about Viking.

Usage
-----
    python scripts/verify_factor_apply.py
"""

import datetime
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf

from factor_engine.french_data import get_ff4_daily
from smart_money import edgar
from smart_money.factor_apply import (
    MIN_QUARTERS_REG,
    MIN_QUARTERS_RELIABLE,
    _aggregate_quarter_factors,
    score_from_returns,
)
from smart_money.models import Filing, Fund, Holding, PriceCache, Security, init_db
from smart_money.returns import FundQuarterReturn, reconstruct_all_quarters

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

VIKING_CIK  = "1103804"
VIKING_NAME = "Viking Global Investors"

KNOWN_SECURITIES: dict[str, dict] = {
    "92826C839": {"ticker": "V",    "security_name": "VISA INC-CLASS A SHARES"},
    "874039100": {"ticker": "TSM",  "security_name": "TAIWAN SEMICONDUCTOR-SP ADR"},
    "808513105": {"ticker": "SCHW", "security_name": "SCHWAB (CHARLES) CORP"},
    "254687106": {"ticker": "DIS",  "security_name": "WALT DISNEY CO/THE"},
    "34959J108": {"ticker": "FTV",  "security_name": "FORTIVE CORP"},
}

# Quarter boundaries for the synthetic check — 2019Q4 through 2024Q4
# Each entry is (prior_period, current_period) defining one return window.
_QUARTER_ENDS = [
    datetime.date(2019, 12, 31),
    datetime.date(2020,  3, 31),
    datetime.date(2020,  6, 30),
    datetime.date(2020,  9, 30),
    datetime.date(2020, 12, 31),
    datetime.date(2021,  3, 31),
    datetime.date(2021,  6, 30),
    datetime.date(2021,  9, 30),
    datetime.date(2021, 12, 31),
    datetime.date(2022,  3, 31),
    datetime.date(2022,  6, 30),
    datetime.date(2022,  9, 30),
    datetime.date(2022, 12, 31),
    datetime.date(2023,  3, 31),
    datetime.date(2023,  6, 30),
    datetime.date(2023,  9, 30),
    datetime.date(2023, 12, 31),
    datetime.date(2024,  3, 31),
    datetime.date(2024,  6, 30),
    datetime.date(2024,  9, 30),
    datetime.date(2024, 12, 31),
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:+.2f}%"


def _sep(char: str = "─", width: int = 72) -> None:
    print(char * width)


# ---------------------------------------------------------------------------
# Check 1: synthetic regression math
# ---------------------------------------------------------------------------

# Known "true" parameters for the synthetic fund.
TRUE_ALPHA    = 0.010   # 1.0% quarterly = 4.0% annualised
TRUE_BETA_MKT = 0.85
TRUE_BETA_SMB = -0.30
TRUE_BETA_HML =  0.10
TRUE_BETA_MOM =  0.15


def _build_synthetic_quarters() -> list[FundQuarterReturn]:
    """
    Build 20 deterministic synthetic quarterly returns using actual FF4 factors.

    R_fund = RF_q + TRUE_ALPHA + TRUE_BETA_MKT * MktExcess_q
                  + TRUE_BETA_SMB * SMB_q + TRUE_BETA_HML * HML_q
                  + TRUE_BETA_MOM * MOM_q

    No noise — OLS must recover parameters exactly (within floating-point).
    """
    ff4 = get_ff4_daily(
        _QUARTER_ENDS[0].isoformat(),
        _QUARTER_ENDS[-1].isoformat(),
    )
    import pandas as pd
    quarters: list[FundQuarterReturn] = []
    for i in range(1, len(_QUARTER_ENDS)):
        boq = _QUARTER_ENDS[i - 1]
        eoq = _QUARTER_ENDS[i]
        fq  = _aggregate_quarter_factors(ff4, boq.isoformat(), eoq.isoformat())
        if fq is None:
            continue
        r_fund = (
            fq["rf"]
            + TRUE_ALPHA
            + TRUE_BETA_MKT * fq["mkt_excess"]
            + TRUE_BETA_SMB * fq["smb"]
            + TRUE_BETA_HML * fq["hml"]
            + TRUE_BETA_MOM * fq["mom"]
        )
        label = f"{eoq.year}Q{(eoq.month - 1) // 3 + 1}"
        quarters.append(FundQuarterReturn(
            fund_cik              = "SYNTHETIC",
            fund_name             = "Synthetic Fund",
            quarter               = label,
            period_start          = boq.isoformat(),
            period_end            = eoq.isoformat(),
            reconstructed_return  = r_fund,
            coverage_pct          = 1.0,
            n_holdings_total      = 50,
            n_holdings_with_price = 50,
            is_valid              = True,
        ))
    return quarters


def run_check_1() -> None:
    _sep("=")
    print("Check 1 — Synthetic regression math")
    _sep("=")
    print(f"\n  True parameters:  alpha={TRUE_ALPHA:.3f}/qtr  β_mkt={TRUE_BETA_MKT:.2f}"
          f"  β_smb={TRUE_BETA_SMB:.2f}  β_hml={TRUE_BETA_HML:.2f}  β_mom={TRUE_BETA_MOM:.2f}")
    print(f"  Method: deterministic (no noise) — OLS must recover exact values\n")

    quarters = _build_synthetic_quarters()
    print(f"  Quarters built: {len(quarters)}  "
          f"({quarters[0]['quarter']} → {quarters[-1]['quarter']})\n")

    score = score_from_returns("SYNTHETIC", "Synthetic Fund", quarters)

    if score is None:
        print("  ERROR: score_from_returns returned None — unexpected.")
        sys.exit(1)

    TOL = 1e-9   # floating-point tolerance for deterministic case

    rows = [
        ("alpha_quarterly",  score["alpha_quarterly"],  TRUE_ALPHA),
        ("beta_market",      score["beta_market"],      TRUE_BETA_MKT),
        ("beta_smb",         score["beta_smb"],         TRUE_BETA_SMB),
        ("beta_hml",         score["beta_hml"],         TRUE_BETA_HML),
        ("beta_mom",         score["beta_mom"],         TRUE_BETA_MOM),
    ]

    print(f"  {'Parameter':<20} {'Recovered':>10}  {'True':>8}  {'|Error|':>10}  {'Pass?'}")
    _sep()
    all_ok = True
    for name, recovered, true_val in rows:
        err    = abs(recovered - true_val)
        passed = err < TOL
        mark   = "✓" if passed else "✗ FAIL"
        if not passed:
            all_ok = False
        print(f"  {name:<20} {recovered:>10.7f}  {true_val:>8.4f}  {err:>10.2e}  {mark}")
    _sep()

    # Decomposition identity: five components must sum to avg_excess_return_q.
    decomp_sum = (
        score["alpha_quarterly"]
        + score["return_from_market"]
        + score["return_from_smb"]
        + score["return_from_hml"]
        + score["return_from_mom"]
    )
    decomp_err = abs(decomp_sum - score["avg_excess_return_q"])
    decomp_ok  = decomp_err < TOL
    if not decomp_ok:
        all_ok = False
    mark = "✓" if decomp_ok else "✗ FAIL"
    print(f"\n  Decomposition identity check:")
    print(f"    alpha + mkt + smb + hml + mom = {decomp_sum:.8f}")
    print(f"    avg_excess_return_q           = {score['avg_excess_return_q']:.8f}")
    print(f"    |error|                       = {decomp_err:.2e}   {mark}\n")

    print(f"  R² = {score['r_squared']:.6f}  (expected 1.000000 with no noise)")
    print(f"  n_quarters = {score['n_quarters']}\n")

    if not all_ok:
        print("  FAILED — regression math has a bug. Stop.")
        sys.exit(1)
    print("  Check 1 PASSED ✓\n")


# ---------------------------------------------------------------------------
# Check 2: Viking EDGAR pipeline
# ---------------------------------------------------------------------------

def _ingest_filing(
    fund: Fund,
    meta: edgar.FilingMeta,
    holdings: list[edgar.HoldingRow],
) -> Filing:
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
    Holding.insert_many([
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
    ]).execute()
    return filing


def _seed_securities() -> None:
    now = datetime.datetime.utcnow()
    for cusip, info in KNOWN_SECURITIES.items():
        Security.get_or_create(
            cusip=cusip,
            defaults={
                "ticker":            info["ticker"],
                "security_name":     info["security_name"],
                "resolution_status": "resolved",
                "resolved_at":       now,
            },
        )


def _fetch_and_cache_prices(price_start: datetime.date, price_end: datetime.date) -> None:
    tickers = [v["ticker"] for v in KNOWN_SECURITIES.values()]
    cusip_by_ticker = {v["ticker"]: k for k, v in KNOWN_SECURITIES.items()}
    sec_by_cusip    = {c: Security.get(Security.cusip == c) for c in KNOWN_SECURITIES}

    end_excl = (price_end + datetime.timedelta(days=1)).isoformat()
    print(f"  [prices] yfinance: {tickers}  {price_start} → {price_end}")
    raw = yf.download(
        tickers,
        start=price_start.isoformat(),
        end=end_excl,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if raw.empty:
        print("  [prices] ERROR: yfinance returned no data")
        return

    import math
    now = datetime.datetime.utcnow()
    for ticker in tickers:
        cusip = cusip_by_ticker[ticker]
        sec   = sec_by_cusip[cusip]
        try:
            close_s    = raw["Close"][ticker]
            adj_close_s = raw["Adj Close"][ticker]
        except (KeyError, TypeError):
            print(f"  [prices] WARNING: no data for {ticker}")
            continue
        rows = []
        for ts, cv, av in zip(close_s.index, close_s.values, adj_close_s.values):
            if math.isnan(float(cv)) or math.isnan(float(av)):
                continue
            rows.append({
                "security":   sec,
                "date":       ts.date() if hasattr(ts, "date") else ts,
                "close":      float(cv),
                "adj_close":  float(av),
                "source":     "yfinance",
                "fetched_at": now,
            })
        if rows:
            for i in range(0, len(rows), 500):
                PriceCache.insert_many(rows[i:i+500]).on_conflict_replace().execute()
            print(f"  [prices] {ticker}: {len(rows)} rows  "
                  f"({rows[0]['date']} → {rows[-1]['date']})")


def run_check_2() -> None:
    _sep("=")
    print("Check 2 — Viking EDGAR pipeline")
    _sep("=")
    print()
    print("  NOTE: coverage_threshold=0.0 and min_quarters=4 are used here.")
    print("  Only 5 CUSIPs have prices — coverage is low (~5-25%).")
    print("  This verifies end-to-end wiring only, not a valid Viking skill estimate.\n")

    # ── Temp DB ──────────────────────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    init_db(db_path)
    print(f"  Temp DB: {db_path}\n")

    _seed_securities()
    print(f"  {len(KNOWN_SECURITIES)} securities seeded\n")

    # ── Fetch Viking filing index ─────────────────────────────────────────────
    print(f"  Fetching 13F index for Viking CIK {VIKING_CIK}…")
    all_filings = edgar.list_13f_filings(VIKING_CIK)
    canonical   = edgar.canonical_filings(all_filings)
    print(f"  {len(canonical)} canonical filings found  "
          f"({canonical[0]['period_of_report']} → {canonical[-1]['period_of_report']})\n")

    # Take the 12 most recent canonical filings (→ 11 return quarters).
    selected = canonical[-12:]
    print(f"  Using most recent 12 filings: "
          f"{selected[0]['period_of_report']} → {selected[-1]['period_of_report']}\n")

    # ── Fetch holdings for each selected filing ───────────────────────────────
    viking = Fund.create(
        name       = VIKING_NAME,
        manager    = "Andreas Halvorsen",
        bucket     = "long_short_equity",
        aum_tier   = "large",
        cik        = VIKING_CIK,
        cik_status = "confirmed",
    )

    for i, meta in enumerate(selected):
        print(f"  [{i+1:02d}/{len(selected)}] Fetching holdings  "
              f"period={meta['period_of_report']}  type={meta['form_type']}")
        holdings = edgar.fetch_holdings(VIKING_CIK, meta["accession_number"])
        _ingest_filing(viking, meta, holdings)
        print(f"         → {len(holdings)} rows ingested")
    print()

    # ── Fetch prices ──────────────────────────────────────────────────────────
    # Cover all BOQ/EOQ dates used in reconstruction.
    earliest_boq = datetime.date.fromisoformat(selected[0]["period_of_report"])
    latest_eoq   = datetime.date.fromisoformat(selected[-1]["period_of_report"])
    # Widen by one month on each side so boundary prices are never missing.
    price_start  = (earliest_boq.replace(day=1)
                    - datetime.timedelta(days=1)).replace(day=1)
    price_end    = latest_eoq
    print(f"  Fetching prices  {price_start} → {price_end}…")
    _fetch_and_cache_prices(price_start, price_end)
    print()

    # ── Reconstruct quarterly returns ─────────────────────────────────────────
    print("  Reconstructing quarterly returns (coverage_threshold=0.0)…")
    all_quarters = reconstruct_all_quarters(viking, coverage_threshold=0.0)
    print(f"\n  {'Quarter':<10} {'Return':>8}  {'Coverage':>9}  {'Valid?'}")
    _sep()
    for q in all_quarters:
        print(f"  {q['quarter']:<10} {_pct(q['reconstructed_return']):>8}  "
              f"{q['coverage_pct']:>8.1%}  {'yes' if q['is_valid'] else 'no '}"
              f"  ({q['n_holdings_with_price']}/{q['n_holdings_total']} holdings)")
    _sep()
    print(f"  {len(all_quarters)} quarters total  "
          f"({sum(1 for q in all_quarters if q['is_valid'])} is_valid=True)\n")

    # ── Skill decomposition ───────────────────────────────────────────────────
    valid = [q for q in all_quarters if q["is_valid"]]
    print(f"  Running score_from_returns  (n={len(valid)} quarters, min_quarters=4)…\n")

    score = score_from_returns(
        VIKING_CIK,
        VIKING_NAME,
        valid,
        min_quarters=4,
    )

    if score is None:
        print("  score_from_returns returned None — fewer than 4 valid quarters.")
        print("  End-to-end wiring verified (no regression output to check).\n")
        db_path.unlink(missing_ok=True)
        print("  Check 2 complete (no score).\n")
        return

    # ── Print skill score ─────────────────────────────────────────────────────
    print("=" * 72)
    print(f"  {VIKING_NAME}  —  FF4 Skill Decomposition  [LOW COVERAGE / VERIFY ONLY]")
    print("=" * 72)
    print(f"  Quarters used   : {score['n_quarters']}  ({score['quarters_used'][0]} → {score['quarters_used'][-1]})")
    print(f"  Reliable?       : {score['is_reliable']}  (threshold = {MIN_QUARTERS_RELIABLE})")
    print(f"  Confidence      : {score['confidence_label']}")
    print()
    print(f"  Alpha (quarterly)   : {_pct(score['alpha_quarterly'])}")
    print(f"  Alpha (annualised)  : {_pct(score['alpha_annualized'])}")
    print(f"  Alpha t-stat        : {score['alpha_t_stat']:+.3f}")
    print(f"  Alpha p-value       : {score['alpha_p_value']:.4f}")
    print()
    print(f"  β_market   : {score['beta_market']:+.4f}  (t = {score['t_stat_market']:+.3f})")
    print(f"  β_smb      : {score['beta_smb']:+.4f}  (t = {score['t_stat_smb']:+.3f})")
    print(f"  β_hml      : {score['beta_hml']:+.4f}  (t = {score['t_stat_hml']:+.3f})")
    print(f"  β_mom      : {score['beta_mom']:+.4f}  (t = {score['t_stat_mom']:+.3f})")
    print(f"  R²         : {score['r_squared']:.4f}")
    print()
    print("  Historical attribution (avg quarterly excess return over sample):")
    print(f"  {'Component':<28} {'Quarterly':>10}")
    _sep()
    print(f"  {'From market beta':<28} {_pct(score['return_from_market']):>10}")
    print(f"  {'From size factor (SMB)':<28} {_pct(score['return_from_smb']):>10}")
    print(f"  {'From value factor (HML)':<28} {_pct(score['return_from_hml']):>10}")
    print(f"  {'From momentum factor (MOM)':<28} {_pct(score['return_from_mom']):>10}")
    print(f"  {'Alpha (skill)':<28} {_pct(score['alpha_quarterly']):>10}")
    _sep()
    decomp_sum = (
        score["return_from_market"]
        + score["return_from_smb"]
        + score["return_from_hml"]
        + score["return_from_mom"]
        + score["alpha_quarterly"]
    )
    print(f"  {'Total (sum)':<28} {_pct(decomp_sum):>10}")
    print(f"  {'avg_excess_return_q':<28} {_pct(score['avg_excess_return_q']):>10}")
    decomp_err = abs(decomp_sum - score["avg_excess_return_q"])
    # Each of the five components is rounded to 6dp independently, so up to
    # 5 × 5e-7 = 2.5e-6 accumulated rounding error is expected on real data.
    DECOMP_TOL = 1e-5
    print(f"  {'Decomposition error':<28} {decomp_err:.2e}   "
          f"{'✓' if decomp_err < DECOMP_TOL else '✗ FAIL'}  (tol {DECOMP_TOL:.0e})")
    print()

    # ── Structural assertions ─────────────────────────────────────────────────
    assert score["n_quarters"] >= 4,         "n_quarters < 4"
    assert 0.0 <= score["r_squared"] <= 1.0, "r_squared out of [0, 1]"
    assert decomp_err < DECOMP_TOL,          "decomposition identity failed"
    assert (score["is_reliable"] ==
            (score["n_quarters"] >= MIN_QUARTERS_RELIABLE)),  "is_reliable inconsistent"
    print("  [check] n_quarters >= 4 ✓")
    print("  [check] r_squared ∈ [0, 1] ✓")
    print("  [check] decomposition identity ✓")
    print("  [check] is_reliable consistent with n_quarters ✓")
    print()

    db_path.unlink(missing_ok=True)
    print("  Check 2 PASSED ✓\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    run_check_1()
    run_check_2()
    _sep("=")
    print("All checks passed.")
    _sep("=")


if __name__ == "__main__":
    main()
