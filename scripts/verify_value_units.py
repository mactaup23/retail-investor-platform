"""
Module 3 — Per-filer value-unit verification

Some 13F filers report the infoTable "value" field in thousands of dollars,
per the literal SEC spec, instead of raw dollars (the empirically-verified
default for this codebase — see smart_money/edgar.py::_VALUE_UNIT_SCALE).

For every confirmed fund in config/fund_universe.yaml, this script:
    1. fetches the latest 13F-HR holdings (value_usd already scaled per
       edgar.py's current _VALUE_UNIT_SCALE table),
    2. samples a few large, liquid positions and resolves their CUSIPs via
       OpenFIGI,
    3. fetches a real closing price near the filing's period_of_report,
    4. compares implied price (value_usd / shares) against the real price.

A ~1000x mismatch means _VALUE_UNIT_SCALE is missing an override (or has a
wrong one) for that CIK. This is an offline, manually-run check — it is not
wired into the pipeline, since it depends on OpenFIGI + yfinance round-trips
that only matter when onboarding a new fund, not on every ingestion run.

Usage:
    .venv/bin/python scripts/verify_value_units.py
    .venv/bin/python scripts/verify_value_units.py --fund "Duquesne Family Office"
"""

import argparse
import datetime
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_money.cusip import resolve_cusips
from smart_money.edgar import fetch_latest_holdings
from smart_money.models import init_db
from smart_money.prices import fetch_prices_for_cusips

YAML_PATH = Path(__file__).parent.parent / "config" / "fund_universe.yaml"

SAMPLE_SIZE = 5          # largest-by-shares holdings to sample per fund
PRICE_WINDOW_DAYS = 7    # window around period_of_report to fetch prices over

# Ratio = implied_price / real_price for one sampled holding.
OK_BAND = (1 / 3, 3)              # plausible: date/volatility noise only
UNIT_MISMATCH_BAND = (300, 3000)  # plausible sign of a missing/wrong x1000 scale


def classify_ratio(ratio: float) -> str:
    """Classify one holding's (implied / real) price ratio."""
    ok_lo, ok_hi = OK_BAND
    if ok_lo <= ratio <= ok_hi:
        return "ok"
    mismatch_lo, mismatch_hi = UNIT_MISMATCH_BAND
    if mismatch_lo <= ratio <= mismatch_hi or mismatch_lo <= (1 / ratio) <= mismatch_hi:
        return "unit_mismatch"
    return "other_mismatch"


def load_funds() -> list[dict]:
    with open(YAML_PATH) as f:
        data = yaml.safe_load(f)
    return [
        fund for fund in data["funds"]
        if fund.get("cik_status") == "confirmed" and fund.get("cik")
    ]


def check_fund(fund: dict) -> str:
    """Returns one of: 'ok', 'mismatch', 'no_data', 'error'."""
    name = fund["name"]
    cik = fund["cik"]

    try:
        meta, holdings = fetch_latest_holdings(cik)
    except Exception as e:
        print(f"  [error] could not fetch holdings: {e}")
        return "error"

    candidates = [h for h in holdings if h["shares"] > 0 and h["cusip"]]
    candidates.sort(key=lambda h: h["shares"], reverse=True)
    sample = candidates[:SAMPLE_SIZE]
    if not sample:
        print(f"  [no_data] no holdings with shares/CUSIP to sample (period {meta['period_of_report']})")
        return "no_data"

    cusips = [h["cusip"] for h in sample]
    figi_results = resolve_cusips(cusips)

    period = datetime.date.fromisoformat(meta["period_of_report"])
    start = period - datetime.timedelta(days=PRICE_WINDOW_DAYS)
    end = period + datetime.timedelta(days=PRICE_WINDOW_DAYS)

    resolved_cusips = [c for c in cusips if figi_results.get(c) and figi_results[c]["ticker"]]
    if not resolved_cusips:
        print(f"  [no_data] none of the sampled CUSIPs resolved to a ticker via OpenFIGI")
        return "no_data"

    prices = fetch_prices_for_cusips(resolved_cusips, start, end)

    checked = 0
    classifications: list[str] = []
    for h in sample:
        cusip = h["cusip"]
        if cusip not in prices or prices[cusip].empty:
            continue
        df = prices[cusip]
        real_price = float(df["close"].iloc[len(df) // 2])  # closest available to period midpoint
        if not real_price:
            continue
        implied_price = h["value_usd"] / h["shares"]
        ratio = implied_price / real_price
        checked += 1
        verdict = classify_ratio(ratio)
        classifications.append(verdict)
        flag = {
            "ok": "",
            "unit_mismatch": "  <-- LIKELY UNIT MISMATCH (~1000x)",
            "other_mismatch": "  <-- MISMATCH (unexplained by a x1000 scale error)",
        }[verdict]
        print(
            f"    {h['issuer_name']:<30} implied=${implied_price:>10,.2f}/sh  "
            f"real=${real_price:>10,.2f}/sh  ratio={ratio:>8.3f}{flag}"
        )

    if checked == 0:
        print(f"  [no_data] resolved tickers but no price data in window {start}..{end}")
        return "no_data"

    if "unit_mismatch" in classifications:
        print(f"  [MISMATCH] {name} (CIK {cik}) — implied price is ~1000x off from real price")
        print(f"             Add/check an entry in smart_money/edgar.py::_VALUE_UNIT_SCALE for CIK {cik}")
        return "mismatch"
    if "other_mismatch" in classifications:
        print(f"  [MISMATCH] {name} (CIK {cik}) — price discrepancy not explained by a x1000 scale error")
        print(f"             Check CUSIP resolution / stock splits before assuming this is a units issue")
        return "mismatch"

    print(f"  [ok] {checked} sampled position(s) consistent with current value scale")
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fund", help="Only check the fund with this exact name")
    args = parser.parse_args()

    init_db()
    funds = load_funds()
    if args.fund:
        funds = [f for f in funds if f["name"] == args.fund]
        if not funds:
            print(f"No confirmed fund named {args.fund!r} found in {YAML_PATH}")
            sys.exit(1)

    print(f"Checking {len(funds)} confirmed fund(s) for value-unit mismatches...\n")

    results: dict[str, list[str]] = {"ok": [], "mismatch": [], "no_data": [], "error": []}
    for fund in funds:
        print(f"=== {fund['name']} (CIK {fund['cik']}) ===")
        outcome = check_fund(fund)
        results[outcome].append(fund["name"])
        print()

    print("─" * 60)
    print(f"ok:       {len(results['ok'])}")
    print(f"mismatch: {len(results['mismatch'])}  {results['mismatch']}")
    print(f"no_data:  {len(results['no_data'])}  {results['no_data']}")
    print(f"error:    {len(results['error'])}  {results['error']}")

    if results["mismatch"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
