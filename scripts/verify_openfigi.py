"""
Verification: resolve real CUSIPs from Viking Global's latest 13F via OpenFIGI.

Fetches Viking's most recent filing, picks the top N positions by USD value,
resolves their CUSIPs, and prints a results table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_money.edgar import fetch_latest_holdings
from smart_money.cusip import resolve_cusips

VIKING_CIK = "1103804"
TOP_N = 5


def main() -> None:
    print(f"Fetching Viking Global (CIK {VIKING_CIK}) latest 13F holdings …")
    filing, holdings = fetch_latest_holdings(VIKING_CIK)
    print(f"  Filing: {filing['period_of_report']}  ({filing['accession_number']})")
    print(f"  Total positions: {len(holdings)}\n")

    top = sorted(holdings, key=lambda h: h["value_usd"], reverse=True)[:TOP_N]

    cusips = [h["cusip"] for h in top]
    print(f"Resolving top {TOP_N} CUSIPs via OpenFIGI …")
    results = resolve_cusips(cusips, skip_resolved=False)

    print()
    print(f"{'#':<3} {'CUSIP':<10} {'Issuer (13F)':<30} {'Value ($M)':<12} "
          f"{'Ticker':<8} {'CompFIGI':<14} {'Type':<15} {'Exch':<6} Status")
    print("─" * 115)

    for rank, holding in enumerate(top, 1):
        cusip  = holding["cusip"]
        issuer = holding["issuer_name"][:28]
        value  = holding["value_usd"] / 1_000_000
        r      = results.get(cusip)
        if r:
            ticker = r["ticker"] or "—"
            figi   = r["composite_figi"] or "—"
            stype  = (r["security_type"] or "—")[:14]
            exch   = r["exchange_code"] or "—"
            status = "resolved"
        else:
            ticker = figi = stype = exch = "—"
            status = "no_match/failed"

        print(
            f"{rank:<3} {cusip:<10} {issuer:<30} {value:<12.1f} "
            f"{ticker:<8} {figi:<14} {stype:<15} {exch:<6} {status}"
        )

    resolved = sum(1 for r in results.values() if r is not None)
    print(f"\n  {resolved}/{len(cusips)} CUSIPs resolved successfully.")


if __name__ == "__main__":
    main()
