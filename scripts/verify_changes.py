"""
Verification: quarter-over-quarter change detection for Viking Global Investors.

Fetches the two most recent canonical 13F-HR filings from EDGAR live,
ingests them into a temporary SQLite DB, runs detect_changes, and prints
a formatted summary — change-type counts, direction breakdown, and the
top-10 moves by current value.

Usage
-----
    python scripts/verify_changes.py
"""

import datetime
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_money import edgar
from smart_money.changes import detect_changes
from smart_money.models import Filing, Fund, Holding, init_db

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VIKING_CIK  = "1103804"
VIKING_NAME = "Viking Global Investors"

# How many holdings to print per change type in the detail table
TOP_N = 10

# ---------------------------------------------------------------------------
# Minimal ingest helpers
# ---------------------------------------------------------------------------

def _ingest_filing(
    fund: Fund,
    meta: edgar.FilingMeta,
    holdings: list[edgar.HoldingRow],
) -> Filing:
    """Insert a Filing and its Holdings into the DB; return the Filing row."""
    total_value = sum(h["value_usd"] for h in holdings)

    filing = Filing.create(
        fund                  = fund,
        period_of_report      = datetime.date.fromisoformat(meta["period_of_report"]),
        filed_date            = datetime.date.fromisoformat(meta["filed_date"]),
        accession_number      = meta["accession_number"],
        form_type             = meta["form_type"],
        total_value_usd       = total_value,
        total_holdings_count  = len(holdings),
    )

    # Rank holdings by descending value (1-based) for rank_by_value
    ranked = sorted(holdings, key=lambda h: h["value_usd"], reverse=True)
    rows = [
        {
            "filing":               filing,
            "cusip":                h["cusip"],
            "issuer_name":          h["issuer_name"],
            "value_usd":            h["value_usd"],
            "shares":               h["shares"],
            "investment_discretion": h["investment_discretion"],
            "put_call":             h["put_call"],
            "other_manager":        h["other_manager"],
            "rank_by_value":        rank + 1,
            "is_price_eligible":    True,
        }
        for rank, h in enumerate(ranked)
    ]
    Holding.insert_many(rows).execute()
    return filing


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_usd(v: int | None) -> str:
    if v is None:
        return "        —"
    billions = v / 1_000_000_000
    if abs(billions) >= 0.1:
        return f"${billions:+8.3f}B"
    millions = v / 1_000_000
    return f"${millions:+8.1f}M"


def _fmt_shares(v: int | None) -> str:
    if v is None:
        return "       —"
    return f"{v:>12,}"


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "     —"
    return f"{v:+7.1f}%"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── 1. Temp DB ──────────────────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    init_db(db_path)
    print(f"[verify_changes] Using temp DB: {db_path}\n")

    # ── 2. Fetch filing metadata from EDGAR ─────────────────────────────────
    print(f"[verify_changes] Fetching 13F filing list for Viking CIK {VIKING_CIK}…")
    all_filings = edgar.list_13f_filings(VIKING_CIK)
    canonical   = edgar.canonical_filings(all_filings)

    if len(canonical) < 2:
        print("ERROR: fewer than 2 canonical filings found — cannot diff.")
        sys.exit(1)

    prior_meta   = canonical[-2]
    current_meta = canonical[-1]

    print(
        f"  Prior period  : {prior_meta['period_of_report']}  "
        f"({prior_meta['form_type']}  filed {prior_meta['filed_date']})"
    )
    print(
        f"  Current period: {current_meta['period_of_report']}  "
        f"({current_meta['form_type']}  filed {current_meta['filed_date']})"
    )
    print()

    # ── 3. Fetch holdings ───────────────────────────────────────────────────
    print("[verify_changes] Fetching prior period holdings…")
    prior_holdings = edgar.fetch_holdings(VIKING_CIK, prior_meta["accession_number"])
    print(f"  {len(prior_holdings)} holding rows")

    print("[verify_changes] Fetching current period holdings…")
    current_holdings = edgar.fetch_holdings(VIKING_CIK, current_meta["accession_number"])
    print(f"  {len(current_holdings)} holding rows")
    print()

    # ── 4. Ingest into temp DB ──────────────────────────────────────────────
    viking = Fund.create(
        name       = VIKING_NAME,
        manager    = "Andreas Halvorsen",
        bucket     = "long_short_equity",
        aum_tier   = "large",
        cik        = VIKING_CIK,
        cik_status = "confirmed",
    )

    print("[verify_changes] Ingesting prior filing…")
    _ingest_filing(viking, prior_meta, prior_holdings)

    print("[verify_changes] Ingesting current filing…")
    current_filing = _ingest_filing(viking, current_meta, current_holdings)
    print()

    # ── 5. Run change detection ─────────────────────────────────────────────
    current_period = datetime.date.fromisoformat(current_meta["period_of_report"])
    changes = detect_changes(viking, current_period)

    # ── 6. Summary counts ───────────────────────────────────────────────────
    counts: dict[str, int] = {}
    for c in changes:
        counts[c["change_type"]] = counts.get(c["change_type"], 0) + 1

    bullish  = sum(1 for c in changes if c["direction"] == "bullish_leaning")
    bearish  = sum(1 for c in changes if c["direction"] == "bearish_leaning")
    total    = len(changes)

    prior_str   = prior_meta["period_of_report"]
    current_str = current_meta["period_of_report"]

    print("=" * 72)
    print(f"  Viking Global Investors  {prior_str} → {current_str}")
    print("=" * 72)
    print(f"  Total changes detected : {total}")
    print(f"  Prior holdings         : {len(prior_holdings)}")
    print(f"  Current holdings       : {len(current_holdings)}")
    print()
    print("  Change-type breakdown:")
    for ct in ("NEW", "INCREASED", "DECREASED", "CLOSED"):
        n = counts.get(ct, 0)
        bar = "█" * min(n, 40)
        print(f"    {ct:<12} {n:>4}  {bar}")
    print()
    print("  Direction breakdown:")
    print(f"    bullish_leaning : {bullish:>4}  (NEW + INCREASED)")
    print(f"    bearish_leaning : {bearish:>4}  (CLOSED + DECREASED)")
    print()

    # ── 7. Detail tables by change type ─────────────────────────────────────
    for ct in ("NEW", "INCREASED", "DECREASED", "CLOSED"):
        subset = [c for c in changes if c["change_type"] == ct]
        if not subset:
            continue

        dir_label = "bullish_leaning" if ct in ("NEW", "INCREASED") else "bearish_leaning"
        print(f"  ── {ct} ({dir_label})  {len(subset)} positions  (top {min(TOP_N, len(subset))} shown)")
        print(
            f"  {'Issuer':<35} {'CUSIP':<10} "
            f"{'Prior Shs':>13} {'Curr Shs':>13} {'Δ Shares':>13} {'Δ%':>8}  {'Curr Val':>11}"
        )
        print("  " + "─" * 106)

        for c in subset[:TOP_N]:
            print(
                f"  {c['issuer_name'][:35]:<35} {c['cusip']:<10} "
                f"{_fmt_shares(c['prior_shares'])} "
                f"{_fmt_shares(c['current_shares'])} "
                f"{_fmt_shares(c['shares_delta'])} "
                f"{_fmt_pct(c['shares_pct_change'])}  "
                f"{_fmt_usd(c['current_value_usd'])}"
            )
        print()

    # ── 8. Spot-check: first_filing flag should be False ────────────────────
    assert all(not c["first_filing"] for c in changes), \
        "BUG: first_filing=True on a two-period diff"
    print("  [check] first_filing=False on all changes ✓")

    # ── 9. Spot-check: direction ↔ change_type consistency ──────────────────
    for c in changes:
        expected = {
            "NEW": "bullish_leaning", "INCREASED": "bullish_leaning",
            "DECREASED": "bearish_leaning", "CLOSED": "bearish_leaning",
            "UNCHANGED": "neutral",
        }[c["change_type"]]
        assert c["direction"] == expected, \
            f"BUG: direction mismatch on {c['cusip']}: {c['direction']} ≠ {expected}"
    print("  [check] direction ↔ change_type consistent on all rows ✓")

    # ── 10. Spot-check: union of changes spans prior + current holdings ──────
    all_cusips_prior   = {h["cusip"] for h in prior_holdings}
    all_cusips_current = {h["cusip"] for h in current_holdings}
    changed_current = {c["cusip"] for c in changes if c["change_type"] != "CLOSED"}
    changed_closed  = {c["cusip"] for c in changes if c["change_type"] == "CLOSED"}

    # Every CLOSED cusip must have been in the prior filing
    assert changed_closed <= all_cusips_prior, \
        "BUG: CLOSED position not in prior holdings"
    print("  [check] All CLOSED cusips were in the prior filing ✓")

    # Every NEW cusip must be in the current filing and not in the prior
    new_cusips = {c["cusip"] for c in changes if c["change_type"] == "NEW"}
    assert new_cusips <= all_cusips_current, \
        "BUG: NEW position not in current holdings"
    print("  [check] All NEW cusips are in the current filing ✓")

    print()
    print(f"  Verification complete — {total} changes, all checks passed.")

    # Cleanup temp file
    db_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
