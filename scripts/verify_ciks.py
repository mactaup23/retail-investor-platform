"""
Module 3 — CIK Verification Pass (v3)

Correctly uses the EDGAR company-name search HTML endpoint to find/verify CIKs.
Handles both single-result (direct redirect) and multi-result (table) responses.
"""

import re
import sys
import time
import json
from difflib import SequenceMatcher
from pathlib import Path

import requests
import yaml

YAML_PATH = Path(__file__).parent.parent / "config" / "fund_universe.yaml"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_COMPANY_SEARCH = "https://www.sec.gov/cgi-bin/browse-edgar"

HEADERS_JSON = {"User-Agent": "retail-investor-platform/1.0 mac.taupier@gmail.com", "Accept": "application/json"}
HEADERS_HTML = {"User-Agent": "retail-investor-platform/1.0 mac.taupier@gmail.com", "Accept": "text/html"}
RATE_DELAY = 0.40  # EDGAR limit: 10 req/s


# ─── EDGAR helpers ───────────────────────────────────────────────────────────

def fetch_submissions(cik: str) -> dict | None:
    padded = str(cik).zfill(10)
    try:
        r = requests.get(SUBMISSIONS_URL.format(cik=padded), headers=HEADERS_JSON, timeout=15)
        time.sleep(RATE_DELAY)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        print(f"    [warn] submissions({cik}): {e}", file=sys.stderr)
        return None


def has_13f_filings(subs: dict) -> bool:
    forms = subs.get("filings", {}).get("recent", {}).get("form", [])
    if any("13F" in f for f in forms):
        return True
    # Check additional filing pages
    for f in subs.get("filings", {}).get("files", []):
        if "13F" in f.get("name", ""):
            return True
    return False


def company_name_search(query: str) -> list[dict]:
    """
    Search EDGAR company database for 13F filers by name.
    Handles both EDGAR response shapes:
      - Single result → redirects to company page (companyName span + CIK in link)
      - Multi result  → table of CIK + company name rows
    """
    params = {
        "company": query, "CIK": "", "type": "13F-HR",
        "dateb": "", "owner": "include", "count": "40", "action": "getcompany",
    }
    try:
        r = requests.get(EDGAR_COMPANY_SEARCH, params=params, headers=HEADERS_HTML, timeout=20)
        time.sleep(RATE_DELAY)
        if r.status_code != 200:
            return []
        html = r.text

        results = []
        seen = set()

        # Shape 1: single-result redirect — companyName span + CIK "see all" link
        cn = re.search(r'class="companyName">\s*(.+?)\s*<acronym', html)
        cik_see_all = re.search(r'CIK=(\d+)[^"]*">\s*\d+\s*\(see all', html)
        if cn and cik_see_all:
            cik = str(int(cik_see_all.group(1)))
            ename = cn.group(1).strip()
            if cik not in seen:
                seen.add(cik)
                results.append({"cik": cik, "edgar_name": ename})

        # Shape 2: multi-result table — CIK as link text, name in adjacent td
        rows = re.findall(
            r'CIK=(\d+)[^"]*"[^>]*>\s*\d+\s*</a></td>\s*<td[^>]*>([^<]+)</td>',
            html
        )
        for cik_raw, ename in rows:
            cik = str(int(cik_raw))
            if cik not in seen:
                seen.add(cik)
                results.append({"cik": cik, "edgar_name": ename.strip()})

        return results
    except Exception as e:
        print(f"    [warn] company_search({query!r}): {e}", file=sys.stderr)
        return []


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def best_match(query: str, candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    q = query.lower()
    def score(c):
        name = c["edgar_name"].lower()
        s = SequenceMatcher(None, q, name).ratio()
        word_hits = sum(1 for w in q.split() if w in name)
        return s * 0.5 + (word_hits / max(len(q.split()), 1)) * 0.5
    ranked = sorted(candidates, key=score, reverse=True)
    best = dict(ranked[0])
    best["match_score"] = score(best)
    return best


# ─── Per-fund logic ───────────────────────────────────────────────────────────

def verify_cik(cik: str, fund_name: str) -> dict:
    subs = fetch_submissions(cik)
    if subs is None:
        return {"result": "api_error", "cik": cik}

    edgar_name = subs.get("name", "")
    name_sim = sim(fund_name, edgar_name)
    files_13f = has_13f_filings(subs)

    if name_sim < 0.25:
        return {"result": "wrong_cik", "cik": cik, "edgar_name": edgar_name,
                "detail": f"CIK belongs to '{edgar_name}' (sim={name_sim:.2f})"}
    if files_13f:
        return {"result": "confirmed", "cik": cik, "edgar_name": edgar_name}
    return {"result": "no_13f_recent", "cik": cik, "edgar_name": edgar_name,
            "detail": "matched entity but no recent 13F-HR"}


def discover_cik(fund_name: str, hints: list[str] | None = None) -> dict:
    words = fund_name.split()
    queries = [fund_name]
    if hints:
        queries.extend(hints)
    if len(words) >= 3:
        queries.append(" ".join(words[:3]))
    if len(words) >= 2:
        queries.append(" ".join(words[:2]))

    for query in queries:
        print(f"    searching '{query}' ...")
        candidates = company_name_search(query)
        if candidates:
            best = best_match(fund_name, candidates)
            if best and best.get("match_score", 0) >= 0.20:
                cik = best["cik"]
                subs = fetch_submissions(cik)
                if subs is None:
                    return {"result": "found_unverified", "cik": cik,
                            "edgar_name": best["edgar_name"]}
                edgar_name = subs.get("name", best["edgar_name"])
                files_13f = has_13f_filings(subs)
                if files_13f:
                    return {"result": "confirmed", "cik": cik,
                            "edgar_name": edgar_name,
                            "detail": f"score={best['match_score']:.2f}",
                            "all_candidates": [c["edgar_name"] for c in candidates[:5]]}
                else:
                    return {"result": "found_no_13f", "cik": cik,
                            "edgar_name": edgar_name,
                            "all_candidates": [c["edgar_name"] for c in candidates[:5]]}
            elif candidates:
                print(f"    low-confidence: {[c['edgar_name'] for c in candidates[:3]]}")
        time.sleep(0.1)

    return {"result": "not_found", "cik": None,
            "detail": "no matching EDGAR filer found"}


# ─── Search hints: alternate name spellings for tricky funds ─────────────────

SEARCH_HINTS: dict[str, list[str]] = {
    "D1 Capital Partners":          ["D1 Capital"],
    "Glenview Capital Management":  ["Glenview Capital"],
    "Whale Rock Capital Management":["Whale Rock"],
    "Baupost Group":                ["Baupost"],
    "ValueAct Capital Management":  ["ValueAct Capital", "Value Act Capital"],
    "Starboard Value":              ["Starboard Value LP"],
    "Horizon Kinetics":             ["Horizon Kinetics LLC"],
    "Ariel Investments":            ["Ariel Investments LLC", "Ariel Capital Management"],
    "Southeastern Asset Management":["Southeastern Asset"],
    "SPO Advisory Corp":            ["SPO Advisory", "SPO Partners"],
    "D.E. Shaw & Co.":              ["DE Shaw", "D E Shaw"],
    "Acadian Asset Management":     ["Acadian Asset"],
    "PDT Partners":                 ["PDT Partners LLC"],
    "Altimeter Capital Management": ["Altimeter Capital"],
    "Dragoneer Investment Group":   ["Dragoneer Investment", "Dragoneer"],
    "Greenoaks Capital Partners":   ["Greenoaks Capital", "Greenoaks"],
    "Perceptive Advisors":          ["Perceptive Life Sciences", "Perceptive Advisors LLC"],
    "OrbiMed Advisors":             ["OrbiMed", "Orbimed"],
    "RA Capital Management":        ["RA Capital"],
    "Kayne Anderson Capital Advisors": ["Kayne Anderson"],
    "SailingStone Capital Partners":["SailingStone Capital", "Sailing Stone"],
    "Corsair Capital Management":   ["Corsair Capital"],
    "Brahman Capital Corp":         ["Brahman Capital"],
    "Pzena Investment Management":  ["Pzena Investment"],
    "Senator Investment Group":     ["Senator Investment"],
    "Situational Awareness LP":     ["Situational Awareness"],
    "Light Street Capital Management": ["Light Street Capital"],
}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    raw = YAML_PATH.read_text()
    data = yaml.safe_load(raw)
    funds = data["funds"]

    results: dict[str, dict] = {}

    for fund in funds:
        name = fund["name"]
        status = fund.get("cik_status", "")
        cik = fund.get("cik")
        hints = SEARCH_HINTS.get(name)

        if status == "confirmed":
            print(f"  CONFIRMED (skip): {name}")
            results[name] = {"result": "already_confirmed", "cik": str(cik)}
            continue

        if status not in ("verify",):
            results[name] = {"result": "skip", "detail": f"cik_status={status}"}
            continue

        print(f"\n── {name}")

        if cik is not None:
            r = verify_cik(str(cik), name)
            if r["result"] in ("wrong_cik", "api_error"):
                print(f"  existing CIK {cik} is bad ({r.get('edgar_name', r['result'])}) — searching by name ...")
                r = discover_cik(name, hints)
            else:
                print(f"  CIK {cik}: {r['result']} [{r.get('edgar_name', '')}]")
        else:
            r = discover_cik(name, hints)

        results[name] = r
        print(f"  → {r['result']}  CIK={r.get('cik')}  [{r.get('edgar_name', '')}]")

    # ─── Print final summary table ────────────────────────────────────────────
    print("\n" + "═" * 105)
    print("  MODULE 3 — CIK VERIFICATION RESULTS")
    print("═" * 105)

    ICONS = {
        "already_confirmed": "✓",
        "confirmed":         "✓",
        "wrong_cik":         "✗",
        "no_13f_recent":     "~",
        "not_found":         "✗",
        "found_no_13f":      "~",
        "found_unverified":  "~",
        "api_error":         "!",
        "skip":              "-",
    }

    confirmed, not_found, manual = [], [], []

    for name, r in results.items():
        res = r.get("result", "?")
        icon = ICONS.get(res, "?")
        cik = r.get("cik", "—")
        ename = r.get("edgar_name", "")
        detail = r.get("detail", "")
        cik_str = f"CIK {cik}" if cik else "no CIK"
        print(f"  [{icon}] {name:<45} {cik_str:<18} {ename[:35]:<35} {detail[:35]}")

        if res in ("already_confirmed", "confirmed"):
            confirmed.append(name)
        elif res in ("not_found", "wrong_cik"):
            not_found.append(name)
        elif res in ("no_13f_recent", "found_no_13f", "found_unverified", "api_error"):
            manual.append(name)

    print("═" * 105)
    print(f"  Confirmed: {len(confirmed)}/39")
    if not_found:
        print(f"\n  NOT FOUND / WRONG CIK ({len(not_found)}):")
        for n in not_found:
            print(f"    - {n}")
    if manual:
        print(f"\n  NEEDS MANUAL REVIEW ({len(manual)}):")
        for n in manual:
            r = results[n]
            print(f"    - {n}: {r.get('result')} | CIK {r.get('cik')} | {r.get('edgar_name', '')}")

    # ─── Write updated YAML (preserve comments via ruamel, fallback pyyaml) ──
    for fund in funds:
        name = fund["name"]
        r = results.get(name, {})
        res = r.get("result", "")
        if res == "confirmed":
            fund["cik"] = r["cik"]
            fund["cik_status"] = "confirmed"
        elif res == "not_found":
            fund["cik"] = None
            fund["cik_status"] = "not_found"
        elif res == "wrong_cik":
            fund["cik"] = None
            fund["cik_status"] = "cik_wrong"
        elif res in ("found_no_13f", "no_13f_recent"):
            if r.get("cik"):
                fund["cik"] = r["cik"]
            fund["cik_status"] = "no_13f_found"

    try:
        from ruamel.yaml import YAML as RYAML
        ry = RYAML()
        ry.preserve_quotes = True
        raw_doc = ry.load(raw)
        for i, fund in enumerate(raw_doc["funds"]):
            updated = data["funds"][i]
            raw_doc["funds"][i]["cik"] = updated.get("cik")
            raw_doc["funds"][i]["cik_status"] = updated.get("cik_status")
        from io import StringIO
        buf = StringIO()
        ry.dump(raw_doc, buf)
        YAML_PATH.write_text(buf.getvalue())
        print("\n  [ruamel] fund_universe.yaml updated (comments preserved).")
    except ImportError:
        YAML_PATH.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True))
        print("\n  [pyyaml] fund_universe.yaml updated (inline comments lost).")

    out = Path(__file__).parent.parent / "data" / "cik_verification_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"  Results → {out}")


if __name__ == "__main__":
    main()
