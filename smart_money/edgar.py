"""
EDGAR client for Module 3 — 13F Smart-Money Positioning & Skill Tracker.

Public interface
----------------
    list_13f_filings(cik)               → list[FilingMeta]
    canonical_filings(filings)          → list[FilingMeta]  (dedup by period)
    fetch_holdings(cik, accession)      → list[HoldingRow]
    fetch_latest_holdings(cik)          → tuple[FilingMeta, list[HoldingRow]]

All functions return plain dicts; no Peewee imports here.

Rate limiting
-------------
SEC policy: ≤ 10 req/sec, User-Agent required.  This module enforces a
0.12 s minimum gap between outbound requests and backs off exponentially
on 429 / 503 (up to 3 retries, max 32 s sleep).

User-Agent
----------
Set EDGAR_USER_AGENT env var to override the default.
"""

import os
import time
import xml.etree.ElementTree as ET
from typing import TypedDict

import requests

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class FilingMeta(TypedDict):
    cik: str
    accession_number: str       # "0001166928-25-000012"
    form_type: str              # "13F-HR" or "13F-HR/A"
    period_of_report: str       # "2024-12-31"
    filed_date: str             # "2025-02-14"


class HoldingRow(TypedDict):
    cusip: str
    issuer_name: str
    title_of_class: str
    value_usd: int              # raw dollars (empirically verified: Viking/Visa 1912630634 → $302.24/sh ✓).
                                 # A minority of filers report in thousands per the literal SEC spec instead —
                                 # see _VALUE_UNIT_SCALE below. Already corrected to raw dollars by the time
                                 # this row is built; no further scaling needed downstream.
    shares: int
    shares_type: str            # "SH" or "PRN"
    investment_discretion: str  # "Sole" | "Shared" | "Other"
    other_manager: str | None
    put_call: str | None        # "Put" | "Call" | None
    voting_sole: int
    voting_shared: int
    voting_none: int
    figi: str | None            # present in recent filings; None otherwise


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_USER_AGENT = os.getenv(
    "EDGAR_USER_AGENT",
    "RetailInvestorPlatform mac.taupier@gmail.com",
)

_BASE_SUBMISSIONS = "https://data.sec.gov/submissions"
_BASE_ARCHIVES    = "https://www.sec.gov/Archives/edgar/data"

_last_request_ts: float = 0.0
_MIN_GAP = 0.12          # seconds between requests (< 10/sec SEC limit)
_MAX_RETRIES = 3


def _get(url: str, **kwargs) -> requests.Response:
    """Throttled GET with exponential backoff on 429/503."""
    global _last_request_ts

    elapsed = time.monotonic() - _last_request_ts
    if elapsed < _MIN_GAP:
        time.sleep(_MIN_GAP - elapsed)

    headers = {"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    headers.update(kwargs.pop("headers", {}))

    delay = 1.0
    for attempt in range(_MAX_RETRIES + 1):
        _last_request_ts = time.monotonic()
        resp = requests.get(url, headers=headers, timeout=30, **kwargs)
        if resp.status_code in (429, 503) and attempt < _MAX_RETRIES:
            time.sleep(delay)
            delay = min(delay * 2, 32)
            continue
        resp.raise_for_status()
        return resp

    resp.raise_for_status()  # will raise after exhausting retries
    return resp              # unreachable; satisfies type checker


# ---------------------------------------------------------------------------
# CIK normalisation
# ---------------------------------------------------------------------------

def _pad_cik(cik: str | int) -> str:
    """Zero-pad CIK to 10 digits as required by the submissions endpoint."""
    return str(int(cik)).zfill(10)


# ---------------------------------------------------------------------------
# Per-filer value-unit overrides
# ---------------------------------------------------------------------------
# The SEC 13F spec says the infoTable "value" field is in thousands of
# dollars. Empirically, the vast majority of filers in this universe ignore
# that and report raw dollars instead (verified: Viking/Visa 1912630634 →
# $302.24/sh). A minority of filers DO follow the literal thousands spec.
# Each entry below was verified by cross-checking a sampled infoTable row's
# (value / shares) against a real contemporaneous market price — re-run
# scripts/verify_value_units.py against any new fund before assuming it
# belongs in (or is absent from) this table.
#
# Keyed by zero-padded CIK. Any CIK not listed here defaults to scale=1
# (raw dollars) — i.e. today's behavior for the existing 38-fund universe
# is completely unchanged.
_VALUE_UNIT_SCALE: dict[str, int] = {
    "0001536411": 1000,   # Duquesne Family Office LLC — verified via AMZN
                          # (value=9539, 45,800 sh → $208/sh only as thousands;
                          # $0.21/sh as raw dollars is impossible)
    "0001897612": 1000,   # T. Rowe Price Investment Management, Inc. —
                          # verified via AGCO Corp (value=59944, 517,336 sh →
                          # $115.90/sh only as thousands; $0.12/sh as raw
                          # dollars is impossible)
}


def _value_scale(cik: str | int) -> int:
    """Return the value-field multiplier for this CIK (1 = raw dollars, the default)."""
    return _VALUE_UNIT_SCALE.get(_pad_cik(cik), 1)


# ---------------------------------------------------------------------------
# Submissions — filing metadata
# ---------------------------------------------------------------------------

def list_13f_filings(cik: str | int) -> list[FilingMeta]:
    """
    Return all 13F-HR and 13F-HR/A filings for a CIK, oldest-first.

    Follows the `filings.files` pagination list so the full history is
    returned, not just the most recent ~40 quarters.
    """
    padded = _pad_cik(cik)
    url = f"{_BASE_SUBMISSIONS}/CIK{padded}.json"
    data = _get(url).json()

    results: list[FilingMeta] = []
    _collect_from_block(cik, data["filings"]["recent"], results)

    # Paginated historical blocks (older filings)
    for file_ref in data["filings"].get("files", []):
        url2 = f"{_BASE_SUBMISSIONS}/{file_ref['name']}"
        block = _get(url2).json()
        _collect_from_block(cik, block, results)

    results.sort(key=lambda r: (r["period_of_report"], r["filed_date"]))
    return results


def _collect_from_block(
    cik: str | int,
    block: dict,
    out: list[FilingMeta],
) -> None:
    """Extract 13F-HR / 13F-HR/A rows from a submissions filings block."""
    forms   = block.get("form", [])
    accnos  = block.get("accessionNumber", [])
    periods = block.get("reportDate", [])
    dates   = block.get("filingDate", [])

    for form, accno, period, filed in zip(forms, accnos, periods, dates):
        if form in ("13F-HR", "13F-HR/A"):
            out.append(
                FilingMeta(
                    cik=str(cik),
                    accession_number=accno,
                    form_type=form,
                    period_of_report=period,
                    filed_date=filed,
                )
            )


def canonical_filings(filings: list[FilingMeta]) -> list[FilingMeta]:
    """
    Deduplicate filings by period_of_report, keeping the latest filing
    (amendments supersede originals).

    Input must be sorted oldest-first (list_13f_filings guarantees this).
    Returns a list sorted oldest-first.
    """
    by_period: dict[str, FilingMeta] = {}
    for f in filings:
        by_period[f["period_of_report"]] = f  # later entry wins
    return sorted(by_period.values(), key=lambda r: r["period_of_report"])


# ---------------------------------------------------------------------------
# Filing document discovery
# ---------------------------------------------------------------------------

def _accession_nodash(accession_number: str) -> str:
    return accession_number.replace("-", "")


def _find_infotable_url(cik: str | int, accession_number: str) -> str:
    """
    Parse the filing index HTML and return the URL of the information table
    XML document.

    The index HTML is always at:
        {BASE}/{cik}/{nodash}/{accession}-index.html

    EDGAR naming is inconsistent across filers and eras:
        - Modern: information_table.xml
        - Older / third-party filers: arbitrary names (e.g. MSFS13F033126.XML)
    We search for the "INFORMATION TABLE" type declaration in the index HTML.

    Two structural quirks handled here:
        1. href values may be absolute paths (/Archives/...) or bare filenames.
        2. EDGAR serves an XSLT-rendered copy under xslForm13F_X02/; we exclude
           that directory because the transformed XML is not valid for parsing.
    """
    import re

    cik_int = int(cik)
    nodash  = _accession_nodash(accession_number)
    index_url = (
        f"{_BASE_ARCHIVES}/{cik_int}/{nodash}/{accession_number}-index.html"
    )
    html = _get(index_url).text

    # Find all XML hrefs, excluding XSLT-rendered copies and primary_doc
    all_xml = re.findall(
        r'href="(/Archives/[^"]+\.(?:xml|XML))"',
        html,
        re.IGNORECASE,
    )
    plain_xml = [
        h for h in all_xml
        if "xslForm" not in h and "primary_doc" not in h.lower()
    ]

    # Pass 1: prefer the file referenced alongside "INFORMATION TABLE" type text
    m = re.search(
        r"INFORMATION TABLE.*?href=\"(/Archives/[^\"]+\.(?:xml|XML))\"",
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if m and "xslForm" not in m.group(1):
        return f"https://www.sec.gov{m.group(1)}"

    # Pass 2: any remaining plain XML that is not the primary cover-page doc
    if plain_xml:
        return f"https://www.sec.gov{plain_xml[0]}"

    raise ValueError(
        f"No information table XML found in filing index "
        f"CIK={cik} accession={accession_number}.  "
        f"Check {index_url}"
    )


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

# EDGAR uses two namespace URIs across filings (differ only in case of last segment).
# We detect the actual URI from the root element at parse time.
_NS_VARIANTS = (
    "http://www.sec.gov/edgar/document/thirteenf/informationtable",  # most filings
    "http://www.sec.gov/edgar/document/thirteenf/informationTable",  # some older filings
)


def _text(el: ET.Element | None, default: str = "") -> str:
    if el is None or el.text is None:
        return default
    return el.text.strip()


def _int(el: ET.Element | None, default: int = 0) -> int:
    t = _text(el)
    return int(t) if t else default


def _detect_ns(root: ET.Element) -> str:
    """
    Extract the namespace URI from the root element tag.

    Handles both default-namespace declarations (xmlns="...") and
    prefixed declarations (xmlns:ns1="...").  The root tag will be
    either '{uri}informationTable' or 'ns1:informationTable' depending
    on the filer; ET normalises prefixed declarations to the Clark
    notation '{uri}localname' so we can always split on '}'.
    """
    tag = root.tag
    if tag.startswith("{"):
        return tag[1:].split("}")[0]
    # Fallback: scan known variants
    for variant in _NS_VARIANTS:
        if root.find(f"{{{variant}}}infoTable") is not None:
            return variant
    return _NS_VARIANTS[0]


def _parse_infotable_xml(xml_bytes: bytes, value_scale: int = 1) -> list[HoldingRow]:
    """
    Parse a 13F information table XML document into holding dicts.

    value_scale corrects filers that report the "value" field in thousands
    per the literal SEC spec instead of raw dollars (see _VALUE_UNIT_SCALE).
    Default of 1 preserves today's raw-dollar behavior for every filer not
    in that table.
    """
    root = ET.fromstring(xml_bytes)
    ns_uri = _detect_ns(root)
    ns = {"ns": ns_uri}
    rows: list[HoldingRow] = []

    for info in root.findall("ns:infoTable", ns):

        def g(tag: str, _info: ET.Element = info) -> ET.Element | None:
            return _info.find(f"ns:{tag}", ns)

        shrs_el = g("shrsOrPrnAmt")
        vote_el = g("votingAuthority")

        row = HoldingRow(
            cusip                 = _text(g("cusip")),
            issuer_name           = _text(g("nameOfIssuer")),
            title_of_class        = _text(g("titleOfClass")),
            value_usd             = _int(g("value")) * value_scale,
            shares                = _int(shrs_el.find("ns:sshPrnamt", ns) if shrs_el is not None else None),
            shares_type           = _text(shrs_el.find("ns:sshPrnamtType", ns) if shrs_el is not None else None, "SH"),
            investment_discretion = _text(g("investmentDiscretion"), "Sole"),
            other_manager         = _text(g("otherManager")) or None,
            put_call              = _text(g("putCall")) or None,
            voting_sole           = _int(vote_el.find("ns:Sole", ns) if vote_el is not None else None),
            voting_shared         = _int(vote_el.find("ns:Shared", ns) if vote_el is not None else None),
            voting_none           = _int(vote_el.find("ns:None", ns) if vote_el is not None else None),
            figi                  = _text(g("figi")) or None,
        )
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# High-level fetch
# ---------------------------------------------------------------------------

def fetch_holdings(cik: str | int, accession_number: str) -> list[HoldingRow]:
    """
    Fetch and parse the information table for a single filing.

    Returns holdings in the order they appear in the XML (not sorted).
    Applies this CIK's value-unit scale (see _VALUE_UNIT_SCALE) so value_usd
    is always in raw dollars regardless of what the filer's software reported.
    """
    infotable_url = _find_infotable_url(cik, accession_number)
    xml_bytes = _get(infotable_url).content
    return _parse_infotable_xml(xml_bytes, value_scale=_value_scale(cik))


def fetch_latest_holdings(cik: str | int) -> tuple[FilingMeta, list[HoldingRow]]:
    """
    Convenience: fetch the canonical most-recent 13F-HR holdings for a CIK.

    Returns (filing_meta, holdings).  Raises ValueError if no 13F-HR filings
    are found for this CIK.
    """
    filings = list_13f_filings(cik)
    if not filings:
        raise ValueError(f"No 13F-HR filings found for CIK {cik}")
    latest = canonical_filings(filings)[-1]
    holdings = fetch_holdings(cik, latest["accession_number"])
    return latest, holdings
