"""
Module 4 — EDGAR filing language-shift scorer.

Fetches consecutive 10-Q or 10-K MD&A sections for portfolio companies and
scores directional language shift across 7 dimensions using the Claude Batch
API (claude-sonnet-4-6).  Results are persisted in NLPCache so repeated runs
are free.

Public interface
----------------
    score_ticker(ticker, *, db_path)           → NLPCache | None
    batch_score_tickers(tickers, *, db_path)   → list[NLPCache]
    load_scores(tickers, *, db_path)           → dict[str, NLPCache]

Caching
-------
Cache key: (ticker, accession_current, accession_prior, scorer_version).
Bump SCORER_VERSION to force a rescore without touching old rows.

Filing fallback
---------------
1. Two most recent 10-Qs filed within 6 months  → 10-Q shift
2. Two most recent 10-Ks filed within 18 months → 10-K shift
3. Neither available                             → skip (return None)

Rate limiting
-------------
All EDGAR requests go through edgar._get(), which enforces the SEC's 10 req/sec
ceiling.  ANTHROPIC_API_KEY must be set in the environment.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import anthropic
from bs4 import BeautifulSoup

from edgar_client import ticker_to_cik

from .edgar import _BASE_ARCHIVES, _BASE_SUBMISSIONS, _get
from .models import DB_PATH, NLPCache, init_db

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCORER_VERSION = "v1"
MODEL = "claude-sonnet-4-6"
MDA_CHAR_LIMIT = 8_000   # chars per MD&A section sent to Claude (~2 k tokens)
MAX_POLL_SECONDS = 3_600  # 1 h ceiling on batch polling
POLL_INTERVAL = 30        # seconds between batch status checks

DIMENSION_WEIGHTS: dict[str, float] = {
    "guidance":                0.25,
    "confidence":              0.20,
    "customer_demand":         0.20,
    "competitive_positioning": 0.15,
    "operational_efficiency":  0.10,
    "risk_factors":            0.05,
    "capital_allocation":      0.05,
}

# ---------------------------------------------------------------------------
# Ticker → CIK lookup
#
# ticker_to_cik is imported from edgar_client (see module docstring there) —
# it's shared with factor_engine's XBRL fetches, both keyed off the same SEC
# company_tickers.json bulk file.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Filing discovery
# ---------------------------------------------------------------------------

class _FilingPair:
    __slots__ = (
        "cik", "current_accession", "prior_accession",
        "form_type", "current_period", "prior_period",
    )

    def __init__(
        self,
        cik: str,
        cur_acc: str,
        prior_acc: str,
        form_type: str,
        cur_period: str,
        prior_period: str,
    ) -> None:
        self.cik              = cik
        self.current_accession = cur_acc
        self.prior_accession   = prior_acc
        self.form_type         = form_type
        self.current_period    = cur_period
        self.prior_period      = prior_period


def _list_company_filings(cik: str, form_type: str) -> list[dict]:
    """
    Return filings of *form_type* for *cik*, newest-first.

    Pulls only the filings.recent block (covers ~3–5 years), which is
    sufficient for the two-filing lookback we need.
    """
    url = f"{_BASE_SUBMISSIONS}/CIK{cik}.json"
    data = _get(url).json()
    recent = data["filings"]["recent"]

    rows = []
    for form, acc, filed, period in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("filingDate", []),
        recent.get("reportDate", []),
    ):
        if form == form_type:
            rows.append({"accession_number": acc, "filed_date": filed, "period_of_report": period})

    # EDGAR recent block is newest-first; preserve that ordering.
    return rows


def _select_filing_pair(cik: str) -> _FilingPair | None:
    """
    Return the best (current, prior) filing pair for shift scoring, or None.

    Priority order:
      1. 10-Q — if the most recent was filed within 6 months
      2. 10-K — if the most recent was filed within 18 months
      3. None  — skip this company
    """
    cutoff_6mo  = (datetime.utcnow() - timedelta(days=182)).strftime("%Y-%m-%d")
    cutoff_18mo = (datetime.utcnow() - timedelta(days=548)).strftime("%Y-%m-%d")

    for form_type, cutoff in [("10-Q", cutoff_6mo), ("10-K", cutoff_18mo)]:
        filings = _list_company_filings(cik, form_type)
        if len(filings) >= 2 and filings[0]["filed_date"] >= cutoff:
            cur, prior = filings[0], filings[1]
            return _FilingPair(
                cik=cik,
                cur_acc=cur["accession_number"],
                prior_acc=prior["accession_number"],
                form_type=form_type,
                cur_period=cur["period_of_report"],
                prior_period=prior["period_of_report"],
            )
    return None


# ---------------------------------------------------------------------------
# MD&A extraction
# ---------------------------------------------------------------------------

# (start_item, end_item_a, end_item_b) — end search stops at the first match
_MDA_BOUNDARIES: dict[str, tuple[str, str, str]] = {
    "10-Q": ("2",  "3",   "4"),
    "10-K": ("7",  "7A",  "8"),
}


def _extract_mda(text: str, form_type: str) -> str:
    """
    Slice the MD&A section out of filing plain text.

    Uses a two-pass search: the first occurrence of 'Item N' is often the
    table-of-contents entry; we prefer the second occurrence if it appears
    within 40 000 chars of the first (typical ToC-to-body gap).
    """
    start_item, end_a, end_b = _MDA_BOUNDARIES.get(form_type, ("2", "3", "4"))

    start_pat = re.compile(
        rf"\bITEM\s+{re.escape(start_item)}\b[.\s–—\-]",
        re.IGNORECASE,
    )
    end_pat = re.compile(
        rf"\bITEM\s+{re.escape(end_a)}\b[.\s–—\-]"
        rf"|\bITEM\s+{re.escape(end_b)}\b[.\s–—\-]",
        re.IGNORECASE,
    )

    m1 = start_pat.search(text)
    if not m1:
        return ""

    # Prefer the second match if it's plausibly in the body (not the ToC).
    m2 = start_pat.search(text, m1.end())
    if m2 and (m2.start() - m1.start()) < 40_000:
        body_start = m2.start()
    else:
        body_start = m1.start()

    m_end = end_pat.search(text, body_start + 200)
    body_end = m_end.start() if m_end else body_start + 60_000

    return text[body_start:body_end].strip()


def _unwrap_ix(href: str) -> str:
    """Strip EDGAR's iXBRL viewer prefix (/ix?doc=...) to get the raw file path."""
    if href.startswith("/ix?doc="):
        return href[len("/ix?doc="):]
    return href


def _find_primary_htm(cik: str, accession_number: str, form_type: str) -> str | None:
    """
    Parse the filing index page and return the URL of the primary .htm document.

    Looks for the table row whose Type cell matches *form_type* exactly.
    Falls back to the first non-index .htm link if the structured lookup fails.
    """
    cik_int = int(cik)
    nodash  = accession_number.replace("-", "")
    index_url = f"{_BASE_ARCHIVES}/{cik_int}/{nodash}/{accession_number}-index.htm"

    try:
        html = _get(index_url).text
    except Exception as exc:
        log.warning("index fetch failed %s/%s: %s", cik, accession_number, exc)
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Structured pass: find the row whose Type cell equals form_type
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue
        for cell in cells:
            if cell.get_text(strip=True) == form_type:
                link = row.find("a", href=True)
                if link:
                    href: str = _unwrap_ix(link["href"])
                    if href.lower().endswith((".htm", ".html")):
                        base = "" if href.startswith("http") else "https://www.sec.gov"
                        return f"{base}{href}"
                break

    # Fallback: first .htm link that is not an index, XSLT view, or iXBRL viewer
    for a in soup.find_all("a", href=True):
        href = _unwrap_ix(a["href"])
        if (
            href.lower().endswith((".htm", ".html"))
            and "index" not in href.lower()
            and "xsl" not in href.lower()
        ):
            base = "" if href.startswith("http") else "https://www.sec.gov"
            return f"{base}{href}"

    return None


def _fetch_mda(cik: str, accession_number: str, form_type: str) -> str:
    """
    Download a filing's primary HTML document and return the MD&A plain text.

    Returns an empty string if the document or section cannot be found.
    """
    primary_url = _find_primary_htm(cik, accession_number, form_type)
    if not primary_url:
        log.warning("No primary HTM found: CIK=%s acc=%s", cik, accession_number)
        return ""

    try:
        html = _get(primary_url).text
    except Exception as exc:
        log.warning("Failed to fetch primary doc %s: %s", primary_url, exc)
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    mda  = _extract_mda(text, form_type)

    if not mda:
        log.warning("MD&A section not found: CIK=%s acc=%s form=%s", cik, accession_number, form_type)

    return mda


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def _composite(scores: dict[str, float]) -> float:
    return sum(scores[d] * w for d, w in DIMENSION_WEIGHTS.items())


# ---------------------------------------------------------------------------
# Claude Batch API helpers
# ---------------------------------------------------------------------------

_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "confidence":              {"type": "number"},
        "guidance":                {"type": "number"},
        "risk_factors":            {"type": "number"},
        "capital_allocation":      {"type": "number"},
        "competitive_positioning": {"type": "number"},
        "customer_demand":         {"type": "number"},
        "operational_efficiency":  {"type": "number"},
        "reasoning":               {"type": "string"},
    },
    "required": [
        "confidence", "guidance", "risk_factors", "capital_allocation",
        "competitive_positioning", "customer_demand", "operational_efficiency",
        "reasoning",
    ],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are a financial language analyst specialising in SEC filing MD&A sections. "
    "Identify meaningful directional shifts in management tone between consecutive filings. "
    "Be conservative: score only shifts clearly supported by specific textual evidence. "
    "Score 0.0 when there is no meaningful change. "
    "Reserve ±1.0 for unambiguous, large shifts."
)


def _build_batch_request(
    custom_id: str,
    prior_mda: str,
    current_mda: str,
    prior_period: str,
    current_period: str,
    form_type: str,
) -> dict:
    user_msg = (
        f"Compare the management discussion language in these two consecutive {form_type} "
        f"filings and score the directional language shift for each dimension.\n\n"
        f"Scale: −1.0 (strongly more negative / cautious in current period) "
        f"to +1.0 (strongly more positive / confident). 0.0 = no detectable shift.\n\n"
        f"<prior_filing period=\"{prior_period}\">\n"
        f"{prior_mda[:MDA_CHAR_LIMIT]}\n"
        f"</prior_filing>\n\n"
        f"<current_filing period=\"{current_period}\">\n"
        f"{current_mda[:MDA_CHAR_LIMIT]}\n"
        f"</current_filing>\n\n"
        "Dimension definitions:\n"
        "• confidence              — overall management certainty and conviction\n"
        "• guidance                — forward-looking performance optimism and clarity\n"
        "• risk_factors            — +1 fewer/softer risks cited; −1 more/harder risks\n"
        "• capital_allocation      — +1 more aggressive/confident deployment signals\n"
        "• competitive_positioning — market share, pricing power, market-leadership language\n"
        "• customer_demand         — demand conditions, pipeline, customer-behaviour signals\n"
        "• operational_efficiency  — cost discipline, margin levers, operational-rigour signals"
    )

    return {
        "custom_id": custom_id,
        "params": {
            "model": MODEL,
            "max_tokens": 1_024,
            "system": _SYSTEM,
            "messages": [{"role": "user", "content": user_msg}],
            "output_config": {
                "format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}
            },
        },
    }


def _parse_score_block(content_blocks: list) -> dict | None:
    """Extract the JSON scores dict from a list of content blocks."""
    for block in content_blocks:
        if getattr(block, "type", None) == "text":
            try:
                return json.loads(block.text)
            except json.JSONDecodeError as exc:
                log.warning("JSON decode error in score block: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def batch_score_tickers(
    tickers: list[str],
    *,
    db_path: Path | None = None,
) -> list[NLPCache]:
    """
    Score language shift for *tickers* using the Claude Batch API.

    Returns NLPCache rows for every ticker that was successfully scored,
    whether from cache or freshly computed.  Tickers with no valid filing
    pair or whose MD&A cannot be extracted are silently skipped.
    """
    init_db(db_path)
    client = anthropic.Anthropic()

    results: list[NLPCache] = []
    batch_requests: list[dict] = []
    meta: dict[str, dict] = {}   # custom_id → {ticker, cik, pair}

    for ticker in tickers:
        cik = ticker_to_cik(ticker)
        if not cik:
            log.warning("No CIK found for ticker %s — skipping", ticker)
            continue

        pair = _select_filing_pair(cik)
        if not pair:
            log.info("No valid filing pair for %s (CIK %s) — skipping", ticker, cik)
            continue

        # Cache hit?
        try:
            row = NLPCache.get(
                (NLPCache.ticker == ticker)
                & (NLPCache.accession_current == pair.current_accession)
                & (NLPCache.accession_prior  == pair.prior_accession)
                & (NLPCache.scorer_version   == SCORER_VERSION)
            )
            log.info("Cache hit for %s", ticker)
            results.append(row)
            continue
        except NLPCache.DoesNotExist:
            pass

        # Fetch MD&A sections
        log.info("Fetching MD&A for %s (CIK %s, %s)", ticker, cik, pair.form_type)
        current_mda = _fetch_mda(cik, pair.current_accession, pair.form_type)
        prior_mda   = _fetch_mda(cik, pair.prior_accession,   pair.form_type)

        if not current_mda or not prior_mda:
            log.warning("MD&A extraction failed for %s — skipping", ticker)
            continue

        custom_id = f"nlp-{ticker}-{SCORER_VERSION}"
        batch_requests.append(
            _build_batch_request(
                custom_id=custom_id,
                prior_mda=prior_mda,
                current_mda=current_mda,
                prior_period=pair.prior_period,
                current_period=pair.current_period,
                form_type=pair.form_type,
            )
        )
        meta[custom_id] = {"ticker": ticker, "cik": cik, "pair": pair}

    if not batch_requests:
        log.info("All %d tickers served from cache or skipped", len(tickers))
        return results

    # Submit batch
    log.info("Submitting Batch API job: %d requests", len(batch_requests))
    batch = client.messages.batches.create(requests=batch_requests)
    log.info("Batch ID: %s", batch.id)

    # Poll until complete
    start = time.monotonic()
    while batch.processing_status == "in_progress":
        elapsed = time.monotonic() - start
        if elapsed > MAX_POLL_SECONDS:
            raise TimeoutError(
                f"Batch {batch.id} still in_progress after {MAX_POLL_SECONDS}s"
            )
        log.info(
            "Batch %s — %s (elapsed %.0fs, next poll in %ds)",
            batch.id, batch.processing_status, elapsed, POLL_INTERVAL,
        )
        time.sleep(POLL_INTERVAL)
        batch = client.messages.batches.retrieve(batch.id)

    if batch.processing_status != "ended":
        raise RuntimeError(
            f"Batch {batch.id} finished with unexpected status: {batch.processing_status}"
        )

    log.info("Batch %s ended — parsing results", batch.id)

    # Parse results and write to cache
    for item in client.messages.batches.results(batch.id):
        if item.result.type != "succeeded":
            log.warning(
                "Request %s — %s", item.custom_id, item.result.type
            )
            continue

        m = meta.get(item.custom_id)
        if not m:
            log.warning("Unknown custom_id in batch results: %s", item.custom_id)
            continue

        scores = _parse_score_block(item.result.message.content)
        if not scores:
            log.warning("Could not parse scores for %s", m["ticker"])
            continue

        pair = m["pair"]
        dim_scores = {d: float(scores.get(d, 0.0)) for d in DIMENSION_WEIGHTS}

        row = NLPCache.create(
            ticker                        = m["ticker"],
            cik_company                   = m["cik"],
            accession_current             = pair.current_accession,
            accession_prior               = pair.prior_accession,
            form_type                     = pair.form_type,
            scorer_version                = SCORER_VERSION,
            confidence_delta              = dim_scores["confidence"],
            guidance_delta                = dim_scores["guidance"],
            risk_factors_delta            = dim_scores["risk_factors"],
            capital_allocation_delta      = dim_scores["capital_allocation"],
            competitive_positioning_delta = dim_scores["competitive_positioning"],
            customer_demand_delta         = dim_scores["customer_demand"],
            operational_efficiency_delta  = dim_scores["operational_efficiency"],
            composite_score               = _composite(dim_scores),
            reasoning                     = scores.get("reasoning", ""),
            scored_at                     = datetime.utcnow(),
        )
        results.append(row)
        log.info(
            "Scored %s: composite=%.3f  guidance=%.2f  confidence=%.2f  "
            "demand=%.2f  competitive=%.2f  efficiency=%.2f  "
            "risk=%.2f  capalloc=%.2f",
            m["ticker"],
            row.composite_score,
            row.guidance_delta,
            row.confidence_delta,
            row.customer_demand_delta,
            row.competitive_positioning_delta,
            row.operational_efficiency_delta,
            row.risk_factors_delta,
            row.capital_allocation_delta,
        )

    return results


def score_ticker(ticker: str, *, db_path: Path | None = None) -> NLPCache | None:
    """Score a single ticker. Convenience wrapper around batch_score_tickers."""
    rows = batch_score_tickers([ticker], db_path=db_path)
    return rows[0] if rows else None


def load_scores(
    tickers: list[str],
    *,
    db_path: Path | None = None,
) -> dict[str, NLPCache]:
    """
    Return cached NLP scores keyed by ticker (most recent SCORER_VERSION only).

    Used by convergence.py to attach nlp_shift enrichments without re-running
    the batch scorer.
    """
    init_db(db_path)
    out: dict[str, NLPCache] = {}
    for ticker in tickers:
        try:
            row = (
                NLPCache.select()
                .where(
                    (NLPCache.ticker == ticker)
                    & (NLPCache.scorer_version == SCORER_VERSION)
                )
                .order_by(NLPCache.scored_at.desc())
                .get()
            )
            out[ticker] = row
        except NLPCache.DoesNotExist:
            pass
    return out
