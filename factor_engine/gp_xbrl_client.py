"""
Raw SEC EDGAR XBRL companyfacts fetch + cache layer for the GP factor.

This module owns exactly one thing: getting a company's full XBRL fact
history onto disk, cheaply on every run after the first. It knows nothing
about revenue/COGS tag selection, TTM windows, or gross-profitability math —
that derivation logic lives in gp_fundamentals.py and reads the cache this
module writes.

Why a separate raw-JSON cache layer instead of caching straight to the
derived (period_end, revenue, cogs, total_assets, gp_ratio) shape
gp_fundamentals.py used under yfinance
-----------------------------------------------------------------------
The XBRL tag-fallback and duration-filtering logic in gp_fundamentals.py is
exactly the kind of thing that needs iteration once real edge cases turn up
across ~1500 companies (tag switches, non-calendar fiscal years, YTD-only
filers). Caching the raw companyfacts JSON means fixing that derivation logic
costs zero network calls to re-apply — only the (cheap, local) re-derivation
re-runs. Without this layer, every fix to the tag-fallback list would mean
re-fetching 1500 companies from SEC.

API choice: companyfacts, not frames
-------------------------------------
data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json returns *all* tagged facts
for one company across its full filing history in a single request — this
matches the per-ticker fetch loop directly (1500 tickers -> 1500 requests,
one company's full timeline each). The frames API (one concept/period across
*all* filers) is built for the opposite query shape (cross-sectional) and
would require iterating years x quarters x fallback tags per concept, each
returning a huge cross-section to filter down to our ~1500 CIKs — far more
requests for the same result.

Rate limiting
-------------
Goes through edgar_client.get(), the same throttle smart_money's 13F/NLP
fetches share (0.12s minimum gap, exponential backoff on 429/503) — see that
module's docstring.
"""

import json
import os

from edgar_client import get as _get

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gp", "xbrl_raw")

_BASE_COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts"

# A CIK with this exact marker cached means "fetched, but SEC has no
# companyfacts for it" (e.g. no XBRL ever filed, CIK not a registrant) —
# distinguishes a confirmed-empty result (don't retry every resumed run) from
# "never attempted" (no cache file at all). Mirrors gp_fundamentals.py's
# _EMPTY_SENTINEL pattern for the same reason.
_NOT_FOUND_MARKER = {"_not_found": True}


def _cache_path(cik: str) -> str:
    return os.path.join(_CACHE_DIR, f"{cik}.json")


def fetch_company_facts(cik: str, force: bool = False) -> "dict | None":
    """
    Fetch (or load cached) raw XBRL companyfacts JSON for one CIK.

    Returns the full companyfacts dict (facts live under
    result["facts"]["us-gaap"][tag]["units"]["USD"]), or None if SEC has no
    XBRL data for this CIK (never raises for a single missing company).

    force=True re-fetches over the network even if a cache file exists —
    use when re-running after a prior partial/failed run, not for normal
    re-derivation (which should read the cache, not bypass it).
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache = _cache_path(cik)

    if not force and os.path.exists(cache):
        with open(cache) as f:
            data = json.load(f)
        return None if data.get("_not_found") else data

    url = f"{_BASE_COMPANYFACTS}/CIK{cik}.json"
    try:
        resp = _get(url)
    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 404:
            with open(cache, "w") as f:
                json.dump(_NOT_FOUND_MARKER, f)
            return None
        raise

    data = resp.json()
    with open(cache, "w") as f:
        json.dump(data, f)
    return data


def fetch_universe_company_facts(
    ciks: dict[str, str],
    force: bool = False,
) -> dict[str, "dict | None"]:
    """
    Fetch companyfacts for a ticker -> CIK mapping, resumable.

    ciks: dict[ticker -> CIK] (tickers with no resolvable CIK should already
    be filtered out by the caller — see gp_fundamentals.py).

    Returns dict[ticker -> companyfacts dict, or None if unavailable].
    Prints progress the same way gp_fundamentals.py's yfinance-era fetcher
    did (per-ticker line + rolling ETA once fetched >= 5), since a full
    ~1500-company run is long enough that silent progress looks stalled.
    """
    import time

    os.makedirs(_CACHE_DIR, exist_ok=True)
    results: dict[str, "dict | None"] = {}
    n_fetched = 0
    n_cached = 0
    n_not_found = 0
    start_time = time.time()
    items = list(ciks.items())
    total = len(items)

    for i, (ticker, cik) in enumerate(items):
        already_cached = os.path.exists(_cache_path(cik)) and not force

        if already_cached:
            n_cached += 1
            results[ticker] = fetch_company_facts(cik, force=force)
            continue

        remaining = total - i - 1
        elapsed = time.time() - start_time
        eta_str = ""
        if n_fetched >= 5:
            rate = elapsed / n_fetched
            eta_min = (rate * remaining) / 60.0
            eta_str = f", ETA ~{eta_min:.0f} min"
        print(f"  [gp_xbrl_client] fetching {ticker:<8s} CIK {cik} "
              f"({i + 1}/{total}, {remaining} remaining{eta_str})...", flush=True)

        data = fetch_company_facts(cik, force=force)
        results[ticker] = data
        n_fetched += 1
        if data is None:
            n_not_found += 1
            print(f"  [gp_xbrl_client]   -> {ticker}: no XBRL companyfacts", flush=True)

        if n_fetched % 50 == 0:
            elapsed_min = elapsed / 60.0
            print(f"  [gp_xbrl_client] === checkpoint: {i + 1}/{total} processed, "
                  f"{n_fetched} freshly fetched ({n_not_found} not found), "
                  f"{elapsed_min:.1f} min elapsed ===", flush=True)

    total_elapsed_min = (time.time() - start_time) / 60.0
    print(f"  [gp_xbrl_client] Done in {total_elapsed_min:.1f} min. "
          f"{n_fetched} companies freshly fetched ({n_not_found} with no XBRL data), "
          f"{n_cached} loaded from cache.")
    return results
