"""
Portfolio concentration and correlation-overlap analysis.

Four pieces, per CLAUDE.md's Pre-Launch Polish List item "Concentration/
Correlation Analysis":

1. Top-N concentration       — top_n_concentration()
2. HHI (+ effective N)       — compute_hhi()
3. Trailing correlation      — trailing_correlation_matrix() + find_correlation_clusters()
                                + find_correlation_cliques()
4. Stress-period correlation — stress_period_factor_correlation()

Correlation threshold: 0.70 (confirmed) — standard portfolio-risk-literature
bar (≈49% shared variance), balancing false positives at lower thresholds
against missing real clustering (e.g. NVDA/QQQM/QTUM shared semiconductor/AI
exposure) at higher ones.

Clusters vs. cliques (both reported, per explicit decision after a real-
portfolio sanity check surfaced this): find_correlation_clusters() uses
connected components, which is transitive — a broad-market "hub" holding
(e.g. a total-market fund, correlated with several sector/style slices simply
because it partially contains them) can chain otherwise-unrelated groups into
one mega-cluster. On the real 9-holding portfolio this sanity-checked
project uses, that produced an 8-of-9-ticker cluster driven by VTI's hub
correlations, burying a genuine mutual overlap ({SCHD, VTV, XLI}, all three
directly >0.70 with each other) inside it — and it also revealed that
NVDA and QTUM are only 0.52 correlated directly; QQQM is the actual bridge
between them, not a three-way mutual tie. find_correlation_cliques() is the
stricter complement: it requires EVERY pair within a group to individually
exceed threshold, which isolates genuinely mutual overlaps like
{SCHD, VTV, XLI} from hub-and-spoke artifacts. Report both — clusters answer
"how much of the portfolio moves as one block," cliques answer "which
specific positions are mutually redundant with each other."

HHI convention: computed on the standard 0–1 scale (sum of squared normalized
weights), not the antitrust 0–10,000 convention, paired with effective N =
1/HHI — the number of equal-sized positions the current concentration is
equivalent to. This is more directly actionable for a retail reader than a
bare HHI number ("your 9 nominal holdings behave like 5.2 effective
positions").

Trailing correlation (item 3) uses ACTUAL daily returns over the trailing
252 trading days ending at the analysis window's end date — a real, current
sample, not thin, so raw pairwise Pearson correlation is used directly (no
model needed here).

Stress-period correlation (item 4) is deliberately NOT raw pairwise price
correlation computed over the narrow 4–8 week crisis window — that would be
a thin, noisy sample (the same small-sample problem this session's DCF pilot
and 3-fund tier test already ran into), and several current holdings didn't
exist yet in 2008/2020 (QQQM inception Oct 2020, QTUM inception Sept 2018),
so raw price data is simply missing for parts of the portfolio in those
windows. Instead this reuses the already-validated stress_test.py machinery:
each holding's CURRENT per-holding factor betas (from
factor_engine/portfolio.py::run_per_holding_regressions(), estimated over the
live analysis window) are applied to the ACTUAL historical factor returns
during each scenario window (factor_engine/stress_test.py::estimate_daily_returns()) to
reconstruct a factor-implied daily return series per holding — exactly the
same "current betas × historical factor path" model the portfolio-level
stress test already uses, just applied per-holding instead of only at the
portfolio level. This has two benefits: it's far less noisy (each implied
series is a deterministic linear combination of the shared factor path, so
the resulting correlation isolates the SYSTEMATIC co-movement implied by
shared factor exposure, with no idiosyncratic noise term), and it
sidesteps the inception-date gap entirely — every current holding gets an
implied return for every scenario, including QQQM/QTUM in 2008, because the
reconstruction never touches each holding's own raw historical prices.

This means stress-period correlation numbers are NOT directly comparable in
magnitude to the trailing (raw, noisy, idiosyncratic-inclusive) correlation
numbers from item 3 — they answer a related but distinct question ("how much
would CURRENT factor exposures have co-moved under THAT historical factor
regime" vs. "how much did these holdings actually co-move recently"). Present
them side by side with clearly distinct labels, and let cluster MEMBERSHIP
GROWTH (not a numeric correlation delta) carry the "diversification
evaporates in crises" narrative — that comparison is apples-to-apples since
both use the same >0.70 threshold and clustering logic.
"""

import numpy as np
import pandas as pd

_DEFAULT_CORRELATION_THRESHOLD = 0.70
_DEFAULT_TRAILING_WINDOW_DAYS = 252


# ---------------------------------------------------------------------------
# 1. Top-N concentration
# ---------------------------------------------------------------------------

def top_n_concentration(weights: dict[str, float], ns: tuple[int, ...] = (3, 5, 10)) -> dict:
    """
    Percentage of portfolio value held in the top N positions, for each N.

    If N exceeds the number of holdings, it's capped at the holding count
    (e.g. top_10 on a 9-holding portfolio reports the same figure as "all
    holdings" — flagged via the paired top_{n}_capped_at field rather than
    silently returning a misleading top_10 == 100% without context).
    """
    sorted_weights = sorted(weights.values(), reverse=True)
    n_holdings = len(sorted_weights)
    result: dict = {"n_holdings": n_holdings}
    for n in ns:
        capped_n = min(n, n_holdings)
        result[f"top_{n}"] = float(sum(sorted_weights[:capped_n]))
        result[f"top_{n}_capped_at"] = capped_n
    return result


# ---------------------------------------------------------------------------
# 2. HHI and effective N
# ---------------------------------------------------------------------------

def compute_hhi(weights: dict[str, float]) -> dict:
    """
    Herfindahl-Hirschman Index on the 0-1 scale (sum of squared normalized
    weights) plus effective N = 1/HHI, the number of equal-sized positions
    this concentration is equivalent to.

    meaningfully_more_concentrated_than_count flags when effective N falls
    below 60% of the actual holding count — i.e. the portfolio behaves like
    meaningfully fewer independent bets than its position count suggests.
    """
    w = np.array(list(weights.values()), dtype=float)
    n_holdings = len(w)
    hhi = float(np.sum(w ** 2)) if n_holdings > 0 else 0.0
    effective_n = (1.0 / hhi) if hhi > 0 else 0.0
    flag = bool(n_holdings > 0 and effective_n < 0.6 * n_holdings)

    return {
        "hhi": hhi,
        "effective_n": effective_n,
        "n_holdings": n_holdings,
        "meaningfully_more_concentrated_than_count": flag,
    }


# ---------------------------------------------------------------------------
# Shared helpers — correlation matrix summary + clustering
# ---------------------------------------------------------------------------

def _average_pairwise_correlation(corr_matrix: pd.DataFrame) -> float:
    """Mean of the off-diagonal upper-triangle entries. NaN if <2 tickers."""
    n = len(corr_matrix)
    if n < 2:
        return float("nan")
    mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    return float(corr_matrix.values[mask].mean())


def _union_find_clusters(pairs: list[tuple[str, str]], all_tickers: list[str]) -> list[list[str]]:
    """
    Connected components over an above-threshold-correlation edge list, via a
    plain union-find (no networkx dependency in this codebase). Returns only
    clusters with 2+ members — a ticker with no above-threshold pair to
    anything else isn't a "cluster."
    """
    parent = {t: t for t in all_tickers}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        union(a, b)

    groups: dict[str, list[str]] = {}
    for t in all_tickers:
        groups.setdefault(find(t), []).append(t)

    return [g for g in groups.values() if len(g) > 1]


def find_correlation_clusters(
    corr_matrix: pd.DataFrame,
    weights: dict[str, float],
    threshold: float = _DEFAULT_CORRELATION_THRESHOLD,
) -> list[dict]:
    """
    Group tickers into correlation clusters (connected components of the
    >threshold pairwise-correlation graph). Each cluster reports its combined
    portfolio weight and internal average pairwise correlation, sorted by
    combined weight descending so the largest hidden-overlap exposure shows
    first.

    Caveat (confirmed via real-portfolio sanity check, see
    scripts/run_concentration_risk_sanity.py): connected components is
    transitive, so a single broad-market "hub" holding correlated with several
    otherwise-unrelated positions (e.g. a total-market fund correlated with
    both a value-tilted ETF and a growth-tilted ETF, each of which it
    partially contains by construction) can chain everything into one mega-
    cluster, burying a more specific, genuinely mutual overlap elsewhere in
    the portfolio. See find_correlation_cliques() below for the stricter
    complement to this view — report both rather than picking one.
    """
    tickers = list(corr_matrix.columns)
    pairs = [
        (a, b)
        for i, a in enumerate(tickers)
        for b in tickers[i + 1:]
        if corr_matrix.loc[a, b] > threshold
    ]
    clusters = _union_find_clusters(pairs, tickers)

    result = []
    for cluster in clusters:
        sub = corr_matrix.loc[cluster, cluster]
        result.append({
            "tickers": sorted(cluster),
            "combined_weight": float(sum(weights.get(t, 0.0) for t in cluster)),
            "avg_pairwise_correlation": _average_pairwise_correlation(sub),
        })
    result.sort(key=lambda c: c["combined_weight"], reverse=True)
    return result


def _bron_kerbosch(r: set, p: set, x: set, adj: dict[str, set], cliques: list[set]) -> None:
    """Bron-Kerbosch without pivoting — exact maximal-clique enumeration."""
    if not p and not x:
        if len(r) > 1:
            cliques.append(set(r))
        return
    for v in list(p):
        _bron_kerbosch(r | {v}, p & adj[v], x & adj[v], adj, cliques)
        p = p - {v}
        x = x | {v}


def find_correlation_cliques(
    corr_matrix: pd.DataFrame,
    weights: dict[str, float],
    threshold: float = _DEFAULT_CORRELATION_THRESHOLD,
) -> list[dict]:
    """
    Stricter complement to find_correlation_clusters(): a maximal CLIQUE
    requires EVERY pair within the group to individually exceed threshold, not
    just a chain of pairwise links. This is what isolates a genuinely mutual
    overlap (e.g. three positions all directly >0.70 correlated with each
    other) from a hub-and-spoke artifact where one broad holding bridges two
    otherwise loosely-related groups into a single connected component.

    Exact enumeration via Bron-Kerbosch (no pivoting) — worst-case exponential
    in ticker count, but this codebase caps portfolios at
    dashboard/holdings.py::MAX_POSITIONS = 20, well within the practical range
    for exact clique enumeration on a real correlation graph.

    Returns maximal cliques only (size >= 2), sorted by combined weight
    descending, same shape as find_correlation_clusters()'s output.
    """
    tickers = list(corr_matrix.columns)
    adj = {
        t: {o for o in tickers if o != t and corr_matrix.loc[t, o] > threshold}
        for t in tickers
    }
    cliques: list[set] = []
    _bron_kerbosch(set(), set(tickers), set(), adj, cliques)

    result = []
    for clique in cliques:
        clique_list = sorted(clique)
        sub = corr_matrix.loc[clique_list, clique_list]
        result.append({
            "tickers": clique_list,
            "combined_weight": float(sum(weights.get(t, 0.0) for t in clique_list)),
            "avg_pairwise_correlation": _average_pairwise_correlation(sub),
        })
    result.sort(key=lambda c: c["combined_weight"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# 3. Trailing (actual) correlation
# ---------------------------------------------------------------------------

def trailing_correlation_matrix(
    all_returns: pd.DataFrame,
    window_days: int = _DEFAULT_TRAILING_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Pairwise Pearson correlation of actual daily returns over the trailing
    `window_days` (default 252 ≈ 12 months) ending at all_returns' last date.

    all_returns is expected to already span the full analysis window (e.g.
    factor_engine/portfolio.py::analyze_portfolio()'s all_returns) — this
    slices its tail rather than issuing a new fetch.
    """
    return all_returns.tail(window_days).corr()


# ---------------------------------------------------------------------------
# 4. Stress-period factor-implied correlation
# ---------------------------------------------------------------------------

def stress_period_factor_correlation(
    per_holding: list[dict],
    weights: dict[str, float],
    threshold: float = _DEFAULT_CORRELATION_THRESHOLD,
) -> list[dict]:
    """
    For each of stress_test.py's three scenarios (2008/2020/2022), reconstruct
    every holding's factor-implied daily return series (current per-holding
    betas × that scenario's actual historical factor path — see module
    docstring for why this replaces raw pairwise price correlation), then
    compute the resulting correlation matrix, average pairwise correlation,
    and both correlation clusters and cliques (see find_correlation_clusters()
    / find_correlation_cliques()) at the same threshold used for the trailing
    (actual) analysis.

    gp_available flags whether the GP factor had coverage for that scenario's
    full window (structurally impossible for 2008/2020, since GP's coverage
    starts 2013 — see stress_test.py's own module docstring); the gp term is
    simply omitted from the reconstruction when unavailable, not zeroed.
    """
    from factor_engine.french_data import get_ff7_daily
    from factor_engine.stress_test import SCENARIOS, estimate_daily_returns, gp_available as _gp_available_check

    results = []
    for key, scenario in SCENARIOS.items():
        factors = get_ff7_daily(scenario["start"], scenario["end"])
        if factors.empty:
            continue

        implied = {}
        for h in per_holding:
            daily_r, _ = estimate_daily_returns(
                factors,
                h["beta_market"], h["beta_smb"], h["beta_hml"],
                h["beta_rmw"], h["beta_cma"], h["beta_mom"], h["beta_gp"],
                h["alpha_daily"],
            )
            implied[h["ticker"]] = daily_r
        implied_df = pd.DataFrame(implied)

        corr_matrix = implied_df.corr()
        clusters = find_correlation_clusters(corr_matrix, weights, threshold=threshold)
        cliques = find_correlation_cliques(corr_matrix, weights, threshold=threshold)

        results.append({
            "key": key,
            "label": scenario["label"],
            "start": scenario["start"],
            "end": scenario["end"],
            "gp_available": _gp_available_check(factors),
            "avg_pairwise_correlation": _average_pairwise_correlation(corr_matrix),
            "clusters": clusters,
            "cliques": cliques,
            "correlation_matrix": corr_matrix.round(4).to_dict(),
        })
    return results


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_concentration_analysis(
    weights: dict[str, float],
    all_returns: pd.DataFrame,
    per_holding: list[dict],
    correlation_threshold: float = _DEFAULT_CORRELATION_THRESHOLD,
    trailing_window_days: int = _DEFAULT_TRAILING_WINDOW_DAYS,
) -> dict:
    """
    Run the full concentration/correlation analysis.

    Parameters
    ----------
    weights      : normalized ticker -> weight dict (same as analyze_portfolio()'s "weights")
    all_returns  : per-ticker daily return DataFrame spanning the analysis
                   window (analyze_portfolio()'s "all_returns")
    per_holding  : list of per-ticker regression dicts (analyze_portfolio()'s
                   "per_holding") — supplies the betas/alpha for the
                   stress-period reconstruction.
    """
    trailing_corr = trailing_correlation_matrix(all_returns, window_days=trailing_window_days)
    trailing_clusters = find_correlation_clusters(trailing_corr, weights, threshold=correlation_threshold)
    trailing_cliques = find_correlation_cliques(trailing_corr, weights, threshold=correlation_threshold)

    return {
        "top_n": top_n_concentration(weights),
        "hhi": compute_hhi(weights),
        "trailing_window_days": trailing_window_days,
        "correlation_threshold": correlation_threshold,
        "trailing_avg_pairwise_correlation": _average_pairwise_correlation(trailing_corr),
        "trailing_clusters": trailing_clusters,
        "trailing_cliques": trailing_cliques,
        "trailing_correlation_matrix": trailing_corr.round(4).to_dict(),
        "stress_period_correlation": stress_period_factor_correlation(
            per_holding, weights, threshold=correlation_threshold
        ),
    }
