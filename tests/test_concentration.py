"""
Sanity checks for factor_engine/concentration.py.

Tests use synthetic weights/returns/betas so no network call is needed.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from factor_engine.concentration import (
    top_n_concentration,
    compute_hhi,
    _average_pairwise_correlation,
    find_correlation_clusters,
    find_correlation_cliques,
    trailing_correlation_matrix,
    stress_period_factor_correlation,
)


# ---------------------------------------------------------------------------
# Top-N concentration
# ---------------------------------------------------------------------------

def test_top_n_concentration_basic():
    weights = {"A": 0.30, "B": 0.20, "C": 0.15, "D": 0.10, "E": 0.10,
               "F": 0.05, "G": 0.04, "H": 0.03, "I": 0.02, "J": 0.01}

    result = top_n_concentration(weights, ns=(3, 5, 10))

    assert result["n_holdings"] == 10
    assert result["top_3"] == pytest.approx(0.65)
    assert result["top_5"] == pytest.approx(0.85)
    assert result["top_10"] == pytest.approx(1.0)


def test_top_n_concentration_caps_at_holding_count():
    weights = {"A": 0.4, "B": 0.35, "C": 0.25}  # only 3 holdings

    result = top_n_concentration(weights, ns=(3, 5, 10))

    assert result["top_5_capped_at"] == 3
    assert result["top_10_capped_at"] == 3
    assert result["top_5"] == pytest.approx(1.0)
    assert result["top_10"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# HHI / effective N
# ---------------------------------------------------------------------------

def test_hhi_equal_weights_gives_full_effective_n():
    weights = {t: 0.20 for t in ["A", "B", "C", "D", "E"]}  # 5 equal-weighted

    result = compute_hhi(weights)

    assert result["hhi"] == pytest.approx(0.20)
    assert result["effective_n"] == pytest.approx(5.0)
    assert result["meaningfully_more_concentrated_than_count"] is False


def test_hhi_concentrated_portfolio_flags_low_effective_n():
    weights = {"A": 0.70, "B": 0.05, "C": 0.05, "D": 0.05, "E": 0.05,
               "F": 0.05, "G": 0.05}  # one dominant position among 7

    result = compute_hhi(weights)

    assert result["effective_n"] < 0.6 * result["n_holdings"]
    assert result["meaningfully_more_concentrated_than_count"] is True


def test_hhi_single_holding_is_maximally_concentrated():
    result = compute_hhi({"A": 1.0})

    assert result["hhi"] == pytest.approx(1.0)
    assert result["effective_n"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Correlation clustering
# ---------------------------------------------------------------------------

def test_average_pairwise_correlation_known_matrix():
    corr = pd.DataFrame(
        [[1.0, 0.5, 0.2], [0.5, 1.0, 0.8], [0.2, 0.8, 1.0]],
        index=["A", "B", "C"], columns=["A", "B", "C"],
    )
    # off-diagonal upper triangle: 0.5, 0.2, 0.8 → mean 0.5
    assert _average_pairwise_correlation(corr) == pytest.approx(0.5)


def test_find_correlation_clusters_groups_above_threshold():
    corr = pd.DataFrame(
        [[1.00, 0.85, 0.10],
         [0.85, 1.00, 0.05],
         [0.10, 0.05, 1.00]],
        index=["NVDA", "QQQM", "SCHD"], columns=["NVDA", "QQQM", "SCHD"],
    )
    weights = {"NVDA": 0.10, "QQQM": 0.15, "SCHD": 0.20}

    clusters = find_correlation_clusters(corr, weights, threshold=0.70)

    assert len(clusters) == 1
    assert clusters[0]["tickers"] == ["NVDA", "QQQM"]
    assert clusters[0]["combined_weight"] == pytest.approx(0.25)
    assert clusters[0]["avg_pairwise_correlation"] == pytest.approx(0.85)


def test_find_correlation_clusters_transitive_grouping():
    # A-B correlated, B-C correlated, A-C not directly >threshold —
    # union-find should still merge all three into one cluster.
    corr = pd.DataFrame(
        [[1.00, 0.75, 0.40],
         [0.75, 1.00, 0.72],
         [0.40, 0.72, 1.00]],
        index=["A", "B", "C"], columns=["A", "B", "C"],
    )
    weights = {"A": 0.1, "B": 0.1, "C": 0.1}

    clusters = find_correlation_clusters(corr, weights, threshold=0.70)

    assert len(clusters) == 1
    assert clusters[0]["tickers"] == ["A", "B", "C"]


def test_find_correlation_clusters_no_pairs_above_threshold():
    corr = pd.DataFrame(
        [[1.0, 0.1], [0.1, 1.0]], index=["A", "B"], columns=["A", "B"],
    )
    clusters = find_correlation_clusters(corr, {"A": 0.5, "B": 0.5}, threshold=0.70)

    assert clusters == []


def test_find_correlation_clusters_sorted_by_combined_weight_desc():
    corr = pd.DataFrame(
        [[1.00, 0.90, 0.05, 0.05],
         [0.90, 1.00, 0.05, 0.05],
         [0.05, 0.05, 1.00, 0.80],
         [0.05, 0.05, 0.80, 1.00]],
        index=["BIG1", "BIG2", "SM1", "SM2"], columns=["BIG1", "BIG2", "SM1", "SM2"],
    )
    weights = {"BIG1": 0.30, "BIG2": 0.30, "SM1": 0.02, "SM2": 0.02}

    clusters = find_correlation_clusters(corr, weights, threshold=0.70)

    assert len(clusters) == 2
    assert clusters[0]["tickers"] == ["BIG1", "BIG2"]  # heavier cluster first
    assert clusters[1]["tickers"] == ["SM1", "SM2"]


# ---------------------------------------------------------------------------
# Correlation cliques (stricter complement to clusters)
# ---------------------------------------------------------------------------

def test_find_correlation_cliques_splits_hub_from_true_mutual_group():
    # HUB is >threshold with everyone (like VTI), but A/B and C/D are NOT
    # cross-correlated with each other — only within their own pair. A plain
    # connected-components cluster would merge all 5 into one; cliques should
    # correctly separate {HUB,A,B} and {HUB,C,D} as the two maximal mutual
    # groups (HUB participates in both since it's directly tied to all four).
    tickers = ["HUB", "A", "B", "C", "D"]
    corr = pd.DataFrame(1.0, index=tickers, columns=tickers)
    for t in tickers:
        corr.loc["HUB", t] = corr.loc[t, "HUB"] = 0.85
    corr.loc["A", "B"] = corr.loc["B", "A"] = 0.90
    corr.loc["C", "D"] = corr.loc["D", "C"] = 0.88
    corr.loc["A", "C"] = corr.loc["C", "A"] = 0.10
    corr.loc["A", "D"] = corr.loc["D", "A"] = 0.10
    corr.loc["B", "C"] = corr.loc["C", "B"] = 0.10
    corr.loc["B", "D"] = corr.loc["D", "B"] = 0.10

    weights = {t: 0.2 for t in tickers}

    clusters = find_correlation_clusters(corr, weights, threshold=0.70)
    cliques = find_correlation_cliques(corr, weights, threshold=0.70)

    # Connected components merges everything via the hub.
    assert len(clusters) == 1
    assert clusters[0]["tickers"] == ["A", "B", "C", "D", "HUB"]

    # Cliques correctly split into the two genuinely-mutual groups.
    clique_sets = [set(c["tickers"]) for c in cliques]
    assert {"HUB", "A", "B"} in clique_sets
    assert {"HUB", "C", "D"} in clique_sets
    assert len(cliques) == 2


def test_find_correlation_cliques_real_portfolio_shape():
    # Mirrors the real sanity-check finding: NVDA-QTUM directly below
    # threshold (bridged only via QQQM) should NOT appear as a 3-way clique.
    tickers = ["NVDA", "QQQM", "QTUM"]
    corr = pd.DataFrame(
        [[1.00, 0.73, 0.52],
         [0.73, 1.00, 0.77],
         [0.52, 0.77, 1.00]],
        index=tickers, columns=tickers,
    )
    weights = {t: 0.1 for t in tickers}

    cliques = find_correlation_cliques(corr, weights, threshold=0.70)

    clique_sets = [set(c["tickers"]) for c in cliques]
    assert {"NVDA", "QQQM", "QTUM"} not in clique_sets
    assert {"NVDA", "QQQM"} in clique_sets
    assert {"QQQM", "QTUM"} in clique_sets


def test_find_correlation_cliques_no_pairs_above_threshold():
    corr = pd.DataFrame(
        [[1.0, 0.1], [0.1, 1.0]], index=["A", "B"], columns=["A", "B"],
    )
    assert find_correlation_cliques(corr, {"A": 0.5, "B": 0.5}, threshold=0.70) == []


# ---------------------------------------------------------------------------
# Trailing correlation window
# ---------------------------------------------------------------------------

def test_trailing_correlation_matrix_uses_only_tail_window():
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(5)

    # First 148 rows: A and B are independent noise.
    # Last 252 rows: A and B are the same series (correlation ~1).
    early_a = rng.normal(0, 1, n - 252)
    early_b = rng.normal(0, 1, n - 252)
    late_shared = rng.normal(0, 1, 252)

    a = np.concatenate([early_a, late_shared])
    b = np.concatenate([early_b, late_shared])
    all_returns = pd.DataFrame({"A": a, "B": b}, index=idx)

    corr = trailing_correlation_matrix(all_returns, window_days=252)

    assert corr.loc["A", "B"] > 0.99  # trailing window only sees the shared segment


# ---------------------------------------------------------------------------
# Stress-period factor-implied correlation
# ---------------------------------------------------------------------------

def _synthetic_factors(n=40, start="2022-01-03"):
    idx = pd.date_range(start, periods=n, freq="B")
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "mkt_excess": rng.normal(0.0, 0.02, n),
        "smb": rng.normal(0.0, 0.005, n),
        "hml": rng.normal(0.0, 0.005, n),
        "rmw": rng.normal(0.0, 0.003, n),
        "cma": rng.normal(0.0, 0.003, n),
        "mom": rng.normal(0.0, 0.004, n),
        "gp": rng.normal(0.0, 0.003, n),
        "rf": 0.00005,
    }, index=idx)


@patch("factor_engine.stress_test.SCENARIOS", {
    "fake_scenario": {"label": "Fake Scenario", "start": "2022-01-01", "end": "2022-03-01",
                       "description": "synthetic test scenario"},
})
@patch("factor_engine.french_data.get_ff7_daily")
def test_stress_period_factor_correlation_two_similar_beta_holdings_cluster(mock_factors):
    factors = _synthetic_factors()
    mock_factors.return_value = factors

    # Two holdings with nearly identical beta vectors should produce a
    # near-deterministic factor-implied correlation close to 1.0.
    per_holding = [
        {"ticker": "HIGH_BETA_1", "beta_market": 1.5, "beta_smb": 0.1, "beta_hml": -0.2,
         "beta_rmw": 0.0, "beta_cma": 0.0, "beta_mom": 0.3, "beta_gp": 0.1, "alpha_daily": 0.0001},
        {"ticker": "HIGH_BETA_2", "beta_market": 1.45, "beta_smb": 0.12, "beta_hml": -0.18,
         "beta_rmw": 0.0, "beta_cma": 0.0, "beta_mom": 0.28, "beta_gp": 0.1, "alpha_daily": 0.0002},
        {"ticker": "DEFENSIVE", "beta_market": 0.1, "beta_smb": -0.3, "beta_hml": 0.4,
         "beta_rmw": 0.2, "beta_cma": 0.1, "beta_mom": -0.1, "beta_gp": -0.05, "alpha_daily": 0.0},
    ]
    weights = {"HIGH_BETA_1": 0.2, "HIGH_BETA_2": 0.2, "DEFENSIVE": 0.6}

    results = stress_period_factor_correlation(per_holding, weights, threshold=0.70)

    assert len(results) == 1
    scenario_result = results[0]
    assert scenario_result["key"] == "fake_scenario"
    assert scenario_result["gp_available"] is True

    cluster_tickers = [set(c["tickers"]) for c in scenario_result["clusters"]]
    assert {"HIGH_BETA_1", "HIGH_BETA_2"} in cluster_tickers


@patch("factor_engine.stress_test.SCENARIOS", {
    "no_gp_scenario": {"label": "Pre-GP Scenario", "start": "2008-09-01", "end": "2008-10-01",
                        "description": "synthetic pre-GP-coverage scenario"},
})
@patch("factor_engine.french_data.get_ff7_daily")
def test_stress_period_factor_correlation_flags_gp_unavailable(mock_factors):
    factors = _synthetic_factors(start="2008-09-01")
    factors["gp"] = np.nan  # simulate pre-2013 GP coverage gap
    mock_factors.return_value = factors

    per_holding = [
        {"ticker": "A", "beta_market": 1.0, "beta_smb": 0.0, "beta_hml": 0.0,
         "beta_rmw": 0.0, "beta_cma": 0.0, "beta_mom": 0.0, "beta_gp": 0.0, "alpha_daily": 0.0},
        {"ticker": "B", "beta_market": 0.8, "beta_smb": 0.0, "beta_hml": 0.0,
         "beta_rmw": 0.0, "beta_cma": 0.0, "beta_mom": 0.0, "beta_gp": 0.0, "alpha_daily": 0.0},
    ]
    weights = {"A": 0.5, "B": 0.5}

    results = stress_period_factor_correlation(per_holding, weights, threshold=0.70)

    assert results[0]["gp_available"] is False
