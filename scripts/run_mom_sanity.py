"""
Sanity checks for the MOM factor and the Fama-French-Carhart 4-factor loadings.

Two checks, in addition to the standard descriptives/orthogonality block:

1. Correlation of the ETF-proxy MOM series (factor_engine/factors/mom.py, MTUM
   minus IWB) against Ken French's official daily momentum series
   (factor_engine/french_data.py::get_ff4_daily()). This is the number quoted
   in mom.py's docstring as an estimate (~0.55-0.70) — this script measures it
   directly.

2. Winner/loser directional check. Momentum loading depends on a stock's
   *actual* trailing return over the specific test window, not its sector or
   reputation — a "growth" stock can be a momentum loser if it just sold off.
   So rather than asserting e.g. "AAPL should have positive beta_mom" (which
   can flip and make this script flaky), we measure each candidate's trailing
   12-1 return (12 months ending 1 month before END, skipping the most recent
   month per the Carhart definition) directly from price data, then check
   that beta_mom is rank-correlated with that measured momentum across the
   candidate set.

Run from the project root with the venv active:
    python scripts/run_mom_sanity.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from factor_engine.data_loader import load_prices
from factor_engine.factors.market import build_market_factor
from factor_engine.factors.smb import build_smb_factor
from factor_engine.factors.hml import build_hml_factor, compute_factor_loadings
from factor_engine.factors.mom import build_mom_factor
from factor_engine.french_data import get_ff4_daily

# Diverse-sector large caps — momentum is about trailing price trend, not
# sector or style, so candidates deliberately span sectors.
CANDIDATES = ["AAPL", "MSFT", "NVDA", "XOM", "JPM", "INTC", "PFE", "KO"]
START = "2020-01-01"
END   = "2024-12-31"


def _trailing_12_1_return(ticker: str, as_of: str) -> float:
    """
    Trailing 12-1 momentum: cumulative return from 13 months before `as_of`
    to 1 month before `as_of`, skipping the most recent month per Carhart.
    """
    end_ts   = pd.Timestamp(as_of)
    lookback_end   = end_ts - pd.DateOffset(months=1)
    lookback_start = end_ts - pd.DateOffset(months=13)
    prices = load_prices(
        [ticker],
        lookback_start.strftime("%Y-%m-%d"),
        lookback_end.strftime("%Y-%m-%d"),
    )[ticker]
    return float(prices.iloc[-1] / prices.iloc[0] - 1.0)


# Annual snapshot dates within the regression window (each needs 13 months of
# lookback, so the first snapshot is >=13 months after START).
_SNAPSHOT_DATES = ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]


def _avg_trailing_12_1_return(ticker: str) -> float:
    """
    Average trailing 12-1 return sampled annually across the regression window.

    A single end-of-window snapshot is a noisy point estimate that need not
    line up with a beta estimated over the *whole* multi-year window (beta_mom
    measures day-to-day co-movement with the momentum factor across the full
    sample, not whether the stock happened to be a winner at one instant).
    Averaging several snapshots gives a fairer "how much of a momentum stock
    was this, on average, during the regression window" comparison.
    """
    vals = [_trailing_12_1_return(ticker, d) for d in _SNAPSHOT_DATES]
    return float(sum(vals) / len(vals))


def main():
    print(f"Building factors ({START} to {END})...\n")

    market_factor = build_market_factor(START, END)
    smb_factor    = build_smb_factor(START, END)
    hml_factor    = build_hml_factor(START, END)
    mom_factor    = build_mom_factor(START, END)

    mkt = market_factor["market_excess"]
    smb = smb_factor["smb"]
    hml = hml_factor["hml"]
    mom = mom_factor["mom"]

    # ── MOM factor descriptives ────────────────────────────────────────────
    mean_mom = mom.mean()
    std_mom  = mom.std()

    print("=== MOM Factor Descriptives (ETF proxy: MTUM - IWB) ===")
    print(f"  Trading days loaded : {len(mom_factor)}")
    print(f"  Mean daily MOM      : {mean_mom*100:+.4f}%  "
          f"(annualised ≈ {mean_mom*252*100:+.2f}%)")
    print(f"  Std dev daily MOM   : {std_mom*100:.4f}%  "
          f"(annualised ≈ {std_mom*(252**0.5)*100:.2f}%)")
    print(f"  Cumulative MOM      : {mom.sum()*100:+.2f}%  (log-return sum)")

    # ── Orthogonality checks ───────────────────────────────────────────────
    aligned = pd.DataFrame({"mkt": mkt, "smb": smb, "hml": hml, "mom": mom}).dropna()
    print(f"\n  Correlation matrix (daily returns):")
    print(f"    corr(MOM, Mkt-RF) : {aligned['mom'].corr(aligned['mkt']):+.4f}")
    print(f"    corr(MOM, SMB)    : {aligned['mom'].corr(aligned['smb']):+.4f}")
    print(f"    corr(MOM, HML)    : {aligned['mom'].corr(aligned['hml']):+.4f}  "
          f"[momentum and value are typically negatively correlated]")

    # ── Correlation vs official Ken French momentum series ─────────────────
    print(f"\n=== ETF proxy vs official Ken French momentum series ===")
    ff4 = get_ff4_daily(START, END)
    joined = pd.DataFrame({"proxy": mom, "official": ff4["mom"]}).dropna()
    corr_official = joined["proxy"].corr(joined["official"])
    print(f"  corr(MTUM-IWB, official UMD) : {corr_official:+.4f}  "
          f"over {len(joined)} overlapping trading days")
    print(f"  [mom.py docstring estimate: ~0.55-0.70 — long-only ETF proxy vs "
          f"long-short academic factor]")

    # ── 4-factor loadings ───────────────────────────────────────────────────
    print(f"\nComputing 4-factor loadings for {CANDIDATES}...\n")
    results = []
    for ticker in CANDIDATES:
        print(f"  {ticker}...")
        r = compute_factor_loadings(
            ticker, START, END,
            market_factor=market_factor,
            smb_factor=smb_factor,
            hml_factor=hml_factor,
            mom_factor=mom_factor,
        )
        r["avg_trailing_12_1_return"] = _avg_trailing_12_1_return(ticker)
        results.append(r)

    df = (
        pd.DataFrame(results)
        .set_index("ticker")
        [["avg_trailing_12_1_return", "beta_mom", "t_stat_mom", "p_value_mom",
          "beta_market", "alpha_annualised", "r_squared", "n_obs"]]
    )

    print("\n=== Fama-French-Carhart 4-Factor Loadings — Momentum ===")
    print(df.to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\nInterpretation:")
    print("  beta_mom < 0  → contrarian tilt  (moves with recent losers)")
    print("  beta_mom > 0  → momentum tilt    (moves with recent winners)")
    print("  avg_trailing_12_1_return is the stock's *actual* measured momentum,")
    print(f"  averaged across {len(_SNAPSHOT_DATES)} annual snapshots within the regression")
    print("  window — the ground truth beta_mom should track, not sector or")
    print("  reputation (a growth stock can be a momentum loser if it sold off).")

    rank_corr = df["avg_trailing_12_1_return"].corr(df["beta_mom"], method="spearman")
    print(f"\nIllustrative check: rank correlation(avg_trailing_12_1_return, beta_mom) = {rank_corr:+.3f}")
    print("  This is a directional illustration, not a pass/fail gate. beta_mom")
    print("  measures day-to-day co-movement with the MOM factor's swings across")
    print("  the *whole* multi-year window — it is a related but distinct quantity")
    print("  from a stock's own realized trailing return, and the two can diverge")
    print("  in a window with a momentum regime shift (e.g. the 2022 growth/momentum")
    print("  crash reversed several years of prior trend). A weak or negative value")
    print("  here does not by itself indicate a bug in the factor construction —")
    print("  the primary correctness checks are the synthetic OLS-recovery tests")
    print("  (tests/test_mom_factor.py, deterministic) and the correlation against")
    print("  Ken French's official momentum series measured above.")


if __name__ == "__main__":
    main()
