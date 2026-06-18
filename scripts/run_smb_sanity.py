"""
Sanity checks for the SMB factor and 2-factor loadings.

Run from the project root with the venv active:
    python scripts/run_smb_sanity.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from factor_engine.factors.market import build_market_factor
from factor_engine.factors.smb import build_smb_factor, compute_smb_loading

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "GLD"]
START = "2020-01-01"
END   = "2024-12-31"


def main():
    print(f"Building factors ({START} to {END})...\n")

    market_factor = build_market_factor(START, END)
    smb_factor    = build_smb_factor(START, END)

    # ── SMB factor descriptives ────────────────────────────────────────────
    smb = smb_factor["smb"]
    mkt = market_factor["market_excess"]

    corr_smb_mkt = smb.corr(mkt)
    mean_daily_smb = smb.mean()
    std_daily_smb  = smb.std()
    mean_daily_mkt = mkt.mean()
    std_daily_mkt  = mkt.std()
    cumulative_smb = smb.sum()

    print("=== SMB Factor Descriptives ===")
    print(f"  Trading days loaded : {len(smb_factor)}")
    print(f"  Mean daily SMB      : {mean_daily_smb*100:+.4f}%  "
          f"(annualised ≈ {mean_daily_smb*252*100:+.2f}%)")
    print(f"  Std dev daily SMB   : {std_daily_smb*100:.4f}%  "
          f"(annualised ≈ {std_daily_smb*(252**0.5)*100:.2f}%)")
    print(f"  Cumulative SMB      : {cumulative_smb*100:+.2f}%  (log-return sum)")
    print(f"  Mean daily Mkt-RF   : {mean_daily_mkt*100:+.4f}%")
    print(f"  Std dev daily Mkt-RF: {std_daily_mkt*100:.4f}%")
    print(f"\n  Correlation(SMB, Mkt-RF): {corr_smb_mkt:+.4f}")
    print(f"  [Target: |r| < 0.20 — SMB should be largely orthogonal to market]\n")

    # ── 2-factor loadings ──────────────────────────────────────────────────
    results = []
    for ticker in TICKERS:
        print(f"  Computing 2-factor loading for {ticker}...")
        r = compute_smb_loading(ticker, START, END,
                                market_factor=market_factor,
                                smb_factor=smb_factor)
        results.append(r)

    df = (
        pd.DataFrame(results)
        .set_index("ticker")
        [["beta_market", "beta_smb", "t_stat_smb", "p_value_smb",
          "alpha_annualised", "r_squared", "n_obs"]]
    )

    print("\n=== 2-Factor (Mkt-RF + SMB) Loadings ===")
    print(df.to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\nInterpretation:")
    print("  beta_market > 1   → more volatile than the market")
    print("  beta_smb < 0      → large-cap tilt  (stock moves like 'big' stocks)")
    print("  beta_smb > 0      → small-cap tilt  (stock moves like 'small' stocks)")
    print("\nKey checks:")

    for r in results:
        t = r["ticker"]
        b = r["beta_smb"]
        p = r["p_value_smb"]
        sig = "**significant**" if p < 0.05 else "not significant"
        direction = "negative ✓" if b < 0 else "positive"
        if t in ("AAPL", "MSFT"):
            print(f"  {t}: β_smb = {b:+.4f}  ({direction})  p={p:.4f}  [{sig}]"
                  f"  ← large-cap; expect negative")
        else:
            print(f"  {t}: β_smb = {b:+.4f}  p={p:.4f}  [{sig}]")


if __name__ == "__main__":
    main()
