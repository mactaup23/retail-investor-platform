"""
Sanity checks for the HML factor and Fama-French 3-factor loadings.

Run from the project root with the venv active:
    python scripts/run_hml_sanity.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from factor_engine.factors.market import build_market_factor
from factor_engine.factors.smb import build_smb_factor
from factor_engine.factors.hml import build_hml_factor, compute_factor_loadings

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "GLD"]
START = "2020-01-01"
END   = "2024-12-31"


def main():
    print(f"Building factors ({START} to {END})...\n")

    market_factor = build_market_factor(START, END)
    smb_factor    = build_smb_factor(START, END)
    hml_factor    = build_hml_factor(START, END)

    mkt = market_factor["market_excess"]
    smb = smb_factor["smb"]
    hml = hml_factor["hml"]

    # ── HML factor descriptives ────────────────────────────────────────────
    mean_hml = hml.mean()
    std_hml  = hml.std()

    print("=== HML Factor Descriptives ===")
    print(f"  Trading days loaded : {len(hml_factor)}")
    print(f"  Mean daily HML      : {mean_hml*100:+.4f}%  "
          f"(annualised ≈ {mean_hml*252*100:+.2f}%)")
    print(f"  Std dev daily HML   : {std_hml*100:.4f}%  "
          f"(annualised ≈ {std_hml*(252**0.5)*100:.2f}%)")
    print(f"  Cumulative HML      : {hml.sum()*100:+.2f}%  (log-return sum)")

    # ── Orthogonality checks ───────────────────────────────────────────────
    # Align all three to the same trading days before computing correlations
    aligned = pd.DataFrame({"mkt": mkt, "smb": smb, "hml": hml}).dropna()
    corr_hml_mkt = aligned["hml"].corr(aligned["mkt"])
    corr_hml_smb = aligned["hml"].corr(aligned["smb"])
    corr_smb_mkt = aligned["smb"].corr(aligned["mkt"])

    print(f"\n  Correlation matrix (daily returns):")
    print(f"    corr(HML, Mkt-RF) : {corr_hml_mkt:+.4f}  [target: |r| < 0.20]")
    print(f"    corr(HML, SMB)    : {corr_hml_smb:+.4f}  [target: |r| < 0.20 — 4-ETF averaging reduces size tilt]")
    print(f"    corr(SMB, Mkt-RF) : {corr_smb_mkt:+.4f}")

    # ── 3-factor loadings ──────────────────────────────────────────────────
    print(f"\nComputing 3-factor loadings for {TICKERS}...\n")
    results = []
    for ticker in TICKERS:
        print(f"  {ticker}...")
        r = compute_factor_loadings(
            ticker, START, END,
            market_factor=market_factor,
            smb_factor=smb_factor,
            hml_factor=hml_factor,
        )
        results.append(r)

    df = (
        pd.DataFrame(results)
        .set_index("ticker")
        [["beta_market", "beta_smb", "beta_hml",
          "t_stat_hml", "p_value_hml",
          "alpha_annualised", "r_squared", "n_obs"]]
    )

    print("\n=== Fama-French 3-Factor Loadings ===")
    print(df.to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\nInterpretation:")
    print("  beta_hml < 0  → growth tilt  (low B/M, stock moves like 'growth')")
    print("  beta_hml > 0  → value tilt   (high B/M, stock moves like 'value')")

    print("\nKey checks (HML loading):")
    expectations = {
        "AAPL": ("negative", "large-cap growth; expect β_hml < 0"),
        "MSFT": ("negative", "large-cap growth; expect β_hml < 0"),
        "JPM":  ("positive", "classic value/bank stock; expect β_hml > 0"),
        "XOM":  ("positive", "energy/value tilt; expect β_hml > 0"),
        "GLD":  (None,       "gold — no B/M signal; expect β_hml ≈ 0"),
    }
    for r in results:
        t   = r["ticker"]
        b   = r["beta_hml"]
        p   = r["p_value_hml"]
        sig = "significant ✓" if p < 0.05 else "not significant"
        exp_dir, note = expectations[t]

        if exp_dir == "negative":
            direction = "negative ✓" if b < 0 else "positive ✗ (unexpected)"
        elif exp_dir == "positive":
            direction = "positive ✓" if b > 0 else "negative ✗ (unexpected)"
        else:
            direction = f"{b:+.4f} (near-zero expected)"

        print(f"  {t:4s}: β_hml = {b:+.4f}  ({direction})  p={p:.4f}  [{sig}]  ← {note}")


if __name__ == "__main__":
    main()
