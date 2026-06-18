"""
Demo script: compute CAPM beta for a small set of stocks.

Run from the project root with the venv active:
    python scripts/run_market_factor.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from factor_engine.factors.market import build_market_factor, compute_beta

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "GLD"]
START = "2020-01-01"
END   = "2024-12-31"


def main():
    print(f"Building market factor ({START} to {END})...")
    market_factor = build_market_factor(START, END)
    print(f"  {len(market_factor)} trading days loaded.\n")

    results = []
    for ticker in TICKERS:
        print(f"  Computing beta for {ticker}...")
        result = compute_beta(ticker, START, END, market_factor=market_factor)
        results.append(result)

    df = pd.DataFrame(results).set_index("ticker")
    print("\n--- CAPM Beta Results ---")
    print(df.to_string())
    print("\nInterpretation:")
    print("  beta > 1  →  more volatile than the market")
    print("  beta < 1  →  less volatile than the market")
    print("  alpha     →  annualised excess return unexplained by market exposure")


if __name__ == "__main__":
    main()
