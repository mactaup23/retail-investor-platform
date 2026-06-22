"""
Verification: fetch real price history for Viking Global's top-5 holdings.

Resolves CUSIPs from Viking's latest 13F (already in Security table from the
CUSIP verification pass), fetches prices via yfinance, stores in PriceCache,
and prints a summary of date range, trading-day count, and sample rows.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_money.models import init_db
from smart_money.prices import fetch_prices

# Top-5 Viking holdings resolved by verify_openfigi.py
# Visa (V), TSMC (TSM), Charles Schwab (SCHW), Disney (DIS), Fortive (FTV)
TICKERS = ["V", "TSM", "SCHW", "DIS", "FTV"]

START = "2023-01-01"
END   = "2025-03-31"   # covers last two full 13F filing periods


def main() -> None:
    init_db()

    print(f"Fetching price history for Viking top-5 tickers: {TICKERS}")
    print(f"Date range: {START} → {END}\n")

    prices = fetch_prices(TICKERS, START, END, skip_cached=False)

    print()
    print(f"{'Ticker':<8} {'Start':<12} {'End':<12} {'Days':>6}  {'First Adj Close':>16}  {'Last Adj Close':>15}")
    print("─" * 75)

    for ticker in TICKERS:
        if ticker not in prices:
            print(f"{ticker:<8}  — no data returned")
            continue
        df = prices[ticker]
        n          = len(df)
        first_date = df.index[0]
        last_date  = df.index[-1]
        first_adj  = df["adj_close"].iloc[0]
        last_adj   = df["adj_close"].iloc[-1]
        print(
            f"{ticker:<8} {str(first_date):<12} {str(last_date):<12} {n:>6}  "
            f"{first_adj:>16.4f}  {last_adj:>15.4f}"
        )

    print()
    print("Sample rows (first 3 trading days) for each ticker:")
    for ticker in TICKERS:
        if ticker not in prices:
            continue
        df = prices[ticker]
        print(f"\n  {ticker}")
        print(f"  {'Date':<12} {'Close':>10} {'Adj Close':>12}")
        print(f"  {'─'*12} {'─'*10} {'─'*12}")
        for date, row in df.head(3).iterrows():
            print(f"  {str(date):<12} {row['close']:>10.4f} {row['adj_close']:>12.4f}")

    resolved = sum(1 for t in TICKERS if t in prices)
    print(f"\n  {resolved}/{len(TICKERS)} tickers fetched and cached successfully.")


if __name__ == "__main__":
    main()
