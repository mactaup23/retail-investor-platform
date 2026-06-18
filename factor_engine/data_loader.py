"""
Fetches and caches historical adjusted-close price data via yfinance.

Prices are saved as CSVs under data/ so repeat runs don't re-hit the network.
Call `load_prices` with a list of tickers and a date range; it returns a
DataFrame indexed by date with one column per ticker.
"""

import os
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _cache_path(ticker: str, start: str, end: str) -> str:
    return os.path.join(DATA_DIR, f"{ticker}_{start}_{end}.csv")


def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Return a DataFrame of daily adjusted-close prices.

    Parameters
    ----------
    tickers : list of str
        e.g. ["AAPL", "SPY"]
    start : str
        ISO date, e.g. "2018-01-01"
    end : str
        ISO date, e.g. "2023-12-31"

    Returns
    -------
    pd.DataFrame
        Index = DatetimeIndex, columns = tickers, values = adjusted close prices.
        Rows with any NaN are dropped.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = {}

    for ticker in tickers:
        cache = _cache_path(ticker, start, end)
        if os.path.exists(cache):
            series = pd.read_csv(cache, index_col=0, parse_dates=True).squeeze()
        else:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty:
                raise ValueError(f"No data returned for {ticker!r} ({start} to {end})")
            series = raw["Close"].squeeze()
            series.to_frame(ticker).to_csv(cache)

        series.name = ticker
        frames[ticker] = series

    prices = pd.DataFrame(frames).dropna()
    prices.index.name = "date"
    return prices


def load_returns(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Daily log returns, computed from adjusted-close prices."""
    prices = load_prices(tickers, start, end)
    import numpy as np
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns
