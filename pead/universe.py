"""
Stock universe for the PEAD signal.

Deliberately reuses factor_engine.gp_universe rather than re-scraping —
the brief for this signal is explicitly to run over "the same ~1,448-1,500
stock universe already used by the GP factor and convergence signal", and
gp_universe.py already solves the S&P Composite 1500 sourcing problem
(see that module's docstring for why Wikipedia's index tables are used
instead of iShares holdings CSVs).
"""

from factor_engine.gp_universe import get_universe, get_universe_tickers

__all__ = ["get_universe", "get_universe_tickers"]
