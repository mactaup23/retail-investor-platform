"""
MOM (Momentum) factor — Fama-French-Carhart style daily return series, ETF proxy.

Construction
------------
Momentum proxy : MTUM (iShares MSCI USA Momentum Factor ETF)
Benchmark      : IWB  (iShares Russell 1000) — the same large-cap-blend leg
                 used as SMB's "big" side in factor_engine/factors/smb.py
Daily MOM      : log_ret(MTUM) − log_ret(IWB)

Academic momentum (Carhart's UMD, "up minus down") ranks stocks on trailing
return from month t-12 to t-2 (skipping the most recent month to avoid
short-term reversal), goes long the winner decile and short the loser decile,
and reforms the portfolio monthly.

ETF proxy vs. pure Carhart UMD — important structural caveat
--------------------------------------------------------------
Unlike the SMB and HML proxies (which pair a genuine long side against a
genuine short side via Russell size/style ETFs), there is no liquid
"low-momentum" or "loser" ETF to short here. MTUM is long-only, so this
factor is constructed as MTUM's return in excess of a broad large-cap
benchmark rather than a true long-short winners-minus-losers spread.  MTUM
also reconstitutes semi-annually with turnover-dampening buffer rules (vs.
UMD's monthly reformation) and ranks stocks on a risk-adjusted momentum
score rather than raw 12-1 return.

For these reasons, expect empirical correlation between this ETF-based
series and Ken French's published daily momentum factor to run meaningfully
lower than the SMB (~0.85–0.90) and HML (~0.80–0.88) proxies. Measured via
scripts/run_mom_sanity.py over 2020-01-01–2024-12-31: corr = +0.71 — at the
upper end of what a long-only proxy can achieve against a long-short
academic factor.  This is appropriate for a retail investor platform
illustrating factor exposure on individual holdings, but should be flagged
whenever comparing loadings to academic benchmarks.

MTUM inception is April 2013, which bounds how far back this proxy can be
computed — unlike the official Ken French momentum series (back to the
1920s) used for fund skill scoring and stress testing in
factor_engine/french_data.py::get_ff4_daily().

Factor loading regression (Fama-French-Carhart 4-factor model)
------------------------------------------------------------------
compute_factor_loadings() in factor_engine/factors/hml.py fits a joint
4-factor OLS:

    r_i − r_f = α + β_mkt·(Mkt-RF) + β_smb·SMB + β_hml·HML + β_mom·MOM + ε

All four betas are estimated in a single regression — running mom as a
separate or paired regression would omit a correlated regressor and bias
every other estimate, the same argument already made for SMB and HML.
"""

import pandas as pd

from factor_engine.data_loader import load_returns

MOMENTUM_ETF = "MTUM"  # iShares MSCI USA Momentum Factor ETF
BENCHMARK_ETF = "IWB"  # iShares Russell 1000 (large-cap blend)


def build_mom_factor(start: str, end: str) -> pd.DataFrame:
    """
    Construct the daily MOM factor series.

    Returns a DataFrame with columns:
        momentum_return  — MTUM daily log return
        benchmark_return — IWB daily log return
        mom              — momentum_return − benchmark_return  (the MOM factor)
    """
    tickers = [MOMENTUM_ETF, BENCHMARK_ETF]
    returns = load_returns(tickers, start, end)
    return pd.DataFrame({
        "momentum_return":  returns[MOMENTUM_ETF],
        "benchmark_return": returns[BENCHMARK_ETF],
        "mom":              returns[MOMENTUM_ETF] - returns[BENCHMARK_ETF],
    }).dropna()
