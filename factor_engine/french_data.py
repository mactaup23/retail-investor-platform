"""
Official Fama-French data downloaded directly from Ken French's data library —
the 3-factor file, the separately-published momentum file, and the 5-factor
file (which adds RMW and CMA), merged into progressively richer panels up to
the full 7-factor model (Fama-French 5 + Carhart momentum + this platform's
proprietary GP factor).

We use the published US daily series (Mkt-RF, SMB, HML, RMW, CMA, Mom, RF)
rather than the ETF-proxy series built in factor_engine/factors/.  The
official series uses actual CRSP stock returns and proper B/M breakpoints —
strictly more accurate than the IWM/IWB/IWD/IWF/IWN/IWO/MTUM proxies.

GP (Gross Profitability) has no Ken French analog — it's a proprietary factor
constructed in factor_engine/factors/gp.py from SEC EDGAR XBRL data (~1450
stocks post-exclusions), with 2013-present history — still short of the
Ken French series (full history to 1963), but no longer bounded to
~2021-present as an earlier yfinance-sourced version was.  get_ff7_daily()
left-joins it onto the full ff6 panel so pre-2013 mkt/smb/hml/rmw/cma/mom
history is never truncated — the gp column is simply NaN before GP's actual
coverage starts. Callers that run a single joint regression across the full
panel must dropna consistently, or use get_ff6_daily() (which excludes GP
entirely) if they need the other six factors' full history preserved
without any GP-driven truncation.

VXUS methodology note
---------------------
VXUS tracks the FTSE Global All Cap ex-US index.  Ken French publishes *regional*
daily factor series (Europe, Japan, Asia Pacific ex Japan) but not a combined
"Developed ex-US" daily series, and no regional momentum series at all.
Constructing one would require knowing VXUS's current geographic weights (≈37%
Europe, 27% Pacific, 25% EM, 11% other), which vary over time and would introduce
a time-varying factor matrix — complexity that buys little precision at daily
frequency given that developed markets co-move with the US at r ≈ 0.70–0.85.  We
therefore use the US factor series for VXUS, label it accordingly (see
factor_engine/portfolio.py's _factor_basis_label) everywhere in output, and note
the R² reduction (~0.60–0.75 vs. 0.95+ for domestic ETFs) so the user can
calibrate their confidence.

Cache: data/french/us_ff3_daily.csv + us_mom_daily.csv + us_ff5_daily.csv
(full history from 1926/1963; immutable past data). GP's own cache lives
under data/gp/ — see factor_engine/factors/gp.py.
Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""

import io
import os
import zipfile

import pandas as pd
import requests

_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "french")
_US_FF3_ZIP = "F-F_Research_Data_Factors_daily_CSV.zip"
_US_FF3_CACHE = os.path.join(_CACHE_DIR, "us_ff3_daily.csv")
_US_MOM_ZIP = "F-F_Momentum_Factor_daily_CSV.zip"
_US_MOM_CACHE = os.path.join(_CACHE_DIR, "us_mom_daily.csv")
_US_FF5_ZIP = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
_US_FF5_CACHE = os.path.join(_CACHE_DIR, "us_ff5_daily.csv")

_FF3_COLUMN_MAP = {
    "Mkt-RF": "mkt_excess",
    "MKT-RF": "mkt_excess",
    "SMB":    "smb",
    "HML":    "hml",
    "RF":     "rf",
}
_MOM_COLUMN_MAP = {
    "Mom":   "mom",
    "MOM":   "mom",
    "WML":   "mom",  # some vintages of French's file label this column WML
}
_FF5_COLUMN_MAP = {
    "Mkt-RF": "mkt_excess",
    "MKT-RF": "mkt_excess",
    "SMB":    "smb",
    "HML":    "hml",
    "RMW":    "rmw",
    "CMA":    "cma",
    "RF":     "rf",
}


def _download_zip(zip_name: str) -> str:
    """Download a Ken French daily ZIP and return raw CSV text."""
    url = f"{_BASE_URL}/{zip_name}"
    resp = requests.get(url, timeout=90)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.upper().endswith(".CSV"))
        return zf.open(csv_name).read().decode("latin-1")


def _parse_french_csv(raw: str, column_map: dict[str, str]) -> pd.DataFrame:
    """
    Parse Ken French's non-standard CSV.

    French's files have a title block followed by column headers then data rows
    with YYYYMMDD integer dates.  Newer files are comma-delimited; older files
    use fixed-width whitespace.  Multiple sections (daily / annual) may appear;
    we read only the first data block.
    """
    lines = raw.splitlines()

    # Locate the first row whose initial token is an 8-digit date
    data_start = None
    for i, ln in enumerate(lines):
        tok = ln.strip().replace(",", " ").split()
        if tok and tok[0].isdigit() and len(tok[0]) == 8:
            data_start = i
            break
    if data_start is None:
        raise ValueError("No 8-digit date rows found in French data file")

    # The column header is the last non-empty line before the first data row
    header_idx = data_start - 1
    while header_idx >= 0 and not lines[header_idx].strip():
        header_idx -= 1
    if header_idx < 0:
        raise ValueError("Could not locate column header in French CSV")

    header_line = lines[header_idx]

    # Collect consecutive data rows (stop at blank line or non-date row)
    data_lines = []
    for ln in lines[data_start:]:
        tok = ln.strip().replace(",", " ").split()
        if not tok or not (tok[0].isdigit() and len(tok[0]) == 8):
            break
        data_lines.append(ln)

    block = header_line + "\n" + "\n".join(data_lines)

    # Handle both comma-separated and whitespace-separated formats
    if "," in header_line:
        df = pd.read_csv(io.StringIO(block), index_col=0)
    else:
        df = pd.read_csv(io.StringIO(block), sep=r"\s+", index_col=0, engine="python")

    df.columns = [c.strip() for c in df.columns]
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d")
    df.index.name = "date"
    df = df.sort_index()

    df = df.rename(columns=column_map)

    # French publishes in percent; convert to decimal
    return df / 100.0


def _load_full_history() -> pd.DataFrame:
    """Return the full US FF3 daily history, fetching and caching if needed."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    if os.path.exists(_US_FF3_CACHE):
        df = pd.read_csv(_US_FF3_CACHE, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df
    raw = _download_zip(_US_FF3_ZIP)
    df = _parse_french_csv(raw, _FF3_COLUMN_MAP)
    df.to_csv(_US_FF3_CACHE)
    return df


def _load_full_mom_history() -> pd.DataFrame:
    """Return the full US daily momentum (UMD/Mom) factor history, fetching and caching if needed."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    if os.path.exists(_US_MOM_CACHE):
        df = pd.read_csv(_US_MOM_CACHE, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df
    raw = _download_zip(_US_MOM_ZIP)
    df = _parse_french_csv(raw, _MOM_COLUMN_MAP)
    df.to_csv(_US_MOM_CACHE)
    return df


def _load_full_ff5_history() -> pd.DataFrame:
    """
    Return the full US daily 5-factor (Mkt-RF, SMB, HML, RMW, CMA, RF) history,
    fetching and caching if needed.  Only the rmw/cma columns are actually
    consumed by get_ff6_daily()/get_ff7_daily() below — mkt/smb/hml/rf come
    from the 3-factor file (_load_full_history()) to keep a single source of
    truth for those three, consistent with how get_ff4_daily() already only
    takes 'mom' from the separately-published momentum file.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    if os.path.exists(_US_FF5_CACHE):
        df = pd.read_csv(_US_FF5_CACHE, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df
    raw = _download_zip(_US_FF5_ZIP)
    df = _parse_french_csv(raw, _FF5_COLUMN_MAP)
    df.to_csv(_US_FF5_CACHE)
    return df


def get_ff3_daily(start: str, end: str) -> pd.DataFrame:
    """
    Return official Fama-French US 3-factor data for the given date range.

    Columns
    -------
    mkt_excess : daily market excess return (Mkt-RF), decimal
    smb        : daily SMB return, decimal
    hml        : daily HML return, decimal
    rf         : daily risk-free rate, decimal

    Both US holdings and VXUS use this series; VXUS is labeled accordingly
    in output to signal reduced precision (see module docstring).

    Kept 3-factor-only for callers that specifically want FF3; see
    get_ff4_daily() for the Carhart 4-factor version (adds momentum).
    """
    df = _load_full_history()
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    cols = [c for c in ("mkt_excess", "smb", "hml", "rf") if c in df.columns]
    return df.loc[mask, cols].copy()


def get_ff4_daily(start: str, end: str) -> pd.DataFrame:
    """
    Return official Fama-French-Carhart 4-factor data for the given date range.

    Columns
    -------
    mkt_excess : daily market excess return (Mkt-RF), decimal
    smb        : daily SMB return, decimal
    hml        : daily HML return, decimal
    mom        : daily momentum (UMD, "up minus down") factor return, decimal
    rf         : daily risk-free rate, decimal

    mom comes from Ken French's separately-published daily momentum file, not
    the 3-factor file — it is merged in here on the date index.  Both series
    share the same underlying CRSP universe, so this is the genuine academic
    momentum factor (not an ETF proxy) with history back to the 1920s,
    unlike the ETF-proxy MOM series in factor_engine/factors/mom.py which is
    limited by MTUM's 2013 inception.
    """
    ff3 = _load_full_history()
    mom = _load_full_mom_history()
    df = ff3.join(mom[["mom"]], how="inner")
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    cols = [c for c in ("mkt_excess", "smb", "hml", "mom", "rf") if c in df.columns]
    return df.loc[mask, cols].copy()


def _load_full_ff6_history() -> pd.DataFrame:
    """Full-history join of ff3 + mom + (rmw, cma from the 5-factor file)."""
    ff3 = _load_full_history()
    mom = _load_full_mom_history()
    ff5 = _load_full_ff5_history()
    df = ff3.join(mom[["mom"]], how="inner").join(ff5[["rmw", "cma"]], how="inner")
    return df


def get_ff6_daily(start: str, end: str) -> pd.DataFrame:
    """
    Return the Fama-French 5-factor + Carhart momentum ("FF6") daily panel.

    Columns
    -------
    mkt_excess, smb, hml, rmw, cma, mom, rf

    rmw/cma come from Ken French's separately-published 5-factor file (see
    _load_full_ff5_history()); mom comes from the separately-published
    momentum file, exactly as in get_ff4_daily(). All are genuine academic
    series with full history — no ETF proxies involved.
    """
    df = _load_full_ff6_history()
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    cols = [c for c in ("mkt_excess", "smb", "hml", "rmw", "cma", "mom", "rf") if c in df.columns]
    return df.loc[mask, cols].copy()


def get_ff7_daily(start: str, end: str) -> pd.DataFrame:
    """
    Return the full 7-factor panel: FF6 (see get_ff6_daily()) plus this
    platform's proprietary GP (Gross Profitability) factor.

    Columns
    -------
    mkt_excess, smb, hml, rmw, cma, mom, gp, rf

    GP is left-joined onto the full FF6 history, so requesting a [start, end]
    range that predates GP's own coverage (2013-present, see
    factor_engine/factors/gp.py) does NOT truncate the other six factors —
    it simply returns NaN in the gp column for those dates. Callers running a
    single joint regression across the full panel must dropna (which will
    naturally restrict that regression to GP's covered window); callers that
    need the other six factors' full history preserved should use
    get_ff6_daily() for the primary fit and this function only for the
    GP-covered subset — see smart_money/factor_apply.py's two-tier design.
    """
    from factor_engine.factors.gp import build_gp_factor

    ff6 = _load_full_ff6_history()
    # Wide net: GP's real coverage floor is unknown until the factor is built,
    # so request from well before its plausible start through today, then
    # let the left-join below place NaNs wherever GP has no data.
    gp = build_gp_factor(start="2015-01-01", end=pd.Timestamp.today().date().isoformat())
    df = ff6.join(gp[["gp"]] if not gp.empty else pd.DataFrame(columns=["gp"]), how="left")

    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    cols = [c for c in ("mkt_excess", "smb", "hml", "rmw", "cma", "mom", "gp", "rf") if c in df.columns]
    return df.loc[mask, cols].copy()
