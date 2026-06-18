"""
Official Fama-French 3-factor data downloaded directly from Ken French's data library.

We use the published US daily series (Mkt-RF, SMB, HML, RF) rather than the
ETF-proxy series built in factor_engine/factors/.  The official series uses
actual CRSP stock returns and proper B/M breakpoints — strictly more accurate
than the IWM/IWB/IWD/IWF/IWN/IWO proxies.

VXUS methodology note
---------------------
VXUS tracks the FTSE Global All Cap ex-US index.  Ken French publishes *regional*
daily factor series (Europe, Japan, Asia Pacific ex Japan) but not a combined
"Developed ex-US" daily series.  Constructing one would require knowing VXUS's
current geographic weights (≈37% Europe, 27% Pacific, 25% EM, 11% other), which
vary over time and would introduce a time-varying factor matrix — complexity that
buys little precision at daily frequency given that developed markets co-move with
the US at r ≈ 0.70–0.85.  We therefore use the US FF3 factors for VXUS, label it
"US FF3 (intl. approx.)" everywhere in output, and note the R² reduction (~0.60–
0.75 vs. 0.95+ for domestic ETFs) so the user can calibrate their confidence.

Cache: data/french/us_ff3_daily.csv  (full history from 1926; immutable past data)
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


def _download_us_ff3() -> str:
    """Download the Ken French US daily FF3 ZIP and return raw CSV text."""
    url = f"{_BASE_URL}/{_US_FF3_ZIP}"
    resp = requests.get(url, timeout=90)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.upper().endswith(".CSV"))
        return zf.open(csv_name).read().decode("latin-1")


def _parse_french_csv(raw: str) -> pd.DataFrame:
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

    df = df.rename(columns={
        "Mkt-RF": "mkt_excess",
        "MKT-RF": "mkt_excess",
        "SMB":    "smb",
        "HML":    "hml",
        "RF":     "rf",
    })

    # French publishes in percent; convert to decimal
    return df / 100.0


def _load_full_history() -> pd.DataFrame:
    """Return the full US FF3 daily history, fetching and caching if needed."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    if os.path.exists(_US_FF3_CACHE):
        df = pd.read_csv(_US_FF3_CACHE, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df
    raw = _download_us_ff3()
    df = _parse_french_csv(raw)
    df.to_csv(_US_FF3_CACHE)
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
    """
    df = _load_full_history()
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    cols = [c for c in ("mkt_excess", "smb", "hml", "rf") if c in df.columns]
    return df.loc[mask, cols].copy()
