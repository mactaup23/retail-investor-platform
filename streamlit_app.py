"""
Smart Money — Retail Investor Platform
Entry point for the Streamlit multi-page app.

Page order: Signals (default) → Portfolio → Tax Lots → Backtest → About
"""
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="Smart Money",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Database bootstrap — must run before get_db()/init_db() touch the file.
_DB_URL = "https://pub-ac106313846e4e0c8b8550be55d431f3.r2.dev/module3.db"
_MIN_DB_SIZE_BYTES = 500 * 1024 * 1024


def _ensure_db_downloaded() -> None:
    from smart_money.models import DB_PATH

    if DB_PATH.exists():
        return  # fast path — normal use

    import requests

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = DB_PATH.with_suffix(".db.part")

    status = st.empty()
    status.info("Downloading database — this may take a minute on first launch...")
    progress = st.progress(0.0)

    try:
        with requests.get(_DB_URL, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        progress.progress(min(downloaded / total, 1.0))

        if tmp_path.stat().st_size < _MIN_DB_SIZE_BYTES:
            tmp_path.unlink(missing_ok=True)
            status.empty()
            progress.empty()
            st.error(
                "Database download appears incomplete or corrupt (file too small). "
                "The app cannot run without a valid database — please reload to retry."
            )
            st.stop()

        tmp_path.rename(DB_PATH)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        status.empty()
        progress.empty()
        st.error(
            f"Could not load the database: {e}\n\n"
            "The app cannot run without it — please reload the page to retry."
        )
        st.stop()

    status.empty()
    progress.empty()


_ensure_db_downloaded()

# Initialise DB once per server process (cached resource — no-op on re-runs)
from dashboard.db import get_db
get_db()

# Pre-warm portfolio FF3 betas so the Factor Profile tab is always instant.
# cache_resource means this runs exactly once per server process; subsequent
# page loads (and all other sessions) get an immediate cache hit. Weights come
# from the user's saved portfolio (data/user_prefs.json) so the Signals page's
# "how would this change my exposure" callouts compare against the portfolio
# the user actually owns, not a hardcoded stand-in.
from dashboard.factor import current_portfolio_betas as _prewarm_ff3

_warmup_slot = st.sidebar.empty()
_warmup_slot.caption(":gray[Initializing factor engine…]")
_prewarm_ff3()
_warmup_slot.empty()

# Sidebar header — rendered before every page
with st.sidebar:
    st.markdown("## :material/query_stats: Smart Money")
    st.caption("Factor research · Smart-money signals · Tax-lot modeler")

# Define pages
signals_page   = st.Page("app_pages/signals.py",   title="Signals",   icon=":material/radar:",           default=True)
portfolio_page = st.Page("app_pages/portfolio.py",  title="Portfolio", icon=":material/bar_chart:")
tax_lots_page  = st.Page("app_pages/tax_lots.py",   title="Tax lots",  icon=":material/account_balance:")
backtest_page  = st.Page("app_pages/backtest.py",   title="Backtest",  icon=":material/insights:")
about_page     = st.Page("app_pages/about.py",      title="About",     icon=":material/info:")

page = st.navigation(
    [signals_page, portfolio_page, tax_lots_page, backtest_page, about_page],
    position="sidebar",
)

page.run()
