"""
Smart Money — Retail Investor Platform
Entry point for the Streamlit multi-page app.

Page order: Signals (default) → Portfolio → Tax Lots → About
"""
import streamlit as st

st.set_page_config(
    page_title="Smart Money",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialise DB once per server process (cached resource — no-op on re-runs)
from dashboard.db import get_db
get_db()

# Sidebar header — rendered before every page
with st.sidebar:
    st.markdown("## :material/query_stats: Smart Money")
    st.caption("Factor research · Smart-money signals · Tax-lot modeler")

# Define pages
signals_page   = st.Page("app_pages/signals.py",   title="Signals",   icon=":material/radar:",           default=True)
portfolio_page = st.Page("app_pages/portfolio.py",  title="Portfolio", icon=":material/bar_chart:")
tax_lots_page  = st.Page("app_pages/tax_lots.py",   title="Tax lots",  icon=":material/account_balance:")
about_page     = st.Page("app_pages/about.py",      title="About",     icon=":material/info:")

page = st.navigation(
    [signals_page, portfolio_page, tax_lots_page, about_page],
    position="sidebar",
)

page.run()
