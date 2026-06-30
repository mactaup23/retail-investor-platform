"""About page — platform methodology and documentation."""
import streamlit as st

st.header(":material/info: About this platform")

st.markdown("""
This platform is a factor-based equity analysis and smart-money signal tracker built for
a single investor. It integrates three capabilities that institutional research desks have
separately but retail investors typically do not: factor exposure transparency, institutional
positioning analysis with skill-adjusted weighting, and tax-aware sell decision modeling.
""")

# ---------------------------------------------------------------------------

st.subheader("The problem it solves")

st.markdown("""
Tools like WhaleWisdom and Whalewatcher aggregate 13F filings and surface which stocks
hedge funds are buying. The problem is that most of those funds are not generating
returns through stock selection — they are generating beta to growth, momentum, or size
factors that any index fund captures for free. Treating a Renaissance buy as equivalent
to an Appaloosa buy ignores the fact that one is a quant with thousands of positions and
the other is a concentrated fundamental investor with a verifiable stock-picking record.

This platform solves that by **decomposing each fund's historical returns into factor
exposures and isolating the residual alpha** — the return that cannot be explained by
passive factor exposure. That alpha estimate is then used to weight each fund's current
position changes when computing the convergence signal. A fund with genuine positive
alpha after accounting for factor betas gets 1.0–3.0× weight. A quant fund gets 0.40×
because its 13F is a hedged book with limited directional information.

The second gap it fills: **the 13F signal is backward-looking, so the platform blends it
with a forward-looking NLP language-shift score** extracted from each company's most
recent 10-Q or 10-K MD&A section, scored against the prior filing by Claude. A
convergence signal with corroborating management language shift is a stronger signal than
either alone.

The third gap: **tax-lot-aware sell modeling**. Most retail investors manage their cost
basis with a spreadsheet, if at all. This platform ingests a Fidelity / Schwab / IBKR
CSV export and lets you model the tax consequence of every lot-selection method side by
side before executing.
""")

# ---------------------------------------------------------------------------

st.subheader("Module 1 — factor model construction")

st.markdown("""
The factor model is a self-constructed Fama-French 3-factor (FF3) model built from
freely available ETF proxies. The three factors are:

**Market factor (Mkt−Rf)**
The excess return of the broad equity market over the risk-free rate. Proxy: SPY daily
return minus the 3-month T-bill yield (^IRX), converted to a daily rate. The Ken French
data library's official US market factor is used for portfolio-level analysis (via
`factor_engine/french_data.py`) in preference to the ETF proxy for higher accuracy.

**Size factor (SMB — Small Minus Big)**
The return spread between small-cap and large-cap stocks. Proxy: IWM (iShares Russell
2000) minus IWB (iShares Russell 1000). Correlation with the academic FF SMB factor is
~0.85–0.90. The gap exists because ETF-based SMB captures a blended small-cap premium
rather than a pure size-sorted factor portfolio.

**Value factor (HML — High Minus Low)**
The return spread between value and growth stocks. Proxy: average of IWD + IWN (value
ETFs) minus average of IWF + IWO (growth ETFs). Correlation with the academic FF HML
factor is ~0.80–0.88. The four-ETF averaging reduces noise from any single value/growth
index's idiosyncratic construction choices.

**Why build from scratch instead of using the Ken French data library directly?**
For the smart-money fund skill scoring (Module 3), we need quarterly factor returns
aligned exactly with each fund's quarterly reporting period. The Ken French library
provides monthly and daily series, but constructing them from ETF proxies gives full
control over the alignment. The ETF-proxy approach is also the stronger interview story —
it demonstrates the methodology, not just the ability to download a CSV.

**Regression specification**
For a return series *r* and risk-free rate *Rf*:

> excess_return = α + β_mkt × (Mkt − Rf) + β_smb × SMB + β_hml × HML + ε

Estimated via OLS. The intercept α is Jensen's alpha — the average daily excess return
after accounting for factor exposures. Alpha is annualised as α × 252.
""")

# ---------------------------------------------------------------------------

st.subheader("Module 2 — portfolio factor exposure")

st.markdown("""
The portfolio consists of nine holdings with the following weights (as of the most recent
rebalancing):

| Ticker | Weight | Description |
|--------|--------|-------------|
| VTI    | 24.4%  | US total market ETF |
| QQQM   | 11.4%  | Nasdaq-100 ETF |
| SCHD   | 11.8%  | Dividend-weighted large-cap ETF |
| VXUS   | 15.6%  | Ex-US international equity ETF |
| NVDA   | 2.9%   | Nvidia Corporation |
| GOOGL  | 5.2%   | Alphabet Inc. |
| QTUM   | 10.3%  | Quantum computing / AI thematic ETF |
| VTV    | 8.2%   | Vanguard Value ETF |
| XLI    | 5.1%   | Industrials sector ETF |

**Two-tier analysis**
Tier 1 produces a single headline set of betas by constructing the daily portfolio return
as a weighted sum of individual holding log returns, then regressing the combined series
against the FF3 factors. This captures diversification effects (cross-holding
correlations).

Tier 2 runs FF3 independently on each holding and computes the weighted beta contribution
(weight × beta) for each. The sum of weighted betas approximates — but won't exactly
match — the Tier 1 betas, because independent regressions use the same factor matrix but
different residual structures.

**VXUS treatment**
VXUS is an international ETF. Strictly, it should be regressed against international
Fama-French factors (Global FF3 from the Ken French library). The platform uses US FF3 as
an approximation and labels it explicitly in the attribution table. The beta estimates are
directionally useful but carry additional noise.

**Stress tests**
The stress tests apply the portfolio's current FF3 betas to actual factor returns that
occurred during three historical episodes. The daily estimated return is:

> r̂ₜ = Rfₜ + α_daily + β_mkt × Mkt_excessₜ + β_smb × SMBₜ + β_hml × HMLₜ

Period return: R = exp(Σ r̂ₜ) − 1

This answers: *given this portfolio's current factor loadings, how much would the factor
model have predicted it to lose under each historical macro shock?* It is a **risk
characterisation, not a backtest** — the portfolio did not exist in 2008. The label
"estimated portfolio return" is used throughout, not "backtested return."

Three scenarios: 2008 Financial Crisis (Sep 2008 – Mar 2009, Lehman collapse to S&P
trough), 2020 COVID Crash (Feb 19 – Mar 23 2020, peak to trough), 2022 Rate Hike Bear
Market (full calendar year 2022, Fed +425bp).
""")

# ---------------------------------------------------------------------------

st.subheader("Module 3 — EDGAR ingestion and fund skill scoring")

st.markdown("""
**13F universe**
38 confirmed hedge funds across four strategy buckets:
- Long/short equity: Appaloosa, Viking Global, Pershing Square, D1 Capital, Light Street,
  Tiger Global, Coatue, Lone Pine, Third Point, Greenlight, Eminence, Glenview, Maverick,
  Highfields, Baupost, Greenlight, Duquesne
- Fundamental value: Berkshire Hathaway, Ariel, Fairholme, Gotham
- Quant/systematic: Renaissance, Two Sigma, DE Shaw, AQR, Citadel, Millennium
- Sector specialist: Baker Bros (biotech), Whale Rock (tech), Durable Capital (tech),
  Senator Investment (diversified), Sachem Head (industrials)

**EDGAR ingestion**
Quarterly 13F-HR filings are fetched from SEC EDGAR's full-text search index. Each
filing's XML information table is parsed to extract (CUSIP, issuer name, value, shares,
investment discretion, put/call indicator). The EDGAR value field is empirically in raw
dollars, not thousands as the SEC spec states — verified by cross-checking Viking Global's
Visa position against contemporaneous market prices.

CUSIP-to-ticker resolution uses the OpenFIGI API. Price data comes from yfinance (adjusted
close). Pre-June 2013 filings are in plain-text format and skipped.

**Skill scoring**
For each fund, quarterly portfolio returns are reconstructed using the Grinblatt-Titman
"buy-and-hold beginning-of-quarter" methodology: take the positions as of the 13F date
(which is the quarter *end*), assume those positions were held at the start of the quarter,
and compute their buy-and-hold return over the quarter using prices at both ends.

A fund's quarterly excess return series is then regressed against the FF3 factors to
decompose its returns:

> r_fund_q − Rf_q = α_q + β_mkt × Mkt_q + β_smb × SMB_q + β_hml × HML_q + ε_q

The intercept α is the quarterly alpha — the return attributable to stock selection after
removing factor beta. It is annualised as (1 + α_q)⁴ − 1 for display.

Skill thresholds:
- **Reliable** (≥12 quarters): sufficient history for OLS to be meaningful
- **Scored** (≥8 quarters): enough data to estimate but wide confidence intervals
- α is a **directional signal, not a significance test**. The t-statistic and
  confidence label are always shown alongside alpha.

Coverage limitations for quant funds: Renaissance, Two Sigma, DE Shaw, and AQR hold
thousands of foreign and delisted positions that yfinance cannot price. Their coverage
rate (fraction of beginning-of-quarter holdings with available end-of-quarter prices) is
structurally below the 80% gate required for return computation. These funds are tracked
for crowding analysis but their skill scores are unreliable.
""")

# ---------------------------------------------------------------------------

st.subheader("Module 4 — convergence signal and NLP")

st.markdown("""
**Convergence score**
For each security in the universe, the convergence score for a quarter measures the
weighted net directional sentiment across all funds that hold or recently changed their
position:

- **Direction** for each fund: bullish_leaning (new position, add, maintained large) or
  bearish_leaning (trim, exit, no position)
- **Weight** for each fund: derived from the skill score tier (see below)

The raw convergence score ranges from −1 (all weight bearish) to +1 (all weight bullish).
Additional signals: breadth (fraction of funds with a position), position size as a
fraction of each fund's portfolio, and trend (new / accelerating / stable / fading based
on comparison to the prior two quarters).

**Fund skill weight tiers**

| Fund situation | Weight multiplier |
|----------------|------------------|
| Scored, reliable, positive alpha | clamp(1.0 + α_ann / 0.20, 0.5, 3.0) |
| Scored, reliable, negative alpha | Same formula, minimum 0.10 |
| Scored but unreliable (<12 quarters) | 0.80 |
| Unscored long/short equity | 0.85 |
| Unscored fundamental value | 0.90 |
| Unscored sector specialist | 0.85 |
| Unscored quant/systematic | 0.40 |

The quant/systematic 0.40 weight reflects the fact that a 13F for a quant fund is a
hedged book snapshot, not a directional view. A Citadel 13F long is not equivalent to an
Appaloosa 13F long.

**NLP scoring**
The NLP scorer reads the MD&A section of each portfolio company's most recent 10-Q or
10-K, compares it to the prior period's filing, and scores the language shift across seven
dimensions using Claude (claude-sonnet-4-6 via the Anthropic Batch API for 50% cost
reduction):

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Guidance | 0.25 | Forward-looking language about revenue / earnings trajectory |
| Confidence | 0.20 | Tone certainty vs hedging language |
| Customer demand | 0.20 | Discussion of demand environment, order books, bookings |
| Competitive positioning | 0.15 | Market share language, differentiation claims |
| Operational efficiency | 0.10 | Margin language, cost structure discussion |
| Risk factors | 0.05 | Change in disclosed risk factor severity |
| Capital allocation | 0.05 | Buyback, dividend, capex language shifts |

The composite NLP score is the weighted sum of dimension deltas, ranging from −1
(language meaningfully more negative) to +1 (language meaningfully more positive). A
score near 0 means **no language shift detected, not a negative signal** — large-cap IR
teams use formulaic MD&A language that scores close to zero even in healthy periods.

**Signal blending**
Normal blend:
> final_score = 0.70 × convergence_score + 0.30 × nlp_score

Contradiction override (fires when |conv| ≥ 0.25 AND |nlp| ≥ 0.25 AND opposite signs):
> final_score = (0.85 × convergence_score + 0.15 × nlp_score) × 0.80

The 0.80 dampening reflects elevated uncertainty. The convergence direction is preserved
but confidence is reduced. When NLP data is unavailable, the convergence score is used
unmodified (no penalty).

**Status logic**
Each security is assigned a status based on the current final_score and the change from
the prior quarter:

| Status | Condition |
|--------|-----------|
| EXIT SIGNAL | Prior signal exists and current score dropped below 0.30 |
| STRENGTHENING | New entry with score ≥ 0.55, OR quarter-over-quarter delta ≥ +0.10, OR delta ≥ +0.05 and trend = accelerating |
| WEAKENING | Delta ≤ −0.10, OR delta ≤ −0.05 and trend = fading |
| HOLDING | All other above-threshold cases |

Securities with no prior signal row and score < 0.30 are filtered — you cannot exit a
position that was never signalled.
""")

# ---------------------------------------------------------------------------

st.subheader("Module 5 — tax-lot engine")

st.markdown("""
The tax-lot engine ingests lot-level cost basis data from brokerage CSV exports
(Fidelity, Schwab, IBKR auto-detected; generic format via column mapping) and models the
tax cost of proposed sell decisions.

**Lot-selection methods**

| Method | Description |
|--------|-------------|
| FIFO | Oldest lots first (IRS default when Specific ID not elected) |
| LIFO | Newest lots first |
| MIN_TAX | Tax-minimising priority: LT losses first → ST losses → LT gains smallest per-share → ST gains smallest per-share. Disallowed-loss lots are excluded |
| SPEC_ID | Caller specifies lot IDs in the order provided |

**Important MIN_TAX vs LIFO nuance:** LIFO can numerically show lower taxes than MIN_TAX
when disallowed-loss lots are selected. This is correct behaviour. MIN_TAX deliberately
excludes those lots because the "savings" are deferred into the replacement lot's adjusted
cost basis — not actually realised. LIFO savings on disallowed lots are deferred, not
permanent. MIN_TAX is the economically correct choice for genuine tax harvesting.

**Wash-sale detection**
A capital loss is disallowed when a substantially identical security is purchased within
the 61-day window centred on the sale date (30 days before + sale date + 30 days after).
The engine checks same-ticker purchases only; options on the same underlying are outside
scope.

Two kinds of flags:
- **Disallowed:** A same-ticker lot was acquired within 30 days *before* the proposed sale
  date. The loss is already disallowed; the disallowed amount is added to the replacement
  lot's cost basis.
- **Warning:** No disqualifying prior purchase found. Prospective risk only — buying back
  the same ticker within 30 days *after* the sale will retroactively disallow the loss.

**Tax rates**
The model accepts: `st_rate` (federal short-term / ordinary income), `lt_rate` (federal
LT capital gains), `state_rate` (additive to both ST and LT), and `niit` (adds 3.8% NIIT
surcharge). The engine does not compute AGI or determine which bracket applies — you
provide your marginal rates.
""")

# ---------------------------------------------------------------------------

st.subheader("Valuation — FCF yield comparison")

st.markdown("""
**FCF Yield Historical Comparison:** To assess whether a company's current free cash flow
yield is unusually high or low, the platform compares trailing-twelve-month (TTM) FCF yield
against the company's own 3-year historical average — using the 3 fiscal years prior to
the most recent completed year (not including the most recent year itself). This avoids
comparing the current figure against a baseline that overlaps with the same period, which
would understate genuine changes. If fewer than 3 years of historical data are available,
the platform falls back to a general absolute benchmark (FCF yield above 4% is generally
considered strong, below 1.5% generally weak) rather than a company-specific comparison.

TTM FCF is computed by summing the four most recent quarterly free cash flow figures rather
than annualising a single quarter — this correctly captures seasonal variation that a
single-quarter ×4 annualisation would distort (e.g. a retailer with heavy Q4 cash
generation). The TTM FCF margin uses the same four-quarter sum in both numerator and
denominator (TTM FCF ÷ TTM Revenue) for consistency.
""")

# ---------------------------------------------------------------------------

st.subheader("Data sources and limitations")

st.markdown("""
**Data sources**

| Source | What it provides | Latency |
|--------|-----------------|---------|
| SEC EDGAR | 13F-HR quarterly filings | 45 days after quarter end |
| Ken French Data Library | US FF3 daily factors for portfolio analysis | Monthly updates |
| yfinance | Adjusted daily closing prices | Daily (T+0) |
| OpenFIGI API | CUSIP → ticker resolution | Near real-time |
| Anthropic Batch API | NLP scoring of 10-Q / 10-K MD&A | Computed at pipeline run |

**Known limitations (by design)**

*13F data is long-only and 45-day lagged.* 13F filings report equity long positions only.
Short positions, derivatives, non-US equities, and futures are not disclosed. A fund
appearing "bullish" via 13F could be net short via derivatives. The lag means data
reflects the portfolio at the prior quarter-end, not today.

*Bearish signals come from trims and exits, not from actual short positions.* When the
convergence score is negative for a security, it means funds have been reducing or
exiting long positions — not that funds are actively short.

*Quant fund coverage is structurally limited.* Renaissance, Two Sigma, DE Shaw, and AQR
report thousands of positions spanning foreign-listed stocks, delisted securities, and
obscure instruments that yfinance cannot price. Their 13F-derived return reconstructions
are unreliable regardless of pipeline parameters.

*ETF proxy factors vs pure factor sorts.* The self-constructed FF3 factors (IWM−IWB for
SMB, four-ETF blend for HML) correlate 0.80–0.90 with the academic factor sorts but are
not identical. The ETF proxies include transaction costs and tracking error that the
academic sorts do not. The effect on beta estimates is small (within 0.05 for most
holdings) but real.

*Skill scores require 12+ quarters to be reliable.* Below 12 quarters, the OLS alpha
estimate has wide confidence intervals. Alpha t-statistics and confidence labels are always
shown alongside the point estimate. An alpha of +8% annualised with a t-stat of 1.2 and
"low confidence" label is not the same signal as the same alpha with a t-stat of 3.1.

*NLP score of 0 means no language shift, not a negative signal.* Large-cap IR teams use
formulaic MD&A language that changes little quarter to quarter. A composite NLP score
near zero is not a bearish signal — it means the model detected no meaningful change in
tone, guidance, or risk factor language.

*Baupost files incomplete 13Fs.* Baupost Asset Management has historically requested
confidential treatment on portions of its 13F holdings. The disclosed book is materially
incomplete and should not be interpreted as representative of its full portfolio.
""")

# ---------------------------------------------------------------------------

st.subheader("Why this is different from WhaleWisdom and similar tools")

st.markdown("""
**WhaleWisdom and similar 13F aggregators** show you which funds own which stocks, and
track changes quarter over quarter. This is useful as a starting point but has two
critical gaps:

**Gap 1: No skill separation**
A WhaleWisdom "top buys" list weights a Two Sigma add equally with a Viking Global add.
Two Sigma runs thousands of algorithmically selected positions across its 13F book. Viking
runs a concentrated, fundamentally-driven long/short portfolio. Treating these signals as
equivalent is like averaging a coin flip with a skilled analyst's recommendation.

This platform's skill-weighted convergence signal uses each fund's historical alpha
(excess return after stripping out factor beta) to determine how much weight its position
changes receive. A fund that has genuinely added value through stock selection after
accounting for its market, size, and value tilts gets up to 3× weight. A quant fund gets
0.40× weight because its 13F is not a directional signal.

**Gap 2: No confirmation layer**
Even a well-calibrated convergence signal is a backward-looking indicator — it tells you
what sophisticated institutions owned 45 days ago. This platform adds the NLP layer as a
forward-looking confirmation: does management's own language in the most recent filing
corroborate the bullish or bearish positioning? A convergence signal with corroborating
management language is a stronger thesis. A contradiction flags elevated uncertainty.

**Gap 3: No tax integration**
Even if you have a strong sell signal on a position, the right sell decision depends on
your tax situation. A position you've held for 11 months and 15 days might be worth
waiting 16 days to cross the long-term threshold. This platform integrates that
consideration directly — the Tax lots page surfaces near-LT lots alongside the signal
data, and the Sell Modeler shows the after-tax consequence of every lot-selection method
before you call your broker.
""")

# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Built by Mac Taupier as a portfolio project.  "
    "Data sources: SEC EDGAR · Ken French Data Library · yfinance · OpenFIGI · Anthropic API.  "
    "Not investment advice."
)
