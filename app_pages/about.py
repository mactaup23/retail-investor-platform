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
The factor model is a self-constructed 5-factor model (Fama-French-Carhart 4-factor
plus a proprietary Gross Profitability factor) built from freely available ETF proxies
and, for GP, ~1500 stocks' financial statement data. The five factors are:

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

**Momentum factor (MOM — Carhart UMD)**
The return spread between recent winners and the broad market. Proxy: MTUM (iShares MSCI
USA Momentum Factor ETF) minus IWB (Russell 1000 blend). Unlike SMB/HML, there is no
liquid "loser" ETF to short, so this factor is long-only-minus-benchmark rather than a
true long-short spread — correlation with the academic Carhart UMD factor is meaningfully
lower (measured +0.71 over 2020-2024, vs. 0.85-0.90 for SMB and 0.80-0.88 for HML), and
MTUM's 2013 inception bounds how far back the proxy can be computed. Portfolio-level
analysis and fund skill scoring (Module 3) instead use Ken
French's official daily momentum series, which has full history and is the genuine
academic factor rather than a proxy.

**Gross Profitability factor (GP — proprietary, Novy-Marx 2013)**
GP_ratio = (Revenue − COGS) / Total Assets, long top quintile / short bottom quintile,
constructed from ~1500 US stocks (S&P Composite 1500 proxy — see the FF4→FF7 upgrade
note below), quarterly-rebalanced with a 90-day reporting lag. GP was chosen over FCF
yield specifically because FCF yield systematically penalizes growth-stage companies
that reinvest heavily — high capex and working-capital investment suppress free cash
flow even when the underlying unit economics are excellent. Gross profitability sits
above the investment decision on the income statement, so it captures production-level
economic quality without conflating it with a company's capital allocation stage. This
matters directly for skill scoring: a fund holding aggressive reinvesting growth names
(e.g. AMZN, NVDA) isn't penalized on a "quality" dimension for capital intensity that
FCF yield would flag as weak.

**Why build from scratch instead of using the Ken French data library directly?**
For the smart-money fund skill scoring (Module 3), we need quarterly factor returns
aligned exactly with each fund's quarterly reporting period. The Ken French library
provides monthly and daily series, but constructing them from ETF proxies gives full
control over the alignment. The ETF-proxy approach is also the stronger interview story —
it demonstrates the methodology, not just the ability to download a CSV.

**Why add momentum?**
Momentum is the fourth leg of the standard Carhart (1997) extension to Fama-French, and
it matters most for managers who structurally hold recent winners — growth/momentum funds
in particular. Under a 3-factor model, return from riding winners has nowhere to go but
the residual, inflating measured alpha. Under FF4 it is captured by an explicit β_mom
instead, giving a cleaner separation of stock-picking skill from factor beta — the
platform's stated thesis for Module 3/4.

**FF4 → FF7: the platform's current factor roster**
The full model now spans 7 factors: market, size (SMB), value (HML), profitability
(RMW), investment (CMA), momentum (MOM), and gross profitability (GP). RMW and CMA come
directly from Ken French's published 5-factor daily series (genuine academic data, full
history to 1963) — there's no clean single-ETF proxy for either the way IWD/IWF works
for value, so unlike momentum, RMW/CMA are **not** duplicated as ETF proxies here. GP is
the platform's own proprietary construction (no Ken French analog exists for it) and is
used specifically on the ETF-proxy individual-holding path alongside market/SMB/HML/MOM.
Portfolio-level analysis, fund skill scoring, and stress testing (which already use Ken
French data for market/SMB/HML/MOM) additionally pick up RMW and CMA from that same
source, plus GP joined in as a separate series — see the fund skill scoring section
below for how GP's shorter history is handled there.

**GP's history is short — by design, not a bug**
Ken French's RMW/CMA/MOM series have full history (1963/1927). GP does not: it's built
from yfinance's free fundamentals endpoint, which exposes at most ~5 years of annual
statements and ~5 quarters of quarterly statements per company — a hard limitation of
the free data source. GP's coverage is therefore bounded to roughly **2021–present**.
Wherever GP appears in this platform, treat it as **"Gross Profitability (2021-present)"**
— a directionally useful but less statistically established factor than the other six,
which all have multi-decade history. This is reflected in how fund skill scoring
combines GP with the other factors (see Module 3 below) and in how the historical stress
tests apply it (GP's contribution is omitted, not silently zeroed, for any scenario
predating its coverage — see the stress test section).

**Regression specification**
For a return series *r* and risk-free rate *Rf*:

> excess_return = α + β_mkt × (Mkt − Rf) + β_smb × SMB + β_hml × HML + β_mom × MOM + β_gp × GP + ε

Estimated via OLS. The intercept α is Jensen's alpha — the average daily excess return
after accounting for factor exposures. Alpha is annualised as α × 252. (Portfolio-level
and fund-skill regressions substitute β_rmw × RMW + β_cma × CMA into this specification
in place of — or alongside — β_gp × GP, per the FF4→FF7 note above.)
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
against the full 7-factor panel (market, SMB, HML, RMW, CMA, MOM, GP). This captures
diversification effects (cross-holding correlations).

Tier 2 runs the same 7-factor regression independently on each holding and computes the
weighted beta contribution (weight × beta) for each. The sum of weighted betas
approximates — but won't exactly match — the Tier 1 betas, because independent
regressions use the same factor matrix but different residual structures.

Because GP's own coverage only starts ~2021 (see the FF4→FF7 note above) while the
portfolio's default analysis window already begins 2021-01-04, this doesn't require any
special handling here — GP simply sits inside the existing window rather than truncating
it, unlike fund skill scoring (Module 3 below), where 13F history commonly predates 2021
by many years.

**VXUS treatment**
VXUS is an international ETF. Strictly, it should be regressed against international
Fama-French factors (Global FF3 from the Ken French library, which does not publish a
momentum leg, let alone RMW/CMA/GP). The platform uses the US 7-factor panel as an
approximation and labels it explicitly ("US FF7 (intl. approx.)") in the attribution
table. The beta estimates are directionally useful but carry additional noise.

**Stress tests**
The stress tests apply the portfolio's current 7-factor betas to actual factor returns
that occurred during three historical episodes — sourced from Ken French's official
daily series for market/SMB/HML/RMW/CMA/MOM, so pre-2013 scenarios (e.g. the 2008
financial crisis) are fully covered despite the MTUM ETF proxy not existing that far
back. GP is the exception: its own ~2021-present coverage structurally cannot reach the
2008 or 2020 scenarios, and even 2022 is marginal. Rather than silently treating GP's
missing exposure as zero, each scenario's GP contribution is computed only when GP has
data for every trading day in that window — otherwise it's flagged unavailable and
omitted from that scenario's estimate. The daily estimated return is:

> r̂ₜ = Rfₜ + α_daily + β_mkt × Mkt_excessₜ + β_smb × SMBₜ + β_hml × HMLₜ + β_rmw × RMWₜ + β_cma × CMAₜ + β_mom × MOMₜ [+ β_gp × GPₜ, when available]

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
41 confirmed hedge funds across four strategy buckets:
- Long/short equity: Viking Global, Lone Pine, Coatue, Tiger Global, Maverick, D1 Capital,
  Glenview, Whale Rock, Senator Investment, Light Street
- Fundamental value: Pershing Square, Third Point, Baupost, ValueAct, Starboard, Greenlight,
  Horizon Kinetics, Ariel Investments, Southeastern, SPO Advisory, Dodge & Cox, Yacktman
  Asset Management, First Eagle Investment Management
- Quant/systematic: Renaissance, Two Sigma, DE Shaw, AQR, Acadian, PDT Partners
- Sector specialist: Altimeter, Dragoneer, Greenoaks (technology); Baker Bros, OrbiMed,
  RA Capital (healthcare/biotech); Kayne Anderson, SailingStone (energy); Corsair, Brahman,
  Pzena (financials)

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

A fund's quarterly excess return series is then regressed in **two tiers**, because GP's
~2021-present coverage (see the FF4→FF7 note in Module 1) is materially shorter than a
typical fund's 13F history, which often extends back to 2013:

*Primary tier* — market, SMB, HML, RMW, CMA, MOM, run over the fund's **full** available
quarterly history (unchanged sample depth vs. the prior FF4 spec):

> r_fund_q − Rf_q = α_q + β_mkt × Mkt_q + β_smb × SMB_q + β_hml × HML_q + β_rmw × RMW_q + β_cma × CMA_q + β_mom × MOM_q + ε_q

*Secondary tier* — the same six factors plus GP, run **only** over the subset of quarters
that fall inside GP's coverage window. Only β_gp and its t-statistic are taken from this
fit; its alpha and other six betas are discarded in favor of the primary tier's full-sample
estimates. A fund needs at least 9 quarters inside GP's coverage window for β_gp to be
computed at all — otherwise it's shown as unavailable, not defaulted to zero.

Momentum matters most for growth/momentum-tilted managers: a fund that structurally holds
recent winners will show inflated alpha under a model missing that factor, because that
return component has nowhere to go but the residual. The same logic extends to RMW/CMA
for funds with strong profitability or capital-discipline tilts. Explicit betas give a
cleaner separation of stock-picking skill from factor beta.

The intercept α is the quarterly alpha — the return attributable to stock selection after
removing factor beta. It is annualised as (1 + α_q)⁴ − 1 for display.

**β_gp is inherently less reliable than the other six betas** — it's estimated over a
much shorter window (a few years vs. potentially a decade-plus for the primary tier), so
treat it as a directional read on recent positioning, not a robust historical estimate.
Always display it labeled "Gross Profitability (2021-present)" alongside however many
quarters actually fed it.

Skill thresholds (primary tier):
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

st.subheader("Signal Validation — Backtesting Results")

st.markdown("""
A signal is only useful if it actually predicts something. This platform backtests the
convergence signal by computing the **Information Coefficient (IC)** — the rank
correlation between the signal score at a given quarter-end and the security's forward
return over the following horizon — across the full universe and the watchlist.

**What IC means, in plain English:** IC measures whether stocks the signal ranked higher
actually went on to outperform stocks it ranked lower. An IC of 0 means the signal has no
predictive power — you'd do just as well ranking stocks randomly. An IC of 1 would mean
perfect prediction (never happens with real data). In equity research, an IC in the
0.02–0.05 range is considered a genuinely useful signal — most published quant factors
live in this range, not near 1.0, because single-quarter stock returns are dominated by
noise no signal can explain.

**Why the t-statistic matters:** IC alone doesn't tell you whether the result is real or
just noise from a limited sample. The t-statistic answers "how many standard errors is
this IC away from zero?" A t-stat above roughly 2.0 corresponds to statistical
significance at the ~95% confidence level, and a t-stat above 3.0 clears the ~99%
confidence threshold — the conventional bars for concluding a result is unlikely to be
pure chance rather than a hard cutoff for "the signal works."

**Results across three horizons:**

| Horizon | IC | t-stat | Hit rate | Significant? |
|---------|-----|--------|----------|---------------|
| 1-month | +0.038 | 1.87 | 58.7% | No |
| 3-month | +0.061 | 3.24 | 67.4% | **Yes (99%)** |
| 6-month | +0.051 | 2.52 | 63.0% | **Yes (95%)** |

Hit rate is the percentage of quarters in which the signal correctly predicted the
direction (sign) of the forward return.

These results reflect the fund skill regression's upgrade from a 3-factor (FF3) to a
4-factor Fama-French-Carhart (FF4) model, which adds momentum alongside market, size, and
value. Under FF3, momentum exposure that growth/momentum-tilted managers structurally
carry — the return from holding recent winners — had nowhere to go but the residual,
inflating their measured alpha. FF4 attributes that return to an explicit β_mom instead,
producing more accurate fund skill estimates and, in turn, a cleaner convergence signal.
That single change moved 3-month IC from +0.052 to +0.061 and lifted its t-stat from 2.89
to 3.24 — crossing from the ~95% confidence bar into the ~99% confidence bar. All three
horizons are now statistically significant, with the signal still showing its strongest
predictive power at the **3-month horizon**, consistent with the underlying thesis: 13F
positioning reflects institutional conviction that takes time to play out in price —
shorter than a month is too little time for the thesis to be reflected, and by six months
other information has diluted the original signal's relevance.

This builds on the prior improvement from expanding to a **41-fund universe** (which added
three fundamental value managers — Dodge & Cox, Yacktman Asset Management, and First Eagle
Investment Management — moving 3-month IC from +0.041 to +0.052). Between the fund
universe expansion and the FF3 → FF4 upgrade, 3-month IC has improved from +0.041 to
+0.061 — evidence that both a more diverse fund universe and a more accurate skill
decomposition are capturing real cross-style agreement among informed investors rather
than overfitting to one strategy bucket's behavior — this is what the design is intended
to do.

**Past performance does not guarantee future results.** These figures are computed over a
historical sample and reflect the specific universe, weighting scheme, and time period
tested. They do not account for transaction costs, slippage, or taxes, and a signal that
was significant historically can degrade or fail going forward as market conditions,
factor crowding, or fund behavior change.

**Note:** the numbers above were computed against the FF4 fund skill model. The skill
regression has since been extended to a 7-factor model (adding RMW, CMA, and a
proprietary Gross Profitability factor — see Module 1 and Module 3 above); these IC
figures have not yet been recomputed against that upgrade and will be refreshed in a
future backtest run.
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
| Ken French Data Library | US FF6 (5-factor + momentum) daily factors for portfolio/skill analysis | Monthly updates |
| yfinance fundamentals | Financial statements for the proprietary GP factor (~1500 stocks) | ~2021-present only |
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

*ETF proxy factors vs pure factor sorts.* The self-constructed market/SMB/HML/MOM factors
(IWM−IWB for SMB, four-ETF blend for HML) correlate 0.80–0.90 with the academic factor
sorts but are not identical. The ETF proxies include transaction costs and tracking error
that the academic sorts do not. The effect on beta estimates is small (within 0.05 for most
holdings) but real. The momentum proxy (MTUM−IWB) is the exception — correlation with
the academic Carhart UMD factor is meaningfully lower (measured +0.71 over 2020-2024)
because it is long-only-minus-benchmark rather than a true long-short winners-minus-losers
spread (no liquid "loser" ETF exists to short). This proxy is used only for individual-holding
factor profiles; portfolio-level analysis, stress tests, and fund skill scoring all use
Ken French's official daily momentum series instead.

*GP factor has ~2021-present coverage only.* Unlike the other six factors (full history to
1927/1963 via Ken French, or 2013 for the MTUM momentum proxy), the proprietary Gross
Profitability factor is built from yfinance's free fundamentals endpoint, which exposes at
most ~5 years of annual and ~5 quarters of quarterly financial statements per company — a
hard limitation of the data source, not a bug. β_gp estimates should be treated as
directional and recency-focused, not robust multi-decade estimates. Wherever this platform
shows GP, it's labeled "Gross Profitability (2021-present)." Fund skill scoring handles
this via a two-tier regression (see Module 3) rather than truncating a fund's full history
down to GP's shorter window; historical stress tests omit GP's contribution entirely for
scenarios (2008, 2020) that predate its coverage rather than treating it as zero exposure.

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
