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

**FF4 → FF7: the platform's factor roster, and where each is used**
The full model spans 7 factors: market, size (SMB), value (HML), profitability (RMW),
investment (CMA), momentum (MOM), and gross profitability (GP). RMW and CMA come directly
from Ken French's published 5-factor daily series (genuine academic data, full history to
1963) — there's no clean single-ETF proxy for either the way IWD/IWF works for value, so
unlike momentum, RMW/CMA are **not** duplicated as ETF proxies here. GP is the platform's
own proprietary construction (no Ken French analog exists for it) and is used on the
ETF-proxy individual-holding path alongside market/SMB/HML/MOM. Portfolio-level analysis
and stress testing (Module 2, which already use Ken French data for market/SMB/HML/MOM)
additionally pick up RMW and CMA from that same source, plus GP joined in as a separate
series. **Fund skill scoring (Module 3/4) uses FF4 only** — market/SMB/HML/MOM, no
RMW/CMA/GP — a deliberate scope decision after a backtest investigation found no
measurable predictive benefit from the three additional factors on that module's short
per-fund quarterly panels; see "Signal Validation" below for the full account, and
CLAUDE.md for why this shouldn't be casually re-upgraded without re-running that
validation.

**GP's data source and coverage**
GP is built from SEC EDGAR's XBRL `companyfacts` API (Revenue, COGS, Total Assets tagged
facts, cross-referenced against SEC's own bulk ticker→CIK mapping) — not yfinance. This
gives GP full **2013–2026** history, matching the platform's other 2013-era history floors
(the 13F XML cutoff, MTUM's ETF inception). An earlier version of this factor was built
from yfinance's free fundamentals endpoint, which exposed at most ~5 years of annual and
~5 quarters of quarterly statements per company — a hard limitation of that data source
that bounded GP to roughly 2021-present. That limitation no longer applies after the
EDGAR XBRL migration.

The migration surfaced real data-quality edge cases, now handled explicitly rather than
silently: **58 tickers are excluded** from the GP universe with documented per-ticker
reasoning (`factor_engine/gp_exclusions.py`) — 35 REITs (their rental-income business
model has no COGS-equivalent concept for Novy-Marx GP to measure at all, the same reason
financials are conventionally excluded from academic profitability factors), 18 tickers
with a pervasive company-specific tag mismatch (e.g. health insurers, whose real cost
driver is claims/medical benefits paid under an entirely different XBRL concept than
standard "cost of revenue" tags), and 5 tickers with an unexplained raw-value magnitude
mismatch versus the prior yfinance-derived data. Individual corrupted observations (e.g.
one placeholder shell-company balance-sheet value from a merger-era filing) are filtered
at the single-quarter level rather than excluding the whole ticker, preserving otherwise-good
history. Every observation also carries a `source` provenance tag — `reported` (directly
tagged), `derived_from_ytd_subtraction` (Q4 is rarely tagged discretely; derived as exact
arithmetic from the fiscal-year and nine-month-YTD facts most filers do tag), or
`estimated_from_margin` (COGS backed out from the company's own historical gross margin,
last resort) — so any future data-quality question can be isolated to a specific tier
rather than guessed at.

**Regression specification**
For a return series *r* and risk-free rate *Rf*:

> excess_return = α + β_mkt × (Mkt − Rf) + β_smb × SMB + β_hml × HML + β_mom × MOM + β_gp × GP + ε

Estimated via OLS. The intercept α is Jensen's alpha — the average daily excess return
after accounting for factor exposures. Alpha is annualised as α × 252. (Module 2's
portfolio-level regressions substitute β_rmw × RMW + β_cma × CMA into this specification
alongside β_gp × GP, per the FF4→FF7 note above; Module 3/4 fund-skill regressions omit
all three.)
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

GP's EDGAR XBRL coverage (2013–2026, see the FF4→FF7 note above) fully spans the
portfolio's default analysis window (beginning 2021-01-04), so no truncation or special
handling is needed here.

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
back. GP is the exception: even with its EDGAR XBRL coverage now spanning 2013–2026 (see
the FF4→FF7 note above), that still doesn't reach the 2008 or 2020 scenarios, both of
which predate it — 2022 is now solidly covered, no longer marginal. Rather than silently
treating GP's missing exposure as zero for the scenarios it can't reach, each scenario's
GP contribution is computed only when GP has data for every trading day in that window —
otherwise it's flagged unavailable and omitted from that scenario's estimate. The daily
estimated return is:

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
60 confirmed funds across eight strategy buckets:
- Long/short equity: Viking Global, Lone Pine, Coatue, Tiger Global, Maverick, D1 Capital,
  Glenview, Whale Rock, Senator Investment, Light Street
- Fundamental value: Pershing Square, Third Point, Baupost, ValueAct, Starboard, Greenlight,
  Horizon Kinetics, Ariel Investments, Southeastern, SPO Advisory, Dodge & Cox, Yacktman
  Asset Management, First Eagle Investment Management
- Quant/systematic: Renaissance, Two Sigma, DE Shaw, AQR, Acadian, PDT Partners
- Sector specialist: Altimeter, Dragoneer, Greenoaks (technology); Baker Bros, OrbiMed,
  RA Capital (healthcare/biotech); Kayne Anderson, SailingStone (energy); Corsair, Brahman,
  Pzena (financials)
- Sovereign wealth: Norges Bank Investment Management, Temasek Holdings
- Insurance portfolio: Berkshire Hathaway, Markel Group, Fairfax Financial Holdings, Loews
- University endowment: Harvard Management Company, UTIMCO
- Long-only institutional: Wellington Management Group, Tweedy Browne, Baron Capital Group
  (BAMCO), Harris Associates, Davis Advisors, Duquesne Family Office, Royce Investment
  Partners, Elliott Investment Management, Ruane, Cunniff & Goldfarb, Capital World
  Investors, T. Rowe Price Investment Management

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

A fund's quarterly excess return series is then regressed against **FF4** (market, size,
value, momentum), run over the fund's full available quarterly history:

> r_fund_q − Rf_q = α_q + β_mkt × Mkt_q + β_smb × SMB_q + β_hml × HML_q + β_mom × MOM_q + ε_q

Momentum matters most for growth/momentum-tilted managers: a fund that structurally holds
recent winners will show inflated alpha under a model missing that factor, because that
return component has nowhere to go but the residual. An explicit momentum beta gives a
cleaner separation of stock-picking skill from factor beta.

The intercept α is the quarterly alpha — the return attributable to stock selection after
removing factor beta. It is annualised as (1 + α_q)⁴ − 1 for display.

**Why FF4 here and FF7 in Module 2** — this module previously ran the fuller FF7 spec
(adding RMW, CMA, and the platform's proprietary Gross Profitability factor), matching
Module 2's portfolio-level model. That was reverted after a backtest investigation — see
"Signal Validation" below for the full account — found no measurable predictive benefit
from the three additional factors on this module's short, per-fund quarterly panels (12-40
quarters per fund), while Module 2 analyzes one portfolio's long daily return history and
doesn't share that same data constraint. FF4 here is a parsimony choice, not a claim that
RMW/CMA/GP are unreliable in general — they remain the production model for Module 2.

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

**Results across three horizons (FF4 fund skill model, 60-fund universe, current production
configuration):**

| Horizon | Universe | IC | t-stat | Hit rate | Significant? |
|---------|----------|-----|--------|----------|---------------|
| 1-month | full | +0.008 | 1.42 | 66% | No |
| 1-month | watchlist | +0.006 | 1.19 | 62% | No |
| 3-month | full | +0.008 | 1.52 | 57% | No |
| 3-month | watchlist | +0.007 | 1.52 | 53% | No |
| 6-month | full | +0.005 | 0.98 | 56% | No |
| 6-month | watchlist | +0.006 | 1.11 | 56% | No |

Hit rate is the percentage of quarters in which the signal correctly predicted the direction
(sign) of the forward return. None of the six (horizon × universe) combinations clear the
~95% confidence bar (t-stat ≈ 2.0) at present.

**An honest investigation: what happened to the +0.061 result, and why it isn't the
benchmark to chase.** This platform previously reported a much stronger number — 3-month IC
+0.061 (t-stat 3.24, 99% significant) — and spent considerable effort trying to recover it
after later changes appeared to erode it. That investigation ultimately traced the *original*
number to a data bug, not a methodology regression. The full account is below, including the
steps that didn't pan out, because that's a more honest record of the platform's research
process than reporting only the parts that resolved cleanly.

*1. The original finding.* On the original 41-fund universe under FF4 (market, size, value,
momentum), the signal backtested at 3-month IC +0.061, t-stat 3.24 — a strong, statistically
significant result across all horizons. This became the benchmark every subsequent change was
measured against.

*2. The investigation.* Two later changes both failed to reproduce or improve on +0.061.
Upgrading the skill regression to FF7 (adding RMW, CMA, and a proprietary Gross Profitability
factor) dropped 3-month IC to +0.007 (t=1.32) on the same 41-fund universe. Expanding the
fund universe from 41 to 60 (adding sovereign wealth funds, insurance portfolios, endowments,
and long-tenured asset managers — see Module 3) left IC flat at +0.008 regardless of factor
model. A controlled isolation test — three separate backtests, each adding exactly one of
RMW, CMA, or GP to FF4 — found all three degraded IC to the same ~0.0075, ruling out any
single factor's data quality or construction as the cause. A follow-up test restricting the
60-fund universe back down to (approximately) the original 41 funds, still under FF4, *also*
failed to reproduce +0.061 (it came back at +0.006, t=1.20) — ruling out fund-universe
composition as the cause too. Neither of the two most obvious explanations held up.

*3. The root cause.* With both hypotheses ruled out, the next step was a git history review:
find the commit where +0.061 was originally measured, and diff everything that changed in the
data pipeline since. `convergence.py`, `signal.py`, and `backtest.py` — the actual signal
computation — were unchanged. But `smart_money/prices.py` had picked up a fix, made in the
same commit as the fund-universe expansion, for dual-class tickers: OpenFIGI/Bloomberg-style
slash notation (`BRK/A`, `BRK/B`, `BF/A`, `BF/B`) was being passed straight to yfinance, which
requires hyphens (`BRK-A`, `BRK-B`) instead. Before the fix, any position in one of these
tickers silently failed to fetch a price.

*4. Why this matters.* Berkshire Hathaway and Brown-Forman are exactly the kind of
widely-held, long-duration value/quality positions this platform's fund universe is built
around. Checking actual holdings data: 63 distinct (ticker, fund) pairs across at least 15 of
the original 41 funds hold one of these four tickers — including Dodge & Cox, First Eagle,
Southeastern Asset Management, Yacktman, Horizon Kinetics, Coatue, Maverick, Two Sigma,
Pershing Square, Greenlight, AQR, Acadian, D.E. Shaw, Renaissance, and PDT Partners. For any
quarter one of these funds held a dual-class position, the return reconstruction either
silently dropped below the 80% coverage gate (excluding that quarter from the regression
entirely) or produced a quarterly return missing that position's contribution outright.

*5. The honest conclusion.* The original +0.061 was very likely inflated by this silent data
incompleteness — not a fair target for later changes to be judged against. On the current,
corrected, complete data, FF4 and FF7 backtest as **statistically indistinguishable** from
each other (~0.008 IC either way). That is a legitimate, if less exciting, empirical result:
neither the FF4→FF7 upgrade nor the 41→60 fund expansion actually broke anything — they were
compared against a number that shouldn't have been trusted as a baseline in the first place.

*6. The final architecture decision.* Module 3/4 (fund skill scoring and the convergence
signal) uses **FF4**. This is a parsimony decision, not a superiority claim — on the
corrected data, FF7's three additional factors earn no measurable improvement in this
module's predictive power, so the simpler, more interpretable model is preferred by default.
If future evidence shows RMW, CMA, or GP add real value here, that would need fresh validation
before changing this again — see CLAUDE.md for the explicit warning against casually
re-upgrading this module without re-running the isolated single-factor backtest diagnostic
that was built for exactly this purpose.

*7. Module 2 keeps FF7.* Portfolio-level risk characterization (`factor_engine/portfolio.py`,
the Factor tab) is architecturally unrelated to Module 3/4 — it analyzes one portfolio's long
daily return history rather than dozens of funds' short quarterly panels, so it doesn't share
the same degrees-of-freedom constraint that made FF7 problematic for skill scoring. More
factors there add genuine risk-decomposition value (AQR independently validates significant
RMW/momentum loadings — see Module 1) without the same statistical penalty. The platform now
deliberately runs two different factor-model complexities for two different jobs, rather than
one unified factor count across the whole system.

**Beyond FF4 vs FF7: three tests of whether the signal can be improved further**

With FF4 established as the production skill-scoring model and the baseline IC
characterized (+0.008, not statistically significant, per the table above), the natural
next question is whether some *other* change to signal construction could improve on
that baseline. This investigation tested three independent, structural hypotheses — each
isolating exactly one variable, evaluated against the same FF4 production configuration,
the same 60-fund universe, and the same convergence/NLP blending logic used everywhere
else on this platform.

*Hypothesis 1 — does the signal predict better at longer horizons?* The production
backtest measured IC at 1/3/6-month horizons (table above). This test extended it to
8/10/12 months to see whether predictive power was still building, or had already peaked.

| Horizon | Universe | Mean IC | t-stat | Hit rate |
|---------|----------|---------|--------|----------|
| 1-month | full | +0.008 | +1.42 | 66% |
| 1-month | watchlist | +0.006 | +1.19 | 62% |
| 3-month | full | +0.008 | +1.52 | 57% |
| 3-month | watchlist | +0.007 | +1.52 | 53% |
| 6-month | full | +0.005 | +0.98 | 56% |
| 6-month | watchlist | +0.006 | +1.11 | 56% |
| 8-month | full | +0.006 | +1.25 | 47% |
| 8-month | watchlist | +0.006 | +1.11 | 47% |
| 10-month | full | +0.002 | +0.35 | 51% |
| 10-month | watchlist | +0.001 | +0.25 | 49% |
| 12-month | full | -0.001 | -0.14 | 59% |
| 12-month | watchlist | -0.002 | -0.29 | 54% |

Predictive power does not improve with horizon — it decays. IC/t-stat peak near 1-3
months and fade toward (and, at 12 months, slightly past) zero. This is a real decay in
signal, not a shrinking-sample artifact: average observations per quarter fall only ~3.5%
from the 1-month horizon to the 12-month horizon, while the t-stat collapses from +1.42
to -0.14 over the same range. A 45-day-lagged quarterly positioning signal apparently
carries real, if weak, information about the near-term (1-3 month) forward path, and that
information decays rather than compounding at longer horizons.

*Hypothesis 2 — does filtering out small, routine position changes improve the signal?*
Production already down-weights small `INCREASED`/`DECREASED` position changes
continuously (a 5% trim contributes at 0.55× weight, a 100%+ change at 0.90×, via a tanh
curve — see "Fund skill weight tiers" above). This test asked whether a *hard* cutoff —
excluding small changes from the convergence calculation entirely, rather than
down-weighting them — would sharpen the signal. Tested at 10% and 25%
`|shares_pct_change|` thresholds, 1- and 3-month horizons, both universes:

| Threshold | Coverage lost (full) | Coverage lost (watchlist) | IC / t-stat range across both horizons |
|-----------|----------------------|----------------------------|------------------------------------------|
| 10% | -7.6% signals | -10.2% signals | +0.006 to +0.007 (baseline: +0.007 to +0.008) |
| 25% | -15.1% signals | -21.2% signals | +0.006 to +0.008 (baseline: +0.007 to +0.008) |

Neither threshold produced a reliable IC or t-stat improvement over the baseline's
continuous down-weighting — results stayed within noise of each other regardless of
threshold. Meanwhile the coverage cost was real: the 25% threshold discarded roughly a
fifth of watchlist-universe signals. Hard-filtering by trade size trades away coverage for
no measurable benefit; the existing continuous weighting already captures what a size cutoff
would add.

*Hypothesis 3 — does restricting to only the highest-skill-tier funds improve the signal?*
Production weights every fund continuously by skill (0.10×-3.00× for scored funds,
0.40×-0.90× bucket defaults for unscored funds — see "Fund skill weight tiers" above).
This test asked whether *excluding* lower-tier funds from the convergence calculation
entirely, rather than continuously down-weighting them, would sharpen the signal. Tested
at the 1-month horizon (the strongest horizon per Hypothesis 1), three tiers:

| Tier | Funds | Full IC / t-stat | Watchlist IC / t-stat |
|------|-------|-------------------|-------------------------|
| Baseline (all funds, continuous weighting) | 61 | +0.008 / +1.41 | +0.006 / +1.19 |
| High confidence only (abs(α t-stat) > 1.5, ≥12 quarters) | 7 | -0.001 / -0.18 | +0.001 / +0.11 |
| Positive alpha only | 28 | +0.005 / +0.97 | +0.003 / +0.56 |
| High confidence AND positive alpha | 3 | +0.030 / +1.53 | NaN (see below) |

The first two tiers underperform baseline on every metric. "High confidence" is a
statistical-*precision* cut, not a skill-*direction* cut — of the 7 funds, 4 have
confidently *negative* alpha, so the tier mixes genuinely-skilled and
genuinely-unskilled-but-precisely-measured funds, sacrificing baseline's diversification
benefit without buying quality. The third tier's full-universe result (+0.030 IC) is the
single best number produced anywhere in this investigation — but it should not be read as
a real finding. It rests on just 66 observations per quarter versus baseline's 2,264 (a
~34x smaller sample), drops one quarter outright, and its watchlist-universe counterpart
produced a mathematically undefined result (too few candidates in at least one quarter for
rank correlation to be computed at all). This is a textbook small-sample mirage: an
attractive point estimate that collapses under the same restriction applied to a
neighboring universe. The continuous skill-weighting already in production handles this
tradeoff more robustly than any hard skill-tier cutoff tested.

**Methodology integrity check: no look-ahead bias.** Before trusting any of the above, the
backtest's return-measurement logic was independently verified. A 13F is not public on its
quarter-end date — funds have up to 45 calendar days to file. The backtest's
`knowledge_date()` function (`smart_money/backtest.py`) adds that 45-day SEC filing lag to
the quarter-end date, and forward returns are measured from the first tradeable price *on
or after* that disclosure date — never from quarter-end itself. Confirmed directly in code
(`_entry_and_exit` sources its price-lookup start date from `knowledge_date(period)`, not
`period`) and empirically (a spot-checked observation for period 2013-06-30 entered at
2013-08-14 — the full 45-day gap). All IC figures in this investigation, and in the
FF4/FF7 investigation above, are computed on this look-ahead-safe basis.

**Overall conclusion.** None of the three structural changes tested — longer horizons,
harder trade-size filtering, harder fund-tier restriction — meaningfully improves the
signal's predictive power over the existing FF4 production configuration's continuous
weighting scheme. In each case, the continuous, smooth approach already in production
(down-weight rather than exclude) outperformed or matched every hard-cutoff alternative
tested, while hard cutoffs consistently cost real signal coverage. Taken together, this
suggests the current architecture is near the practical ceiling for what a single
quarterly institutional-positioning signal, structurally bound to a 45-day reporting lag,
can predict.

**Path forward.** Meaningfully improving the platform's overall predictiveness from here
likely requires adding genuinely *independent* signals rather than continuing to tune this
one — e.g. earnings-surprise / post-earnings-announcement drift, sector-rotation context,
or extending the existing NLP infrastructure toward broader news/sentiment analysis. This
mirrors how real institutional quant strategies operate: combining many weakly-predictive,
independent signals compounds into something more useful than exhaustively tuning a single
dominant one. That is a larger scope of work than a backtest parameter sweep, and is noted
here as a direction rather than committed to as a roadmap item.

**Past performance does not guarantee future results.** These figures are computed over a
historical sample and reflect the specific universe, weighting scheme, and time period
tested. They do not account for transaction costs, slippage, or taxes, and a signal that
was significant historically can degrade or fail going forward as market conditions,
factor crowding, or fund behavior change.
""")

# ---------------------------------------------------------------------------

st.subheader("PEAD — a second, independent signal")

st.markdown("""
The "Path forward" note above pointed at earnings-surprise drift as the natural next step
once the 13F positioning signal appeared to be near its practical ceiling. This section
covers that build: a genuinely independent signal, from a different data domain (earnings
surprises, not institutional positioning), sharing no infrastructure with Module 3/4 beyond
the general backtest methodology.

**Methodology.** SUE — Standardized Unexpected Earnings (Bernard & Thomas, 1989/1990) — the
specific, well-established construction from the PEAD literature, computed over the same
~1,500-ticker S&P Composite 1500 universe the GP factor uses:

```
SUE = (eps_surprise_dollar − mean(prior 8 quarters)) / stddev(prior 8 quarters)
```

standardized against the *dollar* surprise (actual − estimate), not the percentage surprise,
because percentage surprise is unstable for near-zero consensus estimates (observed directly
in the data pull — a name with a $0.02–0.03 estimate swung between −157% and +256% surprise
in consecutive quarters). A rolling 8-quarter window is used rather than an expanding
all-history window, so a company's current surprise volatility isn't blended with a
structurally different volatility regime from years earlier. A 4-quarter minimum (not the
more conventional 8) governs when scoring is allowed to *start* — deliberately loosened
given yfinance's own history depth is the accepted limitation for this first pass. Tickers
without enough history fall back to a cross-sectional percentile rank of the percentage
surprise within the same calendar-quarter cohort, so thin-history names stay in the universe
rather than being dropped.

Two real yfinance data-quality issues surfaced during the build and are handled explicitly,
not silently: (1) yfinance rounds displayed EPS to 2 decimals, which can make a low-EPS
stock's entire trailing window read as an exact zero-surprise streak even though the
underlying `Surprise(%)` shows real movement — this correctly falls back to percentile
rather than producing an undefined/infinite z-score; (2) a small minority of rows report no
consensus estimate at all, yet yfinance still returns an implausible surprise percentage
(observed: 855%, 7900%, and even −4813% in real data) — these rows are excluded from scoring
entirely rather than allowed to pollute the percentile ranking.

Entry timing is anchored to the actual public announcement, classified per-row as
before-market (`bmo`, that day's close already reflects the news) or after-market (`amc`,
next trading day's close is the first to reflect it) from yfinance's announcement timestamp
— verified empirically consistent with known real-world release patterns before being
trusted (a same-day-reporting grocer consistently shows pre-market timestamps; two large-cap
tech reporters consistently show at-or-after-close timestamps).

**Backtest results (1,500-ticker universe, 70 quarterly cohorts spanning 1999–2026):**

| Window | Horizon | IC | t-stat | Hit rate |
|--------|---------|-----|--------|----------|
| **2014Q2+ (primary)** | 1-month | +0.0206 | 3.14 | 69.4% |
| **2014Q2+ (primary)** | 3-month | +0.0176 | 2.56 | 69.4% |
| Full range 1999–2026 | 1-month | +0.0488 | 2.56 | 68.6% |
| Full range 1999–2026 | 3-month | +0.0251 | 1.79 | 68.6% |

Two windows are reported for the same reason the +0.061 figure above is not treated as the
benchmark to chase: cohorts from 2009–2013 clear the minimum-observation gate for a computed
IC, but are built from only 10–59 tickers (whichever legacy names happen to have that much
history), not the full universe — full-universe coverage (1,000+ tickers/quarter) only
starts at 2014Q2. Restricting to 2014Q2+ actually *strengthens* the t-stat despite a lower
raw IC, because it removes a noisy, non-representative early period rather than genuine
signal. **The 2014Q2+ figures are the primary, citable result**; the full-range figures are
kept for the record, not as the headline.

Both windows clear the decision gate this signal was built against (IC 0.02–0.03, t-stat
1.5–2.0), and the aggregate IC held up under individual-cohort spot-checks — real per-stock
score-vs-return examples, not just the summary statistic. A pre-COVID-crash cohort
(announcements Jan–Mar 2020) showed a *negative* IC because a systematic market shock swamped
the idiosyncratic surprise effect entirely; the following COVID-recovery cohort (Apr–May 2020)
showed a cleaner positive pattern; two calmer 2024–2025 cohorts showed the honest, modest
picture a 0.02–0.05 IC actually looks like at the individual-stock level — real population-level
rank correlation across ~1,400 names per quarter, not a stock-picking guarantee on any one name.

**Why this stays yfinance-sourced: an EDGAR extension was investigated and closed, not
skipped.** The GP factor's success migrating from yfinance to SEC EDGAR XBRL data raised the
obvious question of whether the same move would deepen PEAD's history. It doesn't, and the
reason is structural rather than a gap in effort: SEC filings (and their XBRL tags) are a
company's *self*-disclosure of its own financials — actual EPS is confirmed present in the
same cached companyfacts JSON already fetched for the GP factor, requiring zero new EDGAR
calls. But analyst *consensus estimates* are a third-party commercial data product (I/B/E/S,
Zacks, FactSet), compiled by surveying sell-side analysts — there is no EDGAR form or XBRL
concept for it, because it isn't something the filer reports about itself. A surprise needs
both an actual and an estimate for the same quarter, so EDGAR alone cannot extend usable
history no matter how it's used.

Two alternative estimate sources were tested empirically, not just checked in documentation.
Alpha Vantage's EARNINGS endpoint returned genuinely deep, well-structured real data for a
live test ticker — consensus estimates, an explicit announcement date, and even an explicit
pre-market/post-market field, back to 1996 — but its free tier caps at 25 requests/day,
which would take roughly two months to pull this platform's ~1,500-ticker universe once;
a usable rate limit starts at $49.99/month. Financial Modeling Prep's free tier (tested with
a real registered account, not just a demo key) does include a genuine consensus estimate
field, but restricts free-tier symbol access to a small whitelist of mega-caps — several
mid/small-cap names tested were rejected outright regardless of how long one was willing to
wait, a harder blocker than a rate limit. Both vendors confirm the same structural point:
consensus-estimate data is a commercial product industry-wide, and free tiers are built as
tastes-testers, not project-scale data sources. **yfinance's roughly 10–12 year depth is the
practical free ceiling for this signal.** Alpha Vantage's paid tier remains a documented,
available option if circumstances change later — not something this platform pursues now.

PEAD is currently a standalone diagnostic result (`pead/` package, `scripts/run_pead_backtest.py`)
and is not yet blended into the production `FinalSignal` the Signals page surfaces.

**Past performance does not guarantee future results.** As with the figures above, these are
computed over a historical sample and do not account for transaction costs, slippage, or
taxes.
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
| SEC EDGAR XBRL `companyfacts` | Revenue/COGS/Assets for the proprietary GP factor (~1450 stocks post-exclusions) | 2013–2026 |
| Ken French Data Library | US FF6 (5-factor + momentum) daily factors for portfolio analysis (Module 2) and FF4 for fund skill scoring (Module 3/4) | Monthly updates |
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

*GP factor covers 2013–2026, sourced from SEC EDGAR XBRL, with 58 documented exclusions.*
The proprietary Gross Profitability factor is built from SEC EDGAR's XBRL `companyfacts`
API, giving it full 2013-present history — matching the platform's other 2013-era coverage
floors. An earlier version built from yfinance's free fundamentals endpoint was bounded to
roughly 2021-present by that data source's ~5-year statement window; that limitation no
longer applies. 58 tickers are excluded with documented per-ticker reasoning
(`factor_engine/gp_exclusions.py`) — 35 REITs (no COGS-equivalent concept exists for a
rental-income business model), 18 tickers with a pervasive company-specific cost-tag
mismatch (e.g. health insurers), and 5 unexplained magnitude mismatches — and every
remaining observation carries a `source` tag (`reported` / `derived_from_ytd_subtraction` /
`estimated_from_margin`) recording how it was derived. Module 3 fund skill scoring does not
use GP at all (see "Signal Validation" below for why) — GP remains part of Module 1's
individual-holding factor profiles and Module 2's portfolio-level analysis. Historical
stress tests still omit GP's contribution entirely for the 2008 and 2020 scenarios, which
predate even its extended coverage, rather than treating it as zero exposure.

*GP uses an invested-capital denominator, not raw Total Assets — built in two stages, with
mixed but net-positive results.* GP_ratio = (Revenue − COGS) / (Total Assets − Cash −
Short-Term Investments − Non-Interest-Bearing Current Liabilities − Goodwill − Intangible
Assets), where NIBCL = Accounts Payable + Accrued Liabilities. Stage 1 (the NIBCL
adjustment) was undertaken specifically hoping to fix MSFT's counter-intuitive negative
gross-profitability loading — it didn't; MSFT's loading moved further negative, not less.
It was kept anyway because it's more economically correct on its own terms: idle cash and
interest-free supplier financing aren't capital a business had to deploy to earn its gross
profit, so this avoids penalizing capital-light, high-cash names purely for holding cash,
and it correctly credits efficient negative-working-capital retailers (e.g. Kroger, and
McCormick) for a genuine capital efficiency rather than treating large accounts payable as
an ambiguous flaw.

Stage 2 (subtracting Goodwill + Intangible Assets) followed a read-only balance-sheet
diagnostic — no formula change, just pulling actual composition data already cached from
the same XBRL migration — which found MSFT carries goodwill + intangibles at ~20% of total
assets versus ~7% for AAPL and Kroger, roughly 3x, traceable directly to MSFT's acquisition
history (Activision Blizzard, LinkedIn, Nuance, GitHub) rather than AAPL's largely organic
balance sheet. That evidence justified extending the denominator, and this time MSFT
*did* improve substantially — its loading's magnitude fell 78% (from -0.111 to -0.024),
though it remains technically negative. That win came with real collateral cost: AAPL
moved further negative (partly because Apple itself stopped separately disclosing Goodwill
as a distinct XBRL concept after 2017, so it doesn't get the same denominator credit as
peers who still tag it), GOOGL flipped back negative, and both AMZN and NVDA moved further
from the "shouldn't be penalized for reinvestment" result the factor was designed to
deliver. XOM's negative loading nearly doubled in magnitude — checked directly and found
to be driven by its real energy-sector peers (already present in the low-quality quintile
under every version of this formula), not a data bug.

Both stages were kept after full disclosure of their tradeoffs: the formula is more
economically correct on its own terms, and the goodwill stage delivered real, if partial
and not free, progress on the problem it targeted. A single stock's factor loading depends
on its correlation with the long/short portfolio's returns, which is driven by relative
ranking across the ~1,450-ticker universe at every historical rebalance — so a
company-targeted fix predictably has spillover effects on other companies' loadings, in
either direction. Full before/after/after comparison across all three formula versions is
documented in `scripts/run_gp_sanity.py`.

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
