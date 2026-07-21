# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Retail Investor Platform** — a factor-based equity analysis and smart-money signal platform for retail investors. Built as a portfolio project demonstrating investment research methodology.

**Thesis:** Retail investors have access to fragmented institutional-style tools but none that (a) integrate across portfolio analysis, institutional positioning, and tax-aware execution, or (b) separate genuine stock-selection skill from sector/factor beta in 13F tracking data.

**Three-act demo structure:**
1. **See clearly** — Modules 1+2: factor exposure decomposition of real portfolio, historical stress tests
2. **Think ahead** — Modules 3+4: skill-weighted smart-money convergence signal + NLP filing language shifts
3. **Act efficiently** — Module 5: tax-lot-aware sell/harvest decision modeling

## Commands

Use .venv/bin/python for all commands — bare python is not in PATH.

Run all tests: .venv/bin/python -m pytest
Run single test: .venv/bin/python -m pytest tests/test_market_factor.py::test_beta_close_to_true_value
Run full pipeline: .venv/bin/python -m smart_money.pipeline
Pipeline with refresh: .venv/bin/python -m smart_money.pipeline --refresh
Single fund debug: .venv/bin/python -m smart_money.pipeline --fund "Viking Global"
Phase range: .venv/bin/python -m smart_money.pipeline --from-phase 3 --to-phase 4
Verification scripts: .venv/bin/python scripts/verify_nlp.py / verify_returns.py / verify_prices.py

## Architecture

Two independent subsystems share this repo:

factor_engine/ — Modules 1-2, Fama-French-Carhart 4-factor analysis of a retail portfolio. Pure analytics: reads CSV price data from data/, runs OLS regressions, no DB. Entry points: scripts/run_portfolio_analysis.py, scripts/run_market_factor.py.

smart_money/ — Modules 3-5, 13F smart-money signal tracker. Uses SQLite at data/module3.db (Peewee ORM, WAL mode, FK enforcement). All tables defined in models.py; init_db() must be called before any DB access.

## Module 1 — Factor Engine (shared core)

Self-constructed Fama-French-Carhart 4-factor model (market, SMB, HML, MOM) from free ETF proxies:
- Market: SPY vs ^IRX (3-month T-bill risk-free rate)
- SMB: IWM minus IWB — correlation ~0.85-0.90 with academic FF SMB
- HML: 4-ETF averaging IWD+IWN value minus IWF+IWO growth — correlation ~0.80-0.88 with academic FF HML
- MOM: MTUM minus IWB — correlation +0.71 with academic Carhart UMD (measured 2020-2024). Long-only-minus-benchmark, not a true long-short spread (no liquid "loser" ETF exists) — structurally weaker proxy than SMB/HML. MTUM inception (April 2013) bounds how far back this proxy can be computed. See factor_engine/factors/mom.py.

DO NOT fix the ETF proxy approach — deliberate design decision. Building from scratch rather than using paid providers (Barra, Axioma) is the stronger interview story. Correlation caveat is documented in code.

Portfolio-level analysis, stress tests, and fund skill scoring (Module 3) use Ken French's official daily factor series (factor_engine/french_data.py::get_ff4_daily(), merges the 3-factor file with French's separately-published momentum file) rather than the ETF proxies, for full history and accuracy — the ETF proxies above are used only for the individual-holding Factor Profile view (compute_factor_loadings()).

Ken French RF swap is flagged for polish pass — ^IRX used instead of canonical Ken French daily RF. Difference is negligible but swap planned before final launch.

## Module 2 — Portfolio Truth-Teller

Real portfolio: VTI 24.37%, QQQM 11.40%, SCHD 11.78%, VXUS 15.58%, NVDA 2.94%, GOOGL 5.18%, QTUM 10.27%, VTV 8.21%, XLI 5.12%.

VXUS uses Ken French Global FF3 factors (not US factors, and no regional momentum series exists) — labeled "US FF4 (intl. approx.)" in output. All other holdings use US FF4.

Portfolio-level analysis and stress tests run the full FF7 spec (market, SMB, HML, RMW, CMA, MOM, GP — factor_engine/portfolio.py, dashboard/factor.py), unlike Module 3/4 which uses FF4 (see the FF4-vs-FF7 note under Module 3 below for why the two modules deliberately differ). AQR independently validates this module's RMW/momentum loadings as significant.

Stress test methodology: historical factor shock replay against current betas. This is risk characterization, NOT a backtest. Label accordingly in all output.

**GP factor invested-capital refinement (Module 2 only — Module 3/4 no longer uses GP at all).**
GP_ratio = (Revenue - COGS) / (Total Assets - Cash - Short-Term Investments - NIBCL),
where NIBCL (Non-Interest-Bearing Current Liabilities) = Accounts Payable + Accrued
Liabilities, not short-term debt (factor_engine/gp_fundamentals.py). Replaces the original
Total-Assets-only denominator, which penalized capital-light, high-cash businesses
(AAPL/MSFT) and ambiguously treated efficient supplier-financed working capital (KR) as a
flaw rather than the legitimate capital efficiency it is. All tags (Cash, Short-Term
Investments, AP, Accrued Liabilities) were already present in the cached raw XBRL
companyfacts JSON from the original GP/XBRL migration — this refinement required zero new
EDGAR pulls, only re-deriving data/gp/fundamentals/*.csv from the existing cache.
Per-observation `nibcl_source` tracks tag-completeness tier (full / ap_only / none — 20.3%
of the ~1500-ticker universe has no parseable AP/accrued/combined tag at all and defaults
NIBCL to 0 rather than being dropped). `_MAX_PLAUSIBLE_GP_RATIO` in gp_exclusions.py was
recalibrated from 1.0 to 2.5 (empirically reproducing the original threshold's ~0.87%
row-level exclusion rate on the new formula's shifted distribution, not guessed).

**Negative result: did not fix the motivating problem.** This refinement was undertaken
specifically hoping to improve MSFT's beta_gp ranking. Re-running the 2022-2024 directional
sanity check (scripts/run_gp_sanity.py) after the rebuild found MSFT's beta_gp moved
*further* negative (-0.077 → -0.111), not less — same direction of failure, larger
magnitude. AAPL moved the same way (-0.036 → -0.071). The likely mechanical reason: beta_gp
is a regression against the long/short quintile portfolio's daily returns, driven by
relative cross-sectional ranking across the full ~1450-ticker universe at every historical
rebalance — not by one company's own gp_ratio level. Every capital-light company's ratio
rises under the new denominator, not just MSFT's, so a single company's relative rank isn't
guaranteed to improve just because its own ratio improved in isolation. Kept anyway: the
formula is still more economically correct on its own terms (matches the standard
invested-capital convention; KR/MKC's positive loadings are now understood as genuine
negative-working-capital efficiency being rewarded, not something to "fix" — see
run_gp_sanity.py, which was updated to reflect this and no longer expects KR to score
negative). GOOGL crossed from negative to slightly positive; NVDA and XOM both moved
substantially less-negative. The AAPL/MSFT discrepancy remains open and undocumented-away —
flagged in run_gp_sanity.py's output, not silently accepted.

**Goodwill/intangibles extension — mixed result, kept for a real but partial win.**
Before building further, a read-only diagnostic (no formula change) pulled actual balance
sheet composition for AAPL/MSFT/KR as % of total assets, directly from the same cached raw
XBRL — no new EDGAR calls. Finding: MSFT carries Goodwill + Intangible Assets at ~20% of
total assets vs ~7% for AAPL and KR — roughly 3x, tracing to MSFT's acquisition history
(Activision Blizzard, LinkedIn, Nuance, GitHub) vs AAPL's largely organic balance sheet.
This was plausible enough to justify extending the denominator: GP_ratio = (Revenue - COGS)
/ (Total Assets - Cash - Short-Term Investments - NIBCL - Goodwill - Intangible Assets).
Tag coverage is cleaner than NIBCL's (Goodwill 89.4% ever / 85.5% recent, Intangibles 89.5%
ever / 80.9% recent, no combined tag exists anywhere in the universe) but surfaced a
material gap: AAPL's `Goodwill` XBRL tag has zero entries after 2017 — Apple stopped
separately disclosing goodwill as a discrete concept (58 tickers, 3.9% of the universe,
show this same "tagged historically, stopped recently" pattern). Per-observation
`goodwill_source` tracks completeness (full / partial / none); `_MAX_PLAUSIBLE_GP_RATIO`
recalibrated a second time, 2.5 → 4.5, same empirical-exclusion-rate-matching method as the
first recalibration.

Result: **MSFT genuinely improved** — beta_gp -0.111 → -0.024, a 78% reduction in magnitude
(still technically negative, but close to neutral for the first time). This was real,
targeted progress on the motivating problem, not another miss. But it came with real
collateral cost: AAPL moved further negative (-0.071 → -0.171, partly because AAPL's own
goodwill-tag gap means it doesn't get the same denominator credit as peers who still tag
Goodwill), GOOGL flipped back negative (+0.011 → -0.090), AMZN moved much more negative
(-0.048 → -0.284, now failing the reinvestor check it used to narrowly pass), and NVDA moved
more negative (-0.492 → -0.811). XOM's magnitude nearly doubled (-0.594 → -1.147) — this was
investigated directly (compared short-basket composition and overall factor volatility
between the NIBCL-only and goodwill-extended versions) and found to be **not a bug**: XOM's
actual energy-sector peers (EQT, HLX, VAL, KNTK, CVI, PBF) have consistently occupied the
bottom GP quintile across every version of this formula — the same "capital-intensive
commodity businesses score low" pattern, amplified, not new or spurious.

**Kept anyway**, per explicit decision after reviewing the full mixed picture: the formula
is more economically correct on its own terms, and it measurably helped the specific problem
it was built to address, even though the fix didn't isolate to just its target. `beta_gp` is
driven by cross-sectional rank across the ~1450-ticker universe at every historical
rebalance, not by one company's own ratio in isolation — a company-specific denominator fix
predictably has spillover effects on other companies' loadings, in either direction. Full
before/after/after comparison across all three formula versions is in run_gp_sanity.py's
module docstring.

## Module 3 — EDGAR Ingestion and Fund Skill

Pipeline phases: edgar → cusip → prices → returns → skill

Key constraints:
- Pre-2013-06-01 filings are plain-text EDGAR format, skipped via _XML_CUTOFF
- Quant/systematic funds: only top-200 positions by USD value (rank_by_value <= 200) are is_price_eligible
- Coverage gate for returns: 80% of BOQ positions must have prices; first filings yield None

CRITICAL DATA FACTS — do not change without understanding why:
- EDGAR 13F value field is in RAW DOLLARS, not thousands. Empirically verified: Viking/Visa raw XML value 1912630634 implies $302.24/share matching actual price. SEC spec says thousands but actual filings use raw dollars.
- EDGAR filing index uses -index.html not -index.json (JSON endpoint does not exist)
- iXBRL viewer URLs (/ix?doc=...) must be stripped via _unwrap_ix() before fetching raw HTML
- Namespace detection must handle both informationTable (mixed case) and informationtable (lowercase) across filers and eras
- Viking Global CIK is 1103804, not 1166928 (which is West Bancorporation — wrong entity caught during verification)
- Dual-class tickers (BRK/A, BRK/B, BF/A, BF/B, etc.) use OpenFIGI/Bloomberg slash notation in Security.ticker, but yfinance requires hyphens (BRK-A). smart_money/prices.py::_to_yfinance_symbol() translates at the price-fetch boundary — do not pass Security.ticker straight to yfinance. Before this fix existed, positions in these tickers silently failed to fetch a price, which understated or excluded affected funds' quarterly returns without erroring. This is the confirmed root cause of an early backtest figure (3-month IC +0.061, t=3.24, see below) that could not be reproduced after the fix landed — the fix is correct; +0.061 was measured on incomplete data, not a target to chase back.

Fund skill regression is FF4 (market, size, value, momentum — Fama-French-Carhart, adds momentum to the historical FF3 spec) — see smart_money/factor_apply.py. Momentum matters most for growth/momentum-tilted managers (e.g. Greenoaks, Altimeter): under FF3 their return from holding recent winners was mis-attributed to alpha; FF4 captures it as beta_mom instead.

**FF4 (Module 3/4) vs FF7 (Module 2) — deliberate split, do not unify without re-validation.**
Module 3/4 skill scoring ran the fuller FF7 spec (adds RMW, CMA, and the proprietary GP
factor — see Module 2 below) for a period, then reverted to FF4. The investigation (full
account in app_pages/about.py's "Signal Validation" section):
1. FF4 on the original 41-fund universe backtested at 3-month IC +0.061 (t=3.24) — the
   number every later change was compared against.
2. Neither the FF4→FF7 upgrade nor the 41→60 fund-universe expansion reproduced or improved
   on it (both landed around IC +0.008 regardless of factor model).
3. A controlled isolation test — three backtests, each adding exactly one of RMW/CMA/GP to
   FF4 (smart_money/factor_apply_diagnostic.py + scripts/run_ff4_plus_one_diagnostic.py) —
   found all three degraded IC to the same ~0.0075, ruling out any single factor's data
   quality/construction as the cause.
4. A follow-up test restricting the 60-fund universe back to the original 41
   (scripts/run_41fund_ff4_diagnostic.py) also failed to reproduce +0.061 (+0.006, t=1.20),
   ruling out fund-universe composition too.
5. Git history review found the real cause: +0.061 was measured before the dual-class-ticker
   price-fetch fix above existed. 63 (ticker, fund) pairs across >=15 of the original 41
   funds hold BRK/A, BRK/B, BF/A, or BF/B — including Dodge & Cox, First Eagle, Southeastern,
   Yacktman, Horizon Kinetics, Coatue, Maverick, Two Sigma, Pershing Square, Greenlight, AQR,
   Acadian, D.E. Shaw, Renaissance, PDT Partners. Before the fix, those positions silently
   failed to fetch a price, corrupting affected funds' quarterly return reconstruction.

Conclusion: +0.061 was likely inflated by silent data incompleteness, not a fair baseline.
On corrected data, FF4 and FF7 are statistically indistinguishable for Module 3/4 (~0.008 IC
either way) — **FF4 is kept for parsimony, not proven superiority.** Module 2 keeps FF7
(portfolio-level risk characterization over one long daily return history has no comparable
per-fund degrees-of-freedom constraint, and AQR independently validates significant
RMW/momentum loadings there — see Module 2 below). **Do not casually re-upgrade Module 3/4
back to FF7** without re-running the isolated single-factor backtest diagnostic — the
evidence that FF4 is the right choice here is specific and already established; reintroducing
RMW/CMA/GP without fresh evidence would repeat a change already shown not to earn its keep.

Skill score thresholds: MIN_QUARTERS_REG=8, MIN_QUARTERS_RELIABLE=12. Alpha is a directional decomposition signal NOT a significance test. Always show t-stat and confidence_label alongside alpha.

## Module 4 — Convergence Signal and NLP

Fund skill weight tiers — DO NOT CHANGE without understanding rationale:
- Scored, reliable, positive alpha: clamp(1.0 + alpha_annualized/0.20, 0.5, 3.0)
- Scored, reliable, negative alpha: same formula, clamps at 0.10
- Scored, unreliable (<12q): 0.80
- Unscored long_short_equity: 0.85
- Unscored fundamental_value: 0.90
- Unscored sector_specialist: 0.85
- Unscored quant_systematic: 0.40 (13F is a hedged book, not directional)

NLP scorer (nlp.py): claude-sonnet-4-6 via Batch API (asynchronous, 50% cheaper). 7 dimensions with weights: guidance 0.25, confidence 0.20, customer_demand 0.20, competitive_positioning 0.15, operational_efficiency 0.10, risk_factors 0.05, capital_allocation 0.05. Cache key: (ticker, accession_current, accession_prior, scorer_version). Bump SCORER_VERSION to force rescore.

Signal blending (signal.py): 70% convergence + 30% NLP. Contradiction override: when |conv| >= 0.25 AND |nlp| >= 0.25 AND opposite signs, use (0.85 x conv + 0.15 x nlp) x 0.80. EXIT SIGNAL only fires when a prior FinalSignal row exists for the CUSIP. Always use display_name property (ticker ?? issuer_name) for UI display, never .ticker directly. _SHORT_NAMES dict handles TwoSigma, DEShaw, D1, LightSt.

## PEAD Signal — Post-Earnings-Announcement Drift (new, standalone)

A third independent research pillar (`pead/` package), alongside factor_engine/ and
smart_money/ — motivated by the Module 3/4 signal-improvement investigation (see
app_pages/about.py "Path forward") finding the 13F positioning signal had hit its practical
ceiling. Genuinely independent data domain (earnings surprises, not institutional
positioning); shares no infrastructure with Module 3/4 beyond backtest methodology. **Not
yet wired into signal.py's blend** — a standalone diagnostic result.

Universe: same ~1,500-ticker S&P Composite 1500 universe as the GP factor
(factor_engine/gp_universe.py, reused directly via pead/universe.py — no re-scrape).

Data source: yfinance `Ticker.get_earnings_dates(limit=50)` — pass a high limit explicitly,
the default limit=12 is shallow (~3yr); empirically most tickers return 40-50 quarters
(10+yr) of history. Cached per-ticker to data/pead/surprises/*.csv (pead/surprises.py).

SUE (Standardized Unexpected Earnings, Bernard & Thomas 1989/1990) construction
(pead/signal.py): `(eps_surprise_dollar - mean(prior window)) / stddev(prior window)`,
standardized against the *dollar* surprise, not percentage (percentage surprise is unstable
for near-zero consensus estimates — empirically confirmed, e.g. a name swinging between
-157%/+256% surprise on a $0.02-0.03 estimate). WINDOW=8 trailing quarters (rolling, not
expanding — avoids blending a company's early/mature volatility regimes). MIN_QUARTERS=4
(not the more conventional 8) is the floor for when scoring may *start*, deliberately
loosened given yfinance's shallow-history limitation is already accepted for this pass — do
not conflate this floor with WINDOW, which is a separate ceiling on how much history feeds
the std/mean (confirmed intentional, not a bug, after explicit user review). Tickers below
MIN_QUARTERS fall back to cross-sectional percentile rank of `eps_surprise_pct` within the
same calendar-quarter cohort (`score_method` column records which path each row took).

Two real yfinance data-quality issues found and handled explicitly (do not "fix" by
reverting to raw percentage surprise — that's the less robust construction, not a fix):
(1) yfinance's displayed EPS Estimate/Reported EPS are rounded to 2 decimals, producing a
spurious zero-variance trailing window for low-EPS stocks (e.g. NVDA 2014-2016) even though
the underlying Surprise(%) shows real movement — correctly falls back to percentile via the
std>0 guard, not a crash. (2) A minority of rows have a null eps_estimate but yfinance still
returns an implausible Surprise(%) anyway (observed: 855%, 7900%, -4813%) — these are
excluded from scoring entirely (`score_method="no_estimate"`) and from the percentile
ranking pool, not allowed to pollute it.

Entry timing (pead/backtest.py `entry_date()`): session classified per-row (not per-ticker)
from yfinance's announcement hour — `bmo` (hour<16 ET, that day's close already reflects the
news) vs `amc`/`unknown` (hour>=16 ET or missing, next trading day's close is first to
reflect it — "unknown" defaults conservatively to amc, never risking same-day look-ahead).
Verified empirically consistent with real-world release patterns before being trusted.

Backtest (pead/backtest.py, mirrors smart_money/backtest.py's Spearman IC / coverage-gate /
t-stat methodology, grouped by `quarter_cohort` — calendar quarter of announcement date —
instead of a 13F period, since PEAD events scatter across company fiscal calendars rather
than sharing one quarter-end grid). Horizons: 21/63 trading days (1mo/3mo) only, per this
signal's first-pass scope.

**Result (1,500-ticker universe, 70 quarterly cohorts, 1999-2026) — two windows reported,
2014Q2+ is primary:**

| Window | Horizon | IC | t-stat | Hit rate |
|--------|---------|-----|--------|----------|
| 2014Q2+ (primary) | 1mo | +0.0206 | 3.14 | 69.4% |
| 2014Q2+ (primary) | 3mo | +0.0176 | 2.56 | 69.4% |
| Full range 1999-2026 | 1mo | +0.0488 | 2.56 | 68.6% |
| Full range 1999-2026 | 3mo | +0.0251 | 1.79 | 68.6% |

Full-range cohorts from 2009-2013 clear MIN_COHORT_OBS=10 but are computed from only 10-59
tickers (whichever legacy names have that much yfinance history), not the full universe
(full coverage starts 2014Q2, 1,000+ tickers/quarter) — this inflates the full-range mean IC
while widening its variance. Restricting to 2014Q2+ *strengthens* the t-stat despite a lower
raw IC. **Cite the 2014Q2+ figures as the primary result** — same "don't chase an
unrepresentative baseline" lesson as the Module 3/4 +0.061 investigation above. Both windows
clear the pre-committed decision gate (IC 0.02-0.03, t-stat 1.5-2.0). Validated via
individual-cohort spot-checks (scripts/spotcheck_pead_cohorts.py) — e.g. 2020Q1 (pre-COVID
crash) showed a *negative* IC because a systematic market shock swamped the idiosyncratic
surprise effect, while 2020Q2 (COVID recovery) showed a clean positive pattern — real
per-stock examples, not just the aggregate statistic.

**EDGAR extension investigated and closed — do not re-attempt without a new estimates
source.** The GP factor's yfinance-to-EDGAR migration raised the obvious question of
whether PEAD's history could deepen the same way. It structurally cannot: SEC/XBRL data is a
company's self-disclosure of its own financials (actual EPS confirmed present via
us-gaap:EarningsPerShareBasic/Diluted in the *already-cached* GP factor companyfacts JSON —
zero new EDGAR calls needed), but analyst *consensus estimates* are a third-party commercial
product (I/B/E/S, Zacks, FactSet) with no EDGAR form or XBRL concept, because they aren't
something the filer reports about itself. A surprise needs both sides for the same quarter,
so EDGAR alone cannot extend usable history regardless of how it's used.

Two alternative estimate sources were empirically tested (live API calls, not just
documentation): Alpha Vantage's EARNINGS endpoint returned genuinely deep data (consensus
estimate, explicit announcement date, explicit pre-market/post-market field, history to
1996) but its free tier caps at 25 requests/day (~2 months to pull the full ~1,500-ticker
universe once); a usable rate limit starts at $49.99/month. Financial Modeling Prep (tested
with a real registered account, not a demo key) has a genuine consensus-estimate field but
restricts free-tier symbol access to a small mega-cap whitelist — mid/small-cap tickers
(tested: HELE, CVI, HLX, KR) were rejected outright regardless of wait time, a harder
blocker than a rate limit. Conclusion: consensus-estimate data is a commercial product
industry-wide, not a gap specific to either vendor — **yfinance's ~10-12yr depth is the
practical free ceiling for this signal.** Alpha Vantage's paid tier ($49.99/mo) is a
documented, available option if circumstances change later; not pursued now.

## Composite Signal — 13F + PEAD Combination Test (standalone diagnostic, not wired in)

`scripts/run_composite_backtest.py` — tests whether combining the 13F skill-weighted
convergence signal (`ConvergenceScore.convergence_score`) and PEAD's SUE score produces
stronger predictive power than either alone. Motivated by the same "genuinely independent
signals" logic that produced PEAD in the first place.

**Combination method: fixed 50/50 average of percentile ranks, not a fitted weighting.**
Ranks computed within each quarter's paired cohort. Deliberately not a regression-fit or
IC-proportional weighting — fitting weights on this dataset would risk overfitting the
blend itself, the same discipline already applied to the FF4-vs-FF7 decision above and to
not chasing the +0.061 baseline. Zero parameters estimated from this backtest's own data.

**Universe alignment.** Intersection of tickers with a resolved ticker in that quarter's
`ConvergenceScore` and tickers with a scoreable PEAD event — 1,432 tickers, 58,185 paired
(quarter, ticker) observations across 48 quarters (2014Q2-2026Q1, matching PEAD's own
primary window). A ticker with only one of the two scores is excluded from that quarter
entirely — no imputation, same philosophy as every other coverage gate in this codebase.

**Timing alignment.** 13F signal for `period` isn't knowable until `period + 45 days`
(`smart_money/backtest.py::knowledge_date`). Each (period, ticker) is paired to the most
recent PEAD event already knowable by that date (`pead/backtest.py::entry_date`). The
forward-return window for all three backtest variants (13F-alone, PEAD-alone, composite)
starts from `max(13F knowledge_date, PEAD entry_date)` — the later of the two — so the
comparison isolates the effect of the score, not a difference in when the return clock
starts. This shared-anchor choice is specific to this comparison; it does not change either
signal's own canonical standalone backtest (`smart_money/backtest.py`, `pead/backtest.py`).

**Staleness bug found and fixed — 120-day cap on PEAD pairing.** The first pass paired each
quarter to the *most recent knowable* PEAD event with no floor on recency. A handful of
thinly-covered tickers (BFS, UVV) have almost no scoreable yfinance earnings history, so
pairing silently fell back to a single announcement from 2008/2012 and reused it across
dozens of unrelated later quarters — a stale, meaningless pairing. Fixed with a 120-day cap
(~1 quarter of slack past knowledge_date): if the only knowable event is staler than that,
the pairing is dropped (`_MAX_PAIRING_STALENESS_DAYS` in the script). Only ~1.5% of raw
pairs exceeded 120 days (median gap 17 days), so the fix barely moved the aggregate numbers
— but it was wrong before the cap existed, the same "exclude rather than fabricate"
standard as PEAD's own `no_estimate` exclusion.

**Corrected result:**

| Horizon | Variant | Mean IC | t-stat | Hit rate |
|---|---|---|---|---|
| 1-month | 13F-alone | +0.0060 | 0.89 | 50.0% |
| 1-month | PEAD-alone | +0.0062 | 0.82 | 54.2% |
| 1-month | Composite | +0.0089 | 1.33 | 58.3% |
| 3-month | 13F-alone | +0.0064 | 1.05 | 51.1% |
| 3-month | PEAD-alone | +0.0083 | 1.44 | 66.0% |
| 3-month | Composite | +0.0101 | 1.77 | 61.7% |

Composite beats both individually-recomputed baselines on t-stat at both horizons — the
pre-committed success bar (point IC alone is not sufficient given how much this project's
other backtests have moved on noise; see the +0.061 investigation). But t = 1.33-1.77 sits
close to, not clearly past, the decision-gate language used elsewhere (t-stat 1.5-2.0) —
closest at 3-month, short at 1-month.

**Both individual baselines are weaker here than their own documented standalone
figures** (13F ~0.006 vs. its usual ~0.008; PEAD ~0.006-0.008 vs. its 2014Q2+ primary
+0.02) — not a red flag, but a structural consequence of this test's constraints, not a
data problem:
1. The intersection universe filters toward exactly where PEAD is weakest: PEAD's drift is
   classically stronger in smaller, less-analyst-covered names, while the 13F intersection
   skews toward larger, actively institutionally-traded names by construction.
2. The shared entry timing delays PEAD's entry past its own tighter natural window — the
   13F 45-day lag is the binding constraint on the shared anchor 99.7% of the time (median
   17 days after the actual earnings date), likely after some drift has already played out.

**Conclusion: combination doesn't destroy value on this harder overlap subset, but is not
evidence that blending would lift the production pipeline's IC.** Both signals individually
look stronger on their own broader, natural universes than either does on the narrow
intersection this test had to restrict to. The honest reading is that these two signals
currently work better run separately than combined. **Not wired into `FinalSignal`** — kept
as a second standalone diagnostic alongside PEAD, same status.

## DCF Valuation Engine

A fourth independent research pillar (`dcf/` package), alongside factor_engine/,
smart_money/, and pead/ — projects unlevered free cash flow forward per-ticker and compares
the resulting per-share intrinsic value range to current market price. Feeds a valuation
badge into the Discovery/Watchlist signal cards and the existing Valuation tab
(app_pages/signals.py), alongside its current trading-multiples/FCF-yield/analyst-consensus
content.

**Data source: SEC EDGAR XBRL, reusing the GP factor's cache, not yfinance.** For any ticker
already in the GP universe, pulling the additional concepts DCF needs (EBIT, D&A, capex,
interest expense, total debt, effective tax rate, diluted shares — see dcf/fundamentals.py)
costs zero new EDGAR calls: it reads the same already-cached raw companyfacts JSON
(data/gp/xbrl_raw/{cik}.json) GP's own fetch populated, plus new tag-parsing logic. Revenue
and cash are read directly from GP's own cached fundamentals CSV rather than re-derived.
yfinance's ~5yr annual / ~5-quarter statement ceiling (documented under Module 2's GP factor
migration) was the deciding factor against using it here: 5 years of lookback is right at
that ceiling with zero cushion; XBRL clears it comfortably.

**FCF construction (approved simplification, stated explicitly, not silent):**
`FCF = EBIT x (1 - effective_tax_rate) + D&A - Capex`. Working-capital changes are NOT
modeled — a granular NWC build-up would inherit the same AP/accrued-liabilities XBRL
tag-coverage gaps the GP factor's NIBCL work already found (only 76%/65% standalone
coverage, 20% resolving neither). EBIT margin, D&A%, and capex% are each a blended 60% TTM /
40% trailing-3-year-average baseline, held flat across the full 10-year projection — one
blend per ratio, not a second fade curve stacked on top of the growth fade.

**Growth**: Year-1 growth = the company's own trailing 5-year revenue CAGR, clamped to
[-15%, +30%] to prevent one distortive year from being extrapolated indefinitely (same
discipline as PEAD's dollar-based SUE standardization). Linearly faded to a terminal growth
rate by year 10. Terminal growth = the current 10-year US Treasury yield
(dcf/wacc.py::fetch_risk_free_rate()) — a new fetch, deliberately distinct from the
platform's existing short-duration Ken French/^IRX RF used in factor regressions
(french_data.py), since a 10yr+ DCF cash flow stream should be discounted/faded against a
duration-matched rate, not a 3-month bill.

**WACC**: cost of equity via CAPM, reusing Module 1's own beta
(dashboard.factor.ticker_ff3_profile) rather than a second beta computation. Cost of debt =
interest expense / total debt, after-tax — deliberately simple for v1 (equity value is far
more sensitive to cost of equity than cost of debt in a properly-weighted WACC). Weighted by
market-cap equity / book-value total debt.

**Bull/Base/Bear, not Monte Carlo**: three explicit, labeled scenarios varying Year-1
growth — sampling from an assumed probability distribution would itself just be a guess,
implying false statistical rigor this model doesn't have.

**Clamp-collapse bug found and fixed.** Bull/bear originally derived their spread by adding
GROWTH_SPREAD to the already-clamped base growth value, reusing base's own clamp ceiling.
For a name whose raw growth already exceeds that ceiling — NVDA: 66.9% raw unclamped 5-year
revenue CAGR against a 30% base ceiling — base saturates at 30%, and bull (30% + 5pp,
reclamped to the same 30% ceiling) collapsed onto it: no bull/base differentiation for
exactly the highest-growth names, where a reader would most want to see the bull case
reasoned through. A purely proportional/multiplicative fix (bull = base x 1.3) was
considered and rejected: it has a real correctness problem, not just a stylistic one — for a
negative or near-zero base growth rate, multiplying by a factor > 1 either inverts the
direction (-5% x 1.3 = -6.5%, making "bull" *worse* than base) or barely moves. Fixed by
deriving the spread from the RAW unclamped CAGR (not the clamped base) and giving bull/bear
their own wider clamp bands (`BULL_GROWTH_CLAMP_MAX` / `BEAR_GROWTH_CLAMP_MIN`, +/-15pp
beyond the base band) — stays additive (simple, sign-robust) but fixes the actual bug. NVDA
now shows bear=base=30% (there is no more-bearish story the data supports — even
raw-minus-spread is still above the base ceiling) and bull=45% (genuinely differentiated,
reaching toward but still well below its true trailing growth).

**Business-model exclusions (`dcf/exclusions.py`) — different mechanism from GP's REIT/
insurer exclusions.** Banks, insurers, and REITs are excluded via live GICS sector/industry
classification (yfinance's `info` dict: `Financial Services`+`Bank`/`Insurance`, or
`Real Estate`+`REIT`), not a hand-maintained ticker list. GP's exclusions
(factor_engine/gp_exclusions.py) are a DATA-AVAILABILITY problem — no COGS-equivalent XBRL
concept exists at all for these business models. DCF's is a METHODOLOGICAL-VALIDITY
problem: EBIT resolves just fine for a bank, insurer, or REIT, so the engine would happily
produce a per-share number — that number would be conceptually invalid, which is worse than
missing data, since it's indistinguishable from a valid result. Bank capital structure is
regulatory- (Basel), not market-driven, breaking the CAPM/WACC framing; insurer float and
claims reserves function as operating and financing leverage simultaneously; REITs must
distribute ~90% of taxable income (breaking the "reinvest FCF for growth" framing) and their
large D&A doesn't track real economic depreciation of often-appreciating property. Standard
practice for all three is DDM/embedded-value/FFO-based valuation, not enterprise DCF.
Confirmed empirically (scripts/run_dcf_sanity.py) that GICS classification correctly
excludes JPM (bank) / MET (insurer) / O (REIT) while correctly NOT excluding UNH — a "health
insurer" colloquially, but GICS-classified Healthcare/Healthcare Plans, not Financial
Services, since its economics don't share banks/insurers' regulatory-capital-structure
problem.

**Two real XBRL tag bugs found and fixed during sanity testing**, both the same "filer
changes or drops a tag over time" pattern the GP factor's own docstring already documents
for Revenue (ASC 606 adoption wave) and Goodwill (AAPL stopped tagging it after 2017):
- Diluted shares outstanding resolved to nothing for every ticker — XBRL reports share
  counts under `units.shares`, not `units.USD`; the GP-factor helper reused for tag
  resolution only checks `units.USD`. Fixed with a local `_shares_facts_by_tag` helper
  rather than generalizing the shared GP helper (every other tag here is genuinely
  USD-denominated).
- AAPL's interest expense silently defaulted to $0, making its cost of debt come out as
  0.00% despite $90.68B of real debt. Root cause: AAPL's `InterestExpense`/
  `InterestExpenseDebt` XBRL tags have no entries after FY2023 — Apple folded interest
  expense into "Other income/expense, net" starting FY2024, the same kind of disclosure
  change as its Goodwill tag dropping off in 2017. Fixed with a carry-forward-with-
  source-flag (`interest_expense_source: reported | carried_forward | none`) rather than a
  silent zero, mirroring the `debt_source`/`tax_rate_source` pattern already used elsewhere.
- KO showed $0 total debt despite $1.65B of reported interest expense — implausible on its
  face, and confirmed as a real gap: KO switched XBRL tags in 2025 from
  `LongTermDebtNoncurrent`/`ShortTermBorrowings` to lease-inclusive
  `LongTermDebtAndCapitalLeaseObligations*`/`OtherShortTermBorrowings`. Fixed by adding the
  new-era tags as fallbacks in the existing per-period tag-priority resolution (same
  mechanism gp_fundamentals.py already uses for Revenue's ASC-606-era tag switch).

**Known limitation — terminal value dominance is a well-documented limitation of
single-stage DCF for mega-cap, wide-moat companies, not a bug in this implementation.**
Sanity-check results across real portfolio/watchlist holdings (AAPL, MSFT — real portfolio
positions; KO — mature slow-grower control; NVDA — real portfolio + watchlist "stretched
valuation" candidate), figures as of the sanity run and will drift with market prices:

| Ticker | Base case vs. current price | % of value from terminal value |
|--------|------------------------------|--------------------------------|
| AAPL   | -54%                         | 60% |
| MSFT   | -39%                         | 63% |
| KO     | +35%                         | 81% |
| NVDA   | -44% base / -11% bull        | 50-53% |

High terminal-value share (50-81% here) is not, by itself, unique to mega-caps — it's a
normal property of any 10-year DCF, and KO's own TV share is the highest of the four despite
its result matching intuition cleanly. What IS specific to mega-cap, wide-moat companies is
that this same structural TV-dependency interacts with a terminal growth rate deliberately
capped at a conservative risk-free-rate proxy (see Growth above) applied uniformly to every
company — while near-term explicit cash flows (the part of a DCF that's actually reliably
projectable from a company's own reported financials) are a comparatively small share of
total value for a business whose market price may reflect continued competitive moat, real
optionality, or capital-allocation flexibility (e.g. buyback-driven per-share compounding an
enterprise-value DCF doesn't model) extending well past a decade-then-fade-to-GDP story.

Checked against real data, not just theory: analyst consensus price targets (yfinance) for
AAPL sit modestly below current price too ($315.79 mean vs. $332.51 current) — a partial,
weaker echo of this model's direction, and a reverse-solve found even generous
analyst-forward growth estimates (21.8% earnings growth) fall well short of the ~30% growth
this model would need to match AAPL's current price, so the gap here looks more like a
genuine structural limitation (no buyback modeling) than an overly conservative assumption.
MSFT's picture is different and more concerning: the *entire* analyst consensus range
($400-$870) sits above where this model lands ($240.60) — a real, evidenced divergence, not
a hand-wave. The reverse-solve here is more diagnostic: MSFT's forward consensus growth
(18.3% revenue / 23.4% earnings) is meaningfully higher than this model's 14.5%
trailing-5yr-CAGR-based assumption, which is diluted by pre-AI-acceleration years and misses
a real recent inflection; compounding this, MSFT's capex-of-revenue baseline (21%, reflecting
current AI infrastructure buildout) is held flat for the full 10-year projection, a
meaningfully conservative choice for capex intensity that's widely expected to at least
partially normalize rather than persist a full decade.

**Practical recommendation: read this engine's output alongside relative valuation (peer
multiples / comps) for mega-cap names specifically, rather than as a standalone verdict** —
see the new Pre-Launch Polish List item below. This mirrors real institutional equity
research practice: DCF is one input among several (DCF, comps, precedent transactions,
sum-of-the-parts), each understood to carry different reliability by company type, not a
single number treated as ground truth.

## DCF Standalone Backtest — Pilot Result (inconclusive-leaning-negative; see Full-Universe Result below — this pattern did not hold up at scale)

Follow-on to the DCF Valuation Engine above: does the Base-case valuation gap (`(base_per_share
- price) / price`) actually predict forward returns? Same motivating question already asked of
13F convergence and PEAD, and same "genuinely independent signal" logic — but DCF needed new
infrastructure first, since `run_dcf()` only values a company **as of right now** (live price/
market cap, a beta from a fixed 2021-2024 window, current risk-free rate) — unlike 13F filing
periods or PEAD announcement dates, it has no time axis at all to backtest against.

**`dcf/backtest.py`** (new) builds a point-in-time replay: `compute_point_in_time_dcf(ticker,
as_of)` truncates fundamentals by **actual filing date** (not `period_end` — a fiscal year
isn't public the day it ends), reconstructs a trailing-window beta ending at `as_of` (same
joint FF7 OLS as `ticker_ff3_profile`, just parameterized by end date), and pulls historical
price and risk-free rate — then reuses `dcf/valuation.py`'s `compute_baseline`/`run_scenario`
unchanged. Score is Base case only, not a Bull/Base/Bear blend (approved — avoids stacking an
unvalidated blend-weighting choice on top of the reconstruction itself).

**Three real bugs found and fixed during this work, none of them in the DCF math itself:**

1. **Share-count/price basis bug (dcf/backtest.py).** `pead.prices`' cached price series is
   split-adjusted to TODAY's share count, but XBRL `diluted_shares` is the true nominal count
   as originally reported — never retroactively adjusted for a LATER split. Uncorrected, AAPL
   as-of 2018-06-29 (pre its 2020 4:1 split) computed a nonsensical +601.6% valuation gap.
   Fixed by converting the raw share count onto the price series' basis via
   `yf.Ticker(ticker).splits` before computing market cap or per-share value — verified against
   NVDA's two-split history (4x in 2021, 10x in 2024) landing within ~2% of its actual current
   share count.
2. **XOM ticker→CIK collision (edgar_client.py, not fixed — confirmed isolated).** SEC's own
   `company_tickers.json` currently maps `XOM` to CIK 2115436 ("ExxonMobil Holdings Corp", a
   shell entity with `tickers: []` in its own submissions record), not the real operating
   company's CIK 34088. Cross-checked 300 other tickers' resolved CIKs against each CIK's own
   reported `tickers` field: 0/300 mismatches — this looks like an isolated quirk of Exxon's
   corporate structure, not a systemic problem with `ticker_to_cik()`, so the shared utility
   (also used by the GP factor's full universe pull) was deliberately left unchanged rather than
   patched on unvalidated evidence. XOM simply drops out via the existing `no_xbrl_fundamentals`
   gate, same as any other data-availability exclusion.
3. **No shared yfinance throttle (new: yfinance_client.py).** Unlike SEC EDGAR
   (`edgar_client.py`'s proven 0.12s-gap + backoff, shared across the whole process), nothing
   throttled yfinance calls anywhere in this codebase. A naive ~300-ticker pilot run (one
   `fetch_prices` call per ticker instead of one batched call for the whole sample, plus
   per-ticker `.info`/`.splits`/`load_returns` calls) triggered "Too Many Requests" from Yahoo
   after ~50 tickers and silently dropped 233/300 (78%) — not a random subset, but whichever
   tickers processed before the block, which would have corrupted the backtest's coverage
   invisibly if the per-ticker error log hadn't been read closely. Fixed with `yfinance_client.py`
   (mirrors `edgar_client.py`'s pattern: shared process-wide clock, exponential backoff on
   rate-limit-shaped errors) plus batching the pilot's price fetch into one call for the whole
   sample instead of one per ticker. Even after both fixes, a 300-ticker run still throttled to
   a crawl with zero errors logged — diagnosed as Yahoo applying a session/rolling-window
   soft-throttle (slower responses, not hard errors, invisible to a quick isolated retest of the
   ticker where progress had stalled) that total request *volume* trips, not just instantaneous
   rate. Cut further by deriving the beta regression's daily return series from the
   already-batched price data instead of a third independent per-ticker fetch via
   `factor_engine.data_loader.load_returns` — a deliberate, flagged methodology choice (that
   series and `pead.prices`' aren't provably identical, though both are standard total-return
   closes), not a silent substitution.

**Pilot scope (deliberately not the full ~1,500-ticker universe — see CLAUDE.md's general
"pilot before full-scale compute" discipline):** 100-ticker stratified sample (proportional
across `index_source`'s sp500/sp400/sp600 size tiers), quarterly evaluation grid 2014-06-30
through the most recent completed quarter, all six of `smart_money/backtest.py`'s
`HORIZONS_TRADING_DAYS` (21/63/126/168/210/252 trading days = 1/3/6/8/10/12 months) — DCF/value
signals are academically documented to work on longer horizons than 13F convergence or PEAD
drift, so this was tested explicitly wider than PEAD's own (21, 63), flagged going in as a
different kind of signal with a different expected horizon so a null short-horizon result
wouldn't be misread as failure. (Originally scoped to 300 tickers; cut to 100 mid-session once
the throttling above made even a fixed pilot size take multiple hours — see `--n-tickers` CLI
flag on `scripts/run_dcf_pilot_backtest.py`.)

**Coverage confirmed legitimate before reading any IC number** (the whole point of fixing bug
#3 first): 100/100 tickers had price coverage, both skip categories are ordinary data gaps
(`unsuitable_business_model`: 27, `no_xbrl_fundamentals`: 12 — zero `fetch_failure` or
`no_price_coverage` drops), 61 tickers scored, 2,211 (ticker, quarter) rows across 46 quarters,
every count reconciling exactly (61 tickers x up to 49 quarters = 2,211 scored + 99 no_beta +
636 insufficient_history + 43 no_diluted_shares = 2,989).

**Result:**

| Horizon | Mean IC | t-stat | Hit rate |
|---|---|---|---|
| 1mo | -0.0169 | -0.58 | 57.8% |
| 3mo | -0.0337 | -1.32 | 53.3% |
| 6mo | -0.0320 | -1.06 | 56.8% |
| 8mo | -0.0332 | -1.10 | 55.8% |
| 10mo | -0.0441 | -1.41 | 54.8% |
| 12mo | -0.0372 | -1.17 | 57.1% |

**Inconclusive-leaning-negative, not scaled to the full universe.** Every horizon's IC is
negative, and none reach significance (max |t| = 1.41 at 10mo) — this is NOT the "weak short
horizon, strengthening at 6-12mo" pattern the pre-registered hypothesis expected for a value
signal; the long horizons don't show a positive IC emerging either. At the same time, N=61
tickers is modest (smaller than PEAD's ~1,500-ticker or the composite test's 1,432-ticker
samples), so this isn't powered to call a confident negative result either — consistently
negative in sign, but statistically indistinguishable from zero throughout. Explicit decision
after reviewing this: **stop here rather than scale to the full universe.** The pilot's own
purpose (catch a methodology bug before burning full-universe compute) was served three times
over; the result itself doesn't look promising enough to justify several more hours of
Yahoo-throttled compute on the current evidence. Not wired into any signal blend, same status
as PEAD's EDGAR-extension spike (closed) rather than the composite test (closed but with a
usable, if modest, positive result). Revisit only with a new idea for why the Base-case
valuation gap specifically might be biased (e.g. the terminal-value dominance and flat
margin/capex assumptions already documented under DCF Valuation Engine above), not by simply
re-running at larger N on the current methodology.

## DCF Standalone Backtest — Full-Universe Result (final)

Follow-on to the pilot above. The pilot's own "stop here" decision cited compute cost/risk
(several more hours of Yahoo-throttled compute) as the reason not to scale up, not a
determination that N=61 was already sufficient to trust the negative lean — so this wasn't
revisited because of a new methodological idea (the bar the pilot section sets for a
re-run); it was revisited because the compute-cost/risk objection itself was removable with
better infrastructure, which is what got built first.

**New infrastructure (`scripts/run_dcf_full_backtest.py`), specifically to de-risk a
~1,500-ticker run against the pilot's own documented Yahoo session-level soft-throttle:**
- Resumable state (`data/dcf/full_backtest_state.json`, one entry per ticker: `scored` or a
  skip reason) plus incremental per-ticker panel writes — a stall or crash loses at most the
  ticker in flight, not the run.
- Chunked execution (250 tickers/chunk, one batched price fetch per chunk, an 8s cooldown
  between chunks) instead of one monolithic fetch-everything-then-score pass.
- A `--canary` mode: times a small sample of tickers NOT in the pilot's own 61-ticker sample
  (a genuine cold read, not a warm-cache hit) through the same throttled call path, before
  committing to the full run. Run first: 15 tickers, 1.95s/ticker average, zero backoff
  warnings — healthy, cleared to launch.
- A business-model classification cache (`dcf/exclusions.py::check_business_model_fit`,
  `data/dcf/business_model_cache.csv`) — found during this work that
  `compute_point_in_time_dcf()` was calling this function fresh on every single
  (ticker, as_of) pair, not once per ticker, so an uncached ~46-quarter backtest grid was
  making ~46x more yfinance `.info` calls per ticker than the pilot's own rate-limiting
  investigation assumed. GICS sector/industry is static enough not to need refetching; this
  benefits every `check_business_model_fit` call site (`dcf/valuation.py`,
  `dcf/backtest.py`), not just this script.

**Run result: clean, fast, no throttle recurrence.** Completed in ~55 minutes (one pass, no
restarts needed) across the full 1,506-ticker universe. Zero `fetch_failure` entries in the
final state file — the resumable chunking held up at scale; the pilot's own throttle problem
did not recur. Coverage: 1,066 scored (70.8%), 266 `unsuitable_business_model`, 136
`no_xbrl_fundamentals`, 35 `no_scored_rows` (loaded fine, no quarter cleared every
per-observation gate), 3 `no_price_coverage`. (Exclusion/no-fundamentals rates differ
somewhat from the pilot's 27%/12% — expected sampling variance from N=100 vs the full
universe, not a methodology difference.)

**Result (1,066 tickers, 39,970 rows, ~42-45 quarters per horizon):**

| Horizon | Mean IC | t-stat | Hit rate |
|---|---|---|---|
| 1mo  | -0.0040 | -0.27 | 51.1% |
| 3mo  | -0.0101 | -0.73 | 53.3% |
| 6mo  | -0.0072 | -0.61 | 52.3% |
| 8mo  | -0.0001 | -0.01 | 51.2% |
| 10mo | +0.0008 | +0.07 | 50.0% |
| 12mo | +0.0032 | +0.25 | 50.0% |

**The key methodological lesson: the pilot's consistent negative lean was a small-sample
artifact, not a real effect.** At N=61, every horizon leaned negative and the pattern
deepened toward longer horizons (max |t| = 1.41 at 10mo) — a shape that looked, on its face,
like it might be an emerging value-signal-at-long-horizon story just short of significance.
At N=1,066, that shape doesn't survive: every IC sits close to zero, no horizon clears even
|t| = 1, and the longer horizons (8-12mo) come out essentially flat-to-slightly-positive
rather than continuing the pilot's negative trend. Properly read, the pilot result was never
"weak evidence of a negative effect" — it was too underpowered to distinguish a real effect
from noise in either direction, and the full-universe run is what actually answers the
question. This is the same "don't trust an underpowered baseline" discipline as the
Module 3/4 +0.061 investigation and PEAD's full-range-vs-2014Q2+ windowing above, applied to
this signal's own pilot-vs-full-scale step specifically.

**Final conclusion: the standalone Base-case DCF valuation gap shows no detectable
predictive power at any tested horizon (1-12 months) on point-in-time reconstructed data.**
Practical implications:
- **Not worth testing in the 13F+PEAD composite framework** (`scripts/
  run_composite_backtest.py` above) — that test's own value came from combining two signals
  that each independently cleared a real bar; blending in a signal with no standalone edge
  can't improve a combination, so this isn't pursued.
- **DCF remains valuable as a standalone valuation display, independent of this result.** The
  Discovery/Watchlist valuation badge and Valuation tab answer "is this cheap or expensive
  relative to modeled intrinsic value" — a legitimate research question in its own right
  (see DCF Valuation Engine's own worked AAPL/MSFT/KO/NVDA discussion above), not a
  return-prediction claim. A valuation framework failing to predict near-term returns is
  consistent with, not contradicted by, standard equity-research practice — DCF is normally
  read as a long-horizon fundamentals check, not a trading signal, and this backtest's null
  result is itself evidence for presenting it that way rather than as a scored predictive
  input.

Not wired into any signal blend — same status as the pilot, PEAD's EDGAR-extension spike, and
every other closed diagnostic in this document. Panel: `data/dcf/full_backtest_panel.csv`.
State: `data/dcf/full_backtest_state.json`.

## Module 5 — Tax-Lot Engine

taxlot.py ingests Fidelity/Schwab/IBKR/generic CSV exports. Supports FIFO/LIFO/MIN_TAX/SPEC_ID lot selection. Flags wash-sale risk (retrospective disallowed + prospective warning). Persists to TaxLot table.

MIN_TAX vs LIFO nuance: LIFO can numerically show lower taxes when disallowed-loss lots are selected. This is CORRECT behavior — MIN_TAX deliberately excludes disallowed lots. LIFO savings on disallowed lots are deferred into replacement basis, not realized. DO NOT fix this.

## Configuration

- Fund universe: config/fund_universe.yaml — 41 active + 1 conditional funds across 4 strategy buckets
- API keys in .env: ANTHROPIC_API_KEY (NLP scorer), OPENFIGI_API_KEY (CUSIP resolution)
- EDGAR rate limit: 0.12s minimum between requests enforced in edgar.py
- data/ is gitignored — DB, cached CSVs, price data never commit
- WAL mode set in init_db() — DO NOT REMOVE — prevents DB corruption from mid-write crashes

## DB Schema

Fund → Filing → Holding → Security → PriceCache
Fund → FundSkillResult
ConvergenceScore (cusip + period, unique)
NLPCache (ticker + accession keys + scorer_version, unique)
FinalSignal (cusip + period, unique)
Watchlist
TaxLot

## Known Limitations — by design, do not treat as bugs

- 13F data is long-only and 45-day lagged. No shorts, no futures, no non-US positions. Bearish-leaning signals come from trims/exits not actual short positions.
- Quant fund coverage ceiling is structural. Renaissance, Two Sigma, DE Shaw, AQR hold thousands of foreign/delisted positions yfinance cannot price. Will remain unreliable regardless of pipeline re-runs.
- Baupost filing completeness: historically requests confidential treatment on portions of 13F. Disclosed book is materially incomplete.
- NLP score of 0.000 (e.g. META) means insufficient language shift detected, not a negative signal. Large-cap IR teams use formulaic MD&A language.
- EDGAR value field empirical finding: SEC spec says thousands, empirically raw dollars. See edgar.py comments.

## Development Conventions

- Check in before writing code. Explain approach and key design decisions first. User reviews proposals before approving builds.
- Verify with real data before committing. Every module has a verification script.
- Commit at meaningful checkpoints — after verified modules or significant fixes, not after every file edit.
- Explain what you are changing and why before editing existing files, especially models.py.
- Never silently change methodology — flag better approaches for discussion rather than implementing unilaterally.
- Use .venv/bin/python for all shell commands. System Python 3.14 lacks required packages.

## Pre-Launch Polish List

1. Ken French RF series swap (replace ^IRX with canonical FF RF)
2. Completed — momentum factor (MOM) added across compute_factor_loadings(), portfolio.py, stress_test.py, dashboard/factor.py, and factor_apply.py; FundSkillResult migrated with beta_mom/t_stat_mom/return_from_mom
3. Staleness check in prices.py _is_cached
4. Delisted ticker handler improvements
5. Quarterly pipeline automation (Mac launchd/cron)
6. Full CIK re-verification pass across all 38 funds
7. Completed — fund universe expanded from 41 to 60 (sovereign wealth, insurance, university endowments, 11 long-tenured RIAs including Norges Bank, Capital Group, T. Rowe Price, Wellington)
8. Additional signal sources — Schedule 13D/13G, Form 4 insider transactions, active ETF daily holdings
9. Completed — replaced with Ariel Investments (CIK 936753)
10. Tax-lot engine enhancements — cross-account wash-sale detection, NYC state/city tax rates, tax-efficient rebalancing optimization
11. Dashboard error handling and graceful degradation
12. README and methodology documentation
13. CLAUDE.md update after dashboard is built
14. Completed — DCF valuation engine (`dcf/`); see full section above
15. Relative valuation / comps (peer trading multiples) — recommended companion to the DCF
    engine for mega-cap names specifically, where single-stage DCF's terminal-value
    dependency is least reliable (see DCF Valuation Engine section above)
