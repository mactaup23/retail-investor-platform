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
