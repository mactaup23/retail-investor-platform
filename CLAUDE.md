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

Stress test methodology: historical factor shock replay against current betas. This is risk characterization, NOT a backtest. Label accordingly in all output.

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

Fund skill regression is Fama-French-Carhart 4-factor (adds momentum to the historical FF3 spec) — see smart_money/factor_apply.py. Momentum matters most for growth/momentum-tilted managers (e.g. Greenoaks, Altimeter): under FF3 their return from holding recent winners was mis-attributed to alpha; FF4 captures it as beta_mom instead.

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
7. Fund universe expansion — long-only institutional managers (Norges Bank, Capital Group, T. Rowe Price, Wellington, endowments)
8. Additional signal sources — Schedule 13D/13G, Form 4 insider transactions, active ETF daily holdings
9. Completed — replaced with Ariel Investments (CIK 936753)
10. Tax-lot engine enhancements — cross-account wash-sale detection, NYC state/city tax rates, tax-efficient rebalancing optimization
11. Dashboard error handling and graceful degradation
12. README and methodology documentation
13. CLAUDE.md update after dashboard is built
