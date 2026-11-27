---
name: trading-system-conventions
description: Conventions for the Stock Momentum's / Strategy A intraday trading system: architecture, time zones, table and column names, version and variant tags, journaling, and test patterns. Use for any code, test or doc work in this repo.
---

# Trading system conventions

Merged from the planned `trading-project-conventions` + `testing-patterns` skills. Two other skills cover the dangerous parts. Load them too when the task touches their area:
- **backtest-and-experiment-rigor**: features, signals, backtests, variants, results.
- **order-risk-and-execution**: anything that can size, place, change or cancel an order.

## 1. Sources of truth (read, don't remember)
Strategy rules change every cycle, so they are NOT copied here. Read the project docs (claude.ai Project "Stock Momentum's"):
| Doc | Decides |
|---|---|
| `preregistration-v1.md` | LOCKED backtest rules, cost model, pass gate, DEV 2018–22 / HOLDOUT 2023–25. A change means v2 with a date and a reason, never after seeing results. |
| `strategy-A-spec-draft.md` | The team's demo/live spec (early entry, tranches, filters). |
| `strategy-A-design-plan.md` | Risk parameters, universe thresholds, architecture, forming-candle model. |
| `program-spec-and-skills.md` | The 12 components and the experiment protocol. |
| `expert-verdicts-strategy-A.md` | Team decisions and the EXPERIMENT MODE principle. |
| `data-sources-decision.md` | Where each dataset comes from. |
| `ideas-backlog.md` | Every new idea goes here. |
| `code/schema.py` | The existing DuckDB catalog. Follow its names. |

- Strategy parameters live in ONE file per version: `config/strategy_a/A-v<N>.yaml`. Never hard-code a threshold in `signal/`.
- If two docs disagree, do not pick silently. Name the conflict and ask which is the baseline; the other becomes a variant. Known open conflicts are listed in `skills-research-and-merge-plan.md`.

## 2. Architecture: one engine, same code for replay and live
```
src/
  config.py  paths, NY = "America/New_York"
  data/      schema.py (DuckDB catalog), loaders
  feeds/     IEX websocket, REST poller, HistoricalFeed (replay)
  clock/     Clock interface: LiveClock, ReplayClock (injected)
  bars/      1-min -> 5-min builder, partial-bar snapshots
  signal/    PURE functions: (closed bars, state, params) -> Decision
  variants/  registry + evaluator (every signal through every variant)
  risk/      risk.py: approves or rejects every real order
  exec/      AlpacaPaperBroker and SimBroker (same interface)
  journal/   append-only event log
  bus/  analysis/  reports/  ops/
config/  tests/  experiments/  runs/  data/ (gitignored)
```
- Live and replay differ ONLY in which Feed, Clock and Broker are injected.
- `signal/` and `variants/` never import from `feeds/`, `exec/` or network libraries, and never call `datetime.now()`, `time.time()` or `pd.Timestamp.now()`. Time arrives as an argument. A purity test enforces this.
- Storage: Parquet + DuckDB for history and the journal; SQLite (WAL) for live state.
- One Alpaca account = one live data stream (a second connection gets error 406). Each partner's instance uses its own account.

## 3. Data model (from `code/schema.py`, keep these names)
- Bars: `source, ticker, security_id, ts_start_utc, o, h, l, c, v, vwap, n_trades, ingested_at`; Parquet partitioned by `source/year/month`.
- Views: `bars_raw` (deduped, latest ingest wins), `bars` (adds `ts_ny`, `trade_date_ny`, `session` = PRE/RTH/POST/OFF), `bars_adj` (split-adjusted `*_adj` columns).
- Use `bars` (raw) for intraday runs, VWAP and P&L. Use `bars_adj` ONLY for cross-day features (gap, ATR/σ windows, 20-day $ volume, RVOL baseline).
- Tables: `securities` (ticker ↔ permanent `security_id` over time, `exchange`, `active`, `delisted_utc`), `corp_actions`, `events` (point-in-time `ts_utc`), `early_closes`.
- Join and group by `security_id`, not `ticker`: tickers change and get reused.
- Raw prices are never overwritten. Re-ingest appends; the view dedupes.

## 4. Time
- Store every timestamp as tz-aware UTC. A naive datetime is a bug. Suffix columns `_utc` / `_ny`.
- All market logic runs in `America/New_York` (`config.NY`), converted at the edge. Israel time never appears in logic: IL and US switch DST on different dates, so the offset changes for a few weeks a year.
- Bars are labelled by START time: the 10:00 5-min bar covers [10:00, 10:05). It is closed only when `clock.now() >= bar_end`.
- Half days come from the `early_closes` table (still a TODO in schema.py: load it before any backtest). Never assume a 16:00 close.
- Session anchors (ET): open 09:30, streak counter resets at 09:30, no signals 09:30–09:45, live flatten job 15:50, last backtest bar 15:55.

## 5. Names and tags
- Strategy version: `A-v1`, `A-v2`… Variant id: `<version>__<knob>=<value>`, e.g. `A-v1__entry_min=4.0`. Baseline: `A-v1__base`.
- Every journal row, order, fill and report row carries: `strategy_version`, `variant_id`, `code_sha`, `feed` (`iex`/`sip`), and either `run_id` (backtest) or `session_date_ny` + `instance` (`pc`/`vps`) + `account` (`acct1`/`acct2`) for live.
- Units in column names: `pnl_r`, `slip_bps`, `px_usd`.
- Money at the broker boundary: `Decimal`, rounded to the tick. Floats are fine in analysis.
- `client_order_id` format: see order-risk-and-execution.

## 6. Journal (append-only)
Tables: `raw_events`, `partial_bar_snapshots` (every 15 s), `decision_snapshots` (including NO_TRADE and VETOED with a reason code), `orders`, `fills` (with NBBO at fill and the shadow fill), `finals` (SIP bars after the close plus an `iex_sip_agree` flag).
Never UPDATE a row. A correction is a new row that references the old one.

## 7. Testing patterns
Layout: `tests/unit`, `tests/integration`, `tests/parity`, `tests/fixtures`.

Synthetic generators in `tests/fixtures/synth.py`, always seeded:
- `random_walk_day(seed)`: no edge by construction. Gross edge vs the random-entry null must be about zero. A clear profit here means lookahead.
- `red_run_then_green(n, drop_atr)`, `trend_down_day()`, `noise_day()`: known shapes.

Known-answer tests: hand-built bar sequences, with the right decision, fill and P&L worked out in the docstring. Must-have cases:
- N red bars then green: signal on exactly the right bar. N−1 reds: no signal.
- A bar with a tiny body (|c − o| < ε) inside a run, handled per the current version's rule.
- Stop and target hit in the same 1-min bar: stop first.
- Signal at 09:35 (blocked), on a half day, on both DST-change days.
- A split inside a cross-day window: `bars_adj` gives a continuous series, `bars` does not.

Required tests (CI runs them):
1. **Future-perturbation:** replace all data after time t with noise; every decision at or before t must be unchanged.
2. **Parity:** replay a recorded live day; decisions must equal the journal's decisions.
3. **Purity:** no forbidden imports or clock calls in `signal/` and `variants/`.
4. **Risk limits:** each limit at, just below and just above its boundary (see order-risk-and-execution).

## 8. Check your work before saying "done"
1. Run the tests and report the pass/fail counts.
2. Recompute one number from any report or table by hand.
3. Touched features, signals, backtests or results? Run the backtest-and-experiment-rigor checklist.
4. Touched the order path? Run the order-risk-and-execution checklist.
5. Changed a parameter? Decide: new version or new variant. Log it in `experiments/changelog.md`.
6. Say plainly what you did NOT verify.

## 9. Team rules
- Demo/paper first. Test alternatives side by side instead of debating them.
- EXPERIMENT MODE: change anything, any time, but version it.
- Trade-offs and judgement calls go to the two partners. Only stable per-stock traits are auto-tuned, under the rules in backtest-and-experiment-rigor.
- New ideas go to `ideas-backlog.md`.
