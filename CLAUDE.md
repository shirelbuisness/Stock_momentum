# Stock Momentum's: intraday US-stock trading system (paper first)

Two partners: Shirel (developer, CS + biology student; prefers Hebrew in chat, code and docs in English) and a trader partner.
Goal: a machine that, after ALL costs, makes more money than it loses. Fail fast on real data; paper trade; only then real money.

## Where things stand (27.11.2026)
- **Strategy B (ORB on Stocks in Play) is the current focus.** It has the strongest published evidence (Zarattini, Barbon & Aziz 2024).
- Code: `strategy_b/` (standalone; not yet on the shared engine). **136 tests pass** (the Parquet one needs pyarrow; CI runs them all on every push: `.github/workflows/tests.yml`).
- **NOTHING has run on real market data yet.** Every result so far is synthetic. The next step is the real Alpaca probe.
- Strategy A (6 red candles → reversal) and C (SPY momentum) exist in the claude.ai Project "Stock Momentum's", not here.
  Another cloud session built `scripts/run_all.py` + `src/` for A in a repo that is NOT on GitHub. If it is needed, ask Shirel.

## Next step (do this first)
1. `pip install -r requirements.txt && cd strategy_b && python -m pytest -q` → expect `136 passed` (~5 min; `135 passed, 1 skipped` without pyarrow).
2. Shirel sets `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` (free Alpaca account) in her shell or a gitignored `.env`.
3. `python run_dev_a.py` (one command: probe → download → validate → DEV-A → ticks → rerun → shortlist → draft `docs/results-DEV-A.md`; stops at NO-GO). Step by step instead: `docs/strategy-B-data-runbook.md`.
4. Read the draft with her: probe (GO / NO-GO, delisted-names line, 500-symbol line), data gate (+ `identity_warnings.csv`), the B0-replication sanity gate, then the numbers.

## Files
| Path | What |
|---|---|
| `strategy_b/fetch_b_data.py` | Alpaca downloader (probe, assets, daily, opening, fullday, validate, ticks). Resumable, atomic writes |
| `strategy_b/b_mvb.py` | Backtester: 15 versions in one day-major pass, all costs, nulls, paired tests + Holm, holdout guard |
| `strategy_b/viz_b.py` | Independent re-implementation of the base rules (verifier) + trade screenshots + replay video |
| `strategy_b/fake_alpaca.py`, `synth.py` | Offline fake Alpaca + seeded synthetic market (tests only) |
| `strategy_b/variants_b.json` | The 14 frozen versions + base |
| `strategy_b/gate_stats.py` | DSR, random-skip null, Hansen SPA, PBO (CSCV), expected cone (iid + day-block) |
| `strategy_b/gate_b.py` | Mechanical DEV-A shortlist / DEV-B confirmation / holdout pass gate per the prereg |
| `strategy_b/run_dev_a.py` | The whole runbook as one command + a draft `docs/results-DEV-A.md` |
| `strategy_b/beta_b.py` | Market exposure of the long-only base vs SPY (plan v3 §5) |
| `strategy_b/paper_gate_b.py` | Paper → live gate (slippage vs model, top-N overlap, expectancy CI), offline |
| `strategy_b/nyse_calendar.py` | NYSE holidays + 13:00 closes 2015–2023 (default calendar of b_mvb.py) |
| `strategy_b/dry_run_b.py` | Offline 09:35 selection → order intents file (no broker code) |
| `strategy_b/paper_vs_cone.py` | Paper (shadow-fill) trades vs the expected cone |
| `strategy_b/to_parquet.py` | Optional Parquet copy of the data (identical results) |
| `strategy_b/reference/b_mvb_ref.py` | Frozen pre-speed-work engine; never edit (equivalence test) |
| `docs/strategy-B-final-plan-v3.md` | Authoritative plan: versions, testing protocol, regime/portfolio |
| `docs/preregistration-B-draft.md` | Locked rules + pass gate (NOT signed yet; open choices marked [ ]) |
| `docs/strategy-B-data-runbook.md` | Exact commands, time and disk estimates |
| `docs/build-log-strategy-B-mvb.md` | Boss reviews: bugs found and fixed, open issues (R9 same-bar finding) |
| `.claude/skills/` | Project rules. Load them: conventions (all work), rigor (any backtest/result), order-risk (any order code) |

## Non-negotiable rules
- **Always include costs** in every number (commission + half spread + slippage). Report gross only when labelled.
- **Holdout 2024–2025 is opened ONCE**, for base + at most 2 promoted versions. Both scripts refuse 2024+ without `--open-holdout`. Never bypass this, and never download 2024+ data early.
- **"Zero losing trades" is not a goal.** It is overfitting. Targets: positive after costs, ≥ 60% positive months, max DD ≤ ~15R, positive at 2× slippage.
- **Every backtest run is a trial**, logged automatically in `experiments/trials.csv` (it feeds the Deflated Sharpe). Don't delete rows.
- **One knob per version.** New ideas go to the backlog, not into the running cycle.
- **No decision between base and the stop variants on pessimistic numbers alone** (finding R9): resolve the ambiguous minutes with `fetch_b_data.py ticks` and rerun with `--ticks`.
- **Paper only.** Never write code that can place live orders.
- **Before building anything with a UI** (dashboard, app), show Shirel design options and let her choose.
- **Check your work like "the boss":** run the tests, recompute one number by hand, say plainly what was NOT verified.
- Secrets: env or `.env` only (gitignored). Never print keys.

## Known open issues
- Alpaca's asset list is today's view, so long-delisted names may be missing (survivorship). The probe measures it.
- Ticker reuse can mix companies: `validate` now lists likely cases (`data/identity_warnings.csv`) for a spot check, it does not remove them. The ETF filter is a name heuristic.
- Built on the night of 27.11 (synthetic-tested only): DSR, random-skip null, Hansen SPA, PBO, day-block cone, `gate_b.py`, NYSE calendar, short shadow, dry run, Parquet. Session 2 the same night: market beta, B0-replication sanity gate, holdout pass-gate checker, `run_dev_a.py`, identity warnings, paper → live gate. See `docs/night-log-2026-11-27.md`.
- The cloud environment blocks `data.alpaca.markets` and has no keys: the real run must happen on Shirel's machine, or after the host + key are added to the environment.
- The prereg must still pick: the Sharpe sizing, the breakeven-slippage threshold (1 or 2 ticks), the net-R bar (+0.05R or +0.08R), and the cost-filter multiple (3× or 5×).
