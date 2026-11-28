# Build log: Strategy B backtester (`strategy_b/b_mvb.py`) and downloader (`strategy_b/fetch_b_data.py`)

## Round 6: night session 3 (27.11.2026, late). Details: `night-log-2026-11-27.md`
- `shadow_day_b.py`: paper shadow logger without orders; shadow trades = backtest trades (7 versions, exact), verifier 283/283 on a 74-session shadow run.
- **Finding (IEX):** the free plan's 09:35 view is IEX. IEX volume ÷ a SIP baseline → RVOL ≈ IEX share → the RVOL ≥ 100% filter selects nothing. Fixed in the logger (IEX baseline); a live engine needs the same. Day-to-day IEX-share noise (σ 0.5, synthetic) gave 79% overlap, under the paper→live gate's 80%.
- `fetch_b_data.py quotes` + `spread_b.py`: half-spread calibration from SIP NBBO at the model's trade moments (close exits measured at the END of the 15:58 bar).
- **Boss findings fixed:** `model_refs.csv` first used trades.csv prices, which already include half spread + k → the paper slippage check would have been lenient by the whole model cost; the IEX baseline cache was shared across OR lengths (F9 would have read 10-minute bars as its 15-minute baseline).
- **161 tests** (160 + Parquet). Still never run on real data.

## Round 5: night session 2 (27.11.2026, evening). Details: `night-log-2026-11-27.md`
- Market exposure of the base vs SPY + F8 red-signal-day diagnostic (plan v3 §5); B0-replication sanity gate (`--replication`); mean net R without the top-5 tickers.
- `gate_b.py holdout`: the whole pass gate, with the open prereg choices as required arguments; `--nulls-all-versions` for item 3.
- `run_dev_a.py` (runbook as one command, draft results doc); `validate` lists likely ticker reuse; `paper_gate_b.py` (paper → live gate).
- SPA size measured at DEV-A length: ~6% at 5%, ~11% at 10% (slightly liberal).
- **Boss findings fixed:** `SPY_PREV` leaked between runs in one process; a new summary column broke a test (CI red for 4 runs, fixed; full suite now runs before every push).
- **136 tests** (135 + Parquet). Still never run on real data.

## Round 4: unattended night session (27.11.2026). Details: `night-log-2026-11-27.md`
- Gate tools built: DSR, random-skip null, Hansen SPA, PBO (CSCV), expected cone (iid + day-block), `gate_b.py`.
- NYSE calendar 2015–2023 is now the default; breakeven in bp; scratches; short shadow (SSR/ETB); 09:35 dry run (no broker code); Parquet option; CI on every push.
- Speed 2.6× with bit-identical results (frozen reference engine + equivalence test).
- **Boss findings fixed:** (1) the independent verifier did not know half days; (2) the daily download could drop its end day (Dec 31 in 2015, 2018–2021) if the API reads a date-only `end` as 00:00 UTC; (3) the calendar state leaked between runs in one process.
- **112 tests pass** (CI green). Still never run on real data (no keys; host blocked in the cloud environment).

## Round 3: downloader + tick resolution (27.11.2026)
- `fetch_b_data.py`: probe, assets, daily (+ splits), opening, fullday (top-100 + 30 random + SPY), validate (data gate), ticks.
- `fake_alpaca.py`: an in-process fake of the Alpaca endpoints, used only by the tests.
- **End-to-end:** download from the fake, then backtest → **identical trades** to the source market.
- **Ticks:** resolving the ambiguous minutes with prints gives exactly the result predicted by the fake's print order, and 100% of ambiguous trades are resolved.
- **Boss finding (fixed):** per-day `asof` symbol mapping was used in the minute stages but not the daily stage, so a renamed company would get two identities. Now one mapping is used everywhere, and a test guards it.
- **42 tests pass.** Never run against the real Alpaca API: the probe is the first real check.

## Round 2: visual proof + independent re-derivation (27.11.2026)
**Verdict: the engine does what the plan says.** One modelling assumption is strong enough to flip a decision, and resolving it is a cheap data task.

### Proof that it works
- **Independent verifier** (`viz_b.py`, `independent_trade`): the base rules re-written from the plan text with plain loops, sharing no code with `simulate_trade`. It re-derived every trade of the demo run (entry minute, entry price, exit minute, exit price, reason, net R):

| Version | Trades checked | Match |
|---|---|---|
| base | 626 | **626** |
| L1 stop 25% ATR | 626 | **626** |
| L2 stop 50% ATR | 626 | **626** |
| L4 time stop 10:30 | 626 | **626** |
| base with OHLC-path same-bar | 626 | **626** |

- **The verifier can fail** (checked on purpose):
  - with a wrong rule (stop 11% instead of 10%): 604/604 executed trades mismatch;
  - with a wrong rule (2-tick slippage): 486 mismatch;
  - with one exit price moved by 1¢: exactly 1 mismatch caught.
- **Screenshots:** a winner to the close, a stop, a same-minute stop, a wide-stop winner, a time stop, and the equity curves. Each one was read against the rules, and one trade was recomputed by hand: entry 108.97 + ½ spread 0.033 + slip 0.022 = 109.024; R = (114.376 − 109.024 − 0.007)/0.436 = +12.26 ✓.
- **Replay video** (14.6 s): one trade minute by minute. It shows the system knows only the past at each minute (no lookahead).
- **Not covered visually:** an intraday stop *gap*. The synthetic market has no intraday gaps; the unit test covers it.

### Finding R9 (important): the same-minute assumption can flip the decision
- At base (10% ATR), **19% of trades** have entry and stop inside the same 1-min bar. The pessimistic rule (locked for B0) counts all of them as stop-outs.
- The standard OHLC-path model is now a sensitivity option (`same_bar="ohlc_path"`, reported under `--stress`). It assumes green bar = O→L→H→C (the low came *before* the entry) and red bar = O→H→L→C.

| Version (demo, synthetic) | mean R pessimistic | mean R OHLC path | same-bar share |
|---|---|---|---|
| base | +0.204 | **+0.361** | 19% |
| L1 stop 25% ATR | +0.316 | +0.316 | 0.3% |
| L2 stop 50% ATR | +0.253 | +0.253 | 0% |
| F2 RVOL ≥ 200% | +0.459 | +0.612 | 33% |
| F6 2nd-bar entry | −0.001 | +0.195 | 24% |

- **The ranking flips:**
  - under the pessimistic rule, L1 beats base;
  - under OHLC path, base beats L1.
- So "widen the stop" may be partly an artifact of the fill assumption.
- Wide stops are immune to this (almost no same-bar cases).
- **Fix (built in Round 3):** `fetch_b_data.py ticks` downloads the real prints for the ambiguous minutes only, and `b_mvb.py --ticks` decides from them which came first.
- **Rule:** no decision between base and the stop variants on pessimistic numbers alone.

## Round 1: boss review (27.11.2026, line-by-line read of the code, the plan and the prereg)
**Verdict: REFINE.** The code is now trustworthy enough for a first real run, but **we have still not produced a single number from real data.** **No new variants or features until the data stage exists and DEV-A has run.**

### Code: found and fixed
| # | Problem | Severity | Fix |
|---|---|---|---|
| R1 | **The random-names null was not random.** It drew only from names with minute bars, and full-day bars exist only for the top-100 RVOL | Blocker for H2 | The pool is now explicitly "names with full-day bars", with a warning in the report. The downloader adds 30 seeded random eligible names per day |
| R2 | **The holdout could be opened many times** | High | Refused if the trial log already holds a run ending ≥ 2024; at most base + 2 variants; date format validated |
| R3 | **Every variant re-read every day file** (hours to days on real data) | High | Day-major engine; tested equal to separate runs; 6× faster |
| R4 | **A truncated file (data gap) was silently treated as a half day** | Medium | `--early-closes` calendar; otherwise only 15:59 / 12:59 endings are accepted; gaps are skipped and listed. Exit = 1 bar before the close on every day |
| R5 | The RVOL baseline counted **rows, not sessions** | Medium | The window is 14 sessions from the daily file; missing = NaN; ≥ 10 valid sessions required |
| R6 | F6 "2nd-bar entry" could trigger BELOW the OR high | Low | Trigger = max(OR high, bar-2 high) (min for shorts) |
| R7 | The data fingerprint covered file names only | Low | Hash over size + mtime of every file |
| R8 | The report mixed sizings without saying so | Low | The header now states it: mean R = equal $ risk; Sharpe/DD 1× = equal notional |

### Round 7 (night 28.11, session 4): boss review
- **Fixed (data, would have hit real data):** `fetch_b_data.py daily` rebuilt `daily.csv` from the years of the CURRENT call only and overwrote `splits.csv` with the current range's splits. A second call (e.g. `daily --start 2021-01-01` later) silently dropped 2015–2020 history and every earlier split factor (wrong ATR/RVOL, wrong split scaling). A year fetched in part was also marked done for good. Now: manifest pieces name the range fetched (`2019:2019-01-01_2019-01-31`), a wider request re-fetches the year's whole span, `daily.csv` is assembled from every part, `splits.csv` is the de-duplicated union of `splits_parts/`. Test: two calls in one year, an overlapping call (split counted once), a narrower call downloads nothing. Found because the DEV-B rehearsal's synthetic DEV-A run had 0 trades.
- **Found, NOT changed (Decision 14):** the per-side cost (half spread + k) is computed once at the trigger price and charged on both legs. The exit leg should strictly be charged at the exit price. On a synthetic run the per-leg cost changes mean net R by −0.002 R (base) / −0.001 R (L2), max −0.026 R on a big winner: small, but optimistic on exactly the trades that pay. The independent verifier makes the same choice, so it cannot catch it.
- `run_dev_b.py`: refuses if the frozen file's knobs differ from the DEV-A run's; `gate_b.confirm` labels PBO from a subset run instead of judging the whole search on it.

### Open, not fixed in code
- **ETFs/ADRs:** the downloader's name heuristic is the only filter.
- ~~Breakeven slippage is in ticks only~~ → bp added in Round 4.
- A breakeven exit (net 0) counts as a loss in win% → Round 4 reports scratches (|R| ≤ 0.05) apart; old win% kept.
- **F9 (15-min OR) ranks by 15-min RVOL.** The paper does not say which it used; decide at signing.
- ~~Gate tools not built~~ → built in Round 4 (DSR, random-skip, SPA, PBO). Still open: 1,000 random-entry/names nulls is just `--nulls 1000` (slow, not timed on real data).
- **The prereg must specify the Sharpe sizing.** Proposal: equal notional 1/N, as implemented.
- **The cost model uses IBKR commission while paper runs on Alpaca (commission-free).** That is pessimistic, so keep it, but say so.
- **Cost at the exit leg** uses the trigger price (see Round 7, Decision 14).
- **Live ranking feed:** on the free plan the 09:35 ranking is IEX-only. Measure IEX-vs-SIP top-N overlap on real data (`shadow_day_b.py --selection-feed iex` vs `sip` over DEV days) before paper; if < 80%, a SIP real-time plan or a different ranking input is needed (team decision).
