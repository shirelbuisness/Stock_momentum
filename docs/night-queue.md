# Night queue (Strategy B)

Work top to bottom. Mark `[x]` when done (commit + push after each).

- [x] a) DSR from experiments/trials.csv, random-skip null, Hansen SPA, PBO via CSCV — all in b_mvb.py's report, with tests
- [x] b) NYSE early-closes calendar 2015–2023, with tests
- [x] c) Breakeven slippage in bp + scratches reported separately
- [x] d) Monte Carlo expected cone per run
- [x] e) Speed profiling, with a test proving identical results
- [x] f) Short side as a backtest-only shadow with SSR and easy-to-borrow flags
- [x] g) GitHub Actions running all tests on push
- [x] h) Offline 09:35 selection dry run against fake_alpaca.py (order intents to a file, no broker code)
- [x] i) Optional Parquet storage with an identical-results test
- [x] j) SKIPPED (no keys; host blocked) — If Alpaca keys exist and data.alpaca.markets is reachable: runbook through DEV-A + ticks → docs/results-DEV-A.md; otherwise log skipped

## Refill 15:01 UTC (blocks: first real number / DEV-B gate / safe paper)
- [x] k) Docs refresh: runbook, plan §6–7, CLAUDE.md, build log → tonight's features and test counts (the runbook still says "42 passed")
- [x] l) `gate_b.py`: DEV-A shortlist + DEV-B confirmation exactly per the prereg protocol (Holm < 0.10 & positive; t ≥ 2, same sign & ≥ ½ size, DSR ≥ 0.95, positive at 2k), with tests
- [x] m) Day-block bootstrap option for the expected cone (trades on one day are correlated), iid vs block on synth
- [x] n) Offline paper-vs-cone checker: a CSV of paper trades (net R) → where it sits in expected_cone.json, with tests

## Refill 19:10 UTC, session 2 (blocks: first real number / DEV-B and holdout gate / safe paper)
- [x] o) Market beta of long-only base vs SPY open→close + F8 red-signal-day diagnostic (plan v3 §5), report section + gate.json, tests
- [x] p) Missing pass-gate numbers in the summary: net R with the top-5 tickers removed; B0-replication sanity (long+short, 4×, commission only, gross Sharpe ≥ 2.0) as a mechanical check
- [x] q) `gate_b.py holdout`: the prereg pass gate (items 1-7 + healthy shape) checked mechanically; the unsigned [ ] choices are REQUIRED arguments with no default; tested on synthetic runs only (never opens the holdout)
- [x] r) One-command DEV-A pipeline (`run_dev_a.py`): probe → assets → daily → opening → fullday → validate → DEV-A → ticks → rerun → gate shortlist → draft `docs/results-DEV-A.md`; stops at NO-GO; rehearsed end-to-end against fake_alpaca
- [x] s) SPA size at n ≈ 1,250 days (DEV-A length) by simulation; record the measured rejection rate
- [x] t) Boss review #4 (done after o–q, see the night log)

## Refill 19:40 UTC, session 2
- [x] u) Ticker reuse / identity breaks: `validate` lists likely reuse (unexplained open/prev-close jump ≥ 3×, or > 30 days without bars then > 50% level change) as WARN rows + `data/identity_warnings.csv` for a spot check
- [x] v) Paper → live gate checker: realised slippage ≤ 1.5× the model; live/backtest top-20 overlap ≥ 80%; live expectancy inside the backtest's 95% CI (prereg "Paper → live gate"), offline, with tests
- [x] w) Docs refresh: CLAUDE.md, plan §6–8, build log Round 5, test counts
- [x] x) Boss review #5 + full suite + verifier
- [x] y) pandas future-proofing: `fillna(False)` on float columns (Pandas4Warning in CI; would raise on a future pandas)

## Next (for the next session)
- [x] z1) SKIPPED in session 3 (no keys in env) — With keys: `python run_dev_a.py` → read the draft `docs/results-DEV-A.md` with Shirel
- [x] z2) A paper shadow logger that writes `paper_trades.csv` / `paper_fills.csv` / `live_selection.csv` in the formats `paper_gate_b.py` and `paper_vs_cone.py` read (no order code; needs the order-risk skill review first)
- [ ] z3) Measure real-data runtime of DEV-A and of `--nulls-all-versions` on the holdout sizes

## Refill 23:10 UTC, session 3 (z1 skipped: no keys; z3 needs real data)
- [x] z2) `shadow_day_b.py`: paper shadow logger WITHOUT orders: 09:35 selection (as the dry run) + after the close the day's SIP bars → the backtest's own fill model → appends `paper_trades.csv` + `live_selection.csv` (formats of paper_gate_b / paper_vs_cone); parity test vs the backtest; selection feed iex|sip (live free plan ranks on IEX)
- [x] aa) Spread calibration from SIP quotes (cost model: "3 bp until calibrated"): `fetch_b_data.py quotes --trades <run>/trades.csv` (NBBO in a window around each entry and exit minute, market data only) + `spread_b.py` → measured half spread in bp (median / by price bucket) with tests; applying it stays a team decision
- [x] ab) Rehearsal: shadow logger over many synthetic DEV days → `paper_gate_b.py` end to end (overlap and expectancy checks pass on honest data, fail on a planted bug)
- [ ] ac) Boss review #6 + docs refresh (CLAUDE.md, runbook, build log Round 6)
