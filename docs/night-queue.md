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
