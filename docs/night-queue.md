# Night queue (Strategy B)

Work top to bottom. Mark `[x]` when done (commit + push after each).

- [x] a) DSR from experiments/trials.csv, random-skip null, Hansen SPA, PBO via CSCV — all in b_mvb.py's report, with tests
- [ ] b) NYSE early-closes calendar 2015–2023, with tests
- [ ] c) Breakeven slippage in bp + scratches reported separately
- [ ] d) Monte Carlo expected cone per run
- [ ] e) Speed profiling, with a test proving identical results
- [ ] f) Short side as a backtest-only shadow with SSR and easy-to-borrow flags
- [ ] g) GitHub Actions running all tests on push
- [ ] h) Offline 09:35 selection dry run against fake_alpaca.py (order intents to a file, no broker code)
- [ ] i) Optional Parquet storage with an identical-results test
- [ ] j) If Alpaca keys exist and data.alpaca.markets is reachable: runbook through DEV-A + ticks → docs/results-DEV-A.md; otherwise log skipped
