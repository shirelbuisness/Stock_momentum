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

## Refill 16:52 (blocks: first real number / DEV-B gate / safe paper)
- [x] k) Docs refresh: runbook, plan §6–7, CLAUDE.md, build log → tonight's features and test counts (the runbook still says "42 passed")
- [x] l) `gate_b.py`: DEV-A shortlist + DEV-B confirmation exactly per the prereg protocol (Holm < 0.10 & positive; t ≥ 2, same sign & ≥ ½ size, DSR ≥ 0.95, positive at 2k), with tests
- [x] m) Day-block bootstrap option for the expected cone (trades on one day are correlated), iid vs block on synth
- [x] n) Offline paper-vs-cone checker: a CSV of paper trades (net R) → where it sits in expected_cone.json, with tests
