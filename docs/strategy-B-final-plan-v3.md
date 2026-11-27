# Strategy B: final plan v3, loss-reduction track + version testing (27.11.2026, status updated 27.11 evening)
Builds on `strategy-B-build-plan.md` (rev 2: rules, data path, fill model, risk) and replaces its variant list and testing protocol. Panel for v3: exit/trade-management expert, statistician, ORB practitioner, regime/portfolio researcher. Code: `strategy_b/b_mvb.py` + `strategy_b/fetch_b_data.py` (42 tests pass).

## 1. Goal, stated honestly
- **Goal:** profitable after ALL costs, with few losing *months*, not zero losing *trades*.
- ORB wins on only ~17–25% of trades at a 10%-ATR stop (QC users, the index replication, the BBT deck; the paper's 48.4% "hit ratio" is undefined), and a few big trend days pay for everything. A backtest with no losers means the rules were bent around past losing days (overfitting).
- **Healthy targets (statistician):** win rate 35–55% (after the wider-stop fix), profit factor 1.2–1.6 (> 2.5 = suspicious), ≥ 60% of months positive, ≥ 6/8 quarters positive, max DD ≤ ~15R, positive at 2× slippage, stable across years.

## 2. Why losses are large today: the stop is inside the noise
- Cost per trade scales with 1/stop. With a $0.25 stop (10% ATR), cost ≈ 0.17R. With a 25% ATR stop it is ≈ 0.07R.
- Confirmed structurally on synthetic data: cost/trade 0.335R → 0.134R → 0.067R at stop 10% → 25% → 50% ATR; win rate 10% → 27% → 42%.
- **Under fixed $-risk sizing, R = $ won per $ risked**, so R stays comparable across stop widths.
- At 10% ATR the 4× cap usually binds (risk per trade < 1%). Wider stops also relieve the cap.
- **Caveat (build log R9):** part of the 10%-ATR losses come from the pessimistic same-minute rule. Decide base vs L1/L2 only on a tick-resolved run.

## 3. The 14 versions (one knob each vs base; frozen before any real run)
| id | knob | why (source) | evidence |
|---|---|---|---|
| base | paper rules, long-only, 1×, full costs | | paper |
| L1 | stop 25% ATR | cost/R ~2.5× lower; stop outside opening noise (exit #1) | arithmetic |
| L2 | stop 50% ATR | same, stronger | arithmetic |
| L3 | stop at OR low, floor max(10% ATR, 2× cost), cap 50% ATR | structural invalidation level | mql5 index study used it |
| L4 | time stop: if the 10:30 close < entry + 0.5R → exit at 10:31 open | stalled breakouts rarely trend | hypothesis |
| L5 | no entries after 10:30 | late breakouts follow chop | practitioners |
| F1 | skip if stop < 5× round-trip cost | removes trades whose costs eat the R (practitioner #1) | mql5 (risk ≥ 2× cost was still net 0) |
| F2 | RVOL ≥ 200% | paper: +0.08R (RVOL > 100%) → +0.38R (RVOL > 30×) | paper Fig. 4 |
| F3 | top-10 | only the strongest "in play" | |
| F4 | skip if OR range > 0.6 ATR | exhausted first bar | lore (BBT) |
| F5 | first bar closes in its top 30% | bar quality | lore (BBT) |
| F6 | trigger at the 2nd 5-min bar's high (never inside the OR) | fewer fake breakouts | |
| F7 | price ≥ $10 | fewer ticks per R | |
| F8 | longs only if SPY prior close → 09:35 is up | long-only = net long; red market days hit all 20 together | hypothesis (Gao et al. is about SPY's last half-hour) |
| F9 | 15-min OR | paper's next best (Sharpe 1.43) | paper Table 3 |

**Stage 2 (only after singles pass), max 2 knobs, each combination = +1 trial:**
- breakeven after +1R and after 10:00, tested **only on top of L1** (at 10% ATR, +1R sits inside the noise);
- trail after +3R at the lowest low of the last three 5-min bars (a diagnostic).
- **Rejected:** partial profit at +2R. It cuts the few trades that pay for everything, and it adds a second cost.

**Not codable yet:** spread-at-09:35 filters (need SIP quotes); VWAP filter; premarket-high break (needs extended-hours bars); breadth sizing; VIX/VIX3M sizing. These are v2 candidates.

## 4. Testing protocol (statistician)
1. **Freeze** the 14 versions in the prereg before touching data. Every run is logged in `experiments/trials.csv` (the script appends automatically), and that log is the DSR trial count.
2. **DEV-A 2016–2020 = explore.** Run all 15 (base + 14) on the same days. Paired **day-level** test (daily R sum of the variant − the base, Newey-West, 5 lags) + **Holm** across the 14. *Added later:* Hansen SPA for "does the best beat base?"; if SPA p > 0.10, nothing beats base.
3. **Shortlist** ≤ 4 with Holm p < 0.10 and a positive effect.
4. **DEV-B 2021–2023 = confirm.** Each needs: paired t ≥ 2; the same sign as DEV-A and ≥ half its size; DSR ≥ 0.95; positive at 2× slippage. *Added later:* PBO via CSCV (> 0.3 = the whole search is overfit).
5. **Stage 2** combinations (§3), then DEV-B again.
6. **Holdout 2024–25, run once:** base + at most 2 promoted variants, Holm on the holdout as well (the first needs p < 0.025).
- **Power:** DEV-A detects about 0.06R after Holm; DEV-B about 0.08R; the holdout about 0.05R. Paired tests are far stronger than unpaired ones, so never compare two variants' raw means.
- **Every filter must beat a random-skip null** (skip the same share of days at random). Skipping days removes losers by construction. *Added later.*

## 5. Regime and portfolio (researcher)
- **Long-only B carries positive market beta** (unverified). First measure B's daily P&L against SPY's open → close return. F8 helps only if long P&L on red-signal days is negative after costs; otherwise half size beats skipping. The real fix is restoring shorts (v2).
- **C (SPY momentum) gained in sell-offs** (+16% Q4 2008, +7% Q1 2020, +12% Q2 2022; beta ≈ −0.05): it offsets long-only B. A (reversal) is the partner for choppy days.
- **Allocation, fixed in advance:** inverse-volatility over 60 days (lagged 1 day); risk-share caps B ≤ 50%, C ≤ 50%, A ≤ 20%; portfolio daily stop −1.5R. Judge by % positive months, worst month, max DD, Calmar, and daily P&L correlation (overall, on SPY-down days, and on the worst 5% of days). Block bootstrap of months for confidence intervals.

## 6. How to run
Exact commands, time and disk estimates: `strategy-B-data-runbook.md`. In short:
```
python -m pytest -q                                   # 42 tests must pass first
python fetch_b_data.py probe                          # GO / NO-GO
python fetch_b_data.py assets | daily | opening | fullday | validate
python b_mvb.py --daily data/daily.csv --minutes data/minutes --variants variants_b.json \
       --start 2016-01-01 --end 2020-12-31 --nulls 20 --stress          # DEV-A
python fetch_b_data.py ticks --trades runs/<run>/trades.csv              # resolve R9
python b_mvb.py ...same... --ticks data/ticks.csv
```
- **Output:** `runs/<stamp>_B/report.md` + `summary.csv` + `trades.csv` + `config.json`, and one row per variant in `experiments/trials.csv`.

## 7. Status of the code (verified with synthetic data + a fake Alpaca only)
- **42 tests pass:**
  - known-answer fills (gap, same-bar stop, 1-tick-through trigger, short mirror, breakeven next-bar only, time stop, cost filter, OR stop cap, OHLC-path same-bar);
  - point-in-time features (split-scaled ATR, RVOL by sessions, RVOL future-perturbation);
  - guards (holdout once, ≤ 2 variants on the holdout, data gaps skipped, half days);
  - nulls (random names drawn only from names with full-day bars);
  - the day-major engine equals separate runs;
  - random walk → no gross edge; planted momentum → detected;
  - downloader: DST, retries, paging, splits, resume, holdout guard, one symbol mapping, data gate;
  - **download → backtest = identical trades**;
  - **tick resolution = the predicted result.**
- **Independent verifier** (`viz_b.py`) re-derived 626/626 trades in each of 5 versions; it fails on a deliberately wrong rule.
- **Never run against the real Alpaca API or real market data.** The probe is the first real check.
- **Not built yet:** SPA, PBO, random-skip null, DSR. They are needed before DEV-B, not before DEV-A.
- **Known approximations:** gross R with a partial exit is approximate (partial is unused in v3); the half spread is an assumed 3 bp until calibrated with SIP quotes.

## 8. Next steps, in order
1. Probe → download → validate (runbook).
2. Sign the prereg (rev 3), filling in the open choices.
3. DEV-A run → ticks → rerun → shortlist → build DSR/SPA/random-skip → DEV-B → holdout once → paper.
4. Trader's opening watch (30+ rows): tag each stop-out as chop / market / spread / news; after ~40 sessions, tags that separate winners from losers become new pre-registered knobs.
