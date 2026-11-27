---
name: backtest-and-experiment-rigor
description: Rules for causal (no-lookahead) features and backtests, the locked cost model and pass gate, random-entry nulls, stress tests, multiple-testing control, one-knob paired variant experiments, and live-vs-backtest monitoring. Use when writing or reviewing any feature, backtest, model, variant or result.
---

# Backtest and experiment rigor

Merged from the planned `no-lookahead-backtesting` + `experiment-variants` skills. Method ideas come from: Alpaca's backtest skill (formalize first, fingerprints), tradermonty's backtest-expert (stress tests), López de Prado (purging, embargo, "don't research under the influence of a backtest"), Harvey & Liu (higher t-bar under multiple testing), the Quantopian 888-algorithm study (more backtests = bigger live gap), Kevin Davey (incubation and failure monitoring), and the reversal literature already summarized in `research-report-phase1b-sentiment.md`.

**Philosophy:** look for the version that breaks least under pessimistic assumptions, not the one that earns most on paper. Spend about 20% of the effort building and 80% trying to break it.

**Team principle (expert-verdicts-strategy-A.md): EXPERIMENT MODE.** Change anything, any time, but version it. Explore fast on history; confirm slowly. Kill rules are signposts, not handcuffs. Rules below that look strict exist so that fast experimenting stays honest, not to slow it down.

**Locked rules win.** `preregistration-v1.md` fixes the signal, exits, costs, vetoes and pass gate for Strategy A's main hypothesis. This skill adds method; it never overrides the preregistration.

## 1. Research order: understand first, backtest last
1. **Event study** (no trading rules): what happens after the event, by run length, drop size, time of day, regime. Answers "is there anything here?" (this is phase 2b in `status-and-open-questions.md`).
2. **Hypothesis written down**: why the edge should exist and who pays us. For reversals the payer is liquidity demand: forced, non-news selling. Edges that fit that story are more believable.
3. **Backtest** = one test of a written rule. Rerunning after a tweak is fine, but it is a new trial (§5), never "the same test again".

## 2. Formalize before code
Write every rule precisely: data field, trigger, inclusive or exclusive bounds, warmup, order type, fill model, exit order, costs. Show it for confirmation unless it is already exact in the preregistration or the version config. Code must match the written rule; any difference is a bug.

## 3. Causal checklist (no lookahead)
A decision at time t may use only data whose end timestamp is ≤ t.
- [ ] Only closed bars. Early entry at minute 4:00 of a forming 5-min candle uses the 1-min bars that ended by 4:00, never the final 5-min bar.
- [ ] Rolling windows (ATR14, σ, 60-day RVOL median, 20-day $ volume) end before the decision. Daily inputs use `shift(1)`.
- [ ] No statistic over the whole sample: no full-period mean, z-score, scaler fit or quantile bucket. Fit on the past only (walk-forward).
- [ ] The expected-reversal table and any shrinkage estimate used as a backtest target are built from earlier dates only.
- [ ] Point-in-time universe from the prior close, including names delisted later (`securities.active = false`, `delisted_utc`). Join on `security_id`. If the source lacks delisted names, label results "survivor-biased".
- [ ] Prices: raw `bars` for intraday runs and P&L; `bars_adj` for cross-day windows; skip split ex-date ±1 day and ex-dividend days with dividend ≥ 1% of price.
- [ ] Events use the time they became public (8-K `acceptanceDateTime`, halt RSS time, news `created_at`, never `updated_at`).
- [ ] Regime inputs (VIX, VIX3M) use YESTERDAY's close unless an intraday value with a timestamp ≤ t exists.
- [ ] `merge_asof(direction="backward")`, and decide on purpose whether exact timestamp matches are allowed.
- [ ] Live decides on IEX; the backtest uses SIP. Measure signal agreement and report it.
- [ ] The future-perturbation test passes (see trading-system-conventions).

**Models (forming-candle logistic, meta-labeling, expected-reversal table):** labels look forward (a triple-barrier label can run 30 min), so a plain random split leaks the answer. Applies from week 4+ of the build plan, not before.
- Split by DATE, never shuffle rows.
- **Purge:** drop training samples whose label window overlaps any test sample's window.
- **Embargo:** also drop training samples from the first trading day after each test block.
- Score with Brier/log-loss plus a calibration plot; keep a model only if it beats the simple baseline (Φ(z)) out of sample.

## 4. Fills and costs (preregistration-v1)
- Signal time ≠ fill time. Fill = next 1-min bar open + slippage.
- Exits on 1-min bars. A limit fills only if price trades through it by ≥ 1 tick; touching is not enough.
- Stop and target in the same bar: assume the stop.
- 30-min time stop; flat by 15:55 ET (live flattens at 15:50). Never overnight.
- Costs per side: ½ spread (5 bp large caps / 20 bp small caps) + slippage at each of {0, 5, 15} bp + $0.005/share with a $0.35/order minimum. Run account sizes $10K / $25K / $50K.
- **Spread at the signal, not the average.** Spreads can widen right after a sharp drop, and that widening alone can erase a contrarian edge. The free Alpaca plan serves full-market SIP quotes older than 15 minutes, back to 2016, so this costs $0: fetch quotes only for a window around each signal minute (not whole days; the limit is 200 requests/min). Rerun costs with the measured spread. Never use IEX quotes for spread (IEX is a small slice of the market).
- Report results by listing exchange too (`securities.exchange`): NYSE and Nasdaq spreads behave differently after drops.
- Halts: block a signal if the symbol had a halt/LULD within ±15 min. Stops can gap through, so model the gap.
- Always report gross and net.

## 5. Null, statistics and multiple testing
- **The metric is edge vs a null, not "it reached ⅓ of the reversal".** That happens by chance 50–90% of the time.
- Null = matched random entries: same symbol, same day, same time-of-day bucket, identical exits and costs. Edge = strategy − null.
- Sanity check: under a random walk, P(target before stop) ≈ S / (S + T).
- Signals on the same day are correlated. Cluster standard errors by day.
- **Strategy A pass gate (prereg, locked):** net expectancy > null, day-clustered CI excludes 0, Deflated Sharpe with the TRUE trial count, and ≥ 500 trades and ≥ 100 distinct days in every reported bucket. Fewer = "not evaluable", never "promising". The DSR already corrects for multiple testing, so do not add a second t-hurdle to this gate.
- **New strategies not yet preregistered (B, D, backlog ideas):** until they have their own preregistration with a DSR gate, use t ≥ 3 rather than 2 as the bar for "worth pursuing". With many ideas tried, t = 2 is too easy to hit by luck.
- **Trial log** (`experiments/trials.csv`): one row per backtest run of any rule, including failed and abandoned ones. It feeds the DSR trial count, so it must be complete. Industry data shows the more backtests run on an idea, the bigger the gap to live results. Signpost (not a cap): past ~20 runs on one idea, pause and ask out loud whether you are fitting noise; write the answer in the changelog.
- The holdout is opened ONCE, only for the frozen primary + one backup.

## 6. Stress tests (before anything is promoted)
- Stop distance at 50 / 75 / 100 / 125 / 150% of baseline; target at 80–120%.
- Entry timing: minute 3 / 4 / 4.5 / candle close, including fake greens that close red.
- Slippage 15 bp and 2× that; worst-case fills (buy at ask + 1 tick, sell at bid − 1 tick); measured signal-time spread (§4).
- Year by year (positive in most years), time of day, large vs small caps, NYSE vs Nasdaq.
- **Regime, split 4 ways** using the labels in `research-report-phase1b-sentiment.md` appendix A: CALM / NORMAL / ELEVATED-STABLE / STRESS-RISING. Your own research found two opposite effects: a HIGH but stable VIX level tends to raise reversal profits, while RISING stress (VIX/VIX3M ≥ 1 with VIX still climbing) favours momentum and reversals lose. So:
  - Keep the prereg STRESS-RISING veto.
  - Shadow-log vetoed signals and report what the veto saved or cost.
  - Any new filter that skips ELEVATED-STABLE days must prove itself in a variant; never add it by intuition.
- **News vs no news:** reversals after non-news drops are expected to be stronger. Report the NONE tag cell separately; if the edge only exists in news cells, suspect a bug.
- Look for plateaus, not sharp peaks. Out-of-sample under 50% of in-sample is a warning.

**Red flags:** win rate > 90% with tiny drawdowns; profit from a handful of days; edge gone with a 1-tick worse fill or the signal-time spread; any profit on `random_walk_day` data; a net Sharpe far above what costs and capacity allow (treat as a bug until proven otherwise).

## 7. Variant experiments
**Design**
- Each variant differs from the baseline in exactly ONE knob.
- Two interacting knobs (e.g. run length N × target): a 2×2 or 3×2 grid. Never 3+ knobs at once.
- 4–8 active variants per cycle, at most. Every variant run also goes in the trial log.
- Where it runs: signal and exit knobs in backtest first, then live shadow. Execution knobs (limit vs market, reprice timing) need real paper orders: acct1 runs variant X, acct2 runs variant Y.

**Cycle file** (`experiments/cycles/C<k>.yaml`), committed BEFORE the cycle starts:
```yaml
cycle: C3
baseline: A-v1__base
start_ny: 2026-12-01
end_ny: 2026-12-12
variants: [A-v1__entry_min=4.0, A-v1__target=0.5]
rules: {kill_min_signals: 100, kill_t: -2, promote_t: 2}
data: {where: backtest+shadow, period: DEV}
```
Thresholds never change mid-cycle.

**Registry** (`experiments/variants.yaml`)
```yaml
- id: A-v1__entry_min=4.0
  base: A-v1__base
  knob: entry_minute
  value: 4.0
  hypothesis: "Earlier entry captures more of the reversal than fake greens cost"
  where: backtest+shadow      # or live_acct2
  cycle: C3
  status: active              # active | killed | candidate | promoted
  created: 2026-11-26
```

**Paired comparison.** Use the union of signals seen by the variant and the baseline; a signal a version skipped counts as 0 R for that version. Δᵢ = R_variant,i − R_base,i.
```python
import numpy as np
from scipy import stats

def paired_delta(df):
    """df columns: day, r_var, r_base (in R, one row per signal)."""
    d = df["r_var"] - df["r_base"]
    n, mean = len(d), d.mean()
    g = (d - mean).groupby(df["day"]).sum()      # day clusters
    G = len(g)
    se = np.sqrt(G / (G - 1) * (g ** 2).sum()) / n
    t = mean / se
    q = stats.t.ppf(0.975, G - 1)
    return {"n": n, "days": G, "mean_dR": mean, "se": se, "t": t,
            "ci95": (mean - q * se, mean + q * se)}
```
Tested on simulated data with no true difference and 40 days: |t| > 2 fired 6.4% of the time instead of 5%. Report the number of days next to every t.

**Fail-fast rules** (signposts: a hit means stop and rethink, written in the changelog)
- **Kill a variant:** ≥ 100 shared signals and t < −2.
- **Candidate:** t > 2. It becomes the baseline only after winning again on the NEXT cycle's fresh data (with K variants, the best one is partly luck).
- **Kill the idea:** backtest gross edge vs the null ≤ 0. No tuning saves a signal with no edge.
- These rules choose between variants. A new BASELINE for real money still needs the prereg pass gate (§5).

**Change log** (`experiments/changelog.md`, append-only):
`date | cycle | action | variant | n | days | mean ΔR | t | decision | decided by`

**Per-stock auto-tuning:** hierarchical shrinkage stock → sector → market; monthly only; a change needs n ≥ 100 and t > 2; capped at ±25%; logged with auto-revert.

## 8. Paper and live monitoring (incubation)
About 35 trades a week gives an SE near 10 bps, so one week proves nothing about the edge. History discovers and chooses rules; paper verifies the implementation.
- **Once the backtest exists** (phase 2c), build an expected range: resample its trades (Monte Carlo, ≥ 5,000 paths, seeded) to get the 5th / 50th / 95th percentile of cumulative R and max drawdown after 25, 50, 100, 200 trades. Save it as `runs/<id>/expected_cone.json`. Until then, paper is judged on implementation checks only.
- **Weekly review** (already the team's rhythm): plot paper results (on shadow fills) against the cone.
  - Below the 5th percentile, or drawdown beyond the backtest's 95th percentile → check the implementation (signal agreement, fills, costs) before blaming the market.
  - ABOVE the 95th percentile → also investigate. Unexpected over-performance is often a bug (lookahead in live code, wrong fills).
- Implementation checks every week: IEX/SIP signal agreement, fill rate, slippage vs the model, trade duration and time-of-day mix vs the backtest. A different mix means live is not running the tested strategy.

## 9. Reproducible runs
Folder: `runs/<YYYYMMDD>_<version>_<run_id>/`
- `notes.md`: the request, the formalized rules, every assumption, and exactly what changed vs the previous run.
- `config.json`, `data_fingerprint.json` (source, feed, adjustment, symbols or universe rule, date range, session, row counts, hash), `trades.parquet`, `summary.json`, `report.md`.
- Seed all randomness, including the null and the Monte Carlo cone.
- Reuse cached data only if the fingerprint matches.

## 10. When reviewing a backtest, answer in this order
1. Verdict: KILL / REFINE / CANDIDATE / NOT EVALUABLE, with blocking reasons.
2. Causal checklist: each item pass / fail / unknown.
3. Trades, days, gross and net mean R, edge vs null with day-clustered t, and the trial count so far.
4. Stress-test table (incl. signal-time spread, 4-way regime split, news vs no-news, exchange).
5. Red flags found.
6. What was not checked.
