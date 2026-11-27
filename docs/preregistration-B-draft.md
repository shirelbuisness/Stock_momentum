# Preregistration B (DRAFT rev 3, not signed): ORB on Stocks in Play
Date: 27.11.2026. Must be signed by both partners before the first real DEV run. Any change after signing = v2 with a date and a reason. rev 3 = rev 2 + the loss-reduction versions and the DEV-A/DEV-B protocol from `strategy-B-final-plan-v3.md`.

## Hypotheses
- **H1:** Trading in the direction of the first 5-min candle on the top-20 RVOL stocks earns positive net expectancy per trade after realistic costs.
- **H2:** The edge comes from RVOL selection. B0 beats (a) a random-entry null and (b) a random top-N drawn from the eligible universe, with the same rules.
- **H3 (loss reduction):** at least one pre-listed version lowers losses (a wider stop, cutting stalled trades, cost/quality filters) and beats B0 on a paired day-level test.

## B0: primary spec (exact replication)
- **Universe:** open > $5, 14-day avg volume ≥ 1M, ATR14 > $0.50, RVOL ≥ 100%, top 20 by RVOL. Common stocks incl. delisted. Every filter uses prior-day data only (raw prices for the price filter; split-scaled for ATR and the volume baseline). The > $5 filter uses day t's 09:30 open.
- **Direction:** green first bar → long, red → short, doji → no trade (rank first, then skip).
- **Two readings:**
  - **B0-replication** (long + short, 4×, commission only): a code check, never the verdict.
  - **B0-primary** (the verdict): long-only, 1×, full cost model. Shorts are a shadow with SSR/ETB flags.
- **Entry:** stop at the OR high or low (needs 1 tick through). **Stop:** 10% ATR from the fill. **Exit:** the close of the 15:58 bar; half days the last bar. Unfilled entries cancelled at the close. Regular hours only.
- **Fill model:** pessimistic 1-min rules (plan rev 2 §5), as implemented in `strategy_b/b_mvb.py`.

## Periods
- **DEV-A:** 2016-01-01 → 2020-12-31 (explore).
- **DEV-B:** 2021-01-01 → 2023-12-31 (confirm).
- **Sanity gate on B0-replication over DEV:** commission-only gross Sharpe ≥ 2.0; failing it points to data or code.
- **HOLDOUT:** 2024-01-01 → 2025-12-31, behind the guard, **run once** (base + at most 2 promoted variants), same code_sha.

## Cost model (locked)
- $0.0035/share + half spread (3 bp until calibrated with SIP quotes, then the measured value) + k slippage per side, with base k = max(1 tick, 2 bp); stress at 2k.
- Borrow 0 in the base case; stress = 30 bp/yr with HTB names skipped.
- **Breakeven slippage** = the slippage per side, on top of commission and half the spread, at which net mean R = 0.

## Pre-registered versions (one knob each vs B0-primary; every run counts as a trial)
- **Loss reduction:** L1 stop 25% ATR · L2 stop 50% ATR · L3 stop at OR low (floor max(10% ATR, 2× cost), cap 50% ATR) · L4 time stop (10:30 close < +0.5R → exit at 10:31 open) · L5 no entries after 10:30.
- **Filters:** F1 skip if stop < [5× or 3×, pick at signing] round-trip cost · F2 RVOL ≥ 200% · F3 top-10 · F4 OR range ≤ 0.6 ATR · F5 first bar closes in its top 30% · F6 trigger at the 2nd 5-min bar · F7 price ≥ $10 · F8 SPY prior close → 09:35 up · F9 15-min OR.
- **Stage 2** (only on DEV-B survivors, ≤ 2 knobs, +1 trial each): L1 + breakeven after +1R after 10:00; L1 + trail after +3R; the best filter + the best exit.

## Protocol
1. DEV-A: run all 15 → paired day-level test (Newey-West, 5 lags) + Holm across 14 → shortlist ≤ 4 with Holm p < 0.10 and a positive effect.
2. DEV-B: each shortlisted version needs all of these:
   - paired t ≥ 2;
   - the same sign as DEV-A and ≥ half its size;
   - DSR ≥ 0.95, with N taken from the trial log;
   - positive at 2k slippage.
3. Every filter must beat a random-skip null (skip the same share of days/trades at random).
4. Holdout: Holm across the promoted versions (with 2, the first needs p < 0.025).

## Pass gate (HOLDOUT). All must hold
1. Mean net R per trade ≥ **[+0.05R or +0.08R, the team picks at signing]**, with t ≥ 2.0 (clustered by day).
2. Annualised net daily Sharpe ≥ 1.0.
3. Beats the 95th percentile of the random-entry and random-names nulls.
4. DSR probability ≥ 0.95, with N = every trial in the log.
5. Still positive at 2k slippage; max DD < 25% at 1×.
6. ≥ 1,000 trades; ≥ 6 of 8 quarters positive; ≥ 60% of months positive; positive with the top-5 tickers removed.
7. Breakeven slippage ≥ **[2 ticks or 1 tick per side, pick at signing]**.
- Passing only at base k but not 2k → "fragile", not "alive".
- **Healthy-shape check:** profit factor 1.2–2.5. Above 2.5, or zero losing months, → treat as a bug until explained.

## Paper → live gate
≥ 3 months of paper, with:
- realised slippage ≤ 1.5× the model;
- live/backtest top-20 overlap ≥ 80% (SIP feed);
- live expectancy inside the backtest's 95% CI.

Only the version that passed, with its filters, runs in paper.

Signatures: ______ (developer) ______ (trader)
