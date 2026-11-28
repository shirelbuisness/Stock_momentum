# Strategy B: paper order path, design for sign-off (DRAFT, no code yet)
Date: 28.11.2026 (night session 4). Status: **proposal. Nothing here is built. No order code is written until both partners sign this doc.**
Rules it follows: `.claude/skills/order-risk-and-execution` (written for Strategy A; where B needs something different it is marked **B-diff** and listed under "Decisions" at the end). CLAUDE.md: paper only, never code that can reach a live account.

## 0. Why this doc exists
The paper → live gate (`paper_gate_b.py`) needs `paper_fills.csv` = real Alpaca **paper** fills joined to the model's reference prices (`model_refs.csv`, written by `shadow_day_b.py`). The shadow logger already reproduces the backtest exactly without orders. The only missing piece is a process that sends the backtest's orders to the paper account and records what came back. This doc fixes how, so that paper differs from the backtest only by real fills.

## 1. Scope
- **In:** the one version that passed the holdout (prereg: "Only the version that passed, with its filters, runs in paper"), **long side only** (B0-primary is long-only; shorts stay a backtest shadow).
- **Out:** live trading, IBKR, shorts, stage-2 combinations, any order not in the backtest (no discretionary adds, no reprices the backtest does not model).
- **Principle:** the paper process must place the *same* orders the backtest simulates, at the *same* times. Every deviation (risk rejection, halt rule, cap) is journalled with a reason code so the gap can be measured, not guessed.

## 2. Daily timeline (ET; half days shift the close to 13:00)
| Time | Step | Source of truth |
|---|---|---|
| 08:30 | Startup: paper-URL assertion (§3), reconcile (§8): must be flat with no open orders; read `HALT` flag | broker |
| 09:00 | History ready (`daily.csv` + opening files up to D−1); loose universe as in `dry_run_b.day_universe` | disk |
| 09:35:00 + ε | Selection = `dry_run_b.prepare/select` on bars closed by 09:35 (F6/F9: 09:30 + 2 × OR). Writes `live_selection.csv` | market data |
| 09:35:05 | For each candidate: `risk_b.approve(intent)` → submit an entry order (§4). ≤ 10 orders/min throttle **B-diff** (see D5) | broker |
| on each fill | Place / correct the protective stop for the filled qty (§4.2) | trade_updates |
| every 30–60 s | Reconcile positions + open orders (`nested=True`) vs local state (§8) | broker |
| 15:59:00 | Close: cancel every open B order (entries and stop legs), wait for confirmed cancels, then market-sell each position (§5) | broker |
| 15:59:30 | Independent safety check: anything still open → flatten path (§7) + alert | broker |
| ~16:20 | `shadow_day_b.py --day D` (model trades + `model_refs.csv`), then the fill joiner writes `paper_fills.csv` (§9) | disk |

The backtest exits on the close of the 15:58 bar = the price at ~15:59:00. A market sell sent at 15:59:00 is the closest live equivalent. (MOC is not used: NYSE/Nasdaq MOC entry cut-offs are at 15:50, and the closing auction price is not the price the backtest uses.)

## 3. Paper-only gate (unchanged from the skill)
- `TradingClient(key, secret, paper=True)` with the literal `True`; never from config or env.
- At startup assert the base URL contains `paper-api`; anything else → exit non-zero + alert.
- The module that can submit orders lives in its own file (`paper_exec_b.py`), imported by nothing in the backtest. A test fails if any other `strategy_b/*.py` imports it, and if any file contains a live trading host string (same style as the existing purity tests of `dry_run_b.py` / `shadow_day_b.py`).
- Keys from env / `.env` only; never printed; account id redacted in every log.

## 4. Orders per name
### 4.1 Entry
The backtest enters with a **stop order 1 tick through the OR high** (`dry_run_b.intents` already builds it: `stop_price = OR high + 0.01`), filled at the trigger or at the bar open if it gapped through, day TIF, regular hours, cancelled at the close if unfilled. No reprice, no chase: a stop entry has nothing to reprice, and the backtest models none.
- Proposed broker form: **OTO** (entry `type=stop` + a `stop_loss` leg), `time_in_force=day`, `extended_hours=false`.
- ⚠ **Not verified:** whether Alpaca accepts `order_class=oto` (or `bracket`) with a `stop` entry. First paper check (§11, P1). Fallback if not: a plain stop entry, and on its fill event a separate stop order for the filled qty (unprotected for the fill → submit latency; measured and logged per trade; reconciliation catches a missing stop).
### 4.2 Protective stop
- Backtest: stop = **fill − 10% ATR14** (version knob; L3 uses the OR low with floor/cap). The distance is known at 09:35; the level only after the fill.
- At submission the OTO stop leg is set from the **trigger**: `trigger − dist`. After the fill event: if `fill > trigger` (gap through), **replace** the leg to `fill − dist` (tighter, never looser; the skill's "never loosen" holds). A replace makes a new order id → store it.
- Partial fills: the stop must cover exactly the filled qty. After the 15:59 cancel of an unfilled rest, read the legs back (the skill's rule; record what Alpaca actually does on paper, P3).
- Every filled share has a resting stop at the broker. A position seen without one by reconciliation → place it now, or flatten (skill §7).
### 4.3 Exit at the close
- 15:59:00: cancel all B orders → wait for each cancel to confirm (shares stay reserved while a sell order is open, so selling first would get `403 insufficient qty`) → market sell the full position.
- If a stop fills while the close sequence runs, the position is already flat: the market sell is computed from the broker's position, not from local state.

## 5. Sizing
- **Paper sizing = the prereg's Sharpe sizing proposal, 1× equal notional:** `qty = floor(equity / top_n / stop_price)` (already `qty_1x_equal_notional` in the dry run). Long-only, so total notional ≤ 1× equity.
- The R of a B trade is its stop distance × qty (= what `r_net` is measured in). It is small at 10% ATR (about 0.025% of equity per name for a 5% ATR/price name) and is reported, not targeted. **B-diff:** the skill's "R_usd = 0.5% of equity, qty from the stop" is A's sizing; B does not use it, because the backtest's Sharpe and the pass gate are computed on equal notional (D1).
- `equity` = the paper account's equity at 09:00 (frozen for the day).

## 6. `risk_b.approve()` contract
Same shape as the skill's `risk.py` (pure, deterministic, no I/O, no clock; time comes in through `state`; every rejection has reason codes and is journalled).
```python
@dataclass(frozen=True)
class OrderIntentB:
    session_date_ny: str; ticker: str; side: str          # "buy" only in paper
    stop_price: Decimal                                   # entry trigger (1 tick through)
    stop_distance: Decimal                                # protective stop distance from the fill
    qty_requested: int; variant_id: str; signal_seq: int

@dataclass(frozen=True)
class RiskDecisionB:
    approved: bool; qty: int; reasons: tuple[str, ...]
```
Checks (proposed values, PAPER; all in `config/risk_b.yaml`, none in code):
| Check | Proposed | Reason code |
|---|---|---|
| Side | long only | `SHORT_NOT_ALLOWED` |
| State fresh (last reconcile ≤ 120 s, no mismatch) | required | `STATE_STALE` |
| `HALT` flag | not set | `HALTED` |
| Max concurrent positions + working entries | top_n (20) **B-diff** (D2) | `MAX_POSITIONS` |
| Notional per name | ≤ equity / top_n (+1 share rounding) | `NOTIONAL_CAP` |
| Total notional incl. working entries | ≤ 1.0 × equity | `GROSS_CAP` |
| Total open risk (Σ qty × dist) | ≤ 2% of equity | `OPEN_RISK` |
| Share of the name's 09:30–09:35 $ volume | ≤ 1% (skill's stricter value) | `LIQUIDITY_CAP` → qty cut, not rejected |
| Daily stop | day P&L ≤ −2% of equity → no new entries + flatten | `DAILY_STOP` |
| Weekly | −4% → pause + human review | `WEEKLY_STOP` |
| One entry per name per day | required (backtest: 1 trade per name) | `DUP_NAME` |
| Entry after the version's entry cut-off | refused | `AFTER_CUTOFF` |
| Halt resumed < 10 min ago / near LULD band | refused **B-diff** (D4) | `HALT_RECENT` / `LULD` |
Each limit gets the skill's boundary tests (at, just below, just above).

## 7. client_order_id, journal, kill switch
- Format (skill): `{version}-{variant}-{SYMBOL}-{YYYYMMDD}-{seq:04d}-{leg}{tranche}-r{attempt}`; B has one tranche. Legs: `E1` entry, `S1` stop leg (and its replacements `-r1`, `-r2`), `C1` close sell, `F1` flatten. Example: `Bv1-base-AAPL-20261126-0007-E1-r0`. `seq` = the candidate's rank at 09:35 (as in `dry_run_b.intents`), so the id is the same on a restart and a duplicate submit is impossible.
- Journal (SQLite): the intent row is written **before** the network call; on a timeout, `get_order_by_client_id` before any retry; never a blind retry.
- Kill switch / flatten: exactly skill §8 (persisted `HALT` → cancel all and confirm → close all → confirm; retry once after 30 s; broker UI as the manual fallback). Flatten is always allowed and skips `approve()`.

## 8. Reconciliation
Skill §7 unchanged: `trade_updates` stream (dedupe on execution id), plus a pull every 30–60 s; the broker wins; unknown order/position or qty mismatch → alert + halt new entries; restart → reconcile first. B adds one check: **after 15:59:30 there must be zero B positions and zero B orders**; otherwise flatten + alert.

## 9. What paper writes (the join with the shadow logger)
- `paper_orders.csv` (journal export): every order event with client_order_id, broker order id, status, qty, fill price, time.
- `paper_fills.csv` (the format `paper_gate_b.py` already reads): `side (+1/−1), ref_px, fill_px, qty`, plus `day, ticker, variant_id, leg` for auditing. Built after the close by joining the paper fills to `model_refs.csv` on `(day, ticker, variant_id, leg)`: entry fill ↔ entry ref; stop or close fill ↔ exit ref.
- A paper fill with no model ref (paper traded, the model did not), or a model ref with no paper fill, is a **mismatch row**, not dropped. These rows are the measurement of "paper ≠ backtest" and feed the overlap and expectancy checks.
- **Shadow fills (skill §9):** the backtest already requires 1 tick through for the entry and charges half spread + k, so the model ref + model cost is the pessimistic shadow. Promotion and the go-live gate use the shadow `r_net` from `paper_trades.csv`; Alpaca's own fills are used only for the slippage ratio.

## 10. Tests required before the first paper order (all offline, against an extended `fake_alpaca.py`)
1. Purity: no live host string anywhere; only `paper_exec_b.py` imports the trading client; `paper=True` literal.
2. A synthetic day through fake broker + fake market → the orders sent equal `dry_run_b.intents` (prices, qty, ids).
3. Gap-through fill → stop leg replaced to fill − dist; the new order id is tracked.
4. Lost response on submit → lookup by client_order_id, no duplicate.
5. 15:59 sequence: cancel confirmed before the sell; a stop that fills during the sequence → no oversell.
6. Kill switch while orders are working → everything cancelled, positions closed, `HALT` persisted.
7. Reconciliation: a position without a stop → stop placed or flatten; an unknown order → halt.
8. `risk_b.approve` boundary tests for every row of §6.
9. End to end: paper fills on the fake = the model refs → `paper_gate_b` slippage ratio ≈ 1; fills 1 cent worse → ratio as computed by hand.

## 11. First checks on the real paper account (manual, logged, before automation)
- P1: does Alpaca accept OTO/bracket with a `stop` entry? What does the order look like with `nested=True`?
- P2: replace of a stop leg after the parent fills: allowed? new id?
- P3: partial fill of the entry, then cancel of the rest: does the stop leg's qty follow the filled qty?
- P4: market sell at 15:59:00 on a thin name: fill time and price vs the 15:58 bar close.
- P5: rate limits and latency from 09:35:00 to 20 accepted orders.

## Decisions for Shirel and the trader (none is decided here)
- **D1 Sizing:** paper at 1× equal notional (matches the prereg's Sharpe sizing proposal and the backtest), not the skill's 0.5%-of-equity R sizing. Equal-$-risk sizing would change every result → only as a new pre-registered version.
- **D2 Position limits:** the skill's "max 4 concurrent / max 2 per sector / one market bet" rules are Strategy A's. Applying them to B would make paper trade a different strategy from the backtest. Proposal: for B, max concurrent = top_n, no sector cap, keep the 2% open-risk cap (does not bind at 10% ATR; can bind for L2 at 50% ATR: ≈ 20 × 0.05 × 0.5 × 5% ≈ 2.5%) and the daily/weekly stops. Every rejection is logged so the effect is measured.
- **D3 Close time:** the skill's independent flatten at **15:50** vs B's exit at **15:59**. Proposal: primary exit 15:59:00, independent safety flatten 15:59:30. Flattening at 15:50 would make every paper trade differ from the backtest by 9 minutes.
- **D4 Halts/LULD:** the skill refuses entries within 10 min after a halt resumes. The backtest does not model halts. Keep the rule (safety), log `HALT_RECENT` rejections, and count them in the overlap check?
- **D5 Throttle:** the skill's ≤ 10 orders/min means 20 entries take 2 minutes; entries late by up to 2 minutes miss early breakouts the backtest takes. Alpaca's documented limit is per API calls (about 200/min). Proposal: a burst of ≤ 20 entries at 09:35, then ≤ 10/min.
- **D6 Feed at 09:35:** same as night-log Decision 12 (IEX vs a paid SIP plan). Paper cannot start until the IEX-vs-SIP overlap is measured on real data.
