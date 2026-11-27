---
name: order-risk-and-execution
description: Safety rules for the Strategy A order path: paper-only gate, R sizing, risk limits, the risk.py contract, Alpaca bracket orders per tranche, client_order_id, reprice, reconciliation, kill switch and shadow fills. Use when code can size, place, change or cancel an order.
---

# Order risk and execution

Merged from the planned `risk-and-safety` + `alpaca-paper-execution` skills, with guardrails from Alpaca's official paper-trading and reconciliation-idempotency skills.
Risk numbers are the PAPER column of `strategy-A-design-plan.md` §2 (still marked "decision open"). Load them from `config/risk.yaml`; if the doc changes, the config changes, not the code.

## 1. Paper-only gate
- Create the client with a literal: `TradingClient(key, secret, paper=True)`. Never read the paper flag from config or env.
- At startup assert the base URL contains `paper-api`. If any live URL or live key is detected, exit non-zero and send a Telegram alert.
- Never write code that switches to live. Going live (IBKR) is a separate team decision, gated by design-plan §8.
- Secrets come only from env / a gitignored `.env`. Never print them in logs, reports, Telegram or chat. Redact account ids.

## 2. Order path (no bypass)
`signal → variants → risk.approve() → exec → broker`
- Every new entry goes through `risk.approve()`.
- Flatten (`/flat`, 15:50 job, daily stop) is always allowed and skips sizing, but still goes through `exec` and is journalled.

## 3. Sizing (R)
```
R_usd     = equity × risk_pct                        # paper: 0.5%
min_dist  = max(1·ATR14(5m), 0.005 × entry)
stop_px   = min(run_low − 1 tick, entry − min_dist)  # the stop actually placed
qty       = floor(R_usd / (entry − stop_px + gap_buffer))
```
- Size on the stop you actually place. Never size on one distance and place the stop at another.
- ⚠ Design-plan wording "≥ 1 ATR or 0.5% of price" is ambiguous; `max` (the wider) is used here as the safer reading until the team confirms.
- Then cap qty: notional ≤ 20% of equity, and a share of the stock's 5-min $ volume. ⚠ The docs disagree (risk table ≤ 5%, universe rules ≤ 1%). Use the stricter 1% until the team decides.
- Tranches split 1 R; they never add to it.
- Add only ABOVE the previous fill. Never average down.

## 4. Portfolio limits (paper)
| Limit | Value |
|---|---|
| Max concurrent positions | 4 |
| Max total open risk | 2% of equity |
| Sector | max 2 per sector; ≥ 3 signals within 10 min with SPY red = one market bet, take only the best |
| Re-entry | 1 per symbol per day, counts as a new R |
| Daily stop | −2% or −4 R → flatten, no new entries today |
| Weekly | −4% → pause and human review |
| Overnight | never; independent flatten job at 15:50 ET |

## 5. `risk.py` contract
```python
@dataclass(frozen=True)
class OrderIntent:
    security_id: str; symbol: str; side: str
    entry_px: Decimal; stop_px: Decimal
    tranche: int; variant_id: str; signal_id: str

@dataclass(frozen=True)
class RiskDecision:
    approved: bool; qty: int; reasons: tuple[str, ...]

def approve(intent: OrderIntent, state: PortfolioState,
            limits: RiskLimits) -> RiskDecision: ...
```
- Pure and deterministic: no I/O, no clock calls (time comes in via `state`).
- Every rejection carries reason codes and is journalled.
- Each limit is unit-tested at, just below and just above its boundary.
- If state is unknown or stale (reconciliation failed or overdue): reject.

## 6. Placing orders (Alpaca, alpaca-py)
**Why one bracket per tranche.** Alpaca reserves shares for every open sell order: the total qty of open sell orders cannot exceed the shares held, otherwise you get `403 insufficient qty available` (`held_for_orders`). Inside one bracket, the second exit leg sits "held" until the first is cancelled. So a separate full-size stop plus extra take-profit orders on the same shares will be rejected.
- Each entry tranche is its own **bracket** (entry limit + take-profit limit + stop) or **OTO** (entry + stop only), `time_in_force=day`, regular hours only. 4 tranches = 4 brackets; their exit legs add up to the position, so nothing is rejected.
- Every fill must have a resting stop at the broker.
- **Breakeven after tranche 1:** replace the stop leg of EACH remaining bracket separately. A leg is its own order (`parent.legs`, fetch with `nested=True`); each needs its own replace call.
- **client_order_id** (deterministic, ≤ 128 chars, unique):
  `{version}-{variant}-{SYMBOL}-{YYYYMMDD}-{signal_seq:04d}-{leg}{tranche}-r{attempt}`
  e.g. `Av1-base-AAPL-20261126-0007-E1-r0` (leg E = entry, X = exit, F = flatten).
- Write the intent row to SQLite BEFORE the network call. On a timeout or lost response, call `get_order_by_client_id` before any retry. Never retry blindly.
- **Reprice:** limit slightly through the price when there is momentum. Not filled within 10 s → `replace_order_by_id` with a new limit price and the next `client_order_id` (`r{attempt+1}`). A replace creates a NEW order id: store it and use it from now on. The old order then gets a `replaced` event. Max 2 reprices; max chase 0.3% from the signal price; then cancel and journal `NO_FILL`.
- **Partial fills:** risk counts filled qty only. After cancelling the unfilled rest of an entry, read the legs back and confirm the stop covers exactly the filled qty; if not, fix it (verify this leg behaviour on paper first and record what you saw).
- Throttle to ≤ 10 orders/min. Error codes: 403 = buying power or shares held by other orders (not auth); 422 = bad input or not tradable; 429 = back off using the rate-limit headers; 401 = stop, do not retry.
- No entry within 10 min after a halt resumes; skip names near LULD bands.

## 7. Reconciliation (the broker is the source of truth)
- Use the `trade_updates` stream for speed. Events can repeat or arrive out of order: dedupe on execution id and check current status before acting.
- Every 30–60 s also pull positions and open orders (`nested=True`) and compare with local state:
  - Position with no protective stop → place the stop now, or flatten.
  - Unknown order or position → Telegram alert and halt new entries.
  - Qty mismatch → broker wins; log a correction row; halt new entries until it matches.
- After any restart: reconcile first, trade later.

## 8. Kill switch and flatten
Triggers: Telegram `/flat`, the dashboard button, the daily stop, the 15:50 job.
1. Set a persisted `HALT` flag in SQLite. The engine refuses new entries until a human resets it.
2. Cancel ALL open orders, including bracket legs, and wait until each cancel is confirmed. Shares stay reserved while sell orders are open, so closing first fails with "insufficient qty".
3. Close all positions (market), then confirm on Telegram with what was closed.
4. If anything is still open after 30 s: alert loudly and retry once; the broker UI is the manual fallback.

## 9. Shadow fills (paper is too generous)
Alpaca paper has no queue, size or impact: a limit fills as soon as it is touched. Log a pessimistic shadow fill next to every paper fill:
- Limit: fills only if price trades 1 tick through it.
- Market: buy at ask + 1 tick, sell at bid − 1 tick.
Report both. Promotion decisions and the go-live gate (≥ 3 months, 150–300 trades, positive net expectancy, slippage ≤ 1.5× the model) use SHADOW results.

## 10. Never do
- Never submit to a live account, or write code that can.
- Never place an entry that bypasses `risk.approve()`.
- Never leave a filled position without a broker-side stop.
- Never retry an order without checking whether the first one arrived.
- Never average down, hold overnight, or loosen a limit in code to "make it fit".
- Never auto-submit in a loop without a rate limit and a journal row.
- Never treat paper P&L as a forecast of live P&L.
- Never print secrets or account numbers.

## 11. Checklist for any change on the order path
1. Does every entry still go through `risk.approve()`?
2. Is every fill protected by a stop at the broker, sized to the filled qty?
3. Is the `client_order_id` deterministic, and checked before any retry?
4. After a replace, does the code track the new order id?
5. Do risk tests still cover each limit's boundary?
6. Does reconciliation still catch an unprotected or unknown position?
7. Does flatten cancel and confirm before closing, and still work while halted?
