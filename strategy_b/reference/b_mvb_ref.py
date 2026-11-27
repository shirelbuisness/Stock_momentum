#!/usr/bin/env python3
# FROZEN REFERENCE (night 27.11.2026): b_mvb.py at commit 5df263c, BEFORE the speed work. Never edit.
# test_speed_equivalence.py runs it next to the optimized b_mvb.py and requires identical outputs.
"""
b_mvb.py: minimum-viable multi-variant backtester for Strategy B
(ORB on Stocks in Play, Zarattini, Barbon & Aziz 2024), with a loss-reduction track.

Throwaway research script (plan step S2/S3). It is folded into the shared engine in S4.
Every result includes ALL costs: commission + half spread + slippage on every side.

INPUTS (all times America/New_York, bars labelled by START time):
  --daily   CSV: ticker,date,open,high,low,close,volume[,split_factor]
            RAW (as-traded) daily bars. split_factor = ratio on its ex-date (2.0 for 2:1), else blank/1.
            The script back-adjusts point-in-time, so no future split leaks into day t.
  --minutes DIR with minutes_YYYY-MM-DD.csv: ticker,time,o,h,l,c,v   (time = "HH:MM", RTH only)
            Must hold 09:30-09:44 bars for the whole loose universe (RVOL baseline needs them every day)
            and full-day bars for the top-RVOL candidates. Optional ticker SPY for the regime filter.

OUTPUT: runs/<stamp>_B/ with report.md, summary.csv, trades.csv, config.json; one row per run appended
to experiments/trials.csv (the trial log that feeds the Deflated Sharpe count).

HOLDOUT GUARD: any date >= 2024-01-01 is refused unless --open-holdout is given.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime

import numpy as np
import pandas as pd

HOLDOUT_START = "2024-01-01"
TICK = 0.01


# ----------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class Knobs:
    # universe (paper, verified)
    min_price: float = 5.0          # day-t 09:30 open (raw)
    min_avg_vol: float = 1_000_000  # 14 prior days, shares, split-adjusted to day t
    min_atr: float = 0.50           # ATR14 of prior days, $ in day-t scale
    rvol_min: float = 1.0           # 100%
    top_n: int = 20
    hist_days: int = 14
    min_hist_days: int = 10         # valid sessions needed in the 14-session window (data plan: >= 10 of 14)
    # signal / entry
    or_minutes: int = 5             # opening range length
    entry_bar2: bool = False        # F6: confirmation. Trigger = max(OR high, 2nd bar high), entries after bar 2
    sides: str = "long"             # "long" | "both"
    entry_cutoff: str = "15:58"     # last bar START at which an entry may trigger
    spy_filter: bool = False        # longs only if SPY prior close -> OR close return > 0 (shorts mirror)
    max_or_range_atr: float = 0.0   # 0 = off; skip if OR range > x * ATR (exhausted move)
    close_loc_min: float = 0.0      # 0 = off; long needs OR close in top (1-x) of OR range, e.g. 0.7
    min_stop_cost_mult: float = 0.0 # 0 = off; skip if stop distance < x * round-trip cost per share
    # stop / exits
    stop_atr: float = 0.10          # stop distance = stop_atr * ATR, from the executed entry price
    stop_mode: str = "atr"          # "atr" | "or_opposite": other side of the OR, floored at
                                    #   max(stop_atr*ATR, 2x round-trip cost) and capped at 0.5*ATR
    breakeven_r: float = 0.0        # 0 = off; after high >= entry + x R, stop -> entry + cost (next bar on)
    trail_after_r: float = 0.0      # 0 = off; after +x R, trail at highest high - trail_atr*ATR
    trail_atr: float = 0.25
    time_stop: str = ""             # "" = off; if the CLOSE of this bar < entry + time_stop_r R,
    time_stop_r: float = 0.5        #   exit at the next bar's open
    breakeven_after: str = "00:00"  # breakeven move only allowed from this bar start on
    same_bar: str = "pessimistic"   # entry-bar ambiguity: "pessimistic" = any low <= stop in the entry bar is a
                                    #   stop-out (LOCKED for B0) | "ohlc_path" = sensitivity: green bar O->L->H->C
                                    #   (low came BEFORE the entry; stopped only if close <= stop), red bar
                                    #   O->H->L->C (stopped if low <= stop)
    partial_r: float = 0.0          # 0 = off; take partial_frac off at +x R (limit, needs 1 tick through)
    partial_frac: float = 0.5
    # costs (per side). k = slippage on stop orders and market exits
    commission_ps: float = 0.0035   # $/share (IBKR Pro tiered, as in the paper). Alpaca = 0
    half_spread_bp: float = 3.0     # ASSUMPTION until calibrated with real SIP quotes
    slip_ticks: float = 1.0         # k = max(slip_ticks * tick, slip_bp)
    slip_bp: float = 2.0


@dataclass
class Variant:
    id: str
    knobs: Knobs
    note: str = ""


def load_variants(path: str | None) -> list[Variant]:
    base = Knobs()
    if not path:
        return [Variant("B-v1__base", base, "paper replica, long-only, full costs")]
    with open(path) as f:
        spec = json.load(f)
    out = []
    for v in spec:
        out.append(Variant(v["id"], replace(base, **v.get("knobs", {})), v.get("note", "")))
    ids = [v.id for v in out]
    if len(set(ids)) != len(ids):
        sys.exit("duplicate variant ids")
    return out


# ----------------------------------------------------------------------------- data
def hm(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


OPEN_MIN = hm("09:30")


def load_daily(path: str) -> pd.DataFrame:
    return load_daily_df(pd.read_csv(path, dtype={"ticker": str}))


def load_daily_df(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    if "split_factor" not in d:
        d["split_factor"] = 1.0
    d["split_factor"] = d["split_factor"].fillna(1.0).replace(0, 1.0)
    return d.sort_values(["ticker", "date"]).reset_index(drop=True)


def daily_features(d: pd.DataFrame, hist: int) -> pd.DataFrame:
    """Point-in-time ATR14 and avg volume for day t from days < t, in day-t's split scale.

    F(d) = cumulative product of split factors up to and including d. A price on day d expressed
    in day-t scale is p_d * F(d) / F(t); a volume is v_d * F(t) / F(d).
    """
    d = d.copy()
    g = d.groupby("ticker", sort=False)
    d["F"] = g["split_factor"].cumprod()
    prev_close_scaled = g["close"].shift(1) * g["F"].shift(1) / d["F"]  # prior close in day-d scale
    tr = np.maximum.reduce([
        d["high"] - d["low"],
        (d["high"] - prev_close_scaled).abs(),
        (d["low"] - prev_close_scaled).abs(),
    ])
    d["tr_F"] = tr * d["F"]                    # constant-scale units
    d["vol_over_F"] = d["volume"] / d["F"]
    g = d.groupby("ticker", sort=False)
    # shift(1): only days strictly before t
    d["atr_F"] = g["tr_F"].transform(lambda s: s.shift(1).rolling(hist, min_periods=hist).mean())
    d["avgvol_overF"] = g["vol_over_F"].transform(lambda s: s.shift(1).rolling(hist, min_periods=hist).mean())
    d["atr14"] = d["atr_F"] / d["F"]
    d["avg_vol14"] = d["avgvol_overF"] * d["F"]
    return d[["ticker", "date", "F", "atr14", "avg_vol14"]]


def minute_files(dirpath: str) -> dict[str, str]:
    out = {}
    for p in glob.glob(os.path.join(dirpath, "minutes_*.csv")):
        day = os.path.basename(p)[len("minutes_"):-4]
        out[day] = p
    return dict(sorted(out.items()))


def load_minutes(path: str) -> pd.DataFrame:
    m = pd.read_csv(path, dtype={"ticker": str, "time": str})
    m["tmin"] = m["time"].map(hm)
    return m.sort_values(["ticker", "tmin"]).reset_index(drop=True)


def opening_table(files: dict[str, str], or_len_set: set[int]) -> pd.DataFrame:
    """Pass 1: opening-range stats per (ticker, day) for every OR length used by any variant."""
    rows = []
    max_len = max(or_len_set) * 2  # entry_bar2 needs the 2nd bar too
    for day, p in files.items():
        m = load_minutes(p)
        m = m[(m["tmin"] >= OPEN_MIN) & (m["tmin"] < OPEN_MIN + max_len)]
        for L in or_len_set:
            w = m[m["tmin"] < OPEN_MIN + L]
            if w.empty:
                continue
            a = w.groupby("ticker").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                                        c=("c", "last"), v=("v", "sum"), first_t=("tmin", "min"),
                                        nbars=("tmin", "size"))
            w2 = m[(m["tmin"] >= OPEN_MIN + L) & (m["tmin"] < OPEN_MIN + 2 * L)]
            a2 = w2.groupby("ticker").agg(h2=("h", "max"), l2=("l", "min"))
            a = a.join(a2, how="left")
            a["day"], a["L"] = day, L
            rows.append(a.reset_index())
    ot = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return ot


def add_rvol(ot: pd.DataFrame, dfe: pd.DataFrame, hist: int, min_hist: int) -> pd.DataFrame:
    """RVOL(t) = OR volume on t / mean OR volume over the `hist` prior SESSIONS of that ticker (split-scaled).

    The window is counted in trading sessions taken from the daily file, not in rows of the opening table.
    A session with no opening data is a missing value (it could be missing data, not zero volume), and the
    baseline needs at least `min_hist` valid sessions out of `hist`.
    """
    sess = dfe[["ticker", "date", "F"]].rename(columns={"date": "day"})
    parts = []
    for L, otL in ot.groupby("L"):
        x = sess.merge(otL[["ticker", "day", "v"]], on=["ticker", "day"], how="left")
        x = x.sort_values(["ticker", "day"])
        x["v_overF"] = x["v"] / x["F"]
        x["base_overF"] = x.groupby("ticker", sort=False)["v_overF"].transform(
            lambda s: s.shift(1).rolling(hist, min_periods=min_hist).mean())
        x["L"] = L
        parts.append(x[["ticker", "day", "L", "F", "base_overF"]])
    base = pd.concat(parts, ignore_index=True)
    ot = ot.merge(base, on=["ticker", "day", "L"], how="left")
    ot["F"] = ot["F"].fillna(1.0)
    ot["rvol"] = (ot["v"] / ot["F"]) / ot["base_overF"]
    return ot


# ----------------------------------------------------------------------------- one trade
@dataclass
class TradeResult:
    filled: bool
    entry_t: int = -1
    exit_t: int = -1
    entry_px: float = np.nan
    exit_px: float = np.nan
    r_ps: float = np.nan        # $ risk per share (stop distance)
    r_net: float = np.nan       # P&L in R, all costs included
    r_gross: float = np.nan     # P&L in R before any cost, same path
    exit_reason: str = ""
    same_bar_stop: bool = False
    ret_net: float = np.nan     # P&L / entry notional (for the 1x portfolio)
    same_bar_source: str = ""   # entry-bar ambiguity decided by: "" (none) | "ticks" | "assumed"


def ticks_decide(prices: np.ndarray, side: int, trig_ls: float, stop_ls: float) -> bool | None:
    """Real prints of the entry minute, in order. In long space: the entry is the first print at least 1 tick
    through the trigger; the trade is stopped in that minute only if a LATER print is at/below the stop.
    Returns None when the prints do not show the entry (feed mismatch) -> the caller keeps its assumption."""
    p = np.asarray(prices, dtype=float)
    if side < 0:
        p = -p
    hit = np.nonzero(p >= trig_ls + TICK - 1e-9)[0]
    if hit.size == 0:
        return None
    after = p[hit[0] + 1:]
    return bool(after.size and after.min() <= stop_ls + 1e-9)


def simulate_trade(bars: pd.DataFrame, side: int, trigger: float, atr: float, k: Knobs,
                   start_t: int, close_t: int, or_other_side: float, tick_lookup=None) -> TradeResult:
    """Pessimistic 1-min fill model. side=+1 long, -1 short. bars sorted by tmin (RTH).

    Entry: stop order at `trigger`; triggers only if the bar trades 1 tick THROUGH it; fills at
      max(trigger, open) (long) + half spread + k. Stop distance measured from the executed price.
    Entry bar: if its range also reaches the stop -> stopped in the same bar (pessimistic).
    Later bars: stop fills at min(stop, open) - half spread - k (gaps fill at the open).
    Breakeven / trailing changes take effect from the NEXT bar after the trigger bar (no same-bar optimism).
    Close: exit at the close of the last bar <= close_t, minus half spread and k.
    """
    b = bars[(bars["tmin"] >= start_t) & (bars["tmin"] <= close_t)]
    if b.empty:
        return TradeResult(False, exit_reason="no_bars")
    t = b["tmin"].to_numpy()
    o, h, l, c = (b[x].to_numpy(dtype=float) for x in ("o", "h", "l", "c"))
    # work in "long space": for shorts flip prices so the same code applies
    if side < 0:
        o, h, l, c = -o, -l, -h, -c
        trig = -trigger
        other = -or_other_side
    else:
        trig = trigger
        other = or_other_side
    cut = hm(k.entry_cutoff)
    can_enter = (t <= cut) & (h >= trig + TICK)
    if not can_enter.any():
        return TradeResult(False, exit_reason="no_trigger")
    j = int(np.argmax(can_enter))
    ref = abs(trig)
    hs = max(0.5 * TICK, ref * k.half_spread_bp / 1e4) if k.half_spread_bp > 0 else 0.0
    kk = max(k.slip_ticks * TICK, ref * k.slip_bp / 1e4)
    comm = k.commission_ps
    rt_cost = 2 * (hs + kk + comm)               # round-trip cost per share, known before the order
    if k.min_stop_cost_mult > 0 and k.stop_mode == "atr" and k.stop_atr * atr < k.min_stop_cost_mult * rt_cost:
        return TradeResult(False, exit_reason="cost_filter")
    raw_fill = max(trig, o[j])
    entry = raw_fill + hs + kk                    # executed price incl. spread + slippage
    if k.stop_mode == "or_opposite":
        r_ps = entry - other
        r_ps = min(max(r_ps, k.stop_atr * atr, 2 * rt_cost), 0.5 * atr)
    else:
        r_ps = k.stop_atr * atr
    if not np.isfinite(r_ps) or r_ps <= TICK:
        return TradeResult(False, exit_reason="stop_too_small")
    stop0 = entry - r_ps
    n = len(t)

    # stop path (vectorised). stop_arr[i] = stop active DURING bar i
    stop_arr = np.full(n, stop0)
    if k.breakeven_r > 0:
        hit = np.where((np.arange(n) >= j) & (t >= hm(k.breakeven_after)) & (h >= entry + k.breakeven_r * r_ps))[0]
        if hit.size:
            be_level = entry + 2 * comm + hs + kk     # breakeven INCLUDING round-trip costs
            stop_arr[hit[0] + 1:] = np.maximum(stop_arr[hit[0] + 1:], be_level)
    if k.trail_after_r > 0:
        hit = np.where((np.arange(n) >= j) & (h >= entry + k.trail_after_r * r_ps))[0]
        if hit.size:
            a = hit[0]
            runmax = np.maximum.accumulate(np.where(np.arange(n) >= j, h, -np.inf))
            trail = runmax - k.trail_atr * atr
            # trail level known at the end of bar i-1 applies to bar i
            stop_arr[a + 1:] = np.maximum(stop_arr[a + 1:], trail[a:n - 1])

    sb_source = ""

    def finish(exit_px_long_space, exit_i, reason, same_bar=False, partial=None):
        # partial = (fraction, exit price long-space) taken earlier
        gross_exit = exit_px_long_space
        if partial:
            f, ppx = partial
            pnl = f * (ppx - entry) + (1 - f) * (gross_exit - entry)
        else:
            pnl = gross_exit - entry
        pnl_net = pnl - 2 * comm
        # gross: undo spread/slippage on both sides (approximate on the same path)
        gross_pnl = pnl + 2 * (hs + kk)
        r_net = pnl_net / r_ps
        res = TradeResult(True, int(t[j]), int(t[exit_i]), side * entry if side > 0 else -entry,
                          side * gross_exit if side > 0 else -gross_exit, r_ps, r_net,
                          gross_pnl / r_ps, reason, same_bar, pnl_net / abs(entry), sb_source)
        return res

    # same-bar stop on the entry bar (works in long space, so shorts are mirrored automatically)
    if k.same_bar == "ohlc_path" and c[j] >= o[j]:
        hit_same = c[j] <= stop0                  # O->L->H->C: after the entry only H->C remains
    else:
        hit_same = l[j] <= stop0                  # pessimistic, or a red bar O->H->L->C
    if l[j] <= stop0:                             # the bar's range covers both levels: ambiguous
        sb_source = "assumed"
        # real prints replace the assumption only in the locked (pessimistic) model; the OHLC-path
        # sensitivity stays a pure assumption so its column keeps its meaning
        prices = tick_lookup(int(t[j])) if (tick_lookup is not None and k.same_bar == "pessimistic") else None
        if prices is not None and len(prices):
            decided = ticks_decide(prices, side, trig, stop0)
            if decided is not None:
                hit_same, sb_source = decided, "ticks"
    if hit_same:
        return finish(stop0 - hs - kk, j, "stop_same_bar", True)

    partial = None
    ts = hm(k.time_stop) if k.time_stop else None
    if ts is not None and t[j] >= ts:
        ts = None                                 # entered after the check time: no time stop
    for i in range(j + 1, n):
        st = stop_arr[i]
        # time stop: decided at the START of bar i from the CLOSE of the ts bar (bar i-1)
        if ts is not None and t[i - 1] == ts and c[i - 1] < entry + k.time_stop_r * r_ps:
            return finish(o[i] - hs - kk, i, "time_stop", partial=partial)
        if o[i] <= st:
            return finish(o[i] - hs - kk, i, "stop_gap", partial=partial)
        if l[i] <= st:
            reason = "stop" if st <= stop0 + 1e-12 else ("breakeven" if k.breakeven_r and st < entry + r_ps else "trail")
            return finish(st - hs - kk, i, reason, partial=partial)
        if k.partial_r > 0 and partial is None and h[i] >= entry + k.partial_r * r_ps + TICK:
            partial = (k.partial_frac, entry + k.partial_r * r_ps - hs)  # limit fill, pays half spread
    return finish(c[n - 1] - hs - kk, n - 1, "close", partial=partial)


# ----------------------------------------------------------------------------- one day
def day_candidates(ot_day: pd.DataFrame, dfe_day: pd.DataFrame, k: Knobs) -> pd.DataFrame:
    x = ot_day[ot_day["L"] == k.or_minutes].merge(
        dfe_day.rename(columns={"date": "day"}), on=["ticker", "day"], how="inner")
    x = x[x["ticker"] != "SPY"]
    x = x[(x["first_t"] == OPEN_MIN) & (x["nbars"] >= max(1, k.or_minutes - 1))]  # halted / late open -> drop
    x = x[(x["o"] > k.min_price) & (x["avg_vol14"] >= k.min_avg_vol) & (x["atr14"] > k.min_atr)]
    x = x[np.isfinite(x["rvol"]) & (x["rvol"] >= k.rvol_min)]
    # rank the universe by RVOL (both colours), THEN apply skip rules to the top-N (paper: doji = no order)
    x = x.sort_values(["rvol", "ticker"], ascending=[False, True]).head(k.top_n).copy()
    x = x[x["c"] != x["o"]]                                  # doji -> no trade
    x["side"] = np.where(x["c"] > x["o"], 1, -1)
    if k.max_or_range_atr > 0:
        x = x[(x["h"] - x["l"]) <= k.max_or_range_atr * x["atr14"]]
    if k.close_loc_min > 0:
        rng_ = (x["h"] - x["l"]).replace(0, np.nan)
        loc = np.where(x["side"] > 0, (x["c"] - x["l"]) / rng_, (x["h"] - x["c"]) / rng_)
        x = x[loc >= k.close_loc_min]
    if k.sides == "long":
        x = x[x["side"] == 1]
    return x


SPY_PREV: dict[str, float] = {}   # day -> SPY prior-session close (raw), filled in main()
EARLY_CLOSES: dict[str, str] = {}  # day -> "HH:MM" official close (e.g. "13:00"), optional calendar
TICKS: dict[tuple, np.ndarray] = {}  # (day, ticker, minute) -> tick prices in time order (fetch_b_data.py ticks)


def load_ticks(path: str) -> int:
    t = pd.read_csv(path, dtype={"ticker": str}).sort_values(["day", "ticker", "minute", "seq"])
    for key, g in t.groupby(["day", "ticker", "minute"]):
        TICKS[(key[0], key[1], int(key[2]))] = g["price"].to_numpy(dtype=float)
    return len(TICKS)
FULL_DAY_MIN_LAST_BAR = hm("15:00")  # a ticker counts as "has full-day bars" if its last bar starts >= 15:00
TRADE_COLS = list(TradeResult.__dataclass_fields__) + ["day", "ticker", "side", "rvol", "atr14", "reason"]


CAL_STATE = {"strict": False, "source": "none (half days inferred from the last bar)"}


@contextmanager
def calendar_context(early: dict[str, str], strict: bool, source: str = ""):
    """Install an early-close calendar for the duration of a run, then restore the previous state (no leaks
    between runs or tests). strict=True: the calendar is COMPLETE for the range, so a file that ends at 12:59 on a
    day the calendar calls a full day is a data gap, not a half day."""
    saved = (dict(EARLY_CLOSES), dict(CAL_STATE))
    EARLY_CLOSES.clear()
    EARLY_CLOSES.update(early)
    CAL_STATE.update(strict=strict, source=source or CAL_STATE["source"])
    try:
        yield
    finally:
        EARLY_CLOSES.clear()
        EARLY_CLOSES.update(saved[0])
        CAL_STATE.clear()
        CAL_STATE.update(saved[1])


def session_close_bar(day: str, m: pd.DataFrame) -> tuple[int | None, str]:
    """START time of the bar whose close we exit on: one bar before the official close (15:58 bar -> 15:59).

    With an early-close calendar entry it is exact (and the file must reach the bar before the close, else it is a
    data gap). Without one the close is inferred from the last bar in the file: 15:59 -> full day, 12:59 -> half
    day (only when no complete calendar is loaded). Anything else = a data gap; the day is skipped, never guessed.
    """
    last = int(m["tmin"].max())
    if day in EARLY_CLOSES:
        close_bar = hm(EARLY_CLOSES[day]) - 1
        if last < close_bar:
            return None, "data_gap"
        return close_bar - 1, "calendar"
    if last >= hm("15:59"):
        return hm("15:58"), "full"
    if last == hm("12:59") and not CAL_STATE["strict"]:
        return hm("12:58"), "half_inferred"
    return None, "data_gap"


def entry_trigger(r, side: int, k: Knobs) -> float:
    """Stop-entry level. Base: the OR high (long) / low (short). F6 confirmation: also beyond bar 2,
    never inside the OR."""
    if not k.entry_bar2:
        return float(r["h"] if side > 0 else r["l"])
    b2 = r["h2"] if side > 0 else r["l2"]
    if not np.isfinite(b2):
        return np.nan
    return float(max(r["h"], b2) if side > 0 else min(r["l"], b2))


def _trades_df(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows).reindex(columns=TRADE_COLS)
    out["filled"] = out["filled"].fillna(False).astype(bool)
    out["same_bar_stop"] = out["same_bar_stop"].fillna(False).astype(bool)
    return out


def run_many(specs: list[tuple[Variant, str, int]], files: dict[str, str], ot: pd.DataFrame,
             dfe: pd.DataFrame, diag: dict | None = None) -> list[pd.DataFrame]:
    """Day-major engine: every minute file is read ONCE and every spec (variant, null_mode, seed) runs on it.

    null_mode: "" (strategy) | "random_entry" (same names, random entry minute 09:35-10:30, market order)
               | "random_names" (random names among universe survivors that HAVE full-day bars, same rules).
    """
    out_rows: list[list[dict]] = [[] for _ in specs]
    ot_by_day = {d: g for d, g in ot.groupby("day")}
    dfe_by_day = {d: g for d, g in dfe.groupby("date")}
    diag = diag if diag is not None else {}
    diag.setdefault("days_skipped_data_gap", [])
    diag.setdefault("null_pool_sizes", [])
    for day, path in files.items():
        if day not in ot_by_day or day not in dfe_by_day:
            continue
        otd, dfd = ot_by_day[day], dfe_by_day[day]
        m = None
        for si, (var, null_mode, seed) in enumerate(specs):
            k = var.knobs
            rng = np.random.default_rng(seed + int(day.replace("-", "")))
            real = day_candidates(otd, dfd, k)
            if null_mode == "random_names":
                if real.empty:
                    continue
                if m is None:
                    m = load_minutes(path)
                full = set(m.groupby("ticker")["tmin"].max().loc[lambda s: s >= FULL_DAY_MIN_LAST_BAR].index)
                pool = day_candidates(otd, dfd, replace(k, rvol_min=-1.0, top_n=10_000))
                pool = pool[pool["ticker"].isin(full)]
                diag["null_pool_sizes"].append(len(pool))
                cand = pool.sample(n=min(len(real), len(pool)), random_state=int(rng.integers(1 << 31)))
            else:
                cand = real
            if cand.empty:
                continue
            if k.spy_filter:
                spy = otd[(otd["ticker"] == "SPY") & (otd["L"] == k.or_minutes)]
                if spy.empty or day not in SPY_PREV:
                    continue
                spy_dir = np.sign(spy["c"].iloc[0] - SPY_PREV[day])   # prior close -> OR close (gap + 1st bar)
                cand = cand[cand["side"] == spy_dir]
                if cand.empty:
                    continue
            if m is None:
                m = load_minutes(path)
            close_t, how = session_close_bar(day, m)
            if close_t is None:
                if day not in diag["days_skipped_data_gap"]:
                    diag["days_skipped_data_gap"].append(day)
                continue
            if "_mb" not in diag or diag["_mb"][0] != day:
                diag["_mb"] = (day, {tk: g for tk, g in m.groupby("ticker")})
            mb = diag["_mb"][1]
            start_t = OPEN_MIN + (2 if k.entry_bar2 else 1) * k.or_minutes
            for _, r in cand.iterrows():
                bars = mb.get(r["ticker"])
                if bars is None:
                    out_rows[si].append(dict(day=day, ticker=r["ticker"], side=int(r["side"]), filled=False,
                                             reason="no_minute_bars", rvol=r["rvol"]))
                    continue
                side = int(r["side"])
                trig = entry_trigger(r, side, k)
                if not np.isfinite(trig):
                    continue
                other = r["l"] if side > 0 else r["h"]
                tl = (lambda mnt, _d=day, _s=r["ticker"]: TICKS.get((_d, _s, mnt))) if TICKS else None
                if null_mode == "random_entry":
                    mins = bars["tmin"][(bars["tmin"] >= start_t) & (bars["tmin"] <= hm("10:30"))].to_numpy()
                    if mins.size == 0:
                        continue
                    t0 = int(rng.choice(mins))
                    op = float(bars.loc[bars["tmin"] == t0, "o"].iloc[0])
                    trig = op - side * TICK    # level just through the open -> fills at the open
                    res = simulate_trade(bars, side, trig, r["atr14"], k, t0, close_t, other)
                else:
                    res = simulate_trade(bars, side, trig, r["atr14"], k, start_t, close_t, other, tick_lookup=tl)
                if not res.filled:
                    out_rows[si].append(dict(day=day, ticker=r["ticker"], side=side, filled=False,
                                             reason=res.exit_reason, rvol=r["rvol"]))
                    continue
                d = asdict(res)
                d.update(day=day, ticker=r["ticker"], side=side, rvol=r["rvol"], atr14=r["atr14"],
                         reason=res.exit_reason)
                out_rows[si].append(d)
    diag.pop("_mb", None)
    return [_trades_df(r) for r in out_rows]


def run_variant(var: Variant, files: dict[str, str], ot: pd.DataFrame, dfe: pd.DataFrame,
                null_mode: str = "", seed: int = 0) -> pd.DataFrame:
    return run_many([(var, null_mode, seed)], files, ot, dfe)[0]


# ----------------------------------------------------------------------------- statistics
def clustered_t(x: pd.Series, day: pd.Series) -> tuple[float, float, int]:
    n = len(x)
    if n < 3:
        return np.nan, np.nan, 0
    mean = x.mean()
    g = (x - mean).groupby(day).sum()
    G = len(g)
    if G < 2:
        return np.nan, np.nan, G
    se = math.sqrt(G / (G - 1) * float((g ** 2).sum())) / n
    return mean / se if se > 0 else np.nan, se, G


def paired_delta(var_tr: pd.DataFrame, base_tr: pd.DataFrame) -> dict:
    """Union of signals; a signal one version skipped counts as 0 R for that version."""
    key = ["day", "ticker"]
    a = var_tr[var_tr["filled"]][key + ["r_net"]].rename(columns={"r_net": "r_var"})
    b = base_tr[base_tr["filled"]][key + ["r_net"]].rename(columns={"r_net": "r_base"})
    u = a.merge(b, on=key, how="outer").fillna({"r_var": 0.0, "r_base": 0.0})
    if u.empty:
        return {"dR": np.nan, "t": np.nan}
    d = u["r_var"] - u["r_base"]
    t, se, G = clustered_t(d, u["day"])
    return {"dR": d.mean(), "t": t, "days": G}


def daily_paired_nw(var_tr: pd.DataFrame, base_tr: pd.DataFrame, all_days: list[str], lags: int = 5) -> dict:
    """Day-level paired test: d_t = sum of variant net R on day t - sum of baseline net R on day t.
    Days with no trades count as 0. Newey-West SE with `lags` lags. Returns a two-sided normal p."""
    from scipy import stats

    d = (daily_R(var_tr, all_days) - daily_R(base_tr, all_days)).to_numpy()
    n = len(d)
    if n < 10:
        return {"d_day": np.nan, "t_nw": np.nan, "p": np.nan}
    m = d.mean()
    e = d - m
    s = (e @ e) / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    se = math.sqrt(max(s, 1e-18) / n)
    t = m / se
    return {"d_day": m, "t_nw": t, "p": float(2 * stats.norm.sf(abs(t)))}


def holm(pvals: list[float]) -> list[float]:
    idx = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    adj = [np.nan] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def daily_R(tr: pd.DataFrame | None, all_days: list[str]) -> pd.Series:
    """Sum of net R per day (equal $ risk per trade); days without trades = 0."""
    if tr is None or tr.empty or not tr["filled"].any():
        return pd.Series(0.0, index=all_days)
    f = tr[tr["filled"]]
    return f.groupby("day")["r_net"].sum().reindex(all_days, fill_value=0.0)


def daily_ret_1x(tr: pd.DataFrame | None, all_days: list[str], top_n: int) -> pd.Series:
    """1x portfolio daily return: equal notional 1/top_n per position; days without trades = 0."""
    if tr is None or tr.empty or not tr["filled"].any():
        return pd.Series(0.0, index=all_days)
    f = tr[tr["filled"]]
    return (f.groupby("day")["ret_net"].sum() / top_n).reindex(all_days, fill_value=0.0)


SCRATCH_R = 0.05   # |net R| <= this = a scratch (neither win nor loss); proposal, team to confirm


def summarize(tr: pd.DataFrame, all_days: list[str], top_n: int, scratch_r: float = SCRATCH_R) -> dict:
    f = tr[tr["filled"]] if not tr.empty and "filled" in tr else tr.head(0)
    out = {"signals": int(len(tr)), "trades": int(len(f))}
    if f.empty:
        return out
    r = f["r_net"]
    t, se, G = clustered_t(r, f["day"])
    wins, losses = r[r > 0], r[r <= 0]
    out.update(
        days_traded=G,
        mean_R_net=r.mean(), t_net=t,
        mean_R_gross=f["r_gross"].mean(),
        win_rate=(r > 0).mean(),
        avg_win_R=wins.mean() if len(wins) else np.nan,
        avg_loss_R=losses.mean() if len(losses) else np.nan,
        profit_factor=wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        same_bar_stop_pct=f["same_bar_stop"].mean(),
        ambiguous_pct=(f["same_bar_source"].fillna("") != "").mean(),
        ambiguous_resolved_by_ticks=float((f["same_bar_source"] == "ticks").sum()
                                          / max(1, (f["same_bar_source"].fillna("") != "").sum())),
        cost_R=(f["r_gross"] - f["r_net"]).mean(),
        # scratches reported apart: a breakeven exit nets ~0 after costs and is neither a win nor a loss
        scratch_pct=float((r.abs() <= scratch_r).mean()),
        win_pct_ex_scratch=float((r > scratch_r).mean()),
        loss_pct_ex_scratch=float((r < -scratch_r).mean()),
        scratch_sum_R=float(r[r.abs() <= scratch_r].sum()),
    )
    # 1x portfolio: equal notional 1/top_n per position, days without trades = 0
    daily = daily_ret_1x(tr, all_days, top_n)
    sd = daily.std(ddof=1)
    out["sharpe_1x"] = daily.mean() / sd * math.sqrt(252) if sd > 0 else np.nan
    out["ann_ret_1x"] = daily.mean() * 252
    eq = daily.cumsum()
    out["max_dd_1x"] = float((eq.cummax() - eq).max())
    out["pct_pos_days"] = float((daily[daily != 0] > 0).mean()) if (daily != 0).any() else np.nan
    monthly = daily.groupby(pd.to_datetime(daily.index).to_period("M")).sum()
    out["pct_pos_months"] = float((monthly > 0).mean())
    out["months"] = int(len(monthly))
    quarterly = daily.groupby(pd.to_datetime(daily.index).to_period("Q")).sum()
    out["quarters_pos"] = f"{int((quarterly > 0).sum())}/{len(quarterly)}"
    ordered = f.sort_values(["day", "entry_t"])["r_net"]
    cumR = ordered.cumsum()
    out["max_dd_R"] = float((cumR.cummax() - cumR).max())
    streak = best = 0
    for x in ordered:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    out["longest_losing_streak"] = best
    pos = np.sort(r[r > 0].to_numpy())[::-1]
    top = max(1, int(math.ceil(0.05 * len(r))))
    out["top5pct_share_of_wins"] = float(pos[:top].sum() / pos.sum()) if pos.sum() > 0 else np.nan
    out["exit_mix"] = f["reason"].value_counts(normalize=True).round(3).to_dict()
    return out


BE_TICKS = [0, 1, 2, 3, 4]
BE_BPS = [0.0, 2.5, 5.0, 10.0, 20.0, 40.0]


def breakeven_specs(var: Variant, unit: str = "ticks") -> list[tuple[Variant, str, int]]:
    """Copies of `var` with slippage per side = x ticks (slip_bp off) or x bp (slip_ticks off)."""
    if unit == "bp":
        return [(Variant(var.id, replace(var.knobs, slip_ticks=0.0, slip_bp=x)), "", 0) for x in BE_BPS]
    return [(Variant(var.id, replace(var.knobs, slip_ticks=kt, slip_bp=0.0)), "", 0) for kt in BE_TICKS]


def breakeven_from(results: list[pd.DataFrame], xs: list[float] | None = None) -> float:
    """Extra slippage per side (on top of commission + half spread) at which mean net R = 0, interpolated
    linearly over the grid `xs` (default BE_TICKS = 0..4 ticks; BE_BPS for basis points).
    0 if already <= 0 with no slippage; np.inf if still positive at the top of the grid."""
    xs_grid = BE_TICKS if xs is None else xs
    xs_, ys = [], []
    for kt, s in zip(xs_grid, results):
        f = s[s["filled"]]
        if f.empty:
            return np.nan
        xs_.append(kt)
        ys.append(f["r_net"].mean())
    ys = np.array(ys)
    if ys[0] <= 0:
        return 0.0
    for i in range(1, len(xs_)):
        if ys[i] <= 0:
            return xs_[i - 1] + ys[i - 1] / (ys[i - 1] - ys[i]) * (xs_[i] - xs_[i - 1])
    return np.inf


# ----------------------------------------------------------------------------- gate statistics
def gate_statistics(variants: list[Variant], results: list[pd.DataFrame], idx: dict, rows: list[dict],
                    all_days: list[str], a) -> dict:
    """DSR (per version, N from the trial log + this run), random-skip null (filters), Hansen SPA (best vs base),
    PBO via CSCV (all versions). Writes dsr / rs_* into `rows` (the summary) and returns the run-level numbers."""
    import gate_stats as G
    prev = []
    if os.path.exists(a.trial_log):
        log = pd.read_csv(a.trial_log, dtype=str)
        prev = log["sharpe_1x"].tolist() if "sharpe_1x" in log else [""] * len(log)
    n_trials, var_sr = G.trial_sharpe_stats(prev, [s.get("sharpe_1x", "") for s in rows])
    sr0 = G.expected_max_sharpe(n_trials, var_sr)
    base_tr = results[idx[("var", variants[0].id)]]
    bf = base_tr[base_tr["filled"]] if not base_tr.empty else base_tr
    dR = {}
    for i, (v, s) in enumerate(zip(variants, rows)):
        tr = results[idx[("var", v.id)]]
        d = G.deflated_sharpe(daily_ret_1x(tr, all_days, v.knobs.top_n).to_numpy(), n_trials, var_sr)
        s["dsr"] = d["dsr"]
        dR[v.id] = daily_R(tr, all_days).to_numpy()
        if i > 0 and not bf.empty:
            vf = tr[tr["filled"]] if not tr.empty else tr
            s.update(G.random_skip_null(bf["r_net"].to_numpy(), vf["r_net"].to_numpy() if not vf.empty else [],
                                        bf["day"].to_numpy(), int(vf["day"].nunique()) if not vf.empty else 0,
                                        reps=a.gate_reps, seed=a.seed + i))
    out = dict(n_trials=n_trials, var_sr=var_sr, sr0_day=sr0)
    if len(variants) > 1:
        ids = [v.id for v in variants[1:]]
        diff = np.column_stack([dR[i] - dR[variants[0].id] for i in ids])
        sp = G.spa_test(diff, reps=a.gate_reps, mean_block=a.spa_block, seed=a.seed)
        out.update(spa_p=sp["spa_p"], spa_best=ids[sp["spa_best"]] if sp["spa_best"] is not None else None)
        pb = G.pbo_cscv(np.column_stack([dR[v.id] for v in variants]), a.pbo_blocks)
        out.update(pbo=pb["pbo"], pbo_splits=pb["splits"], pbo_blocks=pb["blocks"])
    return out


# ----------------------------------------------------------------------------- main
def main(argv=None):
    """Run, then restore the module-level calendar state (EARLY_CLOSES / CAL_STATE) whatever happens."""
    saved = (dict(EARLY_CLOSES), dict(CAL_STATE))
    try:
        return _main(argv)
    finally:
        EARLY_CLOSES.clear()
        EARLY_CLOSES.update(saved[0])
        CAL_STATE.clear()
        CAL_STATE.update(saved[1])


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daily", required=True)
    ap.add_argument("--minutes", required=True)
    ap.add_argument("--variants", default=None, help="JSON list [{id, knobs:{...}, note}]; first = baseline")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--open-holdout", action="store_true", help="allow dates >= 2024-01-01 (ONE run, logged)")
    ap.add_argument("--nulls", type=int, default=20, help="null repetitions per null type (baseline only)")
    ap.add_argument("--breakeven", action="store_true", help="compute breakeven slippage for every variant (5x runtime)")
    ap.add_argument("--stress", action="store_true", help="also run every variant at 2x slippage")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--trial-log", default="experiments/trials.csv")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--early-closes", default="auto",
                    help="'auto' (default) = bundled NYSE calendar (nyse_calendar.py) when the range is inside "
                         "2015-2023; a CSV path date,close (e.g. 2019-11-29,13:00) = a complete calendar; "
                         "'none' = infer half days from the last bar")
    ap.add_argument("--ticks", default=None, help="ticks.csv from `fetch_b_data.py ticks`: resolves same-minute "
                                                  "entry+stop trades from real prints")
    ap.add_argument("--scratch-r", type=float, default=SCRATCH_R, help="|net R| <= this counts as a scratch")
    ap.add_argument("--gate-reps", type=int, default=1000, help="bootstrap / random-skip repetitions (DSR/SPA/PBO)")
    ap.add_argument("--spa-block", type=float, default=10.0, help="mean block length (days) of the SPA bootstrap")
    ap.add_argument("--pbo-blocks", type=int, default=16, help="CSCV blocks for PBO (even)")
    a = ap.parse_args(argv)
    TICKS.clear()
    if a.ticks:
        print(f"ticks loaded for {load_ticks(a.ticks)} ambiguous minutes", flush=True)

    import re
    for name in ("start", "end"):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", getattr(a, name)):
            sys.exit(f"--{name} must be YYYY-MM-DD")
    if a.end >= HOLDOUT_START and not a.open_holdout:
        sys.exit(f"HOLDOUT GUARD: --end {a.end} reaches the holdout (>= {HOLDOUT_START}). "
                 f"Use --end 2023-12-31, or --open-holdout for the ONE pre-registered run.")
    variants = load_variants(a.variants)
    if a.open_holdout:
        if os.path.exists(a.trial_log):
            log = pd.read_csv(a.trial_log, dtype=str)
            if "end" in log and (log["end"] >= HOLDOUT_START).any():
                sys.exit("HOLDOUT GUARD: the trial log already contains a holdout run. The holdout is opened ONCE. "
                         "A second look needs a new preregistration (v2) and a fresh holdout period.")
        if len(variants) > 3:
            sys.exit("HOLDOUT GUARD: at most base + 2 promoted variants may run on the holdout.")
    holiday_days: set[str] = set()
    EARLY_CLOSES.clear()
    if a.early_closes == "auto":
        import nyse_calendar as NC
        if NC.COVERED[0] <= int(a.start[:4]) and int(a.end[:4]) <= NC.COVERED[1]:
            EARLY_CLOSES.update(NC.early_close_map(a.start, a.end))
            holiday_days = NC.holiday_set(a.start, a.end)
            CAL_STATE.update(strict=True, source=f"bundled {NC.COVERED[0]}-{NC.COVERED[1]} (nyse_calendar.py)")
        else:
            CAL_STATE.update(strict=False, source="none: range outside 2015-2023, half days inferred from the last bar")
    elif a.early_closes and a.early_closes != "none":
        ec = pd.read_csv(a.early_closes, dtype=str)
        EARLY_CLOSES.update(dict(zip(pd.to_datetime(ec.iloc[:, 0]).dt.strftime("%Y-%m-%d"), ec.iloc[:, 1])))
        CAL_STATE.update(strict=True, source=f"file {a.early_closes}")
    else:
        CAL_STATE.update(strict=False, source="none (half days inferred from the last bar)")

    files = {d: p for d, p in minute_files(a.minutes).items()}
    # history needs days before --start for the 14-day baselines; trading only inside [start, end]
    trade_files = {d: p for d, p in files.items() if a.start <= d <= a.end}
    if not trade_files:
        sys.exit("no minute files in range")
    or_lens = {v.knobs.or_minutes for v in variants}
    hist = variants[0].knobs.hist_days
    daily = load_daily(a.daily)
    dfe = daily_features(daily, hist)
    spy = daily[daily["ticker"] == "SPY"].sort_values("date")
    SPY_PREV.update(dict(zip(spy["date"].iloc[1:], spy["close"].iloc[:-1])))
    ot = opening_table({d: p for d, p in files.items() if d <= a.end}, or_lens)
    ot = add_rvol(ot, dfe, hist, variants[0].knobs.min_hist_days)
    all_days = sorted(trade_files)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(a.out, f"{stamp}_B")
    os.makedirs(outdir, exist_ok=True)
    h = hashlib.sha256()
    for d, p in sorted(trade_files.items()):
        st_ = os.stat(p)
        h.update(f"{d}:{st_.st_size}:{int(st_.st_mtime)};".encode())
    st_ = os.stat(a.daily)
    h.update(f"daily:{st_.st_size}:{int(st_.st_mtime)}".encode())
    fp = h.hexdigest()[:12]

    # build every spec, then ONE day-major pass
    specs, idx = [], {}
    def add(key, spec):
        idx[key] = len(specs)
        specs.append(spec)
    for v in variants:
        add(("var", v.id), (v, "", 0))
        if a.stress:
            add(("stress", v.id), (Variant(v.id, replace(v.knobs, slip_ticks=2 * v.knobs.slip_ticks,
                                                         slip_bp=2 * v.knobs.slip_bp)), "", 0))
            add(("ohlc", v.id), (Variant(v.id, replace(v.knobs, same_bar="ohlc_path")), "", 0))
        if a.breakeven:
            for kt, sp in zip(BE_TICKS, breakeven_specs(v)):
                add(("be", v.id, kt), sp)
            for x, sp in zip(BE_BPS, breakeven_specs(v, "bp")):
                if x == 0:
                    continue                      # identical to 0 ticks: reuse that run
                add(("be_bp", v.id, x), sp)
    for mode in (["random_entry", "random_names"] if a.nulls > 0 else []):
        for rep in range(a.nulls):
            add(("null", mode, rep), (variants[0], mode, a.seed + 1000 * rep))
    print(f"running {len(specs)} specs over {len(trade_files)} days (each day file read once)", flush=True)
    diag: dict = {"calendar": CAL_STATE["source"], "holiday_files": sorted(set(trade_files) & holiday_days),
                  "early_closes_in_range": sorted(d for d in EARLY_CLOSES if d in trade_files)}
    results = run_many(specs, trade_files, ot, dfe, diag)

    rows, all_tr = [], []
    base_tr = results[idx[("var", variants[0].id)]]
    for i, v in enumerate(variants):
        tr = results[idx[("var", v.id)]]
        s = summarize(tr, all_days, v.knobs.top_n, a.scratch_r)
        s.update(id=v.id, note=v.note)
        if i > 0:
            pdlt = paired_delta(tr, base_tr)
            s["dR_vs_base"], s["t_vs_base"] = pdlt["dR"], pdlt["t"]
            nw = daily_paired_nw(tr, base_tr, all_days)
            s["dR_per_day_vs_base"], s["t_nw_vs_base"], s["p_vs_base"] = nw["d_day"], nw["t_nw"], nw["p"]
        if a.breakeven:
            s["breakeven_ticks"] = breakeven_from([results[idx[("be", v.id, kt)]] for kt in BE_TICKS])
            s["breakeven_bp"] = breakeven_from(
                [results[idx[("be", v.id, 0) if x == 0 else ("be_bp", v.id, x)]] for x in BE_BPS], BE_BPS)
        if a.stress:
            fs = results[idx[("stress", v.id)]]
            fs = fs[fs["filled"]]
            s["mean_R_net_2x_slip"] = fs["r_net"].mean() if not fs.empty else np.nan
            fo = results[idx[("ohlc", v.id)]]
            fo = fo[fo["filled"]]
            s["mean_R_net_ohlc_path"] = fo["r_net"].mean() if not fo.empty else np.nan
        rows.append(s)
        if not tr.empty:
            all_tr.append(tr.assign(variant=v.id))
        print(f"[{v.id}] trades={s.get('trades', 0)} meanR_net={s.get('mean_R_net', float('nan')):.3f} "
              f"t={s.get('t_net', float('nan')):.2f}", flush=True)

    gate = gate_statistics(variants, results, idx, rows, all_days, a)

    null_rows = []
    if a.nulls > 0:
        for mode in ["random_entry", "random_names"]:
            means = []
            for rep in range(a.nulls):
                ntr = results[idx[("null", mode, rep)]]
                f = ntr[ntr["filled"]]
                means.append(f["r_net"].mean() if not f.empty else np.nan)
            means = np.array(means, dtype=float)
            real = rows[0].get("mean_R_net", np.nan)
            null_rows.append(dict(null=mode, reps=a.nulls, null_mean=np.nanmean(means),
                                  null_p95=np.nanpercentile(means, 95), real=real,
                                  share_null_ge_real=float(np.nanmean(means >= real))))

    summ = pd.DataFrame(rows)
    if "p_vs_base" in summ:
        summ["p_holm"] = holm(summ["p_vs_base"].fillna(np.nan).tolist())
    summ.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    if all_tr:
        pd.concat(all_tr, ignore_index=True).to_csv(os.path.join(outdir, "trades.csv"), index=False)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(dict(args=vars(a), variants=[dict(id=v.id, knobs=asdict(v.knobs), note=v.note) for v in variants],
                       data_fingerprint=fp), f, indent=2)
    import gate_stats as G
    cones = {}
    for i, v in enumerate(variants):
        tr = results[idx[("var", v.id)]]
        f = tr[tr["filled"]] if not tr.empty else tr
        cones[v.id] = G.expected_cone(f["r_net"].to_numpy() if not f.empty else [], seed=a.seed + i)
    with open(os.path.join(outdir, "expected_cone.json"), "w") as f:
        json.dump(dict(run=stamp, data_fingerprint=fp, start=a.start, end=a.end, versions=cones), f, indent=2)
    gate["cone_base"] = cones[variants[0].id]
    write_report(outdir, summ, null_rows, base_tr, a, fp, diag, gate)
    os.makedirs(os.path.dirname(a.trial_log) or ".", exist_ok=True)
    new = not os.path.exists(a.trial_log)
    with open(a.trial_log, "a") as f:
        if new:
            f.write("run_stamp,variant_id,start,end,trades,mean_R_net,t_net,sharpe_1x,data_fp\n")
        for s in rows:
            f.write(f"{stamp},{s['id']},{a.start},{a.end},{s.get('trades', 0)},{s.get('mean_R_net', '')},"
                    f"{s.get('t_net', '')},{s.get('sharpe_1x', '')},{fp}\n")
    print(f"\nreport: {os.path.join(outdir, 'report.md')}")
    return outdir


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–" if x is None or np.isnan(x) else ("∞" if x > 0 else "-∞")
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_report(outdir, summ, null_rows, base_tr, a, fp, diag=None, gate=None):
    diag = diag or {}
    gate = gate or {}
    cols = [("id", "variant"), ("trades", "trades"), ("win_rate", "win%"),
            ("win_pct_ex_scratch", "win% ex scratch"), ("scratch_pct", "scratch%"),
            ("loss_pct_ex_scratch", "loss% ex scratch"), ("avg_win_R", "avg win R"),
            ("avg_loss_R", "avg loss R"), ("mean_R_net", "mean R net"), ("t_net", "t (day-clustered)"),
            ("mean_R_gross", "mean R gross"), ("cost_R", "cost R/trade"), ("profit_factor", "PF"),
            ("pct_pos_months", "% pos months"), ("max_dd_R", "max DD (R)"), ("sharpe_1x", "Sharpe 1x"),
            ("max_dd_1x", "max DD 1x"), ("same_bar_stop_pct", "same-bar stop %"),
            ("ambiguous_pct", "ambiguous entry bars %"), ("ambiguous_resolved_by_ticks", "of which resolved by ticks %"),
            ("breakeven_ticks", "breakeven ticks/side"),
            ("breakeven_bp", "breakeven bp/side"), ("mean_R_net_2x_slip", "mean R @2x slip"),
            ("mean_R_net_ohlc_path", "mean R if same-bar = OHLC path"),
            ("quarters_pos", "quarters +"), ("longest_losing_streak", "longest losing streak"),
            ("top5pct_share_of_wins", "top-5% share of wins"),
            ("dR_vs_base", "ΔR/trade vs base"), ("t_vs_base", "t vs base"),
            ("dR_per_day_vs_base", "ΔR/day vs base"), ("t_nw_vs_base", "t NW (day)"), ("p_holm", "p Holm"),
            ("dsr", "DSR"), ("rs_p_trades", "random-skip p (trades)"), ("rs_p_days", "random-skip p (days)")]
    cols = [c for c in cols if c[0] in summ]
    L = [f"# Strategy B backtest {a.start} → {a.end}", "",
         f"Data fingerprint `{fp}`. All numbers are AFTER commission + half spread + slippage unless labelled gross.",
         "Win rate and averages are per trade. Mean R assumes the SAME $ risk per trade; Sharpe and drawdown 1× use",
         "EQUAL NOTIONAL (1/top_n of equity per name). The two sizings differ; read each metric with its own sizing.",
         f"win% = net R > 0 (old definition). Scratch = |net R| ≤ {a.scratch_r:g}R, reported apart from wins and losses.",
         "Breakeven slippage = extra slippage per side on top of commission + half spread at which mean net R = 0, "
         "in ticks (grid 0-4) and in bp (grid 0-40; for a $200 stock 1 tick = 0.5 bp, for a $10 stock 10 bp).",
         "", "| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    gaps = diag.get("days_skipped_data_gap", [])
    pools = diag.get("null_pool_sizes", [])
    warn = [f"- Days skipped as data gaps (last bar does not match the session close): {len(gaps)}"
            + (f" e.g. {gaps[:5]}" if gaps else "")]
    if "calendar" in diag:
        hol = diag.get("holiday_files", [])
        warn.append(f"- NYSE calendar: {diag['calendar']}. Early closes in range: "
                    f"{len(diag.get('early_closes_in_range', []))}. Minute files on NYSE holidays: {len(hol)}"
                    + (f" {hol[:5]} (data error: a closed market has no bars)" if hol else ""))
    if pools:
        warn.append(f"- random_names null pool = universe names WITH full-day bars: median {int(np.median(pools))}/day. "
                    "If full-day bars exist only for top-RVOL names, this null is NOT random and H2 is untested "
                    "(data plan: also download full-day bars for a seeded random sample of eligible names).")
    for _, r in summ.iterrows():
        pct_cols = {"win_rate", "win_pct_ex_scratch", "scratch_pct", "loss_pct_ex_scratch", "pct_pos_months", "same_bar_stop_pct", "ambiguous_pct", "ambiguous_resolved_by_ticks"}
        L.append("| " + " | ".join(fmt(r.get(c[0])) if c[0] not in pct_cols
                                   else (f"{100*r.get(c[0]):.1f}" if pd.notna(r.get(c[0])) else "–") for c in cols) + " |")
    L += ["", "## Data diagnostics", ""] + warn
    if null_rows:
        L += ["", "## Nulls (baseline)", "", "| null | reps | null mean R | null p95 | real | share of nulls ≥ real |",
              "|---|---|---|---|---|---|"]
        for n in null_rows:
            L.append(f"| {n['null']} | {n['reps']} | {fmt(n['null_mean'])} | {fmt(n['null_p95'])} | {fmt(n['real'])} | "
                     f"{fmt(n['share_null_ge_real'], 2)} |")
    if base_tr is not None and not base_tr.empty and base_tr["filled"].any():
        f = base_tr[base_tr["filled"]].copy()
        f["year"] = f["day"].str[:4]
        y = f.groupby("year")["r_net"].agg(["count", "mean"])
        L += ["", "## Baseline by year (mean net R)", "", "| year | trades | mean R |", "|---|---|---|"]
        for yr, rr in y.iterrows():
            L.append(f"| {yr} | {int(rr['count'])} | {rr['mean']:.3f} |")
        if "exit_mix" in summ.columns:
            L += ["", f"Exit mix (baseline): {summ.iloc[0]['exit_mix']}"]
    if gate:
        L += ["", "## Gate statistics (multiple testing / overfitting)", "",
              f"- **Trials in the log (N for the DSR, this run included): {gate['n_trials']}**; cross-trial variance of "
              f"the daily Sharpe: {fmt(gate['var_sr'], 6)}; expected max daily Sharpe under H0: "
              f"{fmt(gate['sr0_day'], 4)} (annualised {fmt(gate['sr0_day'] * math.sqrt(252), 2)}).",
              "- DSR = P(true Sharpe > that expected maximum), on the 1× equal-notional daily returns. Gate: ≥ 0.95.",
              "- Random-skip p: share of random subsets of the base's trades (same count) or days (same count) whose "
              "mean net R ≥ the variant's. Only for versions that keep fewer trades than base (filters). Gate: < 0.05.",
              f"- Hansen SPA (best variant vs base, daily net R, stationary bootstrap, block {a.spa_block:g} d, "
              f"{a.gate_reps} reps): **p = {fmt(gate.get('spa_p'))}**"
              + (f", best = `{gate['spa_best']}`" if gate.get("spa_best") else "")
              + ". p > 0.10 → nothing beats base.",
              f"- PBO via CSCV ({gate.get('pbo_blocks', 0)} blocks, {gate.get('pbo_splits', 0)} splits, all "
              f"{len(summ)} versions, daily net R Sharpe): **PBO = {fmt(gate.get('pbo'))}**. > 0.3 → the search is "
              "overfit."]
    cone = gate.get("cone_base")
    if cone and cone.get("horizons"):
        L += ["", "## Expected cone for paper monitoring (baseline; all versions in expected_cone.json)", "",
              f"{cone['method']}, {cone['paths']} paths, seed {cone['seed']}, from {cone['source_trades']} trades. "
              "Net R after all costs. Below p5 or DD above p95 → check the implementation (signals, fills, costs) "
              "before blaming the market; ABOVE p95 → also investigate.", "",
              "| after N trades | cum R p5 | p50 | p95 | max DD R p5 | p50 | p95 |", "|---|---|---|---|---|---|---|"]
        for h, x in cone["horizons"].items():
            L.append(f"| {h} | " + " | ".join(fmt(v, 2) for v in x["cum_R"] + x["max_dd_R"]) + " |")
    L += ["", "## Reading rules", "- Not preregistered yet: treat t < 3 as 'not worth pursuing' (rigor skill §5).",
          "- A variant beats the base only with t_vs_base > 2 AND wins again on the next period's fresh data.",
          "- Breakeven ticks/side < 2 → B as specified does not get deployed (plan §4).",
          "- Any profit on random-walk data = a bug."]
    with open(os.path.join(outdir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
