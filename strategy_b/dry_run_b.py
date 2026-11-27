#!/usr/bin/env python3
"""
dry_run_b.py: the 09:35 selection of Strategy B for ONE day, as it would run live, written to a file.
OFFLINE / DRY RUN ONLY: there is NO broker code here. It reads MARKET DATA (Alpaca data API, or the in-process fake
in the tests) and writes ORDER INTENTS to CSV/JSON. Nothing is sent anywhere. A purity test enforces that this file
never touches a trading endpoint.

What it does at 09:35 ET on --day D:
  1. History (already on disk, from fetch_b_data.py: daily.csv + minutes_<day>.csv for prior sessions) gives ATR14,
     avg volume and the 14-session RVOL baseline. Only rows dated before D are used.
  2. Downloads D's split factor (corporate actions) and D's 1-min bars 09:30-09:34 for the loose universe + SPY,
     with the request END at 09:35:00: only bars that have CLOSED by 09:35 exist.
  3. Runs b_mvb.py's own functions (opening_table, add_rvol, day_candidates, entry_trigger) for the chosen version,
     so the selection is the backtest's selection by construction (a parity test checks it).
  4. Writes runs/dryrun_<D>_<variant>/order_intents.csv + .json + selection.csv.

Order intent (one per candidate), per the plan: a DAY stop order 1 tick through the opening-range high (long) / low
(short); protective stop = stop_atr x ATR14 from the fill (known only after the fill, so given as a distance);
exit at the 15:59 close (12:59 on half days); cancel unfilled entries at the close. Quantity: 1x equal notional
(equity / top_n / trigger), the prereg's Sharpe sizing proposal; the equal-$-risk quantity is shown next to it.
client_order_id follows the order-risk skill format: {version}-{variant}-{SYMBOL}-{YYYYMMDD}-{seq:04d}-E1-r0.

Usage:
  python dry_run_b.py --data data --day 2019-02-20 [--variants variants_b.json --variant B-v1__base]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

import b_mvb as B
import fetch_b_data as F

DECISION_TIME = "09:35"


def _code_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__)), timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def fetch_today_opening(client, day: str, symbols: list[str], feed: str = "sip") -> pd.DataFrame:
    """1-min bars of `day` that have CLOSED by 09:35 (request end = 09:35:00, then a hard filter on bar start)."""
    df = F.bars_frames(client, symbols, "1Min", F.ny_to_utc(day, "09:30"), F.ny_to_utc(day, DECISION_TIME), feed=feed)
    rows = F.minutes_to_file_rows(df)
    if rows.empty:
        return pd.DataFrame(columns=["ticker", "time", "o", "h", "l", "c", "v"])
    rows = rows[(rows["day"] == day) & (rows["time"] < DECISION_TIME)]
    return rows[["ticker", "time", "o", "h", "l", "c", "v"]]


def fetch_today_split(client, day: str, symbols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for group in F.chunks(symbols, 500):
        for body in F._paged_raw(client, "/v1/corporate-actions", dict(symbols=",".join(group),
                                 types="forward_split,reverse_split", start=day, end=day, limit=1000)):
            for x in F._iter_splits(body):
                if x["date"] == day:
                    out[x["ticker"]] = out.get(x["ticker"], 1.0) * x["split_factor"]
    return out


def prepare(data_dir: str, day: str, k: B.Knobs, today: pd.DataFrame, today_split: dict[str, float],
            minutes_dir: str | None = None):
    """(ot, dfe, daily_before_day): the backtest's opening table with RVOL and daily features for `day`, built from
    history before `day` + today's bars closed by 09:35. select() ranks on it; shadow_day_b.py simulates on it."""
    daily = pd.read_csv(os.path.join(data_dir, "daily.csv"), dtype={"ticker": str})
    daily = daily[daily["date"] < day]
    # placeholder row for day D: only the split factor is known (prices are not used: features use days < D)
    names = sorted(set(daily["ticker"]))
    ph = pd.DataFrame(dict(ticker=names, date=day, open=np.nan, high=np.nan, low=np.nan, close=np.nan,
                           volume=np.nan, split_factor=[today_split.get(t, 1.0) for t in names]))
    dfe = B.daily_features(B.load_daily_df(pd.concat([daily, ph], ignore_index=True)), k.hist_days)
    tmp = tempfile.mkdtemp(prefix="dryrun_")
    try:
        mdir = minutes_dir or os.path.join(data_dir, "minutes")          # the RVOL baseline (same feed as today!)
        hist_files = {d: p for d, p in B.minute_files(mdir).items() if d < day}
        hist_files = dict(list(hist_files.items())[-(k.hist_days + 5):])       # the RVOL window + slack
        p_today = os.path.join(tmp, f"minutes_{day}.csv")
        today.to_csv(p_today, index=False)
        files = {**hist_files, day: p_today}
        ot = B.add_rvol(B.opening_table(files, {k.or_minutes}), dfe, k.hist_days, k.min_hist_days)
        return ot[ot["day"] == day], dfe[dfe["date"] == day], daily
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def spy_prev_close(daily_before: pd.DataFrame) -> float:
    s = daily_before[daily_before["ticker"] == "SPY"].sort_values("date")["close"]
    return float(s.iloc[-1]) if len(s) else float("nan")


def select(data_dir: str, day: str, k: B.Knobs, today: pd.DataFrame, today_split: dict[str, float]) -> pd.DataFrame:
    """The backtest's own selection on (history before `day`) + (today's bars closed by 09:35)."""
    otd, dfd, daily = prepare(data_dir, day, k, today, today_split)
    cand = B.day_candidates(otd, dfd, k)
    if k.spy_filter:
        spy = otd[(otd["ticker"] == "SPY") & (otd["L"] == k.or_minutes)]
        prev = spy_prev_close(daily)
        if spy.empty or not np.isfinite(prev):
            return cand.head(0)
        cand = cand[cand["side"] == np.sign(spy["c"].iloc[0] - prev)]
    return cand


def day_universe(data_dir: str, day: str) -> list[str]:
    """The loose universe for `day` from daily.csv rows dated before `day` (what the opening stage downloads)."""
    daily = pd.read_csv(os.path.join(data_dir, "daily.csv"), dtype={"ticker": str})
    # loose_universe() keys on rows dated `day`: append a copy of the last prior session dated D, so its
    # shift(1)/rolling inputs are exactly the sessions before D
    prior = daily[daily["date"] < day]
    last = prior["date"].max()
    return F.loose_universe(pd.concat([prior, prior[prior["date"] == last].assign(date=day)]), [day])[day]


def intents(cand: pd.DataFrame, day: str, variant_id: str, k: B.Knobs, equity: float, risk_pct: float) -> pd.DataFrame:
    version, _, knob = variant_id.partition("__")
    short = (knob or "base").replace("=", "").replace(":", "").replace(".", "p")[:24]
    close = "13:00" if day in B.EARLY_CLOSES else "16:00"
    rows = []
    for seq, (_, r) in enumerate(cand.iterrows(), start=1):
        side = int(r["side"])
        trig = B.entry_trigger(r, side, k)
        if not np.isfinite(trig):
            continue
        stop_px_order = round(trig + side * B.TICK, 2)                       # 1 tick THROUGH the level
        dist = k.stop_atr * float(r["atr14"]) if k.stop_mode == "atr" else float("nan")
        qty_1x = int(math.floor(equity / k.top_n / stop_px_order))
        qty_risk = int(math.floor(equity * risk_pct / dist)) if dist and np.isfinite(dist) and dist > 0 else 0
        rows.append(dict(
            session_date_ny=day, decision_time_ny=DECISION_TIME, strategy_version=version, variant_id=variant_id,
            ticker=r["ticker"], side="buy" if side > 0 else "sell_short", order_type="stop", time_in_force="day",
            stop_price=stop_px_order, opening_range_high=float(r["h"]), opening_range_low=float(r["l"]),
            protective_stop_distance=round(dist, 4) if np.isfinite(dist) else None,
            protective_stop_rule=(f"fill - {k.stop_atr:g} x ATR14" if side > 0 else f"fill + {k.stop_atr:g} x ATR14")
            if k.stop_mode == "atr" else "or_opposite (floor/cap per b_mvb.simulate_trade)",
            exit_rule=f"close of the bar before the {close} close; cancel if unfilled at the close",
            qty_1x_equal_notional=qty_1x, qty_equal_risk=qty_risk, rvol=float(r["rvol"]), atr14=float(r["atr14"]),
            client_order_id=f"{version.replace('-', '')}-{short}-{r['ticker']}-{day.replace('-', '')}-{seq:04d}-E1-r0",
            feed="sip", sent=False))
    return pd.DataFrame(rows)


def main(argv=None, client=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--day", required=True)
    ap.add_argument("--variants", default=None)
    ap.add_argument("--variant", default=None, help="variant id (default: the first = base)")
    ap.add_argument("--equity", type=float, default=100_000.0)
    ap.add_argument("--risk-pct", type=float, default=0.0025, help="equal-risk sizing shown for info (0.25%%)")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args(argv)
    if a.day >= B.HOLDOUT_START:
        sys.exit("HOLDOUT GUARD: the dry run replays DEV days only (< 2024-01-01).")
    vs = B.load_variants(a.variants)
    v = next((x for x in vs if x.id == a.variant), None) if a.variant else vs[0]
    if v is None:
        sys.exit(f"unknown variant {a.variant}")
    import nyse_calendar as NC
    if not NC.is_trading_day(pd.Timestamp(a.day).date()):
        sys.exit(f"{a.day} is not an NYSE trading day")
    client = client or F.Client()
    if not client.key and client.transport == F.Client._urllib:
        sys.exit("set APCA_API_KEY_ID and APCA_API_SECRET_KEY (market data only)")
    uni = day_universe(a.data, a.day)
    syms = sorted(set(uni) | {"SPY"})
    split = fetch_today_split(client, a.day, syms)
    today = fetch_today_opening(client, a.day, syms)
    with B.calendar_context(NC.early_close_map(a.day, a.day), strict=True, source="bundled"):
        cand = select(a.data, a.day, v.knobs, today, split)
        it = intents(cand, a.day, v.id, v.knobs, a.equity, a.risk_pct)
    out = os.path.join(a.out, f"dryrun_{a.day}_{v.id}")
    os.makedirs(out, exist_ok=True)
    it.to_csv(os.path.join(out, "order_intents.csv"), index=False)
    cand.to_csv(os.path.join(out, "selection.csv"), index=False)
    meta = dict(kind="DRY RUN - order intents only, nothing was sent", session_date_ny=a.day, variant_id=v.id,
                knobs=B.asdict(v.knobs), code_sha=_code_sha(), universe_size=len(uni), bars_today=len(today),
                intents=len(it), data_requests="market data only (bars, corporate actions)")
    with open(os.path.join(out, "order_intents.json"), "w") as f:
        json.dump(dict(meta=meta, intents=it.to_dict(orient="records")), f, indent=2, default=str)
    print(f"dry run {a.day} {v.id}: {len(uni)} names in the loose universe, {len(it)} intents -> {out}")
    return out


if __name__ == "__main__":
    main()
