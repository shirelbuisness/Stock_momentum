#!/usr/bin/env python3
"""
shadow_day_b.py: the paper SHADOW logger of Strategy B for one finished session. NO ORDERS: there is no broker
code here (a purity test enforces it). It reads MARKET DATA only and appends to three CSVs that the offline
paper checks read (paper_gate_b.py, paper_vs_cone.py).

Run it after the session (SIP bars of the day must be older than 15 minutes: after ~16:20 ET; half days ~13:20).

  python shadow_day_b.py --data data --day 2019-02-22 [--variants variants_b.json --variant B-v1__base]
         [--selection-feed iex|sip] [--out paper]

What it does for --day D:
  1. Selection exactly as live would make it at the decision time (09:35 for the base; 09:30 + 2 x OR for F6 and
     F9 needs its 15-min OR): history before D (daily.csv) + D's split factor + D's opening 1-min bars, through
     the dry run's own functions (dry_run_b.prepare -> b_mvb.day_candidates). The F8 SPY filter is applied the
     same way the backtest applies it.
     --selection-feed iex = what a FREE-plan live instance sees at 09:35 (SIP is 15 minutes delayed). IEX volume
     is a small slice of the market, so the RVOL baseline is then built from IEX opening bars too (cached in
     <data>/minutes_iex_opening/); mixing IEX today with a SIP baseline would put every RVOL near 0.03 and the
     RVOL >= 100% filter would reject almost everything.
  2. After the close: D's full-day SIP 1-min bars for the selected names (+ SPY), then b_mvb.run_many - the
     backtest's own pessimistic fill model and full cost model - on exactly those candidates.
  3. Appends (re-running a day replaces that day's rows for that variant; atomic writes):
       <out>/paper_trades.csv   day, ticker, side, variant_id, r_net, r_gross, entry_t, exit_t, entry_px, exit_px,
                                reason, feed_selection, code_sha, kind=shadow   (filled trades only)
       <out>/live_selection.csv day, ticker, side, variant_id, feed_selection  (every signalled name)
       <out>/model_refs.csv     day, ticker, variant_id, leg (entry/exit), side (+1 buy/-1 sell), ref_px, qty_1x:
                                the model's reference prices BEFORE costs. When a paper order path exists, its real
                                fill prices join here into paper_fills.csv (side, ref_px, fill_px, qty).
paper_fills.csv itself is NOT written: real fills need real (paper) orders, which this tool never places.

Holdout guard: D >= 2024-01-01 is refused (the forward shadow run on live dates is a separate, later decision).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

import b_mvb as B
import dry_run_b as D
import fetch_b_data as F

TRADE_OUT = ["day", "ticker", "side", "variant_id", "r_net", "r_gross", "entry_t", "exit_t", "entry_px", "exit_px",
             "reason", "same_bar_stop", "feed_selection", "code_sha", "kind"]
SEL_OUT = ["day", "ticker", "side", "variant_id", "feed_selection"]
REF_OUT = ["day", "ticker", "variant_id", "leg", "side", "ref_px", "qty_1x"]


def opening_end(k: B.Knobs) -> int:
    """Minutes after 09:30 whose bars the selection may read: the OR, plus bar 2 (F6 trigger; opening_table also
    reads it for every version, so fetching it keeps the table identical to the backtest's)."""
    return 2 * k.or_minutes


def fetch_opening(client, day: str, syms: list[str], k: B.Knobs, feed: str) -> pd.DataFrame:
    end = B.OPEN_MIN + opening_end(k)
    df = F.bars_frames(client, syms, "1Min", F.ny_to_utc(day, "09:30"), F.ny_to_utc(day, f"{end // 60:02d}:{end % 60:02d}"),
                       feed=feed)
    rows = F.minutes_to_file_rows(df)
    if rows.empty:
        return pd.DataFrame(columns=["ticker", "time", "o", "h", "l", "c", "v"])
    rows = rows[(rows["day"] == day) & (rows["time"].map(B.hm) < end)]
    return rows[["ticker", "time", "o", "h", "l", "c", "v"]]


def iex_baseline_dir(client, data_dir: str, day: str, syms: list[str], k: B.Knobs) -> str:
    """IEX opening bars for the RVOL window before `day`, cached per session; missing names are fetched once."""
    cdir = os.path.join(data_dir, "minutes_iex_opening")
    os.makedirs(cdir, exist_ok=True)
    days = [d for d in F.trading_days(data_dir) if d < day][-(k.hist_days + 5):]
    for d in days:
        p = os.path.join(cdir, f"minutes_{d}.csv")
        have = pd.read_csv(p, dtype={"ticker": str, "time": str}) if os.path.exists(p) else None
        known = (set(have["ticker"]) if have is not None else set()) | set(_absent(p))
        need = sorted(set(syms) - known)
        if not need:
            continue
        new = fetch_opening(client, d, need, k, "iex")
        parts = [x for x in (have, new) if x is not None and len(x)]
        out = pd.concat(parts, ignore_index=True) if parts else new
        F.atomic_csv(out.sort_values(["ticker", "time"]) if len(out) else out, p)
        _absent_add(p, sorted(set(need) - set(new["ticker"])))     # asked, no IEX bars: do not ask again
    return cdir


def _absent(p: str) -> list[str]:
    q = p + ".absent"
    return open(q).read().split() if os.path.exists(q) else []


def _absent_add(p: str, names: list[str]):
    if names:
        with open(p + ".absent", "a") as f:
            f.write("\n".join(names) + "\n")


def shadow_trades(client, data_dir: str, day: str, var: B.Variant, feed: str):
    """(trades DataFrame from b_mvb.run_many, candidates). Market data only."""
    k = var.knobs
    syms = sorted(set(D.day_universe(data_dir, day)) | {"SPY"})
    split = D.fetch_today_split(client, day, syms)
    today = fetch_opening(client, day, syms, k, feed)
    mdir = iex_baseline_dir(client, data_dir, day, syms, k) if feed == "iex" else None
    otd, dfd, daily = D.prepare(data_dir, day, k, today, split, minutes_dir=mdir)
    cand = B.day_candidates(otd, dfd, k)
    prev = D.spy_prev_close(daily)
    if k.spy_filter:
        spy = otd[(otd["ticker"] == "SPY") & (otd["L"] == k.or_minutes)]
        cand = cand.head(0) if spy.empty or not np.isfinite(prev) else \
            cand[cand["side"] == np.sign(spy["c"].iloc[0] - prev)]
    cand = cand[[np.isfinite(B.entry_trigger(r, int(r["side"]), k)) for _, r in cand.iterrows()]] \
        if len(cand) else cand
    if cand.empty:
        return B._trades_df([]), cand
    close = B.EARLY_CLOSES.get(day, "16:00")
    full = F.minutes_to_file_rows(F.bars_frames(client, sorted(set(cand["ticker"]) | {"SPY"}), "1Min",
                                                F.ny_to_utc(day, "09:30"), F.ny_to_utc(day, close)))
    full = full[full["day"] == day][["ticker", "time", "o", "h", "l", "c", "v"]]
    tmp = tempfile.mkdtemp(prefix="shadow_")
    saved = dict(B.SPY_PREV)
    try:
        p = os.path.join(tmp, f"minutes_{day}.csv")
        full.to_csv(p, index=False)
        B.SPY_PREV.clear()
        if np.isfinite(prev):
            B.SPY_PREV[day] = prev
        tr = B.run_many([(var, "", 0)], {day: p}, otd, dfd)[0]     # recomputes the same candidates from otd
    finally:
        B.SPY_PREV.clear()
        B.SPY_PREV.update(saved)
        shutil.rmtree(tmp, ignore_errors=True)
    return tr, cand


def _replace_rows(path: str, new: pd.DataFrame, day: str, variant_id: str, cols: list[str]):
    old = pd.read_csv(path, dtype={"ticker": str}) if os.path.exists(path) else pd.DataFrame(columns=cols)
    if len(old):
        old = old[~((old["day"] == day) & (old["variant_id"] == variant_id))]
    parts = [x for x in (old, new.reindex(columns=cols)) if len(x)]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    F.atomic_csv(out.sort_values(["day", "variant_id", "ticker"]) if len(out) else out, path)


def write_outputs(out: str, day: str, var: B.Variant, tr: pd.DataFrame, cand: pd.DataFrame, feed: str,
                  equity: float = 100_000.0):
    os.makedirs(out, exist_ok=True)
    sha = D._code_sha()
    f = tr[tr["filled"]].copy() if len(tr) else tr
    f = f.assign(variant_id=var.id, feed_selection=feed, code_sha=sha, kind="shadow")
    _replace_rows(os.path.join(out, "paper_trades.csv"), f, day, var.id, TRADE_OUT)
    sel = cand.assign(variant_id=var.id, feed_selection=feed)[["day", "ticker", "side", "variant_id", "feed_selection"]] \
        if len(cand) else pd.DataFrame(columns=SEL_OUT)
    _replace_rows(os.path.join(out, "live_selection.csv"), sel, day, var.id, SEL_OUT)
    refs = []
    for _, r in f.iterrows():
        q = int(np.floor(equity / var.knobs.top_n / r["entry_px"])) if r["entry_px"] > 0 else 0
        s = int(r["side"])
        refs += [dict(day=day, ticker=r["ticker"], variant_id=var.id, leg="entry", side=s, ref_px=r["entry_px"], qty_1x=q),
                 dict(day=day, ticker=r["ticker"], variant_id=var.id, leg="exit", side=-s, ref_px=r["exit_px"], qty_1x=q)]
    _replace_rows(os.path.join(out, "model_refs.csv"), pd.DataFrame(refs, columns=REF_OUT), day, var.id, REF_OUT)


def main(argv=None, client=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--day", required=True, action="append", help="repeatable; or a range with --to")
    ap.add_argument("--to", default=None, help="last day of a range starting at --day (every NYSE session)")
    ap.add_argument("--variants", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--selection-feed", choices=["iex", "sip"], default="iex",
                    help="feed of the 09:35 ranking (iex = what a free-plan live instance sees)")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args(argv)
    import nyse_calendar as NC
    if any(d >= B.HOLDOUT_START for d in a.day + ([a.to] if a.to else [])):
        sys.exit("HOLDOUT GUARD: the shadow logger replays DEV days only (< 2024-01-01).")
    days = NC.trading_days(a.day[0], a.to) if a.to else a.day
    vs = B.load_variants(a.variants)
    v = next((x for x in vs if x.id == a.variant), None) if a.variant else vs[0]
    if v is None:
        sys.exit(f"unknown variant {a.variant}")
    bad = [d for d in days if not NC.is_trading_day(pd.Timestamp(d).date())]
    if bad:
        sys.exit(f"not NYSE trading days: {bad}")
    client = client or F.Client()
    if not client.key and client.transport == F.Client._urllib:
        sys.exit("set APCA_API_KEY_ID and APCA_API_SECRET_KEY (market data only)")
    for day in days:
        with B.calendar_context(NC.early_close_map(day, day), strict=True, source="bundled"):
            tr, cand = shadow_trades(client, a.data, day, v, a.selection_feed)
        write_outputs(a.out, day, v, tr, cand, a.selection_feed)
        n = int(tr["filled"].sum()) if len(tr) else 0
        print(f"shadow {day} {v.id} [{a.selection_feed}]: {len(cand)} signalled, {n} filled -> {a.out}")
    return a.out


if __name__ == "__main__":
    main()
