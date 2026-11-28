#!/usr/bin/env python3
"""
fetch_b_data.py: download Strategy B's data from Alpaca (free plan, SIP feed) into the format b_mvb.py reads.

Stages (each is resumable; finished pieces are skipped on rerun):
  probe     5-minute go/no-go: keys work, SIP 1-min bars exist for 2016, multi-symbol paging, delisted names.
  assets    securities.csv: active + inactive US equities, common-stock heuristic (ETFs/funds/warrants out).
  daily     daily.csv: RAW 1Day bars for every security + split_factor from /v1/corporate-actions.
  opening   Stage 1: 09:30-09:44 1-min bars for the loose universe of each day (as of the prior close).
  fullday   Stage 2: full-session 1-min bars for the top-100 RVOL names + 30 seeded random eligible names + SPY.
  ticks     For trades whose entry and stop fall in the same minute (from a b_mvb run's trades.csv), download the
            tick prints of that minute so the backtest can see which came first (build log finding R9).
  quotes    SIP NBBO quotes in --window-s seconds of every entry and exit minute of a run's trades (the first
            seconds for entries and stops, the last ones for the 15:58 close exit)
            (seeded sample with --sample) -> data/quotes_spread.csv: the measured spread at the moment the
            model trades, for the cost calibration (spread_b.py). The cost model's 3 bp half spread is an
            assumption until this is measured.

Keys: environment APCA_API_KEY_ID and APCA_API_SECRET_KEY (a free Alpaca account is enough).
Everything is written under --data (default ./data). Nothing in the output depends on today's date except
the survivorship caveat of the asset list, which is written to data/README_data.md.

Usage (typical order):
  python fetch_b_data.py probe
  python fetch_b_data.py assets
  python fetch_b_data.py daily   --start 2015-11-01 --end 2023-12-31
  python fetch_b_data.py opening --start 2015-12-01 --end 2023-12-31
  python fetch_b_data.py fullday --start 2016-01-01 --end 2023-12-31
  python b_mvb.py --daily data/daily.csv --minutes data/minutes ... (DEV-A)
  python fetch_b_data.py ticks --trades runs/<run>/trades.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
DATA_URL = "https://data.alpaca.markets"
TRADING_URL = "https://paper-api.alpaca.markets"

# names that are not operating common stock (conservative; same idea as Strategy A's R2 filter)
NON_COMMON = re.compile(
    r"\b(?:ETF|ETN|FUND|TRUST|WARRANTS?|RIGHTS?|UNITS?|PREFERRED|PFD|DEPOSITARY|NOTES?|BOND|INDEX|ISHARES|"
    r"SPDR|PROSHARES|DIREXION|INVESCO|VANGUARD|ACQUISITION CORP)\b", re.I)
COMMON_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}


# ----------------------------------------------------------------------------- HTTP client
class Client:
    """Thin REST client: auth headers, throttle (free plan = 200 req/min), retries on 429/5xx, paging.
    `transport(url, headers) -> (status, json)` is injectable so tests can use a fake Alpaca."""

    def __init__(self, key=None, secret=None, transport=None, max_per_min=190, sleep=time.sleep):
        self.key = key or os.environ.get("APCA_API_KEY_ID", "")
        self.secret = secret or os.environ.get("APCA_API_SECRET_KEY", "")
        self.transport = transport or self._urllib
        self.min_gap = 60.0 / max_per_min
        self.sleep = sleep
        self._last = 0.0
        self.calls = 0

    @staticmethod
    def _urllib(url, headers):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                return e.code, json.loads(body)
            except ValueError:
                return e.code, {"message": body[:300]}

    def get(self, base, path, params):
        q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{base}{path}?{q}"
        headers = {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret, "Accept": "application/json"}
        for attempt in range(6):
            wait = self.min_gap - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            self.calls += 1
            status, body = self.transport(url, headers)
            if status == 200:
                return body
            if status in (429, 500, 502, 503, 504):
                self.sleep(min(60, 2 ** attempt))
                continue
            raise RuntimeError(f"HTTP {status} for {path}: {body}")
        raise RuntimeError(f"gave up after retries: {path}")

    def paged(self, base, path, params, key):
        """Yield every page's payload[key] following next_page_token."""
        token = None
        while True:
            body = self.get(base, path, {**params, "page_token": token})
            yield body.get(key) or {}
            token = body.get("next_page_token")
            if not token:
                return


# ----------------------------------------------------------------------------- helpers
def ny_to_utc(day: str, hhmm: str) -> str:
    h, m = map(int, hhmm.split(":"))
    d = datetime.strptime(day, "%Y-%m-%d").replace(hour=h, minute=m, tzinfo=NY)
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_to_ny(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(NY)


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def atomic_csv(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


class Manifest:
    """data/manifest.json: which stage finished for which piece. Pieces are only marked after their file is
    written, so a crash never leaves a piece marked done with partial data."""

    def __init__(self, data_dir):
        self.path = os.path.join(data_dir, "manifest.json")
        self.m = json.load(open(self.path)) if os.path.exists(self.path) else {}

    def done(self, stage, piece):
        return piece in self.m.get(stage, [])

    def pieces(self, stage):
        return list(self.m.get(stage, []))

    def mark(self, stage, piece):
        self.m.setdefault(stage, [])
        if piece not in self.m[stage]:
            self.m[stage].append(piece)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.m, f, indent=1)
        os.replace(tmp, self.path)


def bars_frames(client, symbols, timeframe, start, end, asof=None, batch=500, limit=10000, feed="sip"):
    """Multi-symbol bars -> DataFrame(ticker, ts_utc, o,h,l,c,v). Raw prices, SIP feed (feed="iex" only for the
    paper shadow logger's live-like 09:35 ranking: the free plan cannot read SIP within the last 15 minutes).

    Symbol identity: every stage uses Alpaca's DEFAULT mapping (asof = today), so a company keeps ONE ticker across
    the daily file, the minute files and the ticks. Passing asof=<day> in some stages but not others would label the
    same company differently (e.g. a rename) - a silent join failure. Only the probe uses asof, on purpose."""
    out = []
    for group in chunks(list(symbols), batch):
        params = dict(symbols=",".join(group), timeframe=timeframe, start=start, end=end, limit=limit,
                      adjustment="raw", feed=feed, asof=asof, sort="asc")
        for page in client.paged(DATA_URL, "/v2/stocks/bars", params, "bars"):
            for sym, rows in page.items():
                for b in rows:
                    out.append((sym, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]))
    return pd.DataFrame(out, columns=["ticker", "ts_utc", "o", "h", "l", "c", "v"])


def minutes_to_file_rows(df: pd.DataFrame) -> pd.DataFrame:
    """UTC bar timestamps -> NY 'HH:MM' (bar START), regular session only (09:30-15:59)."""
    if df.empty:
        return pd.DataFrame(columns=["ticker", "time", "o", "h", "l", "c", "v"])
    t = df["ts_utc"].map(utc_to_ny)
    df = df.assign(time=t.map(lambda x: x.strftime("%H:%M")), day=t.map(lambda x: x.strftime("%Y-%m-%d")))
    df = df[(df["time"] >= "09:30") & (df["time"] <= "15:59")]
    return df


# ----------------------------------------------------------------------------- stages
def stage_probe(client, data_dir, delisted=("TWTR:2022-06-01", "SIVB:2023-01-05", "BBBY:2022-06-01")):
    ok = True
    print("probe 1: auth + SIP 1-min bars for SPY on 2016-01-04 ...", end=" ")
    df = bars_frames(client, ["SPY"], "1Min", ny_to_utc("2016-01-04", "09:30"), ny_to_utc("2016-01-04", "09:45"))
    n = len(df)
    print(f"{n} bars", "OK" if n >= 14 else "FAIL (need SIP history from 2016)")
    ok &= n >= 14
    print("probe 2: multi-symbol paging (20 names, 1 day of 1-min bars, page size 500) ...", end=" ")
    names = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "XOM", "JNJ",
             "PG", "V", "HD", "KO", "PFE", "INTC", "CSCO", "ORCL", "T", "BAC"]
    calls0 = client.calls
    df = bars_frames(client, names, "1Min", ny_to_utc("2019-03-05", "09:30"), ny_to_utc("2019-03-05", "16:00"), limit=500)
    got = df["ticker"].nunique()
    print(f"{got}/20 symbols, {len(df)} bars, {client.calls - calls0} calls", "OK" if got >= 19 else "FAIL")
    ok &= got >= 19
    for item in delisted:
        sym, d = item.split(":")
        d2 = (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=7)).strftime("%Y-%m-%d")
        n_default = len(bars_frames(client, [sym], "1Day", d, d2))            # the mapping the pipeline uses
        n_asof = len(bars_frames(client, [sym], "1Day", d, d2, asof=d))
        verdict = "OK" if n_default else ("ONLY WITH asof -> delisted names lost by the pipeline (survivorship)"
                                          if n_asof else "MISSING (survivorship risk)")
        print(f"probe 3: delisted {sym} around {d}: default mapping {n_default} bars, asof {n_asof} bars -> {verdict}")
    print("probe 3b: one request with 500 symbols (URL length) ...", end=" ")
    try:
        assets = client.get(TRADING_URL, "/v2/assets", dict(status="active", asset_class="us_equity"))
        many = sorted(a["symbol"] for a in assets if a.get("tradable"))[:500]
        df = bars_frames(client, many, "1Day", "2019-03-05", "2019-03-05", batch=500)
        print(f"{df['ticker'].nunique()} symbols returned OK")
    except Exception as e:                              # noqa: BLE001 - report and suggest a smaller batch
        ok = False
        print(f"FAIL ({e}). Use a smaller batch (edit batch=500 -> 200).")
    print("probe 4: corporate actions (splits) for AAPL 2020 ...", end=" ")
    body = client.get(DATA_URL, "/v1/corporate-actions", dict(symbols="AAPL", types="forward_split",
                                                              start="2020-08-01", end="2020-09-30"))
    splits = list(_iter_splits(body))
    print(splits[:1], "OK" if splits else "FAIL")
    ok &= bool(splits)
    print("\nGO" if ok else "\nNO-GO: fix the failures above before downloading")
    return ok


def stage_assets(client, data_dir):
    rows = []
    for status in ("active", "inactive"):
        body = client.get(TRADING_URL, "/v2/assets", dict(status=status, asset_class="us_equity"))
        for a in body:
            rows.append(dict(ticker=a.get("symbol"), name=a.get("name") or "", exchange=a.get("exchange"),
                             status=a.get("status"), tradable=a.get("tradable"),
                             # TODAY's borrow flags (not point-in-time): used only to label the short shadow
                             shortable=a.get("shortable"), easy_to_borrow=a.get("easy_to_borrow")))
    s = pd.DataFrame(rows).drop_duplicates("ticker")
    s["common"] = s["exchange"].isin(COMMON_EXCHANGES) & ~s["name"].str.contains(NON_COMMON) & \
        ~s["ticker"].str.contains(r"[./ ]", regex=True)
    atomic_csv(s, os.path.join(data_dir, "securities.csv"))
    note = ("# Data notes\n- securities.csv = Alpaca assets (active + inactive) as listed TODAY. Names delisted long ago "
            "may be missing -> results are partly survivor-biased. Cross-check with Massive Basic `active=false` "
            "if available.\n- common = name/exchange heuristic; it also drops some 'Trust' operating companies.\n"
            "- shortable / easy_to_borrow = Alpaca's flags TODAY, not as of the trade date. They label the short "
            "shadow only and are optimistic for the past (names that were hard to borrow then may be easy now).\n")
    with open(os.path.join(data_dir, "README_data.md"), "w") as f:
        f.write(note)
    print(f"securities: {len(s)} rows, {int(s['common'].sum())} common, {int((s['status'] == 'inactive').sum())} inactive")
    return s


def _iter_splits(body):
    ca = body.get("corporate_actions") or {}
    for key, items in ca.items():
        if "split" not in key:
            continue
        for x in items or []:
            if x.get("old_rate") and x.get("new_rate") and x.get("ex_date"):
                yield dict(ticker=x["symbol"], date=x["ex_date"], split_factor=float(x["new_rate"]) / float(x["old_rate"]))


def stage_daily(client, data_dir, start, end, batch=500):
    sec = pd.read_csv(os.path.join(data_dir, "securities.csv"), dtype={"ticker": str})
    syms = sorted(set(sec.loc[sec["common"], "ticker"]) | {"SPY"})
    man = Manifest(data_dir)
    parts_dir = os.path.join(data_dir, "daily_parts")
    os.makedirs(parts_dir, exist_ok=True)
    years = range(int(start[:4]), int(end[:4]) + 1)
    for y in years:
        s, e = max(start, f"{y}-01-01"), min(end, f"{y}-12-31")
        # the piece names the range actually fetched: a year first fetched in part (e.g. Jan-Mar) is fetched
        # again when a later call needs more of it (night 28.11 finding: "2021" was marked done after Jan-Mar)
        piece = f"{y}:{s}_{e}"
        part = os.path.join(parts_dir, f"daily_{y}.csv")
        if os.path.exists(part):
            if _covering(man, "daily", y, s, e):
                continue
            # the part is REWRITTEN, so fetch the span of the old and the new range together
            for s0, e0 in _covering(man, "daily", y, "9999", "0000", any_overlap=True):
                s, e = min(s, s0), max(e, e0)
            piece = f"{y}:{s}_{e}"
        # explicit NY timestamps: a date-only `end` can be read as 00:00 UTC, which drops the end day itself
        # (Dec 31 is a trading day in 2015 and 2018-2021). Night 27.11 finding.
        df = bars_frames(client, syms, "1Day", ny_to_utc(s, "00:00"), ny_to_utc(e, "23:59"), batch=batch)
        if not df.empty:
            df["date"] = df["ts_utc"].map(lambda x: utc_to_ny(x).strftime("%Y-%m-%d"))
        atomic_csv(df, part)
        man.mark("daily", piece)
        print(f"daily {y}: {len(df)} bars, {client.calls} calls so far", flush=True)
    # splits
    if not man.done("splits", f"{start}_{end}"):
        sp = []
        for group in chunks(syms, 500):
            for page_body in _paged_raw(client, "/v1/corporate-actions",
                                        dict(symbols=",".join(group), types="forward_split,reverse_split",
                                             start=start, end=end, limit=1000)):
                sp += list(_iter_splits(page_body))
        sp_dir = os.path.join(data_dir, "splits_parts")
        os.makedirs(sp_dir, exist_ok=True)
        atomic_csv(pd.DataFrame(sp, columns=["ticker", "date", "split_factor"]),
                   os.path.join(sp_dir, f"splits_{start}_{end}.csv"))
        man.mark("splits", f"{start}_{end}")
    # splits.csv = the union of every range ever fetched (a later, narrower call must not drop earlier splits);
    # the same split seen by two overlapping ranges is ONE split (dedupe before the per-day product)
    sp_dir = os.path.join(data_dir, "splits_parts")
    sp_files = sorted(glob.glob(os.path.join(sp_dir, "splits_*.csv")))
    sp_all = pd.concat([pd.read_csv(f, dtype={"ticker": str}) for f in sp_files], ignore_index=True) if sp_files \
        else pd.DataFrame(columns=["ticker", "date", "split_factor"])
    atomic_csv(sp_all.drop_duplicates(["ticker", "date", "split_factor"]).sort_values(["ticker", "date"]),
               os.path.join(data_dir, "splits.csv"))
    # assemble daily.csv from EVERY downloaded year, not only this call's years (a later call for 2021 must not
    # drop 2015-2020 from daily.csv; night 28.11 finding)
    parts = sorted(glob.glob(os.path.join(parts_dir, "daily_*.csv")))
    d = pd.concat([pd.read_csv(f, dtype={"ticker": str}) for f in parts], ignore_index=True)
    d = d.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    sp = pd.read_csv(os.path.join(data_dir, "splits.csv"), dtype={"ticker": str})
    sp = sp.groupby(["ticker", "date"], as_index=False)["split_factor"].prod()
    d = d.merge(sp, on=["ticker", "date"], how="left")
    d["split_factor"] = d["split_factor"].fillna(1.0)
    d = d[["ticker", "date", "open", "high", "low", "close", "volume", "split_factor"]].sort_values(["ticker", "date"])
    atomic_csv(d, os.path.join(data_dir, "daily.csv"))
    print(f"daily.csv: {len(d)} rows, {d['ticker'].nunique()} tickers, {int((d['split_factor'] != 1).sum())} split days")


def _covering(man, stage: str, y: int, s: str, e: str, any_overlap: bool = False) -> list[tuple[str, str]]:
    """Ranges already fetched for year y (manifest pieces 'y:s0_e0') that cover [s, e] (any_overlap: all)."""
    out = []
    for piece in man.pieces(stage):
        yy, _, rng = str(piece).partition(":")
        if yy == str(y) and rng:
            s0, _, e0 = rng.partition("_")
            if any_overlap or (s0 <= s and e <= e0):
                out.append((s0, e0))
    return out


def _paged_raw(client, path, params):
    token = None
    while True:
        body = client.get(DATA_URL, path, {**params, "page_token": token})
        yield body
        token = body.get("next_page_token")
        if not token:
            return


def trading_days(data_dir) -> list[str]:
    d = pd.read_csv(os.path.join(data_dir, "daily.csv"), dtype={"ticker": str}, usecols=["ticker", "date"])
    return sorted(d.loc[d["ticker"] == "SPY", "date"].unique())


def loose_universe(daily: pd.DataFrame, days: list[str], min_px=3.0, min_avgvol=300_000, hist=14) -> dict:
    """Per day t: common names whose PRIOR close >= $3 and prior-14-day avg volume >= 300k (looser than the paper's
    $5 / 1M so every future candidate and the RVOL baseline are covered). Uses data before t only."""
    d = daily.sort_values(["ticker", "date"]).copy()
    g = d.groupby("ticker", sort=False)
    d["prev_close"] = g["close"].shift(1)
    d["avgvol_prev"] = g["volume"].transform(lambda s: s.shift(1).rolling(hist, min_periods=5).mean())
    ok = d[(d["prev_close"] >= min_px) & (d["avgvol_prev"] >= min_avgvol) & (d["ticker"] != "SPY")]
    by = ok.groupby("date")["ticker"].apply(list).to_dict()
    return {day: sorted(by.get(day, [])) for day in days}


def stage_opening(client, data_dir, start, end, batch=500):
    daily = pd.read_csv(os.path.join(data_dir, "daily.csv"), dtype={"ticker": str})
    days = [d for d in trading_days(data_dir) if start <= d <= end]
    uni = loose_universe(daily, days)
    man = Manifest(data_dir)
    mdir = os.path.join(data_dir, "minutes")
    os.makedirs(mdir, exist_ok=True)
    for day in days:
        if man.done("opening", day):
            continue
        syms = uni[day] + ["SPY"]
        df = bars_frames(client, syms, "1Min", ny_to_utc(day, "09:30"), ny_to_utc(day, "09:45"), batch=batch)
        rows = minutes_to_file_rows(df)
        rows = rows[(rows["day"] == day) & (rows["time"] <= "09:44")]
        atomic_csv(rows[["ticker", "time", "o", "h", "l", "c", "v"]], os.path.join(mdir, f"minutes_{day}.csv"))
        man.mark("opening", day)
        print(f"opening {day}: {len(syms)} names, {len(rows)} bars, {client.calls} calls", flush=True)


def stage_fullday(client, data_dir, start, end, top=100, n_random=30, seed=7, batch=100):
    """Top-`top` by 5-min RVOL among names passing the PAPER universe filters (both colours), plus `n_random` seeded
    random names from that same eligible set (so the random-names null has full-day bars), plus SPY."""
    import b_mvb as B
    daily = B.load_daily(os.path.join(data_dir, "daily.csv"))
    dfe = B.daily_features(daily, 14)
    mdir = os.path.join(data_dir, "minutes")
    files = B.minute_files(mdir)
    ot = B.add_rvol(B.opening_table(files, {5}), dfe, 14, 10)
    man = Manifest(data_dir)
    k_any = B.replace(B.Knobs(), rvol_min=-1.0, top_n=10_000, sides="both")
    for day in [d for d in sorted(files) if start <= d <= end]:
        if man.done("fullday", day):
            continue
        otd, dfd = ot[ot["day"] == day], dfe[dfe["date"] == day]
        elig = B.day_candidates(otd, dfd, B.replace(k_any))
        if elig.empty:
            man.mark("fullday", day)
            continue
        ranked = elig.sort_values(["rvol", "ticker"], ascending=[False, True])
        top_names = list(ranked["ticker"].head(top))
        rest = sorted(set(elig["ticker"]) - set(top_names))
        rng = random.Random(f"{seed}-{day}")
        rand_names = rng.sample(rest, min(n_random, len(rest)))
        want = sorted(set(top_names) | set(rand_names) | {"SPY"})
        df = bars_frames(client, want, "1Min", ny_to_utc(day, "09:30"), ny_to_utc(day, "16:00"), batch=batch)
        full = minutes_to_file_rows(df)
        full = full[full["day"] == day][["ticker", "time", "o", "h", "l", "c", "v"]]
        path = os.path.join(mdir, f"minutes_{day}.csv")
        opening = pd.read_csv(path, dtype={"ticker": str, "time": str})
        merged = pd.concat([opening[~opening["ticker"].isin(full["ticker"])], full], ignore_index=True)
        atomic_csv(merged.sort_values(["ticker", "time"]), path)
        pd.DataFrame(dict(ticker=want, why=["top" if t in top_names else "random" if t in rand_names else "spy"
                                            for t in want])).to_csv(os.path.join(mdir, f"selected_{day}.csv"), index=False)
        man.mark("fullday", day)
        print(f"fullday {day}: {len(want)} names ({len(top_names)} top, {len(rand_names)} random), {client.calls} calls",
              flush=True)


def stage_ticks(client, data_dir, trades_csv, only_same_bar=True):
    """Tick prints for the entry minute of every ambiguous trade -> data/ticks.csv (day,ticker,minute,seq,price)."""
    tr = pd.read_csv(trades_csv, dtype={"ticker": str})
    tr = tr[tr["filled"].astype(bool)]
    if only_same_bar:
        tr = tr[tr["same_bar_stop"].astype(bool)]
    need = tr[["day", "ticker", "entry_t"]].drop_duplicates()
    out_path = os.path.join(data_dir, "ticks.csv")
    have = set()
    if os.path.exists(out_path):
        old = pd.read_csv(out_path, dtype={"ticker": str})
        have = set(zip(old["day"], old["ticker"], old["minute"]))
    rows = []
    for (day, minute), grp in need.groupby(["day", "entry_t"]):
        hhmm = f"{int(minute) // 60:02d}:{int(minute) % 60:02d}"
        syms = [s for s in grp["ticker"] if (day, s, int(minute)) not in have]
        if not syms:
            continue
        end_m = int(minute) + 1
        start, end = ny_to_utc(day, hhmm), ny_to_utc(day, f"{end_m // 60:02d}:{end_m % 60:02d}")
        for page in client.paged(DATA_URL, "/v2/stocks/trades",
                                 dict(symbols=",".join(syms), start=start, end=end, limit=10000, feed="sip",
                                      sort="asc"), "trades"):
            for sym, ticks in page.items():
                for i, x in enumerate(ticks):
                    if x.get("u"):            # canceled / corrected prints are not real trades
                        continue
                    rows.append((day, sym, int(minute), x["t"], float(x["p"])))
    new = pd.DataFrame(rows, columns=["day", "ticker", "minute", "ts_utc", "price"])
    if os.path.exists(out_path):
        new = pd.concat([pd.read_csv(out_path, dtype={"ticker": str}), new], ignore_index=True)
    new = new.sort_values(["day", "ticker", "minute", "ts_utc"])
    new["seq"] = new.groupby(["day", "ticker", "minute"]).cumcount()
    atomic_csv(new, out_path)
    print(f"ticks: {len(need)} ambiguous trades, {len(new)} prints stored -> {out_path}")


QUOTE_COLS = ["day", "ticker", "minute", "leg", "window", "n_quotes", "spread_bp_median", "spread_usd_median", "mid_median"]


def quote_spreads(quotes: list[dict]) -> dict | None:
    """Median NBBO spread of a list of Alpaca quotes (bp of the mid and $). Drops one-sided, locked and crossed
    quotes (bp <= 0, ap <= bp): they are not a market a stop order could fill against."""
    bp = np.array([float(q.get("bp", 0) or 0) for q in quotes])
    ap_ = np.array([float(q.get("ap", 0) or 0) for q in quotes])
    ok = (bp > 0) & (ap_ > bp)
    if not ok.any():
        return None
    mid, spr = (ap_[ok] + bp[ok]) / 2, ap_[ok] - bp[ok]
    return dict(n_quotes=int(ok.sum()), spread_bp_median=float(np.median(spr / mid * 1e4)),
                spread_usd_median=float(np.median(spr)), mid_median=float(np.median(mid)))


def stage_quotes(client, data_dir, trades_csv, window_s=10, sample=None, seed=7, open_holdout=False):
    """NBBO in [minute, minute + window_s s) for each entry and exit minute of the run's filled trades."""
    tr = pd.read_csv(trades_csv, dtype={"ticker": str})
    tr = tr[tr["filled"].astype(bool)]
    if "variant" in tr:
        tr = tr.drop_duplicates(["day", "ticker", "entry_t", "exit_t"])
    if not open_holdout and (tr["day"] >= "2024-01-01").any():
        sys.exit("HOLDOUT GUARD: the trades file has 2024+ days.")
    if sample and len(tr) > sample:
        tr = tr.sample(n=sample, random_state=seed)
    # where in the minute the model trades: an entry / a stop at the START (a stop order triggers on the way
    # through; gaps fill at the open), the close exit at the END (it fills at the bar's close)
    at_close = tr["reason"].astype(str).eq("close").to_numpy() if "reason" in tr else np.zeros(len(tr), bool)
    legs = pd.concat([tr[["day", "ticker"]].assign(minute=tr["entry_t"].astype(int), leg="entry", window="start"),
                      tr[["day", "ticker"]].assign(minute=tr["exit_t"].astype(int), leg="exit",
                                                   window=np.where(at_close, "end", "start"))],
                     ignore_index=True).drop_duplicates(["day", "ticker", "minute"])
    out_path = os.path.join(data_dir, "quotes_spread.csv")
    old = pd.read_csv(out_path, dtype={"ticker": str}) if os.path.exists(out_path) else pd.DataFrame(columns=QUOTE_COLS)
    have = set(zip(old["day"], old["ticker"], old["minute"].astype(int)))
    rows = []
    for (day, minute, where), grp in legs.groupby(["day", "minute", "window"]):
        syms = sorted(s for s in grp["ticker"] if (day, s, int(minute)) not in have)
        if not syms:
            continue
        leg_of = dict(zip(grp["ticker"], grp["leg"]))
        hhmm = f"{int(minute) // 60:02d}:{int(minute) % 60:02d}"
        t0 = datetime.fromisoformat(ny_to_utc(day, hhmm).replace("Z", "+00:00"))
        if where == "end":
            t0 += timedelta(seconds=60 - window_s)
        start = t0.strftime("%Y-%m-%dT%H:%M:%SZ")
        end = (t0 + timedelta(seconds=window_s)).strftime("%Y-%m-%dT%H:%M:%SZ")
        got: dict[str, list] = {}
        for page in client.paged(DATA_URL, "/v2/stocks/quotes",
                                 dict(symbols=",".join(syms), start=start, end=end, limit=10000, feed="sip",
                                      sort="asc"), "quotes"):
            for sym, qs in page.items():
                got.setdefault(sym, []).extend(qs)
        for sym in syms:
            st = quote_spreads(got.get(sym, []))
            rows.append(dict(day=day, ticker=sym, minute=int(minute), leg=leg_of[sym], window=where,
                             **(st or dict(n_quotes=0, spread_bp_median=np.nan, spread_usd_median=np.nan,
                                           mid_median=np.nan))))
    new = pd.DataFrame(rows, columns=QUOTE_COLS)
    allq = pd.concat([x for x in (old, new) if len(x)], ignore_index=True) if len(old) or len(new) else new
    atomic_csv(allq.sort_values(["day", "ticker", "minute"]) if len(allq) else allq, out_path)
    print(f"quotes: {len(legs)} trade legs, {len(new)} new rows ({int((new['n_quotes'] == 0).sum())} without a "
          f"two-sided quote) -> {out_path}")
    return out_path


def identity_breaks(d: pd.DataFrame, jump: float = 3.0, gap_days: int = 30, gap_move: float = 0.5) -> pd.DataFrame:
    """Likely ticker reuse / identity breaks in the raw daily bars (known open issue: today's symbol mapping can
    glue two companies together). Per ticker, in date order, after undoing a split on its ex-date:
      jump : open(t) / close(t-1) > `jump` or < 1/`jump` with no split that day (no single stock does that often);
      gap  : > `gap_days` calendar days without a bar, then a level change > `gap_move` (a relisted / new company).
    WARN only: a real 3x gap (biotech news) also shows up. The list is for a human spot check."""
    out = []
    x = d.sort_values(["ticker", "date"])
    sf = x["split_factor"].fillna(1.0).replace(0, 1.0) if "split_factor" in x else pd.Series(1.0, index=x.index)
    prev_close = x.groupby("ticker")["close"].shift(1) / sf
    prev_date = pd.to_datetime(x.groupby("ticker")["date"].shift(1))
    ratio = x["open"] / prev_close
    gap = (pd.to_datetime(x["date"]) - prev_date).dt.days
    j = ratio.notna() & ((ratio > jump) | (ratio < 1 / jump))
    g = gap.notna() & (gap > gap_days) & ((ratio - 1).abs() > gap_move)
    for kind, mask in (("jump", j), ("gap", g & ~j)):
        for _, r in x[mask].assign(ratio=ratio[mask], gap=gap[mask]).iterrows():
            out.append(dict(ticker=r["ticker"], date=r["date"], kind=kind, open_over_prev_close=round(r["ratio"], 3),
                            days_since_prev_bar=int(r["gap"])))
    return pd.DataFrame(out, columns=["ticker", "date", "kind", "open_over_prev_close", "days_since_prev_bar"])


def stage_validate(data_dir, max_bad_share=0.01):
    """The data gate (plan): fail loudly before a backtest can run on broken data."""
    probs = []
    d = pd.read_csv(os.path.join(data_dir, "daily.csv"), dtype={"ticker": str})
    if d.duplicated(["ticker", "date"]).any():
        probs.append(("daily", "-", f"{int(d.duplicated(['ticker', 'date']).sum())} duplicate ticker-days"))
    bad_px = (d[["open", "high", "low", "close"]] <= 0).any(axis=1) | (d["low"] > d[["open", "close"]].min(axis=1) + 1e-9) \
        | (d["high"] < d[["open", "close"]].max(axis=1) - 1e-9)
    if bad_px.any():
        probs.append(("daily", "-", f"{int(bad_px.sum())} rows with impossible OHLC"))
    ib = identity_breaks(d)
    atomic_csv(ib, os.path.join(data_dir, "identity_warnings.csv"))
    for _, r in ib.iterrows():
        probs.append(("identity", r["date"], f"WARN {r['ticker']} {r['kind']}: open/prev close "
                                             f"{r['open_over_prev_close']}, {r['days_since_prev_bar']} days since the "
                                             "previous bar (ticker reuse? spot-check)"))
    mdir = os.path.join(data_dir, "minutes")
    days = sorted(f[8:18] for f in os.listdir(mdir) if f.startswith("minutes_") and f.endswith(".csv"))
    bad_days = 0
    import nyse_calendar as NC
    for day in days:
        # calendar cross-check (WARN only: listed in validation.csv, not counted by the gate; b_mvb.py skips a
        # half-day mismatch as a data gap when the bundled calendar is active)
        y = int(day[:4])
        if NC.COVERED[0] <= y <= NC.COVERED[1]:
            if day in NC.holiday_set(day, day):
                probs.append(("calendar", day, "WARN minute file on an NYSE holiday"))
        m = pd.read_csv(os.path.join(mdir, f"minutes_{day}.csv"), dtype={"ticker": str, "time": str})
        issues = []
        if m.duplicated(["ticker", "time"]).any():
            issues.append("duplicate bars")
        if ((m["l"] > m[["o", "c"]].min(axis=1) + 1e-9) | (m["h"] < m[["o", "c"]].max(axis=1) - 1e-9)).any():
            issues.append("impossible OHLC")
        if (m["time"] < "09:30").any() or (m["time"] > "15:59").any():
            issues.append("bars outside 09:30-15:59")
        spy = m[m["ticker"] == "SPY"]
        if spy.empty or spy["time"].min() != "09:30":
            issues.append("SPY missing or first bar not 09:30")
        elif spy["time"].max() not in ("09:44", "12:59", "15:59"):
            issues.append(f"SPY last bar {spy['time'].max()} (not a full/half day)")
        elif NC.COVERED[0] <= int(day[:4]) <= NC.COVERED[1] and spy["time"].max() != "09:44":
            want = "12:59" if day in NC.early_close_map(day, day) else "15:59"
            if spy["time"].max() != want:
                probs.append(("calendar", day, f"WARN SPY last bar {spy['time'].max()}, NYSE calendar says {want}"))
        if issues:
            bad_days += 1
            probs.append(("minutes", day, "; ".join(issues)))
    rep = pd.DataFrame(probs, columns=["file", "day", "problem"])
    atomic_csv(rep, os.path.join(data_dir, "validation.csv"))
    share = bad_days / max(1, len(days))
    print(f"validate: {len(days)} minute files, {bad_days} with problems ({share:.1%}); daily problems: "
          f"{sum(1 for p in probs if p[0] == 'daily')}; calendar warnings: "
          f"{sum(1 for p in probs if p[0] == 'calendar')}; identity warnings (ticker reuse?): "
          f"{sum(1 for p in probs if p[0] == 'identity')} -> data/validation.csv")
    if share > max_bad_share or any(p[0] == "daily" for p in probs):
        sys.exit("DATA GATE FAILED: fix the problems in data/validation.csv before backtesting")
    print("DATA GATE PASSED")
    return rep


# ----------------------------------------------------------------------------- CLI
def main(argv=None, client=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["probe", "assets", "daily", "opening", "fullday", "validate", "ticks",
                                          "quotes"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--start", default="2015-11-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--trades", help="trades.csv of a b_mvb run (for `ticks` and `quotes`)")
    ap.add_argument("--window-s", type=int, default=10, help="quotes: seconds from the start of each minute")
    ap.add_argument("--sample", type=int, default=None, help="quotes: seeded sample of N trades")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--random", type=int, default=30)
    ap.add_argument("--open-holdout", action="store_true", help="allow downloading dates >= 2024-01-01")
    a = ap.parse_args(argv)
    if a.end >= "2024-01-01" and not a.open_holdout and a.stage not in ("probe", "assets"):
        sys.exit("HOLDOUT GUARD: not downloading 2024+ yet. Pass --open-holdout only when the holdout run is due "
                 "(having the data on disk makes an accidental peek easy).")
    os.makedirs(a.data, exist_ok=True)
    if a.stage == "validate":
        return stage_validate(a.data)
    client = client or Client()
    if not client.key and client.transport == Client._urllib:
        sys.exit("set APCA_API_KEY_ID and APCA_API_SECRET_KEY (free Alpaca account)")
    if a.stage == "probe":
        return stage_probe(client, a.data)
    if a.stage == "assets":
        return stage_assets(client, a.data)
    if a.stage == "daily":
        return stage_daily(client, a.data, a.start, a.end)
    if a.stage == "opening":
        return stage_opening(client, a.data, a.start, a.end)
    if a.stage == "fullday":
        return stage_fullday(client, a.data, a.start, a.end, top=a.top, n_random=a.random)
    if a.stage == "ticks":
        if not a.trades:
            sys.exit("--trades runs/<run>/trades.csv required")
        return stage_ticks(client, a.data, a.trades)
    if a.stage == "quotes":
        if not a.trades:
            sys.exit("--trades runs/<run>/trades.csv required")
        return stage_quotes(client, a.data, a.trades, window_s=a.window_s, sample=a.sample,
                            open_holdout=a.open_holdout)


if __name__ == "__main__":
    main()
