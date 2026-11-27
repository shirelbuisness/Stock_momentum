"""In-process fake of the Alpaca endpoints fetch_b_data.py uses, serving a synth.py market.

It follows the documented shapes: /v2/stocks/bars -> {"bars": {sym: [{t,o,h,l,c,v,n,vw}]}, "next_page_token"},
/v2/stocks/trades -> {"trades": {sym: [{t,p,s,x,c,i,z}]}, ...}, /v1/corporate-actions -> {"corporate_actions": {...}},
/v2/assets -> [ {symbol, name, exchange, status, tradable} ]. Timestamps are UTC RFC-3339; `limit` pages across
symbols; page tokens are opaque strings. Tick prints follow O->L->H->C for green bars and O->H->L->C for red.
"""
from __future__ import annotations

import os
import urllib.parse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")


def _utc(day, hhmm):
    h, m = map(int, hhmm.split(":"))
    return datetime.strptime(day, "%Y-%m-%d").replace(hour=h, minute=m, tzinfo=NY).astimezone(timezone.utc)


def _parse(ts):
    if len(ts) == 10:
        return datetime.strptime(ts, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


class FakeAlpaca:
    def __init__(self, market_dir, splits=None, inactive=None, iex_volume=None, spread_bp=None):
        self.daily = pd.read_csv(os.path.join(market_dir, "daily.csv"), dtype={"ticker": str})
        self.mdir = os.path.join(market_dir, "minutes")
        self.splits = splits or []            # [{"symbol","ex_date","old_rate","new_rate"}]
        self.inactive = set(inactive or [])
        # feed=iex minute bars: volume multiplier, a dict ticker -> share or a callable (ticker, day) -> share
        self.iex_volume = iex_volume or {}
        self.spread_bp = spread_bp or {}      # /v2/stocks/quotes: ticker -> full NBBO spread in bp (default 6)
        self._minutes = {}
        self.requests = []

    def _iex_share(self, ticker, day):
        f = self.iex_volume
        return float(f(ticker, day)) if callable(f) else float(f.get(ticker, 1.0))

    def _day_minutes(self, day):
        if day not in self._minutes:
            p = os.path.join(self.mdir, f"minutes_{day}.csv")
            if not os.path.exists(p):
                self._minutes[day] = None
            else:
                m = pd.read_csv(p, dtype={"ticker": str, "time": str})
                m["ts"] = [_utc(day, t) for t in m["time"]]
                self._minutes[day] = m
        return self._minutes[day]

    # ------------------------------------------------------------------ transport(url, headers)
    def __call__(self, url, headers):
        if headers.get("APCA-API-KEY-ID") in (None, ""):
            return 401, {"message": "unauthorized"}
        u = urllib.parse.urlparse(url)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        self.requests.append((u.path, q))
        if u.path == "/v2/assets":
            return 200, self._assets(q)
        if u.path == "/v2/stocks/bars":
            return 200, self._bars(q)
        if u.path == "/v2/stocks/trades":
            return 200, self._trades(q)
        if u.path == "/v2/stocks/quotes":
            return 200, self._quotes(q)
        if u.path == "/v1/corporate-actions":
            return 200, self._ca(q)
        return 404, {"message": "not found"}

    def _assets(self, q):
        out = []
        for t in sorted(self.daily["ticker"].unique()):
            st = "inactive" if t in self.inactive else "active"
            if st != q.get("status", "active"):
                continue
            etf = t == "SPY"
            out.append(dict(symbol=t, name="SPDR S&P 500 ETF Trust" if etf else f"{t} Holdings Inc",
                            exchange="ARCA" if etf else "NASDAQ", status=st, tradable=st == "active",
                            shortable=st == "active", easy_to_borrow=st == "active" and not t.endswith("7")))
        return out

    def _rows_bars(self, syms, tf, start, end, feed="sip"):
        rows = []
        if tf == "1Day":
            d = self.daily[self.daily["ticker"].isin(syms)]
            for _, r in d.iterrows():
                ts = datetime.strptime(r["date"], "%Y-%m-%d").replace(tzinfo=NY).astimezone(timezone.utc)
                if start <= ts <= end:
                    rows.append((r["ticker"], ts, dict(o=r["open"], h=r["high"], l=r["low"], c=r["close"],
                                                       v=int(r["volume"]), n=1, vw=r["close"])))
        else:
            day = start.astimezone(NY).strftime("%Y-%m-%d")
            m = self._day_minutes(day)
            if m is not None:
                x = m[m["ticker"].isin(syms) & (m["ts"] >= start) & (m["ts"] <= end)]
                for _, r in x.iterrows():
                    v = r["v"] * (self._iex_share(r["ticker"], day) if feed == "iex" else 1.0)
                    rows.append((r["ticker"], r["ts"], dict(o=r["o"], h=r["h"], l=r["l"], c=r["c"], v=int(v),
                                                            n=1, vw=r["c"])))
        rows.sort(key=lambda z: (z[0], z[1]))
        return rows

    def _page(self, rows, q, key):
        limit = int(q.get("limit", 1000))
        off = int(q.get("page_token", "0") or 0)
        page = rows[off:off + limit]
        out = {}
        for sym, ts, payload in page:
            out.setdefault(sym, []).append({"t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), **payload})
        nxt = str(off + limit) if off + limit < len(rows) else None
        return {key: out, "next_page_token": nxt, "currency": "USD"}

    def _bars(self, q):
        syms = q["symbols"].split(",")
        return self._page(self._rows_bars(syms, q["timeframe"], _parse(q["start"]), _parse(q["end"]),
                                          q.get("feed", "sip")), q, "bars")

    def _trades(self, q):
        syms = q["symbols"].split(",")
        start, end = _parse(q["start"]), _parse(q["end"])
        rows = []
        for sym, ts, b in self._rows_bars(syms, "1Min", start, end):
            if ts >= end:
                continue
            green = b["c"] >= b["o"]
            path = [b["o"], b["l"], b["h"], b["c"]] if green else [b["o"], b["h"], b["l"], b["c"]]
            pts = []
            for a, z in zip(path[:-1], path[1:]):            # 5 prints per leg, straight line
                pts += [a + (z - a) * i / 5 for i in range(5)]
            pts.append(path[-1])
            for i, p in enumerate(pts):
                rows.append((sym, ts.replace(second=min(59, i * 3)),
                             dict(p=round(p, 4), s=100, x="V", c=["@"], i=i, z="C")))
        rows.sort(key=lambda z: (z[0], z[1]))
        return self._page(rows, q, "trades")

    def _quotes(self, q):
        """One NBBO quote every 2 s around the bar's close: spread = spread_bp of the mid; every 5th quote is
        locked (bid = ask) and the 7th one-sided (bid 0), which a correct parser must drop."""
        syms = q["symbols"].split(",")
        start, end = _parse(q["start"]), _parse(q["end"])
        minute = start.replace(second=0)
        rows = []
        for sym, ts, b in self._rows_bars(syms, "1Min", minute, minute):
            mid = float(b["c"])
            half = mid * self.spread_bp.get(sym, 6.0) / 2e4
            for i in range(30):
                t = ts.replace(second=2 * i)
                if not (start <= t < end):
                    continue
                bid, ask = mid - half, mid + half
                if i % 5 == 4:
                    bid = ask
                if i == 7:
                    bid = 0.0
                rows.append((sym, t, dict(bp=round(bid, 6), ap=round(ask, 6), bs=1, as_=1, bx="V", ax="V", c=["R"], z="C")))
        rows.sort(key=lambda z: (z[0], z[1]))
        return self._page(rows, q, "quotes")

    def _ca(self, q):
        syms = set(q.get("symbols", "").split(","))
        s, e = q.get("start", "1900-01-01"), q.get("end", "2100-01-01")
        fwd = [x for x in self.splits if x["symbol"] in syms and s <= x["ex_date"] <= e and x["new_rate"] > x["old_rate"]]
        rev = [x for x in self.splits if x["symbol"] in syms and s <= x["ex_date"] <= e and x["new_rate"] < x["old_rate"]]
        return {"corporate_actions": {"forward_splits": fwd, "reverse_splits": rev}, "next_page_token": None}
