"""Seeded synthetic market for testing b_mvb.py (no real data).

random walk (momentum=0): no edge by construction. Any gross profit vs the null = a bug.
planted (momentum>0): on "in play" days the price drifts after 09:35 in the direction of the first
5-min bar. The strategy must detect it.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

MINUTES = [f"{h:02d}:{m:02d}" for h in range(9, 16) for m in range(60) if (h, m) >= (9, 30)]  # 09:30..15:59


def business_days(start: str, n: int) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]


def make_market(outdir: str, n_tickers=60, n_days=45, seed=1, momentum=0.0, start="2019-01-02"):
    rng = np.random.default_rng(seed)
    days = business_days(start, n_days)
    tickers = [f"T{i:03d}" for i in range(n_tickers)] + ["SPY"]
    price = {t: float(rng.uniform(15, 120)) for t in tickers}
    price["SPY"] = 300.0
    base_vol = {t: float(rng.uniform(2e6, 8e6)) for t in tickers}  # shares/day
    os.makedirs(os.path.join(outdir, "minutes"), exist_ok=True)
    nmin = len(MINUTES)
    ushape = 1 + 3 * np.exp(-np.arange(nmin) / 20) + 1.5 * np.exp(-(nmin - 1 - np.arange(nmin)) / 20)
    ushape /= ushape.sum()
    daily_rows = []
    for day in days:
        rows = []
        for t in tickers:
            p0 = price[t] * float(np.exp(rng.normal(0, 0.01)))    # overnight gap
            sig = 0.02 / np.sqrt(nmin)                            # ~2% daily vol
            inplay = t != "SPY" and rng.random() < 0.12
            mult = float(np.exp(rng.normal(2.0, 0.6))) if inplay else float(np.exp(rng.normal(0, 0.25)))
            if inplay:
                sig *= 2.0
            r = rng.normal(0, sig, nmin)
            if momentum and inplay:
                d1 = np.sign(r[:5].sum()) or 1.0
                r[5:] += d1 * momentum * sig
            closes = p0 * np.exp(np.cumsum(r))
            opens = np.concatenate([[p0], closes[:-1]])
            wig = np.abs(rng.normal(0, sig * 0.5, nmin)) * closes
            highs = np.maximum(opens, closes) + wig
            lows = np.minimum(opens, closes) - np.abs(rng.normal(0, sig * 0.5, nmin)) * closes
            vol = base_vol[t] * ushape * rng.lognormal(0, 0.3, nmin)
            vol[:5] *= mult
            vol = np.maximum(1, vol.round())
            rows.append(pd.DataFrame(dict(ticker=t, time=MINUTES, o=opens.round(2), h=highs.round(2),
                                          l=lows.round(2), c=closes.round(2), v=vol.astype(int))))
            daily_rows.append(dict(ticker=t, date=day, open=round(opens[0], 2), high=round(highs.max(), 2),
                                   low=round(lows.min(), 2), close=round(closes[-1], 2), volume=int(vol.sum()),
                                   split_factor=1.0))
            price[t] = float(closes[-1])
        pd.concat(rows).to_csv(os.path.join(outdir, "minutes", f"minutes_{day}.csv"), index=False)
    pd.DataFrame(daily_rows).to_csv(os.path.join(outdir, "daily.csv"), index=False)
    return days
