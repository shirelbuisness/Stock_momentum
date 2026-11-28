"""Visual + independent proof that b_mvb.py does what the plan says.

1. verify_all(): an INDEPENDENT re-implementation of the base trade rules (plain loops, no shared code with
   simulate_trade) re-derives every trade in a run's trades.csv and counts mismatches.
2. shot(): a chart of one trade (1-min candles, OR box, trigger, executed entry, stop, exit) = a "screenshot".
3. equity(): cumulative net R per variant vs time.
4. replay_video(): one trade replayed bar by bar, the way the system sees it live.
"""
from __future__ import annotations

import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

TICK = 0.01
BG, FG, GRID = "#ffffff", "#1f2328", "#e6e8eb"
UP, DOWN = "#1a7f37", "#cf222e"
C_OR, C_TRIG, C_ENTRY, C_STOP, C_EXIT = "#d0d7de", "#0969da", "#0969da", "#cf222e", "#8250df"


def tmin(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def hhmm(x: int) -> str:
    return f"{x // 60:02d}:{x % 60:02d}"


def load_day(minutes_dir: str, day: str) -> pd.DataFrame:
    m = pd.read_csv(os.path.join(minutes_dir, f"minutes_{day}.csv"), dtype={"ticker": str, "time": str})
    m["t"] = m["time"].map(tmin)
    return m


# ----------------------------------------------------------------------------- independent verifier
HALF_DAYS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nyse_early_closes.csv")


def half_days(path: str = HALF_DAYS_CSV) -> set[str]:
    """NYSE 13:00 closes, read straight from the committed CSV (not from b_mvb.py)."""
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path, dtype=str)["date"])


def independent_trade(bars: pd.DataFrame, atr: float, stop_atr=0.10, hs_bp=3.0, slip_ticks=1.0, slip_bp=2.0,
                      comm=0.0035, time_stop=None, time_stop_r=0.5, ohlc_path=False, last_bar="15:58"):
    """Long-only base rules written from the plan text, NOT from simulate_trade:
    OR = 09:30-09:34. Green OR bar -> buy stop at OR high. From 09:35, first 1-min bar whose high is at
    least 1 tick above the trigger fills at max(trigger, open) + half spread + slippage. Stop = fill - stop_atr*ATR.
    Same bar reaching the stop -> stopped. Later bars: open below stop -> exit at open; low at/below stop -> exit at
    stop; both minus half spread + slippage. Otherwise exit at the close of the last bar <= `last_bar`
    (15:58; 12:58 on a 13:00 half day)."""
    orb = bars[(bars.t >= tmin("09:30")) & (bars.t < tmin("09:35"))]
    if orb.empty or orb.c.iloc[-1] <= orb.o.iloc[0]:
        return None
    trig = orb.h.max()
    hs = max(0.005, trig * hs_bp / 1e4)
    k = max(slip_ticks * TICK, trig * slip_bp / 1e4)
    day_bars = bars[(bars.t >= tmin("09:35")) & (bars.t <= tmin(last_bar))].reset_index(drop=True)
    entry = None
    for i, b in day_bars.iterrows():
        if entry is None:
            if b.h >= trig + TICK:
                entry = max(trig, b.o) + hs + k
                stop = entry - stop_atr * atr
                ei = i
                worst_after_entry = b.c if (ohlc_path and b.c >= b.o) else b.l
                if worst_after_entry <= stop:
                    return dict(entry_t=int(b.t), entry_px=entry, exit_t=int(b.t), exit_px=stop - hs - k,
                                reason="stop_same_bar", r=(stop - hs - k - entry - 2 * comm) / (stop_atr * atr))
            continue
        prev = day_bars.iloc[i - 1]
        if time_stop is not None and day_bars.iloc[ei].t < tmin(time_stop) and prev.t == tmin(time_stop) \
                and prev.c < entry + time_stop_r * stop_atr * atr:
            x = b.o - hs - k
            return dict(entry_t=int(day_bars.iloc[ei].t), entry_px=entry, exit_t=int(b.t), exit_px=x,
                        reason="time_stop", r=(x - entry - 2 * comm) / (stop_atr * atr))
        if b.o <= stop:
            x = b.o - hs - k
            return dict(entry_t=int(day_bars.iloc[ei].t), entry_px=entry, exit_t=int(b.t), exit_px=x,
                        reason="stop_gap", r=(x - entry - 2 * comm) / (stop_atr * atr))
        if b.l <= stop:
            x = stop - hs - k
            return dict(entry_t=int(day_bars.iloc[ei].t), entry_px=entry, exit_t=int(b.t), exit_px=x,
                        reason="stop", r=(x - entry - 2 * comm) / (stop_atr * atr))
    if entry is None:
        return dict(reason="no_trigger")
    last = day_bars.iloc[-1]
    x = last.c - hs - k
    return dict(entry_t=int(day_bars.iloc[ei].t), entry_px=entry, exit_t=int(last.t), exit_px=x,
                reason="close", r=(x - entry - 2 * comm) / (stop_atr * atr))


def verify_all(trades: pd.DataFrame, minutes_dir: str, variant: str, **rule) -> dict:
    t = trades[(trades.variant == variant)]
    ok, bad, rows = 0, 0, []
    cache = {}
    halves = half_days()
    for _, tr in t.iterrows():
        if tr.day not in cache:
            cache[tr.day] = load_day(minutes_dir, tr.day)
        bars = cache[tr.day][cache[tr.day].ticker == tr.ticker].sort_values("t")
        ind = independent_trade(bars, tr.atr14, last_bar="12:58" if tr.day in halves else "15:58", **rule)
        if not tr.filled:
            same = ind is not None and ind["reason"] == "no_trigger" and tr.reason == "no_trigger"
        else:
            same = (ind is not None and ind.get("entry_t") == tr.entry_t and ind.get("exit_t") == tr.exit_t
                    and abs(ind["entry_px"] - tr.entry_px) < 1e-6 and abs(ind["exit_px"] - tr.exit_px) < 1e-6
                    and abs(ind["r"] - tr.r_net) < 1e-6 and ind["reason"] == tr.reason)
        ok += same
        bad += not same
        if not same:
            rows.append((tr.day, tr.ticker, tr.reason, ind))
    return dict(variant=variant, checked=ok + bad, match=ok, mismatch=bad, examples=rows[:5])


# ----------------------------------------------------------------------------- screenshots
def candles(ax, bars, t0, t1):
    w = 0.6
    for _, b in bars[(bars.t >= t0) & (bars.t <= t1)].iterrows():
        col = UP if b.c >= b.o else DOWN
        ax.plot([b.t, b.t], [b.l, b.h], color=col, lw=0.8, zorder=2)
        ax.add_patch(Rectangle((b.t - w / 2, min(b.o, b.c)), w, max(abs(b.c - b.o), 0.002),
                               facecolor=col, edgecolor=col, lw=0.5, zorder=3))


def shot(tr: pd.Series, minutes_dir: str, path: str, title: str, upto: int | None = None, ax=None):
    bars = load_day(minutes_dir, tr.day)
    bars = bars[bars.ticker == tr.ticker].sort_values("t")
    orb = bars[(bars.t >= tmin("09:30")) & (bars.t < tmin("09:35"))]
    t0 = tmin("09:30")
    t1 = min(int(tr.exit_t) + 25, tmin("15:59")) if pd.notna(tr.exit_t) else t0 + 120
    if upto is not None:
        t_show = upto
    else:
        t_show = t1
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(12, 6.2), dpi=110)
    fig = ax.figure
    ax.set_facecolor(BG)
    view = bars[(bars.t >= t0) & (bars.t <= t1)]
    lo, hi = view.l.min(), view.h.max()
    stop = tr.entry_px - tr.r_ps
    lo, hi = min(lo, stop, tr.exit_px), max(hi, tr.entry_px, tr.exit_px)
    pad = (hi - lo) * 0.08
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(t0 - 2, t1 + 2)
    ax.add_patch(Rectangle((t0 - 0.5, orb.l.min()), 5, orb.h.max() - orb.l.min(), facecolor=C_OR, alpha=0.55,
                           edgecolor="none", zorder=1))
    ax.text(t0 - 1, orb.l.min(), "OR", fontsize=9, color="#57606a", ha="left", va="top")
    candles(ax, bars, t0, t_show)
    trig = orb.h.max()
    ax.axhline(trig, color=C_TRIG, lw=1, ls=":", zorder=1)
    ax.text(t1, trig, f" trigger (OR high) {trig:.2f}", color=C_TRIG, fontsize=9, va="bottom", ha="right")
    entered = pd.notna(tr.entry_t) and t_show >= tr.entry_t
    if entered:
        e_end = min(t_show, tr.exit_t)
        ax.plot([tr.entry_t, e_end], [stop, stop], color=C_STOP, lw=1.4, ls="--", zorder=4)
        ax.text(tr.entry_t, stop, f" stop {stop:.2f} (fill − 10% ATR = {tr.r_ps:.3f})" if abs(tr.r_ps - 0.1 * tr.atr14) < 1e-9
                else f" stop {stop:.2f} ({tr.r_ps / tr.atr14:.0%} ATR)", color=C_STOP, fontsize=9, va="top")
        ax.scatter([tr.entry_t], [tr.entry_px], marker="^", s=90, color=C_ENTRY, zorder=6, edgecolor="white")
        ax.annotate(f"entry {tr.entry_px:.3f} @ {hhmm(int(tr.entry_t))} (incl. ½ spread + slip)",
                    (tr.entry_t, tr.entry_px), xytext=(14, 26), textcoords="offset points", color=C_ENTRY,
                    fontsize=9, arrowprops=dict(arrowstyle="-", color=C_ENTRY, lw=0.7),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=7)
    if pd.notna(tr.exit_t) and t_show >= tr.exit_t:
        ax.scatter([tr.exit_t], [tr.exit_px], marker="X", s=110, color=C_EXIT, zorder=6, edgecolor="white")
        right = tr.exit_t > (t0 + t1) / 2
        ax.annotate(f"exit {tr.exit_px:.3f} @ {hhmm(int(tr.exit_t))} ({tr.reason})", (tr.exit_t, tr.exit_px),
                    xytext=(-14 if right else 14, -28), textcoords="offset points", ha="right" if right else "left",
                    color=C_EXIT, fontsize=9, arrowprops=dict(arrowstyle="-", color=C_EXIT, lw=0.7),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=7)
    ticks = list(range(t0, t1 + 1, 15 if t1 - t0 <= 150 else 30))
    ax.set_xticks(ticks)
    ax.set_xticklabels([hhmm(x) for x in ticks], fontsize=8)
    ax.grid(color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors="#57606a")
    res = (f"net {tr.r_net:+.2f}R  (gross {tr.r_gross:+.2f}R, cost {tr.r_gross - tr.r_net:.2f}R)"
           if (pd.notna(tr.exit_t) and t_show >= tr.exit_t) else "")
    ax.set_title(f"{title}\n{tr.ticker} {tr.day} · RVOL {tr.rvol:.1f}× · ATR14 {tr.atr14:.2f}   {res}",
                 fontsize=11, color=FG, loc="left")
    if own:
        fig.tight_layout()
        fig.savefig(path, facecolor=BG)
        plt.close(fig)


# ----------------------------------------------------------------------------- equity
def equity(trades: pd.DataFrame, variants: list[str], path: str, title: str):
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=110)
    cols = ["#0969da", "#1a7f37", "#bf8700", "#8250df", "#cf222e"]
    for c, v in zip(cols, variants):
        f = trades[(trades.variant == v) & trades.filled].sort_values(["day", "entry_t"])
        d = f.groupby("day")["r_net"].sum().cumsum()
        x = pd.to_datetime(d.index)
        ax.plot(x, d.values, color=c, lw=2, label=f"{v}  (n={len(f)}, mean {f.r_net.mean():+.3f}R)")
        ax.text(x[-1], d.values[-1], f" {d.values[-1]:+.0f}R", color=c, fontsize=9, va="center")
    ax.axhline(0, color="#57606a", lw=0.8)
    ax.set_ylabel("cumulative net R (all costs)")
    ax.grid(color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- replay video
def replay_video(tr: pd.Series, minutes_dir: str, path: str, fps=30, frames_per_bar=3, hold_s=2.5):
    """One scene file: bars appear one by one; entry / stop / exit appear when the system would know them."""
    W, H, DPI = 1280, 720, 100
    t_end = min(int(tr.exit_t) + 25, tmin("15:59"))
    steps = list(range(tmin("09:30"), t_end + 1))
    tmp = path[:-4] + ".tmp.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    fig, ax = plt.subplots(figsize=(W / DPI, H / DPI), dpi=DPI)

    def frame(upto, n):
        ax.clear()
        shot(tr, minutes_dir, "", "Replay: what the system sees, minute by minute", upto=upto, ax=ax)
        ax.text(0.99, 0.02, f"clock {hhmm(upto)} ET", transform=ax.transAxes, ha="right", fontsize=12,
                color="#57606a")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        for _ in range(n):
            p.stdin.write(buf.tobytes())

    for t in steps:
        frame(t, frames_per_bar)
    frame(steps[-1], int(hold_s * fps))
    p.stdin.close()
    rc = p.wait()
    plt.close(fig)
    if rc == 0:
        os.replace(tmp, path)
    return rc
