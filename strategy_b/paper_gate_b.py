#!/usr/bin/env python3
"""
paper_gate_b.py: the prereg "Paper -> live gate" checked mechanically, offline. No broker code: it reads CSVs
that the paper run (or the shadow logger) wrote, and the backtest run it is compared with.

  python paper_gate_b.py --run runs/<DEV run> --paper paper_trades.csv --fills paper_fills.csv \
         --selection live_selection.csv [--version B-v1__base]

Prereg (rev 3): >= 3 months of paper, with
  (1) realised slippage <= 1.5x the model;
  (2) live/backtest top-20 overlap >= 80% (SIP feed);
  (3) live expectancy inside the backtest's 95% CI.
Only the version that passed the holdout runs in paper; this tool says whether paper supports going live.
It is a gate for REAL money, so missing inputs FAIL (fail closed).

Inputs
  paper_trades.csv : day, r_net  (net R per closed paper trade, from the pessimistic SHADOW fill)
  paper_fills.csv  : side (+1 buy / -1 sell), ref_px, fill_px[, qty]
                     ref_px = the price the backtest model fills at BEFORE costs (the trigger, the bar open if it
                     gapped through, the stop, or the exit bar's close); fill_px = the real paper fill.
                     Realised cost/share = side * (fill_px - ref_px); model cost/share = half spread + k at ref_px,
                     exactly as b_mvb.simulate_trade charges it (commission is a separate, known number).
  live_selection.csv: day, ticker  (the names the live 09:35 selection picked on that day)
The backtest's selection for the same days comes from the run's trades.csv (all signals, filled or not).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import b_mvb as B

MIN_DAYS = 63              # ~3 months of sessions
SLIP_RATIO_MAX = 1.5
OVERLAP_MIN = 0.80


def model_cost_ps(ref_px, k: B.Knobs = B.Knobs()) -> np.ndarray:
    """Per-side cost per share the backtest charges at price ref_px: half spread + slippage k (no commission)."""
    p = np.abs(np.asarray(ref_px, dtype=float))
    hs = np.maximum(0.5 * B.TICK, p * k.half_spread_bp / 1e4) if k.half_spread_bp > 0 else np.zeros_like(p)
    kk = np.maximum(k.slip_ticks * B.TICK, p * k.slip_bp / 1e4)
    return hs + kk


def slippage_ratio(fills: pd.DataFrame, k: B.Knobs = B.Knobs()) -> dict:
    if fills is None or fills.empty:
        return dict(n=0, ratio=float("nan"))
    q = fills["qty"].astype(float).abs() if "qty" in fills else pd.Series(1.0, index=fills.index)
    real = fills["side"].astype(float) * (fills["fill_px"].astype(float) - fills["ref_px"].astype(float))
    model = model_cost_ps(fills["ref_px"], k)
    return dict(n=int(len(fills)), realised_ps=float((real * q).sum() / q.sum()),
                model_ps=float((model * q).sum() / q.sum()), ratio=float((real * q).sum() / (model * q).sum()))


def selection_overlap(live: pd.DataFrame, bt: pd.DataFrame) -> dict:
    """Per day: share of the backtest's selected names that live also picked; mean over the days both have."""
    if live is None or live.empty or bt.empty:
        return dict(days=0, mean=float("nan"), worst=float("nan"))
    L = live.groupby("day")["ticker"].apply(set)
    Bk = bt.groupby("day")["ticker"].apply(set)
    days = sorted(set(L.index) & set(Bk.index))
    ov = pd.Series({d: len(L[d] & Bk[d]) / len(Bk[d]) for d in days if len(Bk[d])}, dtype=float)
    return dict(days=int(len(ov)), mean=float(ov.mean()) if len(ov) else float("nan"),
                worst=float(ov.min()) if len(ov) else float("nan"),
                live_days_without_backtest=int(len(set(L.index) - set(Bk.index))))


def expectancy_in_ci(paper: pd.DataFrame, bt: pd.DataFrame) -> dict:
    f = bt[bt["filled"]]
    _, se, G = B.clustered_t(f["r_net"], f["day"])
    m = float(f["r_net"].mean())
    lo, hi = m - 1.96 * se, m + 1.96 * se
    pm = float(paper["r_net"].mean()) if len(paper) else float("nan")
    return dict(backtest_mean=m, ci95=[lo, hi], paper_mean=pm, paper_trades=int(len(paper)),
                inside=bool(np.isfinite(pm) and lo <= pm <= hi))


def check(bt: pd.DataFrame, paper: pd.DataFrame, fills: pd.DataFrame | None, live_sel: pd.DataFrame | None,
          k: B.Knobs = B.Knobs()) -> dict:
    days = int(paper["day"].nunique()) if len(paper) else 0
    sl = slippage_ratio(fills, k)
    ov = selection_overlap(live_sel, bt)
    ex = expectancy_in_ci(paper, bt)
    checks = {
        f"≥ {MIN_DAYS} paper sessions (~3 months) ({days})": days >= MIN_DAYS,
        f"realised slippage ≤ {SLIP_RATIO_MAX}× model ({sl['ratio']:.2f}×, {sl['n']} fills)":
            bool(sl["n"] > 0 and sl["ratio"] <= SLIP_RATIO_MAX),
        f"top-N overlap with the backtest ≥ {OVERLAP_MIN:.0%} ({ov['mean']:.1%} over {ov['days']} days)":
            bool(ov["days"] > 0 and ov["mean"] >= OVERLAP_MIN),
        f"paper expectancy inside the backtest 95% CI ({ex['paper_mean']:.3f} R in "
        f"[{ex['ci95'][0]:.3f}, {ex['ci95'][1]:.3f}])": ex["inside"],
    }
    return dict(passed=all(checks.values()), checks=checks, slippage=sl, overlap=ov, expectancy=ex, days=days)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--version", default=None)
    ap.add_argument("--paper", required=True)
    ap.add_argument("--fills")
    ap.add_argument("--selection")
    a = ap.parse_args(argv)
    tr = pd.read_csv(os.path.join(a.run, "trades.csv"), dtype={"ticker": str})
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    ver = a.version or cfg["variants"][0]["id"]
    bt = tr[tr["variant"] == ver]
    if bt.empty:
        sys.exit(f"version {ver} not in {a.run}")
    kn = next(v["knobs"] for v in cfg["variants"] if v["id"] == ver)
    k = B.Knobs(**{x: kn[x] for x in ("half_spread_bp", "slip_ticks", "slip_bp") if x in kn})
    rd = lambda p: pd.read_csv(p, dtype={"ticker": str}) if p else None
    res = check(bt, pd.read_csv(a.paper), rd(a.fills), rd(a.selection), k)
    L = [f"# Paper → live gate: {ver}", "", f"**{'PASS: paper supports going live' if res['passed'] else 'NOT YET'}**",
         ""] + [f"- [{'x' if ok else ' '}] {c}" for c, ok in res["checks"].items()]
    L += ["", "Missing inputs fail closed. Passing this gate does not replace the team's decision."]
    p = os.path.join(os.path.dirname(os.path.abspath(a.paper)), "paper_gate.md")
    open(p, "w").write("\n".join(L) + "\n")
    json.dump(res, open(p[:-3] + ".json", "w"), indent=2, default=str)
    print("\n".join(L))
    return res


if __name__ == "__main__":
    main()
