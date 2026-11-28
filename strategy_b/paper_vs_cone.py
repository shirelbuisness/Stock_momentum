#!/usr/bin/env python3
"""
paper_vs_cone.py: where do the paper (shadow-fill) results sit inside the backtest's expected cone? (rigor skill §8)
Offline: reads two CSVs, writes a short report. No broker code.

  python paper_vs_cone.py --run runs/<DEV run> --version B-v1__base --paper paper_trades.csv

paper_trades.csv: one row per closed paper trade, columns day (YYYY-MM-DD) and r_net (net R after all costs, from
the PESSIMISTIC SHADOW fill, not Alpaca's paper fill); optional: reason (exit reason), entry_t (minutes).
The cone is recomputed at EXACTLY the paper trade count n from the run's trades (day-block bootstrap, seeded),
so it is not limited to the 25/50/100/200 horizons of expected_cone.json.

Verdicts (signposts, rigor skill §8):
  cum R < p5 or max DD > p95  -> CHECK IMPLEMENTATION first (signals, fills, costs), then the market
  cum R > p95                 -> ALSO INVESTIGATE (over-performance is often a bug: lookahead, wrong fills)
  otherwise                   -> INSIDE the cone (says nothing about the edge: 1 week of trades proves nothing)
Also compares trades/day and the exit-reason mix with the backtest: a different mix = not the tested strategy.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import gate_stats as G


def check(bt: pd.DataFrame, paper: pd.DataFrame, paths: int = 5000, seed: int = 0) -> dict:
    bt = bt[bt["filled"]].sort_values(["day", "entry_t"])
    paper = paper.sort_values(["day"] + (["entry_t"] if "entry_t" in paper else []))
    n = len(paper)
    if n == 0 or bt.empty:
        return dict(n=n, verdict="NO DATA")
    cone = G.expected_cone(bt["r_net"].to_numpy(), horizons=(n,), paths=paths, seed=seed,
                           days=bt["day"].to_numpy())["horizons"][str(n)]
    cum = paper["r_net"].cumsum().to_numpy()
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    dd = float(np.max(peak - cum))
    lo, mid, hi = cone["cum_R"]
    dd95 = cone["max_dd_R"][2]
    if cum[-1] < lo or dd > dd95:
        verdict = "CHECK IMPLEMENTATION (below p5 or drawdown above p95)"
    elif cum[-1] > hi:
        verdict = "INVESTIGATE (above p95: over-performance is often a bug)"
    else:
        verdict = "INSIDE the cone"
    out = dict(n=n, days=int(paper["day"].nunique()), cum_R=float(cum[-1]), max_dd_R=dd, cone_cum_R=[lo, mid, hi],
               cone_max_dd_R=cone["max_dd_R"], verdict=verdict,
               trades_per_day_paper=n / max(1, paper["day"].nunique()),
               trades_per_day_backtest=len(bt) / max(1, bt["day"].nunique()))
    if "reason" in paper and "reason" in bt:
        a = paper["reason"].value_counts(normalize=True)
        b = bt["reason"].value_counts(normalize=True)
        mix = pd.concat([a.rename("paper"), b.rename("backtest")], axis=1).fillna(0.0)
        out["exit_mix"] = mix.round(3).to_dict(orient="index")
        out["exit_mix_max_gap"] = float((mix["paper"] - mix["backtest"]).abs().max())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--version", default=None, help="variant id in the run (default: the first = base)")
    ap.add_argument("--paper", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    tr = pd.read_csv(os.path.join(a.run, "trades.csv"), dtype={"ticker": str})
    ver = a.version or tr["variant"].iloc[0]
    bt = tr[tr["variant"] == ver]
    if bt.empty:
        sys.exit(f"version {ver} not in {a.run}")
    res = check(bt, pd.read_csv(a.paper), seed=a.seed)
    L = [f"# Paper vs expected cone: {ver}", "", f"**{res['verdict']}**", ""]
    if "cum_R" in res:
        L += [f"- paper: {res['n']} trades on {res['days']} days, cum net R {res['cum_R']:.2f}, max DD {res['max_dd_R']:.2f} R",
              f"- cone after {res['n']} trades (day-block, 5000 paths): cum R p5/p50/p95 = "
              + " / ".join(f"{x:.2f}" for x in res["cone_cum_R"])
              + f"; max DD p95 = {res['cone_max_dd_R'][2]:.2f} R",
              f"- trades/day: paper {res['trades_per_day_paper']:.2f} vs backtest {res['trades_per_day_backtest']:.2f}"]
        if "exit_mix" in res:
            L.append(f"- exit mix (paper vs backtest), largest gap {res['exit_mix_max_gap']:.2f}: {res['exit_mix']}")
    p = os.path.join(os.path.dirname(os.path.abspath(a.paper)), "paper_vs_cone.md")
    open(p, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return res


if __name__ == "__main__":
    main()
