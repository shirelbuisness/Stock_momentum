#!/usr/bin/env python3
"""
spread_b.py: calibrate the cost model's half spread with measured SIP quotes (offline; reads files only).

  python fetch_b_data.py quotes --trades runs/<run>/trades.csv [--sample 3000]   # -> data/quotes_spread.csv
  python spread_b.py --run runs/<run> --quotes data/quotes_spread.csv

The locked cost model (prereg rev 3) charges "half spread (3 bp until calibrated with SIP quotes, then the measured
value)". This tool measures it AT THE MOMENTS THE MODEL TRADES (first seconds of each entry and exit minute) and
shows what the measured value would do to each version's mean net R. It is a DIAGNOSTIC: nothing here changes a
result or logs a trial. Which statistic becomes "the measured value" (median of all legs, entry-leg median, or
per-trade spreads) is a team decision, written into the prereg before the number is used; a rerun with the new
value is then a new trial.

Writes <run>/spread_calibration.md and .json:
  - measured half spread in bp (median / mean / p75 / p90), overall, entry vs exit legs, by price bucket;
  - the model's half spread at the same legs (max(half a tick, 3 bp));
  - per version, on the trades with both legs measured: mean net R as run vs re-costed with each trade's own
    measured half spreads (dR = -(measured - model) $ per share, both legs, / risk per share).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import b_mvb as B

BUCKETS = [(0, 10, "< $10"), (10, 30, "$10–30"), (30, 100, "$30–100"), (100, np.inf, "≥ $100")]


def model_half_spread_usd(px, half_spread_bp: float) -> np.ndarray:
    """The half spread b_mvb charges at price px (same formula as paper_gate_b.model_cost_ps, without k)."""
    p = np.abs(np.asarray(px, dtype=float))
    return np.maximum(0.5 * B.TICK, p * half_spread_bp / 1e4) if half_spread_bp > 0 else np.zeros_like(p)


def _stats(x: pd.Series) -> dict:
    x = x.dropna()
    if x.empty:
        return dict(n=0, median=np.nan, mean=np.nan, p75=np.nan, p90=np.nan)
    return dict(n=int(len(x)), median=float(x.median()), mean=float(x.mean()), p75=float(x.quantile(0.75)),
                p90=float(x.quantile(0.90)))


def calibrate(tr: pd.DataFrame, quotes: pd.DataFrame, knobs_by_version: dict[str, dict]) -> dict:
    q = quotes.copy()
    q["half_bp"] = q["spread_bp_median"] / 2
    q["half_usd"] = q["spread_usd_median"] / 2
    meas = q.dropna(subset=["half_bp"])
    out = dict(legs=int(len(q)), legs_without_quote=int(q["half_bp"].isna().sum()),
               measured_half_bp=_stats(meas["half_bp"]),
               by_leg={lg: _stats(g["half_bp"]) for lg, g in meas.groupby("leg")},
               by_price={lab: _stats(meas.loc[(meas["mid_median"] >= lo) & (meas["mid_median"] < hi), "half_bp"])
                         for lo, hi, lab in BUCKETS})
    hs_bp = float(knobs_by_version.get(next(iter(knobs_by_version), ""), {}).get("half_spread_bp",
                                                                                B.Knobs().half_spread_bp))
    model_bp = model_half_spread_usd(meas["mid_median"], hs_bp) / meas["mid_median"] * 1e4
    out["model_half_bp_at_same_legs"] = _stats(pd.Series(model_bp))
    key = q.set_index(["day", "ticker", "minute"])["half_usd"]
    key = key[~key.index.duplicated()]
    f = tr[tr["filled"].astype(bool)].copy()
    f["hs_in"] = key.reindex(list(zip(f["day"], f["ticker"], f["entry_t"].astype(int)))).to_numpy()
    f["hs_out"] = key.reindex(list(zip(f["day"], f["ticker"], f["exit_t"].astype(int)))).to_numpy()
    both = f.dropna(subset=["hs_in", "hs_out"])
    rows = {}
    for ver, g in both.groupby("variant") if "variant" in both else [("(single)", both)]:
        hs = float(knobs_by_version.get(ver, {}).get("half_spread_bp", B.Knobs().half_spread_bp))
        d_ps = (g["hs_in"] - model_half_spread_usd(g["entry_px"], hs)) + \
               (g["hs_out"] - model_half_spread_usd(g["exit_px"], hs))
        r_new = g["r_net"] - d_ps / g["r_ps"]
        rows[ver] = dict(trades_measured=int(len(g)),
                         trades_in_run=int(len(f[f["variant"] == ver])) if "variant" in f else int(len(f)),
                         mean_r_net_as_run=float(g["r_net"].mean()), mean_r_net_measured_spread=float(r_new.mean()),
                         delta_r=float(r_new.mean() - g["r_net"].mean()))
    out["versions"] = rows
    return out


def fmt(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def report(res: dict) -> str:
    m = res["measured_half_bp"]
    L = ["# Spread calibration (DIAGNOSTIC: no result changed, no trial logged)", "",
         f"Legs: {res['legs']} ({res['legs_without_quote']} without a two-sided quote, excluded).", "",
         "| half spread, bp | n | median | mean | p75 | p90 |", "|---|---|---|---|---|---|"]
    rows = [("measured, all legs", m), ("model at the same legs", res["model_half_bp_at_same_legs"])]
    rows += [(f"measured, {k} legs", v) for k, v in res["by_leg"].items()]
    rows += [(f"measured, mid {k}", v) for k, v in res["by_price"].items()]
    L += [f"| {lab} | {s['n']} | {fmt(s['median'])} | {fmt(s['mean'])} | {fmt(s['p75'])} | {fmt(s['p90'])} |"
          for lab, s in rows]
    L += ["", "## What the measured spread would do (trades with both legs measured)", "",
          "| version | trades measured / in run | mean net R as run | with measured spreads | Δ R |",
          "|---|---|---|---|---|"]
    L += [f"| {v} | {r['trades_measured']} / {r['trades_in_run']} | {fmt(r['mean_r_net_as_run'], 4)} | "
          f"{fmt(r['mean_r_net_measured_spread'], 4)} | {fmt(r['delta_r'], 4)} |" for v, r in res["versions"].items()]
    L += ["", "The quotes are the first seconds of each trade's entry and exit minute (the model's fill minute); the "
          "real fill second is unknown, and the spread right after a breakout can be wider than its minute median.",
          "Using a measured value needs a team decision written into the prereg (which statistic), then a rerun "
          "(= a new trial)."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--quotes", default="data/quotes_spread.csv")
    a = ap.parse_args(argv)
    tr = pd.read_csv(os.path.join(a.run, "trades.csv"), dtype={"ticker": str})
    if (tr["day"] >= B.HOLDOUT_START).any():
        sys.exit("HOLDOUT GUARD: this run has 2024+ trades; calibrate on DEV runs only.")
    q = pd.read_csv(a.quotes, dtype={"ticker": str})
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    kv = {v["id"]: v.get("knobs", {}) for v in cfg.get("variants", [])}
    res = calibrate(tr, q, kv)
    txt = report(res)
    open(os.path.join(a.run, "spread_calibration.md"), "w").write(txt)
    json.dump(res, open(os.path.join(a.run, "spread_calibration.json"), "w"), indent=2, default=float)
    print(txt)
    return res


if __name__ == "__main__":
    main()
