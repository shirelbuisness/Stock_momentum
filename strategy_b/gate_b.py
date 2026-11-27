#!/usr/bin/env python3
"""
gate_b.py: apply the pre-registered DEV-A -> DEV-B protocol (prereg rev 3 + plan v3 §4) to two b_mvb.py runs,
mechanically, so nobody "reads" a shortlist into the numbers.

  python gate_b.py shortlist --dev-a runs/<DEV-A run>
  python gate_b.py confirm   --dev-a runs/<DEV-A run> --dev-b runs/<DEV-B run>

SHORTLIST (DEV-A, 2016-2020): versions with Holm-adjusted p < 0.10 on the paired day-level test AND a positive
  effect (ΔR/day > 0), at most 4 (smallest p first). Plan v3: if Hansen SPA p > 0.10, nothing beats base -> empty.
  Finding R9: a stop version (stop_atr / stop_mode) may only be shortlisted from a tick-resolved run (--ticks).
CONFIRM (DEV-B, 2021-2023), each shortlisted version needs ALL of:
  paired day-level t >= 2 (Newey-West); same sign as DEV-A and >= half its ΔR/day; DSR >= 0.95;
  mean net R at 2x slippage > 0 (the DEV-B run needs --stress); filters (fewer trades than base) also need
  random-skip p < 0.05 (protocol step 3).
Anything missing counts as FAIL (fail closed). Writes gate_<step>.md / .json next to the later run.
No holdout: both runs must end before 2024-01-01.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

DEV_A = ("2016-01-01", "2020-12-31")
DEV_B = ("2021-01-01", "2023-12-31")
STOP_KNOBS = {"stop_atr", "stop_mode"}
HOLM_MAX, SPA_MAX, MAX_SHORTLIST = 0.10, 0.10, 4
T_MIN, DSR_MIN, RS_P_MAX = 2.0, 0.95, 0.05


def load_run(path: str) -> dict:
    cfg = json.load(open(os.path.join(path, "config.json")))
    summ = pd.read_csv(os.path.join(path, "summary.csv"))
    gate = json.load(open(os.path.join(path, "gate.json"))) if os.path.exists(os.path.join(path, "gate.json")) else {}
    args = cfg["args"]
    if args["end"] >= "2024-01-01" or args.get("open_holdout"):
        sys.exit(f"{path}: a holdout run. gate_b.py only handles DEV-A / DEV-B.")
    knobs = {v["id"]: v["knobs"] for v in cfg["variants"]}
    base = cfg["variants"][0]["knobs"]
    changed = {vid: {k for k, x in kn.items() if base.get(k) != x} for vid, kn in knobs.items()}
    return dict(path=path, args=args, summ=summ.set_index("id"), gate=gate, changed=changed,
                base_id=cfg["variants"][0]["id"])


def _f(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return x


def shortlist(a: dict) -> dict:
    s, notes = a["summ"], []
    if not (DEV_A[0] <= a["args"]["start"] and a["args"]["end"] <= DEV_A[1]):
        notes.append(f"WARNING: run period {a['args']['start']}..{a['args']['end']} is not inside DEV-A {DEV_A}")
    rows = []
    for vid in s.index[1:]:
        p, d = _f(s.loc[vid].get("p_holm")), _f(s.loc[vid].get("dR_per_day_vs_base"))
        why = []
        if not (p < HOLM_MAX):
            why.append(f"Holm p {p:.3g} ≥ {HOLM_MAX}" if np.isfinite(p) else "no Holm p")
        if not (d > 0):
            why.append(f"ΔR/day {d:.3g} ≤ 0" if np.isfinite(d) else "no ΔR/day")
        if a["changed"].get(vid, set()) & STOP_KNOBS and not a["args"].get("ticks"):
            why.append("stop version: needs a tick-resolved run (finding R9)")
        rows.append(dict(id=vid, p_holm=p, dR_day=d, eligible=not why, why="; ".join(why)))
    el = sorted([r for r in rows if r["eligible"]], key=lambda r: r["p_holm"])
    spa = _f(a["gate"].get("spa_p"))
    if not np.isfinite(spa):
        notes.append("SPA p missing (run made before gate.json existed?) -> SPA rule not applied")
    elif spa > SPA_MAX:
        notes.append(f"Hansen SPA p = {spa:.3f} > {SPA_MAX}: nothing beats base -> shortlist EMPTY (plan v3 §4)")
        for r in el:
            r["why"] = "SPA: nothing beats base"
        el = []
    if len(el) > MAX_SHORTLIST:
        for r in el[MAX_SHORTLIST:]:
            r["why"] = f"beyond the {MAX_SHORTLIST} smallest Holm p"
        el = el[:MAX_SHORTLIST]
    return dict(step="shortlist", dev_a=a["path"], shortlist=[r["id"] for r in el], rows=rows, notes=notes,
                spa_p=spa, pbo=_f(a["gate"].get("pbo")))


def confirm(a: dict, b: dict) -> dict:
    sl = shortlist(a)
    notes = list(sl["notes"])
    if not (DEV_B[0] <= b["args"]["start"] and b["args"]["end"] <= DEV_B[1]):
        notes.append(f"WARNING: run period {b['args']['start']}..{b['args']['end']} is not inside DEV-B {DEV_B}")
    sb = b["summ"]
    base_trades = _f(sb.iloc[0].get("trades"))
    pbo = _f(b["gate"].get("pbo"))
    if np.isfinite(pbo) and pbo > 0.3:
        notes.append(f"PBO = {pbo:.2f} > 0.3 on DEV-B: the search as a whole looks overfit (plan v3 §4)")
    rows = []
    for vid in sl["shortlist"]:
        if vid not in sb.index:
            rows.append(dict(id=vid, passed=False, checks={"present in DEV-B run": False}))
            continue
        r, da = sb.loc[vid], next(x["dR_day"] for x in sl["rows"] if x["id"] == vid)
        db, t = _f(r.get("dR_per_day_vs_base")), _f(r.get("t_nw_vs_base"))
        checks = {
            f"paired day t ≥ {T_MIN:g} (t = {t:.2f})": t >= T_MIN,
            f"same sign as DEV-A and ≥ ½ size (DEV-A {da:.4f}, DEV-B {db:.4f})": np.sign(db) == np.sign(da)
            and abs(db) >= 0.5 * abs(da),
            f"DSR ≥ {DSR_MIN} (DSR = {_f(r.get('dsr')):.3f})": _f(r.get("dsr")) >= DSR_MIN,
            f"positive at 2× slippage ({_f(r.get('mean_R_net_2x_slip')):.3f} R)": _f(r.get("mean_R_net_2x_slip")) > 0,
        }
        if _f(r.get("trades")) < base_trades:
            checks[f"beats the random-skip null (p = {_f(r.get('rs_p_trades')):.3f} < {RS_P_MAX})"] = \
                _f(r.get("rs_p_trades")) < RS_P_MAX
        if b["changed"].get(vid, set()) & STOP_KNOBS and not b["args"].get("ticks"):
            checks["stop version decided on a tick-resolved run (R9)"] = False
        rows.append(dict(id=vid, passed=all(checks.values()), checks=checks))
    return dict(step="confirm", dev_a=a["path"], dev_b=b["path"], shortlist=sl["shortlist"],
                promoted=[r["id"] for r in rows if r["passed"]], rows=rows, notes=notes, pbo_dev_b=pbo)


def write(res: dict, outdir: str) -> str:
    L = [f"# Gate B: {res['step']} (mechanical, pre-registered rules)", ""]
    L += [f"- {n}" for n in res["notes"]] + ([""] if res["notes"] else [])
    if res["step"] == "shortlist":
        L += [f"**Shortlist: {res['shortlist'] or 'EMPTY'}**", "", "| version | Holm p | ΔR/day | eligible | why not |",
              "|---|---|---|---|---|"]
        for r in res["rows"]:
            L.append(f"| {r['id']} | {r['p_holm']:.4g} | {r['dR_day']:.4g} | {'yes' if r['eligible'] else 'no'} | "
                     f"{r['why']} |")
    else:
        L += [f"Shortlist from DEV-A: {res['shortlist'] or 'EMPTY'}", "",
              f"**Promoted to stage 2 / holdout candidates: {res['promoted'] or 'NONE'}**", ""]
        for r in res["rows"]:
            L.append(f"### {r['id']}: {'PASS' if r['passed'] else 'FAIL'}")
            L += [f"- [{'x' if ok else ' '}] {c}" for c, ok in r["checks"].items()] + [""]
    p = os.path.join(outdir, f"gate_{res['step']}.md")
    open(p, "w").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(outdir, f"gate_{res['step']}.json"), "w"), indent=2, default=str)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["shortlist", "confirm"])
    ap.add_argument("--dev-a", required=True)
    ap.add_argument("--dev-b")
    a = ap.parse_args(argv)
    A = load_run(a.dev_a)
    if a.step == "shortlist":
        res, out = shortlist(A), a.dev_a
    else:
        if not a.dev_b:
            sys.exit("--dev-b required for confirm")
        res, out = confirm(A, load_run(a.dev_b)), a.dev_b
    p = write(res, out)
    print(open(p).read())
    return res


if __name__ == "__main__":
    main()
