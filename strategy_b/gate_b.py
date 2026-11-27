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

  python gate_b.py holdout --run runs/<holdout run> --net-r-bar 0.05|0.08 --breakeven-ticks-bar 1|2

HOLDOUT: the prereg pass gate (items 1-7) per version of the ONE holdout run, plus Holm across the promoted
  versions (with 2, the first needs p < 0.025), the "fragile" label (passes at base k, not at 2k) and the
  healthy-shape check (PF 1.2-2.5; PF > 2.5 or zero losing months = treat as a bug until explained).
  The two values the prereg leaves open ([ ] at signing) are REQUIRED arguments with no default, so this tool
  cannot pick them. It reads a finished run; it never runs a backtest and never opens the holdout.
  --rehearse accepts a DEV run (a dress rehearsal, labelled as such in the output).
  The holdout run needs: --stress (2k), --breakeven (item 7), --nulls > 0 and --nulls-all-versions (item 3).
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


def load_run(path: str, holdout_ok: bool = False) -> dict:
    cfg = json.load(open(os.path.join(path, "config.json")))
    summ = pd.read_csv(os.path.join(path, "summary.csv"))
    gate = json.load(open(os.path.join(path, "gate.json"))) if os.path.exists(os.path.join(path, "gate.json")) else {}
    args = cfg["args"]
    if not holdout_ok and (args["end"] >= "2024-01-01" or args.get("open_holdout")):
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


HOLDOUT = ("2024-01-01", "2025-12-31")
PASS_T, PASS_SHARPE, PASS_DD_1X, PASS_TRADES = 2.0, 1.0, 0.25, 1000
PASS_QUARTERS, PASS_MONTHS, HOLM_HOLDOUT = 6, 0.60, 0.05
PF_HEALTHY = (1.2, 2.5)


def holdout(h: dict, net_r_bar: float, be_ticks_bar: float, rehearse: bool = False) -> dict:
    """Prereg rev 3 pass gate, item by item, for every version in one holdout run. Fail closed."""
    notes = []
    is_holdout = h["args"]["start"] >= HOLDOUT[0] and bool(h["args"].get("open_holdout"))
    if not is_holdout:
        if not rehearse:
            sys.exit(f"{h['path']}: not a holdout run (start {h['args']['start']}, open_holdout "
                     f"{h['args'].get('open_holdout')}). Use --rehearse for a dress rehearsal on a DEV run.")
        notes.append("REHEARSAL on a non-holdout run: these verdicts are NOT the pass gate.")
    s = h["summ"]
    nulls = h["gate"].get("nulls") or []
    rows = []
    for vid in s.index:
        r = s.loc[vid]
        q = str(r.get("quarters_pos", ""))
        qp, qn = (int(q.split("/")[0]), int(q.split("/")[1])) if "/" in q else (-1, 0)
        nv = {n["null"]: n for n in nulls if n.get("version", h["base_id"]) == vid}
        mean, t = _f(r.get("mean_R_net")), _f(r.get("t_net"))
        checks = {
            f"1. mean net R ≥ {net_r_bar:g} with t ≥ {PASS_T:g} (mean {mean:.3f}, t {t:.2f})":
                mean >= net_r_bar and t >= PASS_T,
            f"2. Sharpe 1× ≥ {PASS_SHARPE:g} ({_f(r.get('sharpe_1x')):.2f})": _f(r.get("sharpe_1x")) >= PASS_SHARPE,
        }
        for mode in ("random_entry", "random_names"):
            p95 = _f(nv.get(mode, {}).get("null_p95"))
            checks[f"3. beats the {mode} null p95 ({p95:.3f})" if mode in nv
                   else f"3. beats the {mode} null p95 (MISSING: run with --nulls N --nulls-all-versions)"] = \
                mode in nv and mean > p95
        checks.update({
            f"4. DSR ≥ {DSR_MIN} ({_f(r.get('dsr')):.3f})": _f(r.get("dsr")) >= DSR_MIN,
            f"5a. positive at 2× slippage ({_f(r.get('mean_R_net_2x_slip')):.3f})": _f(r.get("mean_R_net_2x_slip")) > 0,
            f"5b. max DD 1× < {PASS_DD_1X:.0%} ({_f(r.get('max_dd_1x')):.1%})": _f(r.get("max_dd_1x")) < PASS_DD_1X,
            f"6a. ≥ {PASS_TRADES} trades ({_f(r.get('trades')):.0f})": _f(r.get("trades")) >= PASS_TRADES,
            f"6b. ≥ {PASS_QUARTERS} of 8 quarters positive ({q})": qp >= PASS_QUARTERS,
            f"6c. ≥ {PASS_MONTHS:.0%} of months positive ({_f(r.get('pct_pos_months')):.0%})":
                _f(r.get("pct_pos_months")) >= PASS_MONTHS,
            f"6d. positive without the top-5 tickers ({_f(r.get('mean_R_net_ex_top5_tickers')):.3f})":
                _f(r.get("mean_R_net_ex_top5_tickers")) > 0,
            f"7. breakeven slippage ≥ {be_ticks_bar:g} ticks/side ({_f(r.get('breakeven_ticks')):.2f})":
                _f(r.get("breakeven_ticks")) >= be_ticks_bar,
        })
        if vid != h["base_id"]:
            ph = _f(r.get("p_holm"))
            checks[f"Holm across the promoted versions: p < {HOLM_HOLDOUT} ({ph:.4f})"] = ph < HOLM_HOLDOUT
        if qn and qn != 8:
            notes.append(f"{vid}: {qn} quarters in the run, the prereg counts 8 (6 of 8 applied as ≥ 6 positive)")
        passed = all(checks.values())
        at_k = mean >= net_r_bar and t >= PASS_T
        label = "ALIVE" if passed else ("FRAGILE (passes at base k, not at 2k)" if at_k and not
                                        _f(r.get("mean_R_net_2x_slip")) > 0 else "FAIL")
        pf, ppm = _f(r.get("profit_factor")), _f(r.get("pct_pos_months"))
        shape = []
        if not (PF_HEALTHY[0] <= pf <= PF_HEALTHY[1]):
            shape.append(f"PF {pf:.2f} outside {PF_HEALTHY[0]}-{PF_HEALTHY[1]}"
                         + (": treat as a bug until explained" if pf > PF_HEALTHY[1] else ""))
        if ppm == 1.0:
            shape.append("zero losing months: treat as a bug until explained")
        rows.append(dict(id=vid, passed=passed, label=label, checks=checks, shape=shape))
    return dict(step="holdout", run=h["path"], rehearsal=not is_holdout, net_r_bar=net_r_bar,
                be_ticks_bar=be_ticks_bar, passed=[r["id"] for r in rows if r["passed"]], rows=rows, notes=notes)


def write(res: dict, outdir: str) -> str:
    L = [f"# Gate B: {res['step']} (mechanical, pre-registered rules)", ""]
    L += [f"- {n}" for n in res["notes"]] + ([""] if res["notes"] else [])
    if res["step"] == "shortlist":
        L += [f"**Shortlist: {res['shortlist'] or 'EMPTY'}**", "", "| version | Holm p | ΔR/day | eligible | why not |",
              "|---|---|---|---|---|"]
        for r in res["rows"]:
            L.append(f"| {r['id']} | {r['p_holm']:.4g} | {r['dR_day']:.4g} | {'yes' if r['eligible'] else 'no'} | "
                     f"{r['why']} |")
    elif res["step"] == "holdout":
        L += [f"Open prereg choices as given: net-R bar {res['net_r_bar']:g} R, breakeven bar "
              f"{res['be_ticks_bar']:g} ticks/side.", "", f"**Passed the gate: {res['passed'] or 'NONE'}**", ""]
        for r in res["rows"]:
            L.append(f"### {r['id']}: {r['label']}")
            L += [f"- [{'x' if ok else ' '}] {c}" for c, ok in r["checks"].items()]
            L += [f"- healthy-shape warning: {w}" for w in r["shape"]] + [""]
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
    ap.add_argument("step", choices=["shortlist", "confirm", "holdout"])
    ap.add_argument("--dev-a")
    ap.add_argument("--dev-b")
    ap.add_argument("--run", help="holdout: the ONE holdout run (or a DEV run with --rehearse)")
    ap.add_argument("--net-r-bar", type=float, choices=[0.05, 0.08],
                    help="holdout, REQUIRED: the prereg's net-R bar, picked by the team at signing")
    ap.add_argument("--breakeven-ticks-bar", type=float, choices=[1.0, 2.0],
                    help="holdout, REQUIRED: the prereg's breakeven-slippage bar, picked by the team at signing")
    ap.add_argument("--rehearse", action="store_true", help="holdout: accept a DEV run as a dress rehearsal")
    a = ap.parse_args(argv)
    if a.step == "holdout":
        if not a.run or a.net_r_bar is None or a.breakeven_ticks_bar is None:
            sys.exit("holdout needs --run, --net-r-bar and --breakeven-ticks-bar (the open prereg choices have no "
                     "default on purpose)")
        res = holdout(load_run(a.run, holdout_ok=True), a.net_r_bar, a.breakeven_ticks_bar, a.rehearse)
        p = write(res, a.run)
        print(open(p).read())
        return res
    if not a.dev_a:
        sys.exit("--dev-a required")
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
