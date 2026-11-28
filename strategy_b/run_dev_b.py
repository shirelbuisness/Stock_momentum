#!/usr/bin/env python3
"""
run_dev_b.py: DEV-B (2021-2023, "confirm") as ONE command, after DEV-A and after the prereg is signed:

  DEV-A run -> its mechanical shortlist (gate_b.shortlist, recomputed here, never typed in by hand)
     -> fullday + validate for DEV-B (resumable; a no-op when run_dev_a.py already downloaded through 2023)
     -> DEV-B run (base + the shortlisted versions, --stress, --replication, nulls)
     -> ticks for its trades -> tick-resolved DEV-B rerun (finding R9)
     -> gate_b.py confirm -> draft docs/results-DEV-B.md

  export APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...
  python run_dev_b.py --dev-a runs/<DEV-A tick-resolved run> --prereg-signed

- Refuses without --prereg-signed (prereg: signed by both partners before the first real DEV run; DEV-B is a
  confirmation, so its rules must be fixed before it is looked at). The flag is a statement, not a check.
- Refuses an empty shortlist without running anything (no trial is spent): plan v3 says nothing beats base, so the
  holdout runs the base alone.
- By default the DEV-B run holds ONLY base + the shortlist (the protocol confirms the shortlisted versions; every
  extra version is an extra trial that lowers every DSR). --all-versions runs all 15 frozen versions so that PBO
  on DEV-B has the full search to judge (a team choice; see the night log).
- The versions are copied VERBATIM from the frozen variants_b.json into <runs>/variants_devb_<stamp>.json; the
  frozen file is never written.
- No holdout: --start must be >= 2021-01-01 (no overlap with DEV-A) and --end < 2024-01-01. No --open-holdout.
- Each run is logged in experiments/trials.csv by b_mvb.py (2 runs = 2 trials per version, as for DEV-A).
- The results file is a DRAFT built from the run's own files. It is not a verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

import b_mvb as B
import fetch_b_data as F
import gate_b as GB
import run_dev_a as RA

HOLDOUT_START = "2024-01-01"
HERE = os.path.dirname(os.path.abspath(__file__))


def subset_variants(frozen_path: str, ids: list[str], out_path: str) -> list[str]:
    """Base + the given ids, copied verbatim (same dicts, same order as the frozen file)."""
    spec = json.load(open(frozen_path))
    keep = {spec[0]["id"], *ids}
    missing = set(ids) - {v["id"] for v in spec}
    if missing:
        sys.exit(f"shortlisted ids not in {frozen_path}: {sorted(missing)}")
    sub = [v for v in spec if v["id"] in keep]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(sub, open(out_path, "w"), indent=2)
    return [v["id"] for v in sub]


def write_draft(path: str, a, status: str, shortlist: dict, run1: str | None = None, run2: str | None = None,
                conf: dict | None = None, validate_txt: str = "") -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"# Results DEV-B ({a.start} → {a.end}): DRAFT, auto-generated", "",
         f"Generated {now} by `strategy_b/run_dev_b.py` at code `{RA._git_sha()}`. **Status: {status}.**",
         "Nobody has read these numbers yet. Every number is after commission + half spread (3 bp, ASSUMED until "
         "calibrated with SIP quotes) + slippage, unless labelled gross. Not a verdict: the verdict is the holdout.", "",
         f"DEV-A run: `{a.dev_a}`", "", f"**Shortlist from DEV-A (gate_b, mechanical): "
         f"{shortlist['shortlist'] or 'EMPTY'}**", ""]
    L += [f"- {n}" for n in shortlist["notes"]]
    if validate_txt:
        L += ["", "## 1. Data gate", "", "```", validate_txt.strip(), "```"]
    if run2 or run1:
        run = run2 or run1
        s = pd.read_csv(os.path.join(run, "summary.csv"))
        g = json.load(open(os.path.join(run, "gate.json")))
        rp = g.get("replication") or {}
        L += ["", "## 2. Runs", "", f"- first run (pessimistic same-minute rule): `{run1}`",
              f"- tick-resolved rerun (the one the gate reads, R9): `{run2}`" if run2 else "- no tick rerun",
              f"- versions in the run: {', '.join(s['id'])}", "",
              "## 3. Sanity gate (B0-replication, long + short, commission only; code check)", "",
              f"Sharpe {RA._f(rp.get('sharpe_1x'), 2)} on {rp.get('trades', '–')} trades → "
              f"**{'PASS' if rp.get('pass') else 'FAIL: suspect the data or the code before reading on'}** "
              "(prereg: ≥ 2.0).", "",
              "## 4. Versions (DEV-B)", "",
              "| version | trades | mean net R | t | net R at 2× slip | ΔR/day vs base | t NW | DSR |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            L.append(f"| {r['id']} | {int(r.get('trades', 0))} | {RA._f(r.get('mean_R_net'))} | "
                     f"{RA._f(r.get('t_net'), 2)} | {RA._f(r.get('mean_R_net_2x_slip'))} | "
                     f"{RA._f(r.get('dR_per_day_vs_base'), 4)} | {RA._f(r.get('t_nw_vs_base'), 2)} | "
                     f"{RA._f(r.get('dsr'))} |")
        L += ["", f"PBO on DEV-B = {RA._f(g.get('pbo'))} (meaningful only with --all-versions); "
              f"trials in the log N = {g.get('n_trials')}."]
    if conf is not None:
        L += ["", "## 5. Mechanical confirmation (gate_b.py confirm, prereg rules)", "",
              f"**Promoted (stage 2 / holdout candidates): {conf['promoted'] or 'NONE'}**", ""]
        L += [f"- {n}" for n in conf["notes"] if n not in shortlist["notes"]]
        for r in conf["rows"]:
            L.append(f"### {r['id']}: {'PASS' if r['passed'] else 'FAIL'}")
            L += [f"- [{'x' if ok else ' '}] {c}" for c, ok in r["checks"].items()]
    L += ["", "## Not verified / read with care", "",
          "- Half spread is an assumption (3 bp) until SIP quotes are measured (`fetch_b_data.py quotes` + "
          "`spread_b.py`).",
          "- Survivorship and ticker reuse: see the DEV-A draft and `data/identity_warnings.csv`.",
          "- Next: stage 2 on the promoted versions (≤ 2 knobs, +1 trial each), then the holdout ONCE "
          "(base + at most 2), `gate_b.py holdout`."]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "w").write("\n".join(L) + "\n")
    print(f"\ndraft: {path}")
    return path


def main(argv=None, client=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-a", required=True, help="the DEV-A run the shortlist comes from (tick-resolved)")
    ap.add_argument("--prereg-signed", action="store_true",
                    help="REQUIRED: states that docs/preregistration-B-draft.md is signed by both partners")
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--trial-log", default="experiments/trials.csv")
    ap.add_argument("--variants", default=os.path.join(HERE, "variants_b.json"))
    ap.add_argument("--start", default="2021-01-01", help="DEV-B start")
    ap.add_argument("--end", default="2023-12-31", help="DEV-B end")
    ap.add_argument("--all-versions", action="store_true", help="run all 15 frozen versions (PBO on DEV-B)")
    ap.add_argument("--nulls", type=int, default=20)
    ap.add_argument("--gate-reps", type=int, default=1000)
    ap.add_argument("--results", default=os.path.join(HERE, "..", "docs", "results-DEV-B.md"))
    a = ap.parse_args(argv)
    if a.end >= HOLDOUT_START:
        sys.exit(f"--end reaches the holdout ({HOLDOUT_START}+). This tool never touches it.")
    if a.start < GB.DEV_B[0]:
        sys.exit(f"--start {a.start} overlaps DEV-A: DEV-B starts {GB.DEV_B[0]}")
    if not a.prereg_signed:
        sys.exit("DEV-B confirms pre-registered rules: sign docs/preregistration-B-draft.md first, then pass "
                 "--prereg-signed")
    A = GB.load_run(a.dev_a)
    sl = GB.shortlist(A)
    if not sl["shortlist"]:
        write_draft(a.results, a, "NOT RUN: the DEV-A shortlist is empty (nothing to confirm; holdout = base only)",
                    sl)
        return None
    if client is None and not (os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY")):
        sys.exit("set APCA_API_KEY_ID and APCA_API_SECRET_KEY first (free Alpaca account; never commit them)")
    client = client or F.Client()
    dl = ["--data", a.data]
    F.main(["fullday"] + dl + ["--start", a.start, "--end", a.end], client=client)
    try:
        _, val_txt = RA._tee(F.main, ["validate"] + dl)
    except SystemExit as e:
        write_draft(a.results, a, f"STOPPED at the data gate ({e})", sl)
        raise
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if a.all_versions:
        vpath = a.variants
    else:
        vpath = os.path.join(a.runs, f"variants_devb_{stamp}.json")
        subset_variants(a.variants, sl["shortlist"], vpath)
    bt = ["--daily", os.path.join(a.data, "daily.csv"), "--minutes", os.path.join(a.data, "minutes"),
          "--variants", vpath, "--start", a.start, "--end", a.end, "--nulls", str(a.nulls), "--stress",
          "--replication", "--gate-reps", str(a.gate_reps), "--out", a.runs, "--trial-log", a.trial_log]
    sec = os.path.join(a.data, "securities.csv")
    if os.path.exists(sec):
        bt += ["--securities", sec]
    run1 = B.main(bt)
    F.main(["ticks"] + dl + ["--trades", os.path.join(run1, "trades.csv")], client=client)
    run2 = B.main(bt + ["--ticks", os.path.join(a.data, "ticks.csv")])
    B.TICKS.clear()
    conf = GB.confirm(A, GB.load_run(run2))
    GB.write(conf, run2)
    return write_draft(a.results, a, "complete (data gate, DEV-B, tick rerun, confirm)", sl, run1, run2, conf, val_txt)


if __name__ == "__main__":
    main()
