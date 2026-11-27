#!/usr/bin/env python3
"""
run_dev_a.py: the whole data runbook (docs/strategy-B-data-runbook.md) as ONE command, so the first real number
needs only the keys and time:

  probe -> assets -> daily -> opening -> fullday -> validate -> DEV-A run -> ticks -> DEV-A rerun with ticks
        -> gate_b.py shortlist -> draft docs/results-DEV-A.md

  export APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...
  python run_dev_a.py                      # ~3 h download + 2 backtests; every download stage resumes

- Stops at a probe NO-GO (writes the probe output into the draft and exits 2) and at a failed data gate.
- The DEV-A runs use: variants_b.json (frozen), --nulls 20, --stress, --replication (prereg sanity gate),
  --securities (ETB flags for the short shadow). Each run is logged in experiments/trials.csv (2 runs = 2 trials
  per version, as in the runbook).
- No holdout: the downloader and the backtests stop at --dl-end / --end, both refused if >= 2024-01-01. There is
  no --open-holdout here on purpose.
- The results file is a DRAFT built from the run's own files. It is not a verdict and nobody has read it yet.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

import b_mvb as B
import fetch_b_data as F
import gate_b as GB

HOLDOUT_START = "2024-01-01"
PCT = {"win_rate", "pct_pos_months", "max_dd_1x", "ambiguous_resolved_by_ticks"}
HERE = os.path.dirname(os.path.abspath(__file__))


def _tee(fn, *args, **kw):
    """Run fn, print its output live AND return it (for the draft)."""
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            buf.write(s)
            sys.__stdout__.write(s)
            return len(s)
    with contextlib.redirect_stdout(Tee()):
        res = fn(*args, **kw)
    return res, buf.getvalue()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:                                   # noqa: BLE001 - informational only
        return "unknown"


def _f(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if x != x else f"{x:.{nd}f}"


def write_draft(path: str, a, status: str, probe_txt: str = "", run1: str | None = None, run2: str | None = None,
                shortlist: dict | None = None, validate_txt: str = "") -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"# Results DEV-A ({a.start} → {a.end}): DRAFT, auto-generated", "",
         f"Generated {now} by `strategy_b/run_dev_a.py` at code `{_git_sha()}`. **Status: {status}.**",
         "Nobody has read these numbers yet. Every number is after commission + half spread (3 bp, ASSUMED until "
         "calibrated with SIP quotes) + slippage, unless labelled gross. Not a verdict: the verdict is the holdout.", ""]
    L += ["## 1. Probe (GO / NO-GO)", "", "```", probe_txt.strip() or "(probe skipped: --skip-probe)", "```", ""]
    if validate_txt:
        L += ["## 2. Data gate", "", "```", validate_txt.strip(), "```", ""]
    if run2 or run1:
        run = run2 or run1
        s = pd.read_csv(os.path.join(run, "summary.csv"))
        g = json.load(open(os.path.join(run, "gate.json")))
        L += ["## 3. Runs", "", f"- first run (pessimistic same-minute rule): `{run1}`",
              f"- tick-resolved rerun (read this one, finding R9): `{run2}`" if run2 else "- no tick rerun", ""]
        rp = g.get("replication") or {}
        L += ["## 4. Sanity gate (B0-replication, long + short, commission only; code check)", "",
              f"Sharpe {_f(rp.get('sharpe_1x'), 2)} on {rp.get('trades', '–')} trades → "
              f"**{'PASS' if rp.get('pass') else 'FAIL: suspect the data or the code before reading on'}** "
              "(prereg: ≥ 2.0).", ""]
        b = s.iloc[0]
        L += ["## 5. Base (B0-primary: long-only, 1×, full costs)", "", "| metric | value |", "|---|---|"]
        for k, lab in [("trades", "trades"), ("mean_R_net", "mean net R"), ("t_net", "t (day-clustered)"),
                       ("mean_R_gross", "mean gross R"), ("cost_R", "cost R/trade"), ("win_rate", "win% (R > 0)"),
                       ("profit_factor", "profit factor"), ("pct_pos_months", "% positive months"),
                       ("quarters_pos", "quarters positive"), ("max_dd_R", "max DD (R)"), ("sharpe_1x", "Sharpe 1×"),
                       ("max_dd_1x", "max DD 1×"), ("mean_R_net_2x_slip", "mean net R at 2× slippage"),
                       ("mean_R_net_ohlc_path", "mean net R if same-bar = OHLC path"),
                       ("ambiguous_resolved_by_ticks", "ambiguous minutes resolved by ticks"),
                       ("mean_R_net_ex_top5_tickers", "mean net R without the top-5 tickers"), ("dsr", "DSR")]:
            if k in s:
                v = b.get(k)
                txt = (v if isinstance(v, str) else str(int(v)) if k == "trades"
                       else f"{_f(100 * float(v), 1)}%" if k in PCT else _f(v))
                L.append(f"| {lab} | {txt} |")
        L += ["", "## 6. Versions vs base (paired day-level test, Newey-West 5 lags, Holm across 14)", "",
              "| version | trades | mean net R | ΔR/day | t NW | Holm p | DSR |", "|---|---|---|---|---|---|---|"]
        for _, r in s.iloc[1:].iterrows():
            L.append(f"| {r['id']} | {int(r.get('trades', 0))} | {_f(r.get('mean_R_net'))} | "
                     f"{_f(r.get('dR_per_day_vs_base'), 4)} | {_f(r.get('t_nw_vs_base'), 2)} | "
                     f"{_f(r.get('p_holm'), 4)} | {_f(r.get('dsr'))} |")
        L += ["", f"Hansen SPA p = {_f(g.get('spa_p'))} (best `{g.get('spa_best')}`); PBO = {_f(g.get('pbo'))}; "
              f"trials in the log N = {g.get('n_trials')}.", ""]
        for n in g.get("nulls") or []:
            L.append(f"- Null {n['null']} ({n.get('version', 'base')}): p95 {_f(n['null_p95'])} vs real "
                     f"{_f(n['real'])}; share of nulls ≥ real {_f(n['share_null_ge_real'], 2)}")
        be = g.get("beta") or {}
        if "beta" in be:
            L += ["", "## 7. Market exposure (plan v3 §5)", "",
                  f"beta {_f(be['beta'])} (NW t {_f(be.get('beta_t_nw'), 2)}); mean base R/day on SPY-down days "
                  f"{_f(be.get('R_day_spy_down'))} ({be.get('n_spy_down')} days); on F8 red-signal days "
                  f"{_f(be.get('R_day_f8_red'))} ({be.get('n_f8_red')} days)."]
    if shortlist is not None:
        L += ["", "## 8. Mechanical shortlist (gate_b.py, prereg rules)", "",
              f"**Shortlist for DEV-B: {shortlist['shortlist'] or 'EMPTY'}**", ""]
        L += [f"- {n}" for n in shortlist["notes"]]
    L += ["", "## Not verified / read with care", "",
          "- Survivorship: see the probe's delisted-names line; Alpaca's asset list is today's view.",
          "- Ticker reuse can mix two companies; the ETF filter is a name heuristic.",
          "- Half spread is an assumption (3 bp) until SIP quotes are measured.",
          "- SPA is slightly liberal on short samples (measured 6% at 5% with 300 days).",
          "- Next: sign the prereg (open choices), then DEV-B with `--stress`, then `gate_b.py confirm`."]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "w").write("\n".join(L) + "\n")
    print(f"\ndraft: {path}")
    return path


def main(argv=None, client=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--trial-log", default="experiments/trials.csv")
    ap.add_argument("--variants", default=os.path.join(HERE, "variants_b.json"))
    ap.add_argument("--dl-start", default="2015-11-01", help="first day downloaded (history for ATR/RVOL)")
    ap.add_argument("--dl-end", default="2023-12-31", help="last day downloaded (DEV-A + DEV-B; never 2024+)")
    ap.add_argument("--start", default="2016-01-01", help="DEV-A start")
    ap.add_argument("--end", default="2020-12-31", help="DEV-A end")
    ap.add_argument("--nulls", type=int, default=20)
    ap.add_argument("--gate-reps", type=int, default=1000)
    ap.add_argument("--breakeven", action="store_true", help="also breakeven slippage (about 5x the runtime)")
    ap.add_argument("--skip-probe", action="store_true", help="skip the probe (stated in the draft)")
    ap.add_argument("--results", default=os.path.join(HERE, "..", "docs", "results-DEV-A.md"))
    a = ap.parse_args(argv)
    for name in ("dl_end", "end"):
        if getattr(a, name) >= HOLDOUT_START:
            sys.exit(f"--{name.replace('_', '-')} reaches the holdout ({HOLDOUT_START}+). This tool never touches it.")
    if client is None and not (os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY")):
        sys.exit("set APCA_API_KEY_ID and APCA_API_SECRET_KEY first (free Alpaca account; never commit them)")
    client = client or F.Client()
    dl = ["--data", a.data]

    probe_txt = ""
    if not a.skip_probe:
        ok, probe_txt = _tee(F.main, ["probe"] + dl, client=client)
        if not ok:
            write_draft(a.results, a, "NO-GO at the probe: nothing downloaded", probe_txt)
            sys.exit(2)
    F.main(["assets"] + dl, client=client)
    F.main(["daily"] + dl + ["--start", a.dl_start, "--end", a.dl_end], client=client)
    F.main(["opening"] + dl + ["--start", a.dl_start, "--end", a.dl_end], client=client)
    F.main(["fullday"] + dl + ["--start", a.start, "--end", a.dl_end], client=client)
    try:
        _, val_txt = _tee(F.main, ["validate"] + dl)
    except SystemExit as e:
        write_draft(a.results, a, f"STOPPED at the data gate ({e})", probe_txt)
        raise
    bt = ["--daily", os.path.join(a.data, "daily.csv"), "--minutes", os.path.join(a.data, "minutes"),
          "--variants", a.variants, "--start", a.start, "--end", a.end, "--nulls", str(a.nulls), "--stress",
          "--replication", "--gate-reps", str(a.gate_reps), "--out", a.runs, "--trial-log", a.trial_log]
    sec = os.path.join(a.data, "securities.csv")
    if os.path.exists(sec):
        bt += ["--securities", sec]
    if a.breakeven:
        bt.append("--breakeven")
    run1 = B.main(bt)
    F.main(["ticks"] + dl + ["--trades", os.path.join(run1, "trades.csv")], client=client)
    run2 = B.main(bt + ["--ticks", os.path.join(a.data, "ticks.csv")])
    B.TICKS.clear()
    sl = GB.shortlist(GB.load_run(run2))
    GB.write(sl, run2)
    status = "complete (download, data gate, DEV-A, tick rerun, shortlist)" + (" — probe SKIPPED" if a.skip_probe else "")
    return write_draft(a.results, a, status, probe_txt, run1, run2, sl, val_txt)


if __name__ == "__main__":
    main()
