"""Speed work must not change a single number: the optimized b_mvb.py and the frozen pre-optimization copy
(reference/b_mvb_ref.py) run the same synthetic market with every pre-registered version, nulls, the 2x-slippage
and OHLC-path stress runs, breakeven grids and real-print tick resolution; all outputs must be identical."""
import json
import os
import sys

import numpy as np
import pandas as pd

import b_mvb as B
from synth import make_market

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference"))
import b_mvb_ref as R  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _variants(tmp_path):
    v = json.load(open(os.path.join(HERE, "variants_b.json")))
    v += [{"id": "both_sides", "knobs": {"sides": "both"}},
          {"id": "be_trail_partial", "knobs": {"sides": "both", "stop_atr": 0.25, "breakeven_r": 1.0,
                                               "breakeven_after": "10:00", "trail_after_r": 3.0, "partial_r": 2.0}}]
    for x in v:
        x["knobs"]["min_avg_vol"] = 0
    p = tmp_path / "v.json"
    p.write_text(json.dumps(v))
    return str(p)


def _ticks(tmp_path, minutes_dir, trades_csv):
    """Synthetic prints for every ambiguous entry minute: o -> l -> h -> c (green) or o -> h -> l -> c (red)."""
    tr = pd.read_csv(trades_csv)
    amb = tr[tr["filled"] & (tr["same_bar_source"].fillna("") != "")][["day", "ticker", "entry_t"]].drop_duplicates()
    rows = []
    for (day, tk, t) in amb.itertuples(index=False):
        m = pd.read_csv(os.path.join(minutes_dir, f"minutes_{day}.csv"))
        b = m[(m["ticker"] == tk) & (m["time"] == f"{int(t) // 60:02d}:{int(t) % 60:02d}")].iloc[0]
        path = [b.o, b.l, b.h, b.c] if b.c >= b.o else [b.o, b.h, b.l, b.c]
        rows += [dict(day=day, ticker=tk, minute=int(t), seq=i, price=p) for i, p in enumerate(path)]
    p = tmp_path / "ticks.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p), len(amb)


def _run(mod, tmp_path, tag, extra=("--breakeven",)):
    if mod is B:
        extra = (*extra, "--no-short-shadow")           # added after the freeze (task f); not in the reference
    return mod.main(["--daily", str(tmp_path / "daily.csv"), "--minutes", str(tmp_path / "minutes"),
                     "--start", "2019-11-11", "--end", "2019-12-31", "--nulls", "2", "--stress",
                     "--gate-reps", "50", "--variants", _variants(tmp_path), "--out", str(tmp_path / f"runs_{tag}"),
                     "--trial-log", str(tmp_path / f"trials_{tag}.csv"), *extra])


def _same(out_a, out_b):
    ta, tb = pd.read_csv(f"{out_a}/trades.csv"), pd.read_csv(f"{out_b}/trades.csv")
    ta = ta[tb.columns]                                   # columns added after the freeze (ssr, etb) are extra
    pd.testing.assert_frame_equal(ta, tb, check_exact=False, rtol=1e-12, atol=1e-12)
    sa, sb = pd.read_csv(f"{out_a}/summary.csv"), pd.read_csv(f"{out_b}/summary.csv")
    pd.testing.assert_frame_equal(sa, sb, check_exact=False, rtol=1e-12, atol=1e-12)
    ra, rb = open(f"{out_a}/report.md").read(), open(f"{out_b}/report.md").read()
    cut = "## Expected cone"                              # cone section changed after the freeze (task m)
    assert ra.split(cut)[0] == rb.split(cut)[0] and ra.split(cut)[1].split("## ", 1)[1] == rb.split(cut)[1].split("## ", 1)[1]
    ca, cb = json.load(open(f"{out_a}/expected_cone.json")), json.load(open(f"{out_b}/expected_cone.json"))
    # the day-block cone was added after the freeze (task m); the iid cone must be unchanged
    assert {k: v["iid"] for k, v in ca["versions"].items()} == cb["versions"]
    return ta


def test_optimized_engine_gives_identical_results(tmp_path):
    make_market(str(tmp_path), n_tickers=30, n_days=40, seed=99, momentum=0.1, start="2019-10-21")
    new = _run(B, tmp_path, "new")
    ref = _run(R, tmp_path, "ref")
    ta = _same(new, ref)
    assert ta["variant"].nunique() == 17 and ta["filled"].sum() > 300
    assert (ta["day"] == "2019-11-29").any()                        # the calendar half day is covered
    # same again with real-print tick resolution
    ticks, n_amb = _ticks(tmp_path, str(tmp_path / "minutes"), f"{new}/trades.csv")
    assert n_amb > 5
    tb = _same(_run(B, tmp_path, "new_t", ["--ticks", ticks]), _run(R, tmp_path, "ref_t", ["--ticks", ticks]))  # no BE grid
    assert (tb["same_bar_source"] == "ticks").sum() > 5
