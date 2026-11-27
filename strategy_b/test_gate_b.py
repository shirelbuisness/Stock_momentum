"""gate_b.py applies the prereg DEV-A -> DEV-B protocol mechanically. Known-answer cases on hand-built runs."""
import json
import os

import numpy as np
import pandas as pd
import pytest

import b_mvb as B
import gate_b as GB
from synth import make_market

BASE = {"stop_atr": 0.1, "top_n": 20, "rvol_min": 1.0}


def mk(tmp, name, rows, start, end, ticks=None, spa=0.01, pbo=0.1, knobs=None):
    d = tmp / name
    d.mkdir()
    knobs = knobs or {}
    variants = [{"id": r["id"], "knobs": {**BASE, **knobs.get(r["id"], {})}} for r in rows]
    json.dump(dict(args=dict(start=start, end=end, ticks=ticks, open_holdout=False), variants=variants),
              open(d / "config.json", "w"))
    pd.DataFrame(rows).to_csv(d / "summary.csv", index=False)
    json.dump(dict(spa_p=spa, pbo=pbo), open(d / "gate.json", "w"))
    return str(d)


def dev_a(tmp, **kw):
    rows = [dict(id="base", trades=1000),
            dict(id="good", trades=600, p_holm=0.01, dR_per_day_vs_base=0.20),
            dict(id="neg", trades=900, p_holm=0.02, dR_per_day_vs_base=-0.10),
            dict(id="weak", trades=900, p_holm=0.30, dR_per_day_vs_base=0.05),
            dict(id="stop", trades=1000, p_holm=0.001, dR_per_day_vs_base=0.30),
            dict(id="ok2", trades=1000, p_holm=0.05, dR_per_day_vs_base=0.10)]
    return mk(tmp, "a", rows, "2016-01-01", "2020-12-31", knobs={"stop": {"stop_atr": 0.25}}, **kw)


def test_shortlist_known_answer(tmp_path):
    r = GB.shortlist(GB.load_run(dev_a(tmp_path)))
    assert r["shortlist"] == ["good", "ok2"]                         # stop excluded: no ticks (R9)
    why = {x["id"]: x["why"] for x in r["rows"]}
    assert "R9" in why["stop"] and "≤ 0" in why["neg"] and "Holm" in why["weak"]


def test_shortlist_with_ticks_admits_stop_version_and_caps_at_four(tmp_path):
    a = dev_a(tmp_path, ticks="data/ticks.csv")
    assert GB.shortlist(GB.load_run(a))["shortlist"] == ["stop", "good", "ok2"]
    rows = [dict(id="base", trades=1)] + [dict(id=f"v{i}", trades=1, p_holm=0.01 * (i + 1), dR_per_day_vs_base=1)
                                          for i in range(6)]
    r = GB.shortlist(GB.load_run(mk(tmp_path, "six", rows, "2016-01-01", "2020-12-31")))
    assert r["shortlist"] == ["v0", "v1", "v2", "v3"]


def test_spa_above_010_empties_the_shortlist(tmp_path):
    r = GB.shortlist(GB.load_run(dev_a(tmp_path, spa=0.25)))
    assert r["shortlist"] == [] and any("SPA" in n for n in r["notes"])


def test_confirm_known_answer(tmp_path):
    a = dev_a(tmp_path)
    rows = [dict(id="base", trades=800),
            # good: a filter. passes t, sign/size (0.12 >= 0.10), DSR, 2x slip, random skip
            dict(id="good", trades=500, t_nw_vs_base=2.5, dR_per_day_vs_base=0.12, dsr=0.97, mean_R_net_2x_slip=0.02,
                 rs_p_trades=0.01),
            # ok2: same trade count (not a filter), but shrank below half the DEV-A effect
            dict(id="ok2", trades=800, t_nw_vs_base=3.0, dR_per_day_vs_base=0.04, dsr=0.99, mean_R_net_2x_slip=0.1,
                 rs_p_trades=np.nan)]
    b = mk(tmp_path, "b", rows, "2021-01-01", "2023-12-31")
    r = GB.confirm(GB.load_run(a), GB.load_run(b))
    assert r["promoted"] == ["good"]
    ok2 = next(x for x in r["rows"] if x["id"] == "ok2")
    assert [k for k, v in ok2["checks"].items() if not v][0].startswith("same sign")


def test_confirm_fails_closed_on_missing_numbers_and_random_skip(tmp_path):
    a = dev_a(tmp_path)
    rows = [dict(id="base", trades=800),
            dict(id="good", trades=500, t_nw_vs_base=2.5, dR_per_day_vs_base=0.12, dsr=0.97, rs_p_trades=0.20)]
    r = GB.confirm(GB.load_run(a), GB.load_run(mk(tmp_path, "b", rows, "2021-01-01", "2023-12-31")))
    g = next(x for x in r["rows"] if x["id"] == "good")
    fails = [k for k, v in g["checks"].items() if not v]
    assert any("2× slippage" in k for k in fails) and any("random-skip" in k for k in fails)
    assert r["promoted"] == [] and not next(x for x in r["rows"] if x["id"] == "ok2")["passed"]   # ok2 absent


def test_refuses_holdout_runs(tmp_path):
    d = mk(tmp_path, "h", [dict(id="base", trades=1)], "2024-01-01", "2025-12-31")
    with pytest.raises(SystemExit, match="holdout"):
        GB.load_run(d)


def test_end_to_end_on_real_b_mvb_outputs(tmp_path):
    make_market(str(tmp_path), n_tickers=30, n_days=40, seed=71, momentum=0.1)
    v = tmp_path / "v.json"
    v.write_text(json.dumps([{"id": "b", "knobs": {"min_avg_vol": 0}}, {"id": "f3", "knobs": {"min_avg_vol": 0, "top_n": 3}}]))
    run = lambda tag: B.main(["--daily", str(tmp_path / "daily.csv"), "--minutes", str(tmp_path / "minutes"),
                              "--start", "2019-01-23", "--end", "2019-12-31", "--nulls", "0", "--stress",
                              "--gate-reps", "50", "--variants", str(v), "--out", str(tmp_path / tag),
                              "--trial-log", str(tmp_path / "t.csv")])
    a, b = run("a"), run("b")
    assert os.path.exists(os.path.join(a, "gate.json"))
    res = GB.main(["confirm", "--dev-a", a, "--dev-b", b])
    assert res["step"] == "confirm" and any("not inside DEV-B" in n for n in res["notes"])
    assert os.path.exists(os.path.join(b, "gate_confirm.md"))


# ----------------------------------------------------------------------------- holdout pass gate (task q)
GOOD = dict(trades=1500, mean_R_net=0.10, t_net=3.0, sharpe_1x=1.5, dsr=0.97, mean_R_net_2x_slip=0.04,
            max_dd_1x=0.12, quarters_pos="7/8", pct_pos_months=0.7, mean_R_net_ex_top5_tickers=0.06,
            breakeven_ticks=2.5, profit_factor=1.5)


def mk_holdout(tmp, rows, nulls, name="h", start="2024-01-01", end="2025-12-31", open_holdout=True):
    d = tmp / name
    d.mkdir()
    json.dump(dict(args=dict(start=start, end=end, ticks="t.csv", open_holdout=open_holdout),
                   variants=[{"id": r["id"], "knobs": BASE} for r in rows]), open(d / "config.json", "w"))
    pd.DataFrame(rows).to_csv(d / "summary.csv", index=False)
    json.dump(dict(spa_p=0.01, nulls=nulls), open(d / "gate.json", "w"))
    return str(d)


def _nulls(vids, p95=0.0):
    return [dict(null=m, version=v, null_p95=p95) for v in vids for m in ("random_entry", "random_names")]


def test_holdout_all_items_pass_and_each_item_can_fail(tmp_path):
    h = GB.holdout(GB.load_run(mk_holdout(tmp_path, [dict(id="base", **GOOD)], _nulls(["base"])), holdout_ok=True),
                   0.08, 2)
    assert h["passed"] == ["base"] and h["rows"][0]["label"] == "ALIVE" and not h["rehearsal"]
    breaks = {"mean_R_net": 0.07, "t_net": 1.9, "sharpe_1x": 0.9, "dsr": 0.94, "mean_R_net_2x_slip": 0.0,
              "max_dd_1x": 0.25, "trades": 999, "quarters_pos": "5/8", "pct_pos_months": 0.59,
              "mean_R_net_ex_top5_tickers": -0.01, "breakeven_ticks": 1.9}
    item = {"mean_R_net": "1.", "t_net": "1.", "sharpe_1x": "2.", "dsr": "4.", "mean_R_net_2x_slip": "5a.",
            "max_dd_1x": "5b.", "trades": "6a.", "quarters_pos": "6b.", "pct_pos_months": "6c.",
            "mean_R_net_ex_top5_tickers": "6d.", "breakeven_ticks": "7."}
    for i, (k, v) in enumerate(breaks.items()):
        run = mk_holdout(tmp_path, [dict(id="base", **{**GOOD, k: v})], _nulls(["base"]), name=f"b{i}")
        r = GB.holdout(GB.load_run(run, holdout_ok=True), 0.08, 2)["rows"][0]
        failed = [c for c, ok in r["checks"].items() if not ok]
        assert len(failed) == 1 and failed[0].startswith(item[k]), (k, failed)


def test_holdout_bars_matter_nulls_and_holm(tmp_path):
    rows = [dict(id="base", **{**GOOD, "mean_R_net": 0.06}), dict(id="v1", **GOOD, p_holm=0.04),
            dict(id="v2", **GOOD, p_holm=0.06)]
    run = mk_holdout(tmp_path, rows, _nulls(["base", "v1"]))
    h = GB.holdout(GB.load_run(run, holdout_ok=True), 0.05, 1)
    assert h["passed"] == ["base", "v1"]                      # v2: Holm p 0.06 and no nulls of its own
    v2 = [c for c, ok in h["rows"][2]["checks"].items() if not ok]
    assert any("MISSING" in c for c in v2) and any("Holm" in c for c in v2)
    assert GB.holdout(GB.load_run(run, holdout_ok=True), 0.08, 1)["passed"] == ["v1"]   # base 0.06 < 0.08
    run2 = mk_holdout(tmp_path, [dict(id="base", **GOOD)], _nulls(["base"], p95=0.11), name="n")
    assert GB.holdout(GB.load_run(run2, holdout_ok=True), 0.05, 1)["passed"] == []      # does not beat the null


def test_holdout_fragile_label_and_healthy_shape(tmp_path):
    run = mk_holdout(tmp_path, [dict(id="base", **{**GOOD, "mean_R_net_2x_slip": -0.01, "profit_factor": 3.1,
                                                    "pct_pos_months": 1.0})], _nulls(["base"]))
    r = GB.holdout(GB.load_run(run, holdout_ok=True), 0.05, 1)["rows"][0]
    assert r["label"].startswith("FRAGILE") and len(r["shape"]) == 2 and "bug" in r["shape"][0]


def test_holdout_cli_requires_the_open_choices_and_a_real_holdout(tmp_path):
    run = mk_holdout(tmp_path, [dict(id="base", **GOOD)], _nulls(["base"]))
    with pytest.raises(SystemExit):
        GB.main(["holdout", "--run", run, "--net-r-bar", "0.05"])
    with pytest.raises(SystemExit):
        GB.main(["holdout", "--run", run, "--net-r-bar", "0.06", "--breakeven-ticks-bar", "2"])   # not a choice
    dev = mk_holdout(tmp_path, [dict(id="base", **GOOD)], _nulls(["base"]), name="dev", start="2021-01-01",
                     end="2023-12-31", open_holdout=False)
    with pytest.raises(SystemExit, match="not a holdout run"):
        GB.main(["holdout", "--run", dev, "--net-r-bar", "0.05", "--breakeven-ticks-bar", "2"])
    res = GB.main(["holdout", "--run", dev, "--net-r-bar", "0.05", "--breakeven-ticks-bar", "2", "--rehearse"])
    assert res["rehearsal"] and "REHEARSAL" in open(os.path.join(dev, "gate_holdout.md")).read()


def test_holdout_rehearsal_end_to_end_on_b_mvb_output(tmp_path):
    make_market(str(tmp_path), n_tickers=30, n_days=40, seed=71, momentum=0.1)
    v = tmp_path / "v.json"
    v.write_text(json.dumps([{"id": "b", "knobs": {"min_avg_vol": 0}}, {"id": "f3", "knobs": {"min_avg_vol": 0, "top_n": 3}}]))
    out = B.main(["--daily", str(tmp_path / "daily.csv"), "--minutes", str(tmp_path / "minutes"),
                  "--start", "2019-01-23", "--end", "2019-12-31", "--nulls", "2", "--nulls-all-versions",
                  "--stress", "--breakeven", "--gate-reps", "20", "--variants", str(v),
                  "--out", str(tmp_path / "r"), "--trial-log", str(tmp_path / "t.csv")])
    g = json.load(open(os.path.join(out, "gate.json")))
    assert {(n["version"], n["null"]) for n in g["nulls"]} == {(x, m) for x in ("b", "f3")
                                                                for m in ("random_entry", "random_names")}
    assert "| version | null |" in open(os.path.join(out, "report.md")).read()
    res = GB.main(["holdout", "--run", out, "--net-r-bar", "0.05", "--breakeven-ticks-bar", "2", "--rehearse"])
    assert [r["id"] for r in res["rows"]] == ["b", "f3"]
    assert not any("MISSING" in c for r in res["rows"] for c in r["checks"])   # every item had its number
    assert all(not ok for r in res["rows"] for c, ok in r["checks"].items() if c.startswith("6a."))   # < 1000 trades
