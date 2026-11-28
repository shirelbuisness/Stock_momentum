"""paper_gate_b.py: the prereg paper -> live gate, offline, fail closed. Known answers by hand."""
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

import paper_gate_b as P


def test_model_cost_matches_the_backtest_formula_by_hand():
    # $100: half spread max(0.005, 3bp) = 0.03, k = max(1 tick, 2bp) = 0.02 -> 0.05; $10: 0.005 + 0.01 = 0.015
    assert np.allclose(P.model_cost_ps([100.0, 10.0]), [0.05, 0.015])


def test_slippage_ratio_boundary_and_qty_weighting():
    f = pd.DataFrame(dict(side=[1, -1], ref_px=[100.0, 100.0], fill_px=[100.075, 99.925]))
    r = P.slippage_ratio(f)
    assert math.isclose(r["ratio"], 1.5) and math.isclose(r["realised_ps"], 0.075)
    f2 = f.assign(qty=[100, 300], fill_px=[100.05, 99.90])          # 0.05 on 100 sh, 0.10 on 300 sh
    assert math.isclose(P.slippage_ratio(f2)["realised_ps"], (5 + 30) / 400)


def test_selection_overlap_by_hand():
    bt = pd.DataFrame(dict(day=["d1"] * 4 + ["d2"] * 2, ticker=list("ABCD") + list("AB")))
    live = pd.DataFrame(dict(day=["d1"] * 3 + ["d2"] + ["d3"], ticker=list("ABX") + ["A", "Z"]))
    r = P.selection_overlap(live, bt)
    assert r["days"] == 2 and math.isclose(r["mean"], (2 / 4 + 1 / 2) / 2) and r["live_days_without_backtest"] == 1


def _bt(n_days=100, per_day=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = [dict(day=f"D{i:03d}", ticker=f"T{j}", r_net=rng.normal(0.1, 1.0), filled=True)
            for i in range(n_days) for j in range(per_day)]
    return pd.DataFrame(rows)


def _ok_inputs(bt, n_days=70):
    paper = pd.DataFrame(dict(day=[f"P{i:03d}" for i in range(n_days)], r_net=bt["r_net"].mean()))
    fills = pd.DataFrame(dict(side=[1, -1], ref_px=[50.0, 50.0], fill_px=[50.03, 49.97]))
    return paper, fills, bt[["day", "ticker"]]


def test_all_pass_and_each_check_fails_alone():
    bt = _bt()
    paper, fills, sel = _ok_inputs(bt)
    assert P.check(bt, paper, fills, sel)["passed"]
    cases = {
        "sessions": (paper.head(62), fills, sel),
        "slippage": (paper, fills.assign(fill_px=[50.2, 49.8]), sel),
        "overlap": (paper, fills, sel.groupby("day").head(3)),            # 3 of 4 = 75%
        "expectancy": (paper.assign(r_net=5.0), fills, sel),
    }
    for name, (p, f, s) in cases.items():
        res = P.check(bt, p, f, s)
        assert [ok for ok in res["checks"].values()].count(False) == 1 and not res["passed"], name


def test_missing_inputs_fail_closed():
    bt = _bt()
    paper, _, _ = _ok_inputs(bt)
    res = P.check(bt, paper, None, None)
    assert not res["passed"] and sum(not v for v in res["checks"].values()) == 2


def test_cli(tmp_path):
    bt = _bt().assign(variant="B-v1__base")
    os.makedirs(tmp_path / "run")
    bt.to_csv(tmp_path / "run" / "trades.csv", index=False)
    json.dump(dict(variants=[dict(id="B-v1__base", knobs=dict(half_spread_bp=3.0, slip_ticks=1.0, slip_bp=2.0))]),
              open(tmp_path / "run" / "config.json", "w"))
    paper, fills, sel = _ok_inputs(bt)
    for n, df in (("paper", paper), ("fills", fills), ("sel", sel)):
        df.to_csv(tmp_path / f"{n}.csv", index=False)
    res = P.main(["--run", str(tmp_path / "run"), "--paper", str(tmp_path / "paper.csv"),
                  "--fills", str(tmp_path / "fills.csv"), "--selection", str(tmp_path / "sel.csv")])
    assert res["passed"] and "PASS" in open(tmp_path / "paper_gate.md").read()
    with pytest.raises(SystemExit):
        P.main(["--run", str(tmp_path / "run"), "--paper", str(tmp_path / "paper.csv"), "--version", "nope"])
