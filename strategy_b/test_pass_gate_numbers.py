"""Pass-gate numbers added in task p: net R without the top-5 tickers; the B0-replication sanity gate."""
import json
import math
import os

import pandas as pd

import b_mvb as B
from synth import make_market


def _tr(rows):
    return B._trades_df([dict(day=d, ticker=t, side=1, filled=True, r_net=r, r_gross=r + 0.1, ret_net=r / 100,
                              entry_t=600, reason="close", same_bar_stop=False, same_bar_source="")
                         for d, t, r in rows])


def test_mean_without_top5_tickers_by_hand():
    rows = [("2019-01-02", f"T{i}", float(10 - i)) for i in range(7)]          # T0 +10 ... T6 +4
    rows += [("2019-01-03", "T0", 5.0), ("2019-01-03", "T7", -1.0), ("2019-01-04", "T8", -3.0)]
    s = B.summarize(_tr(rows), ["2019-01-02", "2019-01-03", "2019-01-04"], 20)
    # totals: T0 15, T1 9, T2 8, T3 7, T4 6 -> removed; left T5 5, T6 4, T7 -1, T8 -3
    assert math.isclose(s["mean_R_net_ex_top5_tickers"], (5 + 4 - 1 - 3) / 4)


def test_fewer_than_six_tickers_leaves_nothing():
    s = B.summarize(_tr([("2019-01-02", "A", 1.0), ("2019-01-02", "B", -1.0)]), ["2019-01-02"], 20)
    assert math.isnan(s["mean_R_net_ex_top5_tickers"])


def test_replication_run_end_to_end(tmp_path):
    make_market(str(tmp_path), n_tickers=30, n_days=40, seed=71, momentum=0.3)
    v = tmp_path / "v.json"
    v.write_text(json.dumps([{"id": "b", "knobs": {"min_avg_vol": 0}}]))
    out = B.main(["--daily", str(tmp_path / "daily.csv"), "--minutes", str(tmp_path / "minutes"),
                  "--start", "2019-01-23", "--end", "2019-12-31", "--nulls", "0", "--gate-reps", "20",
                  "--replication", "--variants", str(v), "--out", str(tmp_path / "r"),
                  "--trial-log", str(tmp_path / "t.csv")])
    g = json.load(open(os.path.join(out, "gate.json")))["replication"]
    tr = pd.read_csv(os.path.join(out, "trades.csv"))
    rp, base = tr[(tr["variant"] == "b__replication") & tr["filled"]], tr[(tr["variant"] == "b") & tr["filled"]]
    assert set(rp["side"]) == {1, -1} and g["trades"] == len(rp) > len(base)
    # commission only: the cost of a replication trade is lower than the same long trade in the base
    m = rp.merge(base, on=["day", "ticker", "side"], suffixes=("_r", "_b"))
    assert len(m) > 10 and ((m["r_gross_r"] - m["r_net_r"]) < (m["r_gross_b"] - m["r_net_b"])).all()
    assert g["pass"] == (g["sharpe_1x"] >= 2.0)
    assert "## B0-replication sanity gate" in open(os.path.join(out, "report.md")).read()
    log = pd.read_csv(tmp_path / "t.csv")
    assert (log["variant_id"] == "b__replication").sum() == 1          # every backtest run is a trial


def test_no_replication_by_default(tmp_path):
    make_market(str(tmp_path), n_tickers=20, n_days=30, seed=5)
    out = B.main(["--daily", str(tmp_path / "daily.csv"), "--minutes", str(tmp_path / "minutes"),
                  "--start", "2019-01-23", "--end", "2019-12-31", "--nulls", "0", "--gate-reps", "10",
                  "--out", str(tmp_path / "r"), "--trial-log", str(tmp_path / "t.csv")])
    assert json.load(open(os.path.join(out, "gate.json")))["replication"] is None
