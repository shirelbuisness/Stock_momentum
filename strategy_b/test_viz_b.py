"""The independent verifier (viz_b.py) must re-derive 100% of a run's trades, including a calendar half day,
and must FAIL on a deliberately wrong rule."""
import json

import pandas as pd

import b_mvb as B
import viz_b as V
from synth import make_market


def test_independent_verifier_matches_every_trade_incl_half_day(tmp_path):
    make_market(str(tmp_path), n_tickers=40, n_days=40, seed=78, momentum=0.1, start="2019-10-28")
    v = tmp_path / "v.json"
    v.write_text(json.dumps([{"id": "base", "knobs": {"min_avg_vol": 0}},
                             {"id": "L1", "knobs": {"min_avg_vol": 0, "stop_atr": 0.25}},
                             {"id": "L4", "knobs": {"min_avg_vol": 0, "time_stop": "10:30"}}]))
    out = B.main(["--daily", str(tmp_path / "daily.csv"), "--minutes", str(tmp_path / "minutes"),
                  "--start", "2019-11-15", "--end", "2019-12-31", "--nulls", "0", "--gate-reps", "20",
                  "--variants", str(v), "--out", str(tmp_path / "runs"), "--trial-log", str(tmp_path / "t.csv")])
    tr = pd.read_csv(f"{out}/trades.csv")
    assert (tr["day"] == "2019-11-29").any()                       # the half day is inside the test
    for var, rule in [("base", {}), ("L1", dict(stop_atr=0.25)), ("L4", dict(time_stop="10:30"))]:
        r = V.verify_all(tr, str(tmp_path / "minutes"), var, **rule)
        assert r["checked"] > 50 and r["mismatch"] == 0, (var, r["examples"])
    assert V.verify_all(tr, str(tmp_path / "minutes"), "base", stop_atr=0.11)["mismatch"] > 0
