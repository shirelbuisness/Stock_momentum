"""beta_b.py: market exposure of long-only B vs SPY (plan v3 §5). Known answers + an end-to-end planted beta."""
import json
import math
import os

import numpy as np
import pandas as pd

import b_mvb as B
import beta_b
from synth import make_market


def test_exact_linear_relation_gives_exact_beta():
    x = np.array([0.01, -0.02, 0.005, 0.0, -0.01, 0.03])
    r = beta_b.market_exposure(0.5 * x + 0.001, x * 10, x)
    assert math.isclose(r["beta"], 0.5, abs_tol=1e-12) and math.isclose(r["alpha_day"], 0.001, abs_tol=1e-12)
    assert math.isclose(r["corr"], 1.0, abs_tol=1e-12) and r["days"] == 6


def test_nw_with_zero_lags_equals_white_hc0_by_hand():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, 200)
    y = 0.3 * x + rng.normal(0, 0.01, 200)
    r = beta_b.market_exposure(y, y, x, lags=0)
    xc = x - x.mean()
    b = (xc * (y - y.mean())).sum() / (xc ** 2).sum()
    e = y - (y.mean() - b * x.mean()) - b * x
    se = math.sqrt(((xc * e) ** 2).sum()) / (xc ** 2).sum()            # White HC0
    assert math.isclose(r["beta"], b, rel_tol=1e-12) and math.isclose(r["beta_t_nw"], b / se, rel_tol=1e-9)


def test_up_down_and_f8_arithmetic_by_hand():
    spy = np.array([0.01, -0.01, 0.02, -0.03, 0.005])
    R = np.array([2.0, -1.0, 3.0, -4.0, 1.0])
    sig = np.array([1, -1, -1, -1, 1])
    r = beta_b.market_exposure(R / 100, R, spy, sig)
    assert r["n_spy_down"] == 2 and math.isclose(r["R_day_spy_down"], -2.5)
    assert r["n_spy_up"] == 3 and math.isclose(r["R_day_spy_up"], 2.0)
    assert r["n_f8_red"] == 3 and math.isclose(r["R_day_f8_red"], (-1 + 3 - 4) / 3)
    assert math.isclose(r["total_R"], 1.0) and math.isclose(r["total_R_skip_red"], 3.0)
    assert math.isclose(r["total_R_half_red"], 3.0 + 0.5 * (-2.0))
    assert r["n_spy_worst5pct"] == 1 and r["R_day_spy_worst5pct"] == -4.0


def test_missing_spy_days_are_left_out_and_counted():
    x = np.array([0.01, np.nan, -0.01, 0.02])
    r = beta_b.market_exposure([0.02, 9.0, -0.02, 0.04], [1, 1, 1, 1], x)
    assert r["days"] == 3 and r["days_without_spy"] == 1 and math.isclose(r["beta"], 2.0)


def test_end_to_end_planted_beta_in_report(tmp_path):
    """Run once, then rewrite SPY's daily open/close so SPY o->c = 2 x the base's 1x daily return.
    The rerun must report beta = 0.5 exactly (SPY is never a candidate, so the base's trades do not change)."""
    make_market(str(tmp_path), n_tickers=30, n_days=40, seed=71, momentum=0.1)
    v = tmp_path / "v.json"
    v.write_text(json.dumps([{"id": "b", "knobs": {"min_avg_vol": 0}}]))
    run = lambda tag: B.main(["--daily", str(tmp_path / "daily.csv"), "--minutes", str(tmp_path / "minutes"),
                              "--start", "2019-01-23", "--end", "2019-12-31", "--nulls", "0", "--gate-reps", "20",
                              "--no-short-shadow", "--variants", str(v), "--out", str(tmp_path / tag),
                              "--trial-log", str(tmp_path / "t.csv")])
    a = run("a")
    tr = pd.read_csv(os.path.join(a, "trades.csv"))
    days = sorted(f[8:18] for f in os.listdir(tmp_path / "minutes") if "2019-01-23" <= f[8:18] <= "2019-12-31")
    ret = B.daily_ret_1x(tr[tr["variant"] == "b"], days, 20)
    assert (ret != 0).sum() > 5
    d = pd.read_csv(tmp_path / "daily.csv")
    m = (d["ticker"] == "SPY") & d["date"].isin(days)
    d.loc[m, "open"] = 100.0
    d.loc[m, "close"] = 100.0 * (1 + 2 * d.loc[m, "date"].map(ret))
    d.to_csv(tmp_path / "daily.csv", index=False)
    b = run("b")
    g = json.load(open(os.path.join(b, "gate.json")))["beta"]
    assert math.isclose(g["beta"], 0.5, abs_tol=1e-9) and math.isclose(g["corr"], 1.0, abs_tol=1e-9)
    assert g["days"] == len(days) and "n_f8_red" in g
    rep = open(os.path.join(b, "report.md")).read()
    assert "## Market exposure of the base" in rep and "beta = 0.500" in rep
    t2 = pd.read_csv(os.path.join(b, "trades.csv"))
    pd.testing.assert_series_equal(tr["r_net"], t2["r_net"])
