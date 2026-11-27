"""Spread calibration: the quotes stage parses NBBO correctly (drops locked / one-sided quotes, windows, resumes,
holdout guard, market data only) and spread_b.py re-costs trades by hand-checkable arithmetic."""
import json
import os

import numpy as np
import pandas as pd
import pytest

import b_mvb as B
import fetch_b_data as F
import spread_b as SB
from fake_alpaca import FakeAlpaca
from synth import make_market


def _client(fake):
    return F.Client(key="k", secret="s", transport=fake, max_per_min=10**9, sleep=lambda s: None)


def test_quote_spreads_drops_locked_crossed_and_one_sided():
    qs = [dict(bp=10.00, ap=10.02), dict(bp=10.00, ap=10.00), dict(bp=10.01, ap=10.00), dict(bp=0, ap=10.02),
          dict(bp=10.00, ap=10.04), dict(ap=10.03)]
    s = F.quote_spreads(qs)
    assert s["n_quotes"] == 2
    assert s["spread_usd_median"] == pytest.approx(0.03)
    assert s["spread_bp_median"] == pytest.approx(np.median([0.02 / 10.01, 0.04 / 10.02]) * 1e4)
    assert F.quote_spreads([dict(bp=1, ap=1)]) is None and F.quote_spreads([]) is None


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    t = tmp_path_factory.mktemp("m")
    make_market(str(t), n_tickers=30, n_days=40, seed=71, momentum=0.3)
    v = t / "v.json"
    v.write_text(json.dumps([{"id": "b", "knobs": {"min_avg_vol": 0}},
                             {"id": "b2", "knobs": {"min_avg_vol": 0, "stop_atr": 0.25}}]))
    out = B.main(["--daily", str(t / "daily.csv"), "--minutes", str(t / "minutes"), "--start", "2019-01-23",
                  "--end", "2019-12-31", "--nulls", "0", "--gate-reps", "10", "--variants", str(v),
                  "--out", str(t / "r"), "--trial-log", str(t / "t.csv")])
    return str(t), out


def test_quotes_stage_end_to_end(run, tmp_path):
    mkt, out = run
    tr = pd.read_csv(os.path.join(out, "trades.csv"), dtype={"ticker": str})
    names = tr["ticker"].unique()
    fake = FakeAlpaca(mkt, spread_bp={t: 20.0 for t in names})
    data = str(tmp_path / "data")
    F.main(["quotes", "--data", data, "--trades", os.path.join(out, "trades.csv")], client=_client(fake))
    q = pd.read_csv(os.path.join(data, "quotes_spread.csv"), dtype={"ticker": str})
    f = tr[tr["filled"]].drop_duplicates(["day", "ticker", "entry_t", "exit_t"])
    legs = set(zip(f["day"], f["ticker"], f["entry_t"])) | set(zip(f["day"], f["ticker"], f["exit_t"]))
    assert set(zip(q["day"], q["ticker"], q["minute"])) == legs
    # 10 s window = quotes at :00 :02 :04 :06 :08; the one at :08 is locked -> 4 kept; spread exactly 20 bp
    assert (q["n_quotes"] == 4).all()
    assert np.allclose(q["spread_bp_median"], 20.0, atol=1e-3)
    assert {p for p, _ in fake.requests} == {"/v2/stocks/quotes"}
    assert all(x["feed"] == "sip" for _, x in fake.requests)
    n0 = len(fake.requests)                                           # resumable: nothing is asked twice
    F.main(["quotes", "--data", data, "--trades", os.path.join(out, "trades.csv")], client=_client(fake))
    assert len(fake.requests) == n0
    assert len(pd.read_csv(os.path.join(data, "quotes_spread.csv"))) == len(q)
    # a 20 s window reaches the one-sided quote at :14 and the locked ones at :08 and :18 -> 7 kept
    data2 = str(tmp_path / "data2")
    F.main(["quotes", "--data", data2, "--trades", os.path.join(out, "trades.csv"), "--window-s", "20",
            "--sample", "5"], client=_client(fake))
    q2 = pd.read_csv(os.path.join(data2, "quotes_spread.csv"))
    assert (q2["n_quotes"] == 7).all() and len(q2) <= 10


def test_quotes_holdout_guard(tmp_path):
    p = tmp_path / "trades.csv"
    pd.DataFrame(dict(day=["2024-01-03"], ticker=["X"], filled=[True], entry_t=[575], exit_t=[958])).to_csv(p, index=False)
    with pytest.raises(SystemExit, match="HOLDOUT"):
        F.main(["quotes", "--data", str(tmp_path), "--trades", str(p)], client=_client(FakeAlpaca(_tiny(tmp_path))))


def _tiny(tmp_path):
    d = tmp_path / "mk"
    make_market(str(d), n_tickers=2, n_days=3, seed=1)
    return str(d)


def test_calibrate_known_answer():
    """One long trade, $50 in / $51 out, r_ps $0.50, model 3 bp. Measured half spreads $0.05 in, $0.02 out.
    Model half spread = 50 x 3 bp = $0.015 in, 51 x 3 bp = $0.0153 out.
    dR = -((0.05 - 0.015) + (0.02 - 0.0153)) / 0.50 = -0.0794."""
    tr = pd.DataFrame(dict(variant=["b", "b"], day=["2019-01-02"] * 2, ticker=["A", "Z"], filled=[True, True],
                           entry_t=[575, 580], exit_t=[958, 958], entry_px=[50.0, 20.0], exit_px=[51.0, 20.0],
                           r_ps=[0.5, 0.2], r_net=[1.0, -1.0]))
    q = pd.DataFrame(dict(day=["2019-01-02"] * 2, ticker=["A", "A"], minute=[575, 958], leg=["entry", "exit"],
                          n_quotes=[4, 4], spread_bp_median=[0.1 / 50 * 1e4, 0.04 / 51 * 1e4],
                          spread_usd_median=[0.10, 0.04], mid_median=[50.0, 51.0]))
    res = SB.calibrate(tr, q, {"b": {}})
    v = res["versions"]["b"]
    assert v["trades_measured"] == 1 and v["trades_in_run"] == 2       # Z has no quotes -> left out
    assert v["mean_r_net_as_run"] == 1.0
    assert v["delta_r"] == pytest.approx(-((0.05 - 0.015) + (0.02 - 0.0153)) / 0.5)
    assert res["measured_half_bp"]["median"] == pytest.approx(np.median([10.0, 0.02 / 51 * 1e4]))
    assert res["model_half_bp_at_same_legs"]["median"] == pytest.approx(3.0)
    # below $16.67 the model charges half a tick, not 3 bp
    assert SB.model_half_spread_usd([10.0], 3.0)[0] == pytest.approx(0.005)


def test_spread_cli_on_a_run(run, tmp_path):
    mkt, out = run
    tr = pd.read_csv(os.path.join(out, "trades.csv"), dtype={"ticker": str})
    fake = FakeAlpaca(mkt, spread_bp={t: 6.0 for t in tr["ticker"].unique()})     # = the model's 3 bp half spread
    data = str(tmp_path / "d")
    F.main(["quotes", "--data", data, "--trades", os.path.join(out, "trades.csv")], client=_client(fake))
    res = SB.main(["--run", out, "--quotes", os.path.join(data, "quotes_spread.csv")])
    assert res["measured_half_bp"]["median"] == pytest.approx(3.0, abs=1e-3)
    assert set(res["versions"]) == {"b", "b2", "b__shadow_short"}       # the short shadow is costed too
    for v in res["versions"].values():
        assert v["trades_measured"] == v["trades_in_run"] > 0
        # measured = model except where the model's half-tick floor binds or the minute's close != the fill
        assert abs(v["delta_r"]) < 0.02
    assert "DIAGNOSTIC" in open(os.path.join(out, "spread_calibration.md")).read()
