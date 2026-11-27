"""Rehearsal of the paper -> live gate, end to end, offline: a backtest run + the shadow logger over ~3.5 months of
synthetic sessions -> paper_gate_b.py. Honest inputs pass; planted problems fail the check they should fail."""
import json
import os

import numpy as np
import pandas as pd
import pytest

import b_mvb as B
import fetch_b_data as F
import nyse_calendar as NC
import paper_gate_b as PG
import shadow_day_b as S
from fake_alpaca import FakeAlpaca
from synth import make_market


def _client(fake):
    return F.Client(key="k", secret="s", transport=fake, max_per_min=10**9, sleep=lambda s: None)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    t = tmp_path_factory.mktemp("reh")
    mkt, data = str(t / "mkt"), str(t / "data")
    make_market(mkt, n_tickers=40, n_days=100, seed=11, momentum=0.1, start="2019-01-02")
    c = _client(FakeAlpaca(mkt))
    F.main(["assets", "--data", data], client=c)
    F.main(["daily", "--data", data, "--start", "2018-12-01", "--end", "2019-06-30"], client=c)
    F.main(["opening", "--data", data, "--start", "2018-12-01", "--end", "2019-06-30"], client=c)
    days = F.trading_days(data)
    start, end = days[22], days[-1]
    run = B.main(["--daily", os.path.join(mkt, "daily.csv"), "--minutes", os.path.join(mkt, "minutes"),
                  "--start", start, "--end", end, "--nulls", "0", "--gate-reps", "10",
                  "--out", str(t / "runs"), "--trial-log", str(t / "trials.csv")])
    paper = str(t / "paper")
    S.main(["--data", data, "--day", start, "--to", end, "--selection-feed", "sip", "--out", paper], client=c)
    return dict(mkt=mkt, data=data, run=run, paper=paper, start=start, end=end, t=t)


def _bt(run):
    tr = pd.read_csv(os.path.join(run, "trades.csv"), dtype={"ticker": str})
    return tr[tr["variant"] == "B-v1__base"]


def _files(paper):
    rd = lambda n: pd.read_csv(os.path.join(paper, n), dtype={"ticker": str})
    return rd("paper_trades.csv"), rd("live_selection.csv"), rd("model_refs.csv")


def test_honest_shadow_passes_every_check_it_can(world):
    pt, sel, refs = _files(world["paper"])
    bt = _backtest_on_nyse_days(world)
    # every shadow trade is the backtest's trade (the synthetic market has bdays that NYSE closes; the logger
    # refuses those, so compare on NYSE sessions only)
    m = bt[bt["filled"]].merge(pt, on=["day", "ticker"], how="outer", indicator=True, suffixes=("_b", "_s"))
    assert (m["_merge"] == "both").all() and len(m) > 200
    assert np.allclose(m["r_net_b"], m["r_net_s"], rtol=0, atol=1e-12)
    fills = refs.assign(fill_px=refs["ref_px"] + refs["side"] * 0.01, qty=refs["qty_1x"])   # 1 cent worse
    res = PG.check(bt, pt, fills, sel)
    assert res["days"] >= PG.MIN_DAYS
    assert res["overlap"]["mean"] == 1.0 and res["expectancy"]["inside"]
    assert 0 < res["slippage"]["ratio"] < PG.SLIP_RATIO_MAX
    assert res["passed"]
    # no fills at all (no order path yet) -> fails closed
    assert not PG.check(bt, pt, None, sel)["passed"]


def _backtest_on_nyse_days(world):
    bt = _bt(world["run"])
    return bt[bt["day"].isin(set(NC.trading_days(world["start"], world["end"])))]


def test_fills_at_twice_the_model_cost_fail_the_slippage_check(world):
    pt, sel, refs = _files(world["paper"])
    model = PG.model_cost_ps(refs["ref_px"])
    fills = refs.assign(fill_px=refs["ref_px"] + refs["side"] * 2 * model, qty=refs["qty_1x"])
    res = PG.check(_backtest_on_nyse_days(world), pt, fills, sel)
    assert res["slippage"]["ratio"] == pytest.approx(2.0)
    assert not res["passed"]


def test_noisy_iex_ranking_fails_the_overlap_check(world):
    """IEX share of each name's volume varying day to day (lognormal, sigma 1.0 around 3%) reorders RVOL: the
    live (IEX) top-N drifts from the backtest's (SIP) top-N and the >= 80% overlap check fails."""
    rng = np.random.default_rng(3)
    share: dict = {}
    fake = FakeAlpaca(world["mkt"], iex_volume=lambda tk, d: share.setdefault((tk, d), 0.03 * float(np.exp(rng.normal(0, 1.0)))))
    out = str(world["t"] / "paper_iex")
    S.main(["--data", world["data"], "--day", world["start"], "--to", world["end"], "--selection-feed", "iex",
            "--out", out], client=_client(fake))
    pt, sel, refs = _files(out)
    res = PG.check(_backtest_on_nyse_days(world), pt, refs.assign(fill_px=refs["ref_px"], qty=refs["qty_1x"]), sel)
    assert res["overlap"]["mean"] < PG.OVERLAP_MIN
    assert not res["passed"]
    assert set(sel["feed_selection"]) == {"iex"}
