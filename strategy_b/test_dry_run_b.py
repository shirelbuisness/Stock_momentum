"""09:35 dry run (dry_run_b.py) against the fake Alpaca: parity with the backtest's selection, no lookahead,
no trading endpoint, intents written to a file only."""
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

import b_mvb as B
import dry_run_b as D
import fetch_b_data as F
from fake_alpaca import FakeAlpaca
from synth import make_market

DAY = "2019-02-22"          # a Friday inside the synthetic market, > 14 sessions of history


def _client(fake):
    return F.Client(key="k", secret="s", transport=fake, max_per_min=10**9, sleep=lambda s: None)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    mkt = str(tmp_path_factory.mktemp("mkt"))
    make_market(mkt, n_tickers=40, n_days=45, seed=31, momentum=0.1, start="2019-01-02")
    data = str(tmp_path_factory.mktemp("data"))
    c = _client(FakeAlpaca(mkt))
    F.main(["assets", "--data", data], client=c)
    F.main(["daily", "--data", data, "--start", "2018-12-01", "--end", "2019-02-21"], client=c)   # history only
    F.main(["opening", "--data", data, "--start", "2018-12-01", "--end", "2019-02-21"], client=c)
    return mkt, data


def _backtest_candidates(mkt, k):
    files = B.minute_files(os.path.join(mkt, "minutes"))
    dfe = B.daily_features(B.load_daily(os.path.join(mkt, "daily.csv")), k.hist_days)
    ot = B.add_rvol(B.opening_table(files, {k.or_minutes}), dfe, k.hist_days, k.min_hist_days)
    return B.day_candidates(ot[ot["day"] == DAY], dfe[dfe["date"] == DAY], k)


def _run(setup, tmp_path, fake=None, extra=()):
    mkt, data = setup
    fake = fake or FakeAlpaca(mkt)
    out = D.main(["--data", data, "--day", DAY, "--out", str(tmp_path), *extra], client=_client(fake))
    return out, fake


@pytest.mark.parametrize("knobs", [{}, {"sides": "both"}])
def test_dry_run_selection_equals_backtest_selection(setup, tmp_path, knobs):
    mkt, _ = setup
    v = tmp_path / "v.json"
    v.write_text(json.dumps([{"id": "B-v1__base", "knobs": knobs}]))
    out, _ = _run(setup, tmp_path, extra=["--variants", str(v)])
    it = pd.read_csv(os.path.join(out, "order_intents.csv"))
    k = B.replace(B.Knobs(), **knobs)
    bt = _backtest_candidates(mkt, k)
    assert len(bt) > 0
    assert list(it["ticker"]) == list(bt["ticker"])
    assert list(it["side"]) == ["buy" if s > 0 else "sell_short" for s in bt["side"]]
    trig = [B.entry_trigger(r, int(r["side"]), k) for _, r in bt.iterrows()]
    assert np.allclose(it["stop_price"], np.round(np.array(trig) + bt["side"].to_numpy() * B.TICK, 2))
    assert np.allclose(it["rvol"], bt["rvol"]) and np.allclose(it["atr14"], bt["atr14"])
    assert np.allclose(it["protective_stop_distance"], (0.10 * bt["atr14"]).round(4))


def test_dry_run_ignores_everything_after_0935(setup, tmp_path):
    mkt, _ = setup
    a, _ = _run(setup, tmp_path / "a")
    fake = FakeAlpaca(mkt)
    m = fake._day_minutes(DAY)
    late = m["time"] >= "09:35"
    m.loc[late, ["o", "h", "l", "c"]] *= 3.0                # the future becomes absurd
    m.loc[late, "v"] *= 100
    b, _ = _run(setup, tmp_path / "b", fake=fake)
    pd.testing.assert_frame_equal(pd.read_csv(f"{a}/order_intents.csv"), pd.read_csv(f"{b}/order_intents.csv"))


def test_dry_run_only_reads_market_data_and_writes_files(setup, tmp_path):
    out, fake = _run(setup, tmp_path)
    paths = {p for p, _ in fake.requests}
    assert paths <= {"/v2/stocks/bars", "/v1/corporate-actions"}
    src = open(D.__file__).read()
    for bad in ("TRADING_URL", "paper-api", "api.alpaca.markets/v2", "/v2/orders", "/v2/positions", "submit_order",
                "TradingClient", "requests.post", "method=\"POST\""):
        assert bad not in src, bad
    meta = json.load(open(os.path.join(out, "order_intents.json")))["meta"]
    assert meta["kind"].startswith("DRY RUN")
    it = pd.read_csv(os.path.join(out, "order_intents.csv"))
    assert (it["sent"] == False).all()                                     # noqa: E712
    assert it["client_order_id"].str.fullmatch(r"Bv1-base-T\d{3}-20190222-\d{4}-E1-r0").all()
    assert it["client_order_id"].is_unique
    assert (it["qty_1x_equal_notional"] == np.floor(100_000 / 20 / it["stop_price"])).all()


def test_dry_run_refuses_holdout_and_closed_days(setup, tmp_path):
    _, data = setup
    with pytest.raises(SystemExit, match="HOLDOUT"):
        D.main(["--data", data, "--day", "2024-01-03", "--out", str(tmp_path)], client=_client(None))
    with pytest.raises(SystemExit, match="not an NYSE trading day"):
        D.main(["--data", data, "--day", "2019-02-18", "--out", str(tmp_path)], client=_client(None))
