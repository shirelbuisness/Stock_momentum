"""Paper shadow logger (shadow_day_b.py) against the fake Alpaca: the shadow trades of a day equal the backtest's
trades of that day (same selection, same fill and cost model), no lookahead in the selection, IEX ranking with an
IEX baseline, idempotent re-runs, formats read by paper_gate_b / paper_vs_cone, market data only, guards."""
import json
import os
import re

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

DAYS = ["2019-02-21", "2019-02-22", "2019-02-25"]
VARIANTS = [dict(id="B-v1__base", knobs={}), dict(id="B-v1__both", knobs={"sides": "both"}),
            dict(id="B-v1__spy_filter=true", knobs={"spy_filter": True}),
            dict(id="B-v1__time_stop=10:30", knobs={"time_stop": "10:30", "time_stop_r": 0.5}),
            dict(id="B-v1__entry_bar2=true", knobs={"entry_bar2": True}),
            dict(id="B-v1__or_minutes=15", knobs={"or_minutes": 15}),
            dict(id="B-v1__stop_mode=or_opposite", knobs={"stop_mode": "or_opposite"})]
CMP = ["ticker", "side", "entry_t", "exit_t", "entry_px", "exit_px", "r_net", "r_gross", "reason"]


def _client(fake):
    return F.Client(key="k", secret="s", transport=fake, max_per_min=10**9, sleep=lambda s: None)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    mkt = str(tmp_path_factory.mktemp("mkt"))
    make_market(mkt, n_tickers=40, n_days=45, seed=31, momentum=0.1, start="2019-01-02")
    data = str(tmp_path_factory.mktemp("data"))
    c = _client(FakeAlpaca(mkt))
    F.main(["assets", "--data", data], client=c)
    F.main(["daily", "--data", data, "--start", "2018-12-01", "--end", "2019-02-25"], client=c)   # the logger reads only rows < D
    F.main(["opening", "--data", data, "--start", "2018-12-01", "--end", "2019-02-25"], client=c)
    vpath = os.path.join(data, "v.json")
    json.dump(VARIANTS, open(vpath, "w"))
    return mkt, data, vpath


def _backtest_day(mkt, day, var):
    """The backtest on the FULL synthetic market (every name, every minute), restricted to `day`."""
    k = var.knobs
    files = B.minute_files(os.path.join(mkt, "minutes"))
    daily = B.load_daily(os.path.join(mkt, "daily.csv"))
    dfe = B.daily_features(daily, k.hist_days)
    ot = B.add_rvol(B.opening_table({d: p for d, p in files.items() if d <= day}, {k.or_minutes}), dfe,
                    k.hist_days, k.min_hist_days)
    spy = daily[daily["ticker"] == "SPY"].sort_values("date")
    saved = dict(B.SPY_PREV)
    B.SPY_PREV.clear()
    B.SPY_PREV.update(dict(zip(spy["date"].iloc[1:], spy["close"].iloc[:-1])))
    try:
        with B.calendar_context(NC.early_close_map(day, day), strict=True, source="bundled"):
            tr = B.run_many([(var, "", 0)], {day: files[day]}, ot, dfe)[0]
    finally:
        B.SPY_PREV.clear()
        B.SPY_PREV.update(saved)
    return tr[tr["filled"]].sort_values(["ticker"]).reset_index(drop=True)


def _shadow(setup, tmp_path, var_id, feed="sip", fake=None, days=DAYS):
    mkt, data, vpath = setup
    fake = fake or FakeAlpaca(mkt)
    out = str(tmp_path / "paper")
    args = ["--data", data, "--variants", vpath, "--variant", var_id, "--selection-feed", feed, "--out", out]
    for d in days:
        args += ["--day", d]
    S.main(args, client=_client(fake))
    return out, fake


@pytest.mark.parametrize("vid", [v["id"] for v in VARIANTS])
def test_shadow_trades_equal_backtest_trades(setup, tmp_path, vid):
    mkt, _, vpath = setup
    var = next(v for v in B.load_variants(vpath) if v.id == vid)
    out, _ = _shadow(setup, tmp_path, vid)
    pt = pd.read_csv(os.path.join(out, "paper_trades.csv"), dtype={"ticker": str})
    n_total = 0
    for d in DAYS:
        bt = _backtest_day(mkt, d, var)
        sh = pt[pt["day"] == d].sort_values("ticker").reset_index(drop=True)
        assert len(sh) == len(bt), (vid, d)
        n_total += len(bt)
        if len(bt):
            pd.testing.assert_frame_equal(sh[CMP], bt[CMP], check_dtype=False, check_exact=False, rtol=1e-12)
    assert n_total > 0, vid
    assert set(pt["kind"]) == {"shadow"} and set(pt["variant_id"]) == {vid}


def test_selection_ignores_everything_after_the_decision_time(setup, tmp_path):
    """Absurd prices/volume from 09:35 on change the trades (they are simulated on those bars) but never the base
    selection, which is made at 09:35."""
    mkt, _, _ = setup
    a, _ = _shadow(setup, tmp_path / "a", "B-v1__base")
    fake = FakeAlpaca(mkt)
    for d in DAYS:
        m = fake._day_minutes(d)
        late = m["time"] >= "09:35"
        m.loc[late, ["o", "h", "l", "c"]] *= 3.0
        m.loc[late, "v"] *= 100
    b, _ = _shadow(setup, tmp_path / "b", "B-v1__base", fake=fake)
    sa, sb = (pd.read_csv(os.path.join(x, "live_selection.csv")) for x in (a, b))
    pd.testing.assert_frame_equal(sa, sb)
    ta, tb = (pd.read_csv(os.path.join(x, "paper_trades.csv")) for x in (a, b))
    assert len(sa) > 0
    assert len(ta) != len(tb) or not np.allclose(ta["r_net"], tb["r_net"])


def test_iex_selection_uses_an_iex_baseline(setup, tmp_path):
    """IEX = a uniform 3% slice of volume: with an IEX baseline the RVOL ranking is unchanged (identical selection);
    the baseline is cached and the second run asks for no history again."""
    mkt, data, _ = setup
    names = pd.read_csv(os.path.join(mkt, "daily.csv"))["ticker"].unique()
    fake = FakeAlpaca(mkt, iex_volume={t: 0.03 for t in names})
    sip, _ = _shadow(setup, tmp_path / "sip", "B-v1__base")
    iex, _ = _shadow(setup, tmp_path / "iex", "B-v1__base", feed="iex", fake=fake)
    s1, s2 = (pd.read_csv(os.path.join(x, "live_selection.csv")) for x in (sip, iex))
    assert len(s1) > 0
    assert list(s1[["day", "ticker", "side"]].itertuples(index=False)) == \
        list(s2[["day", "ticker", "side"]].itertuples(index=False))
    t1, t2 = (pd.read_csv(os.path.join(x, "paper_trades.csv")) for x in (sip, iex))
    assert np.allclose(t1["r_net"], t2["r_net"])        # simulated on SIP bars either way
    assert os.path.isdir(os.path.join(data, "minutes_iex_opening"))
    n0 = len(fake.requests)
    _shadow(setup, tmp_path / "iex2", "B-v1__base", feed="iex", fake=fake)
    hist_iex = [q for p, q in fake.requests[n0:] if p == "/v2/stocks/bars" and q.get("feed") == "iex"
                and q["start"][:10] not in DAYS]
    assert hist_iex == []
    # the full-day simulation bars are always SIP
    full = [q for p, q in fake.requests if p == "/v2/stocks/bars" and q["end"][11:16] in ("21:00", "18:00")]
    assert full and all(q["feed"] == "sip" for q in full)


def test_iex_today_with_a_sip_baseline_would_select_nothing(setup):
    """Why the IEX baseline exists: IEX bars today / SIP baseline -> RVOL ~0.03 -> the RVOL >= 100% filter empties
    the selection."""
    mkt, data, _ = setup
    names = pd.read_csv(os.path.join(mkt, "daily.csv"))["ticker"].unique()
    c = _client(FakeAlpaca(mkt, iex_volume={t: 0.03 for t in names}))
    k = B.Knobs()
    day = DAYS[1]
    syms = sorted(set(S.D.day_universe(data, day)) | {"SPY"})
    today = S.fetch_opening(c, day, syms, k, "iex")
    otd, dfd, _ = S.D.prepare(data, day, k, today, {})            # SIP baseline (data/minutes)
    assert B.day_candidates(otd, dfd, k).empty
    assert otd["rvol"].dropna().median() < 0.1


def test_rerun_replaces_the_day_and_formats_feed_the_paper_gate(setup, tmp_path):
    out, _ = _shadow(setup, tmp_path, "B-v1__base")
    p = os.path.join(out, "paper_trades.csv")
    before = pd.read_csv(p)
    _shadow(setup, tmp_path, "B-v1__base", days=DAYS[1:2])          # same out dir, one day again
    after = pd.read_csv(p)
    pd.testing.assert_frame_equal(before, after)
    refs = pd.read_csv(os.path.join(out, "model_refs.csv"))
    assert len(refs) == 2 * len(after) and set(refs["leg"]) == {"entry", "exit"}
    assert (refs.groupby(["day", "ticker"])["side"].sum() == 0).all()
    # the paper gate reads the files (the backtest = the logger's own trades -> overlap 100%, mean inside the CI)
    sel = pd.read_csv(os.path.join(out, "live_selection.csv"), dtype={"ticker": str})
    bt = after.assign(filled=True)
    assert PG.selection_overlap(sel, bt)["mean"] == 1.0
    assert PG.expectancy_in_ci(after, bt)["inside"]
    # fills: the model's own reference prices as fills -> realised slippage 0
    fills = refs.assign(fill_px=refs["ref_px"], qty=refs["qty_1x"])
    assert PG.slippage_ratio(fills)["ratio"] == 0.0


def test_guards_and_market_data_only(setup, tmp_path):
    mkt, data, vpath = setup
    with pytest.raises(SystemExit, match="HOLDOUT"):
        S.main(["--data", data, "--day", "2024-01-02"], client=_client(FakeAlpaca(mkt)))
    with pytest.raises(SystemExit, match="HOLDOUT"):
        S.main(["--data", data, "--day", "2023-12-28", "--to", "2024-01-03"], client=_client(FakeAlpaca(mkt)))
    with pytest.raises(SystemExit, match="not NYSE"):
        S.main(["--data", data, "--day", "2019-02-18"], client=_client(FakeAlpaca(mkt)))   # Presidents' Day
    _, fake = _shadow(setup, tmp_path, "B-v1__base")
    assert {p for p, _ in fake.requests} <= {"/v2/stocks/bars", "/v1/corporate-actions"}
    src = open(S.__file__).read()
    for bad in ("paper-api", "/v2/orders", "TradingClient", "submit_order", "/v2/positions", "requests.post"):
        assert bad not in src
    assert not re.search(r"\bimport (alpaca|requests)\b", src)
