"""Tests for fetch_b_data.py against an in-process fake Alpaca. Run: python -m pytest -q test_fetch_b_data.py"""
import os

import numpy as np
import pandas as pd
import pytest

import b_mvb as B
import fetch_b_data as F
from fake_alpaca import FakeAlpaca
from synth import make_market


def client_for(fake, **kw):
    return F.Client(key="k", secret="s", transport=fake, max_per_min=10**9, sleep=lambda s: None, **kw)


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    root = tmp_path_factory.mktemp("mkt")
    make_market(str(root), n_tickers=30, n_days=45, seed=17, momentum=0.02, start="2019-01-02")
    return str(root)


@pytest.fixture(scope="module")
def downloaded(market, tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    fake = FakeAlpaca(market)
    c = client_for(fake)
    for stage in ["assets", "daily", "opening", "fullday"]:
        F.main([stage, "--data", data, "--start", "2018-12-01", "--end", "2019-12-31"], client=c)
    return data, fake, c


# ------------------------------------------------------------------ unit
def test_ny_to_utc_handles_dst():
    assert F.ny_to_utc("2019-01-15", "09:30") == "2019-01-15T14:30:00Z"   # EST
    assert F.ny_to_utc("2019-07-15", "09:30") == "2019-07-15T13:30:00Z"   # EDT
    assert F.ny_to_utc("2019-03-11", "09:30") == "2019-03-11T13:30:00Z"   # day after the US switch


def test_retry_on_429_then_success():
    calls = {"n": 0}

    def flaky(url, headers):
        calls["n"] += 1
        return (429, {"message": "slow down"}) if calls["n"] == 1 else (200, {"ok": 1})
    c = F.Client(key="k", secret="s", transport=flaky, sleep=lambda s: None)
    assert c.get("https://x", "/p", {}) == {"ok": 1} and calls["n"] == 2


def test_http_error_is_loud():
    c = F.Client(key="k", secret="s", transport=lambda u, h: (403, {"message": "forbidden"}), sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        c.get("https://x", "/p", {})


def test_paging_collects_every_page(market):
    fake = FakeAlpaca(market)
    c = client_for(fake)
    day = "2019-02-05"
    syms = ["T000", "T001", "T002", "SPY"]
    big = F.bars_frames(c, syms, "1Min", F.ny_to_utc(day, "09:30"), F.ny_to_utc(day, "16:00"), limit=10000)
    n0 = len(fake.requests)
    small = F.bars_frames(c, syms, "1Min", F.ny_to_utc(day, "09:30"), F.ny_to_utc(day, "16:00"), limit=37)
    assert len(fake.requests) - n0 > 10                      # really paged
    pd.testing.assert_frame_equal(big.reset_index(drop=True), small.reset_index(drop=True))
    assert len(big) == 4 * 390


def test_splits_become_split_factor(market, tmp_path):
    fake = FakeAlpaca(market, splits=[dict(symbol="T003", ex_date="2019-02-11", old_rate=1, new_rate=4),
                                      dict(symbol="T004", ex_date="2019-02-12", old_rate=10, new_rate=1)])
    c = client_for(fake)
    F.main(["assets", "--data", str(tmp_path)], client=c)
    F.main(["daily", "--data", str(tmp_path), "--start", "2019-01-01", "--end", "2019-03-31"], client=c)
    d = pd.read_csv(tmp_path / "daily.csv", dtype={"ticker": str})
    f = d.set_index(["ticker", "date"])["split_factor"]
    assert f[("T003", "2019-02-11")] == 4.0 and f[("T004", "2019-02-12")] == pytest.approx(0.1)
    assert (d["split_factor"] != 1).sum() == 2


def test_holdout_guard_on_download(tmp_path, market):
    with pytest.raises(SystemExit, match="HOLDOUT GUARD"):
        F.main(["opening", "--data", str(tmp_path), "--end", "2024-03-01"], client=client_for(FakeAlpaca(market)))


def test_assets_excludes_etfs(downloaded):
    data, _, _ = downloaded
    s = pd.read_csv(os.path.join(data, "securities.csv"), dtype={"ticker": str})
    assert not s.loc[s["ticker"] == "SPY", "common"].iloc[0]
    assert s.loc[s["ticker"] != "SPY", "common"].all()


# ------------------------------------------------------------------ end to end
def _run_b(daily, minutes, out, extra=()):
    return B.main(["--daily", daily, "--minutes", minutes, "--start", "2019-02-01", "--end", "2019-12-31",
                   "--nulls", "0", "--out", out, "--trial-log", os.path.join(out, "t.csv"), *extra])


def test_resume_skips_finished_pieces(downloaded):
    data, fake, c = downloaded
    n0 = len(fake.requests)
    F.main(["opening", "--data", data, "--start", "2018-12-01", "--end", "2019-12-31"], client=c)
    F.main(["fullday", "--data", data, "--start", "2018-12-01", "--end", "2019-12-31"], client=c)
    assert len(fake.requests) == n0                          # nothing re-downloaded


def test_random_sample_and_selection_log(downloaded):
    data, _, _ = downloaded
    sel = [f for f in os.listdir(os.path.join(data, "minutes")) if f.startswith("selected_")]
    assert sel
    s = pd.read_csv(os.path.join(data, "minutes", sorted(sel)[-1]))
    assert "SPY" in set(s["ticker"]) and set(s["why"]) <= {"top", "random", "spy"}


def test_download_then_backtest_equals_original(market, downloaded, tmp_path):
    """The strongest check: backtesting the downloaded data gives exactly the trades of the source market."""
    data, _, _ = downloaded
    out_src = _run_b(os.path.join(market, "daily.csv"), os.path.join(market, "minutes"), str(tmp_path / "src"))
    out_dl = _run_b(os.path.join(data, "daily.csv"), os.path.join(data, "minutes"), str(tmp_path / "dl"))
    key = ["day", "ticker", "entry_t", "exit_t", "entry_px", "exit_px", "r_net", "reason"]
    a = pd.read_csv(os.path.join(out_src, "trades.csv"), dtype={"ticker": str})
    b = pd.read_csv(os.path.join(out_dl, "trades.csv"), dtype={"ticker": str})
    a, b = a[a["filled"]][key].reset_index(drop=True), b[b["filled"]][key].reset_index(drop=True)
    assert len(a) > 50
    pd.testing.assert_frame_equal(a, b)


def test_ticks_resolve_ambiguous_minutes(downloaded, tmp_path):
    """Fake prints follow O->L->H->C (green) / O->H->L->C (red). Resolving the ambiguous minutes with them must give
    exactly the OHLC-path sensitivity, and every ambiguous trade must be marked as resolved by ticks."""
    data, _, c = downloaded
    out1 = _run_b(os.path.join(data, "daily.csv"), os.path.join(data, "minutes"), str(tmp_path / "r1"), ["--stress"])
    s1 = pd.read_csv(os.path.join(out1, "summary.csv"))
    assert s1.loc[0, "same_bar_stop_pct"] > 0.05            # there is something to resolve
    F.main(["ticks", "--data", data, "--trades", os.path.join(out1, "trades.csv")], client=c)
    out2 = _run_b(os.path.join(data, "daily.csv"), os.path.join(data, "minutes"), str(tmp_path / "r2"),
                  ["--ticks", os.path.join(data, "ticks.csv")])
    s2 = pd.read_csv(os.path.join(out2, "summary.csv"))
    assert s2.loc[0, "mean_R_net"] == pytest.approx(s1.loc[0, "mean_R_net_ohlc_path"], abs=1e-9)
    assert s2.loc[0, "ambiguous_resolved_by_ticks"] == pytest.approx(1.0)
    B.TICKS.clear()


def test_pipeline_never_uses_per_day_asof(downloaded):
    """One symbol mapping everywhere (boss finding): no download stage may send asof."""
    _, fake, _ = downloaded
    assert not [q for path, q in fake.requests if path.startswith("/v2/stocks") and q.get("asof")]


def test_data_gate_passes_clean_and_fails_corrupt(downloaded, tmp_path):
    import shutil
    data, _, _ = downloaded
    F.main(["validate", "--data", data])                       # clean -> passes
    bad = tmp_path / "bad"
    shutil.copytree(data, bad)
    days = sorted(f for f in os.listdir(bad / "minutes") if f.startswith("minutes_"))
    for f in days[:3]:                                         # 3 of ~45 days > 1% -> must fail
        m = pd.read_csv(bad / "minutes" / f)
        pd.concat([m, m.head(5)]).to_csv(bad / "minutes" / f, index=False)   # duplicate bars
    with pytest.raises(SystemExit, match="DATA GATE FAILED"):
        F.main(["validate", "--data", str(bad)])
    rep = pd.read_csv(bad / "validation.csv")
    assert (rep["problem"].str.contains("duplicate")).sum() == 3


def test_probe_runs_and_reports(market, capsys):
    ok = F.stage_probe(client_for(FakeAlpaca(market)), "unused")
    out = capsys.readouterr().out
    assert ok is False and "NO-GO" in out                     # the fake market has no 2016 data: probe must say so


def test_ticks_decide_unit():
    # long: trigger 10.00, stop 9.80. Prints: 9.9, 9.7 (before entry), 10.01 (entry), 10.05, 9.95 -> not stopped
    assert B.ticks_decide(np.array([9.9, 9.7, 10.01, 10.05, 9.95]), 1, 10.00, 9.80) is False
    # entry then a print at the stop -> stopped
    assert B.ticks_decide(np.array([9.9, 10.01, 9.80]), 1, 10.00, 9.80) is True
    # prints never show the entry (feed mismatch) -> undecided
    assert B.ticks_decide(np.array([9.9, 10.00]), 1, 10.00, 9.80) is None
    # short mirror: trigger 10.00 (long space -10.00), stop 10.20 (long space -10.20)
    assert B.ticks_decide(np.array([10.1, 9.99, 10.21]), -1, -10.00, -10.20) is True


def test_validate_lists_calendar_warnings_without_failing_the_gate(downloaded):
    data, _, _ = downloaded
    rep = F.stage_validate(data)
    cal = rep[rep["file"] == "calendar"]
    # the synthetic market uses business days, so MLK day and Presidents' Day 2019 have files
    assert {"2019-01-21", "2019-02-18"} <= set(cal["day"])
    assert cal["problem"].str.startswith("WARN").all()
