"""Tests for b_mvb.py. Run: python -m pytest -q test_b_mvb.py"""
import math
import os

import numpy as np
import pandas as pd
import pytest

import b_mvb as B
from synth import make_market

K0 = B.Knobs(commission_ps=0.0, half_spread_bp=0.0, slip_ticks=0.0, slip_bp=0.0)  # zero-cost for exact checks


def bars(rows):
    """rows: list of (HH:MM, o, h, l, c)."""
    df = pd.DataFrame(rows, columns=["time", "o", "h", "l", "c"])
    df["tmin"] = df["time"].map(B.hm)
    df["ticker"], df["v"] = "X", 1000
    return df


# ------------------------------------------------------------------ known-answer trade cases
def test_close_exit_exact_long():
    # trigger 10.00, ATR 2 -> stop distance 0.20. Fill at 10.00 (needs high >= 10.01). Exit at last close 10.50
    b = bars([("09:35", 9.95, 10.02, 9.94, 10.01), ("09:36", 10.01, 10.20, 10.00, 10.15),
              ("15:58", 10.40, 10.55, 10.38, 10.50)])
    r = B.simulate_trade(b, 1, 10.00, 2.0, K0, B.hm("09:35"), B.hm("15:58"), 9.80)
    assert r.filled and r.exit_reason == "close"
    assert r.entry_px == pytest.approx(10.00)
    assert r.r_net == pytest.approx((10.50 - 10.00) / 0.20)


def test_trigger_needs_one_tick_through():
    b = bars([("09:35", 9.95, 10.00, 9.94, 9.99), ("15:58", 9.99, 10.00, 9.90, 9.95)])  # only touches 10.00
    r = B.simulate_trade(b, 1, 10.00, 2.0, K0, B.hm("09:35"), B.hm("15:58"), 9.80)
    assert not r.filled and r.exit_reason == "no_trigger"


def test_gap_through_fills_at_open():
    b = bars([("09:35", 10.30, 10.40, 10.25, 10.35), ("15:58", 10.35, 10.40, 10.30, 10.35)])
    r = B.simulate_trade(b, 1, 10.00, 2.0, K0, B.hm("09:35"), B.hm("15:58"), 9.80)
    assert r.entry_px == pytest.approx(10.30)


def test_same_bar_entry_and_stop_is_a_loss():
    b = bars([("09:35", 9.95, 10.05, 9.70, 9.90), ("15:58", 11.0, 11.0, 11.0, 11.0)])
    r = B.simulate_trade(b, 1, 10.00, 2.0, K0, B.hm("09:35"), B.hm("15:58"), 9.80)
    assert r.same_bar_stop and r.r_net == pytest.approx(-1.0)


def test_same_bar_ohlc_path_green_bar_low_came_before_entry():
    # green entry bar: O 9.95 -> L 9.70 -> H 10.05 -> C 10.03. The low happened BEFORE the price crossed 10.00
    b = bars([("09:35", 9.95, 10.05, 9.70, 10.03), ("15:58", 10.50, 10.50, 10.50, 10.50)])
    pess = B.simulate_trade(b, 1, 10.00, 2.0, K0, B.hm("09:35"), B.hm("15:58"), 9.80)
    path = B.simulate_trade(b, 1, 10.00, 2.0, B.replace(K0, same_bar="ohlc_path"), B.hm("09:35"), B.hm("15:58"), 9.80)
    assert pess.exit_reason == "stop_same_bar"
    assert path.exit_reason == "close" and path.r_net == pytest.approx(2.5)


def test_same_bar_ohlc_path_red_bar_still_stops():
    # red entry bar: O 9.95 -> H 10.05 (entry) -> L 9.70 (stop) -> C 9.90
    b = bars([("09:35", 9.95, 10.05, 9.70, 9.90), ("15:58", 10.50, 10.50, 10.50, 10.50)])
    path = B.simulate_trade(b, 1, 10.00, 2.0, B.replace(K0, same_bar="ohlc_path"), B.hm("09:35"), B.hm("15:58"), 9.80)
    assert path.exit_reason == "stop_same_bar" and path.r_net == pytest.approx(-1.0)


def test_stop_gap_fills_at_open_worse_than_stop():
    b = bars([("09:35", 9.95, 10.02, 9.94, 10.01), ("09:36", 9.60, 9.65, 9.55, 9.60)])
    r = B.simulate_trade(b, 1, 10.00, 2.0, K0, B.hm("09:35"), B.hm("15:58"), 9.80)
    assert r.exit_reason == "stop_gap" and r.r_net == pytest.approx((9.60 - 10.00) / 0.20)


def test_costs_make_a_flat_trade_negative():
    k = B.Knobs()  # default costs
    b = bars([("09:35", 9.95, 10.02, 9.94, 10.00), ("15:58", 10.00, 10.01, 9.99, 10.00)])
    r = B.simulate_trade(b, 1, 10.00, 2.0, k, B.hm("09:35"), B.hm("15:58"), 9.80)
    assert r.r_net < 0 and r.r_gross == pytest.approx(0.0, abs=1e-9)


def test_breakeven_applies_from_next_bar_only():
    k = B.replace(K0, breakeven_r=1.0)
    # bar 09:36 reaches +1R (10.20) AND dips below entry in the same bar -> must NOT be breakeven-stopped there
    b = bars([("09:35", 9.95, 10.02, 9.94, 10.01), ("09:36", 10.01, 10.25, 9.85, 10.10),
              ("09:37", 10.10, 10.12, 9.99, 10.05), ("15:58", 10.5, 10.5, 10.5, 10.5)])
    r = B.simulate_trade(b, 1, 10.00, 2.0, k, B.hm("09:35"), B.hm("15:58"), 9.80)
    assert r.exit_reason == "breakeven" and r.exit_t == B.hm("09:37")
    assert r.r_net == pytest.approx(0.0, abs=1e-9)


def test_time_stop_uses_bar_close_then_next_open():
    k = B.replace(K0, time_stop="10:30", time_stop_r=0.5)
    b = bars([("09:35", 9.95, 10.02, 9.94, 10.01), ("10:30", 10.01, 10.05, 9.95, 10.03),
              ("10:31", 10.04, 10.06, 10.00, 10.05), ("15:58", 11, 11, 11, 11)])
    r = B.simulate_trade(b, 1, 10.00, 2.0, k, B.hm("09:35"), B.hm("15:58"), 9.80)
    assert r.exit_reason == "time_stop" and r.exit_px == pytest.approx(10.04)


def test_short_mirror():
    b = bars([("09:35", 10.05, 10.06, 9.98, 9.99), ("15:58", 9.60, 9.62, 9.50, 9.50)])
    r = B.simulate_trade(b, -1, 10.00, 2.0, K0, B.hm("09:35"), B.hm("15:58"), 10.20)
    assert r.filled and r.entry_px == pytest.approx(10.00)
    assert r.r_net == pytest.approx((10.00 - 9.50) / 0.20)


def test_cost_filter_skips_tiny_stops():
    k = B.replace(B.Knobs(), min_stop_cost_mult=5.0)
    b = bars([("09:35", 9.95, 10.02, 9.94, 10.01), ("15:58", 10.5, 10.5, 10.5, 10.5)])
    r = B.simulate_trade(b, 1, 10.00, 0.6, k, B.hm("09:35"), B.hm("15:58"), 9.80)  # stop 0.06 vs cost ~0.05
    assert not r.filled and r.exit_reason == "cost_filter"


def test_or_opposite_stop_is_floored_and_capped():
    k = B.replace(K0, stop_mode="or_opposite")
    b = bars([("09:35", 9.95, 10.02, 9.94, 10.01), ("15:58", 10.5, 10.5, 10.5, 10.5)])
    r = B.simulate_trade(b, 1, 10.00, 2.0, k, B.hm("09:35"), B.hm("15:58"), 8.00)  # OR low far away
    assert r.r_ps == pytest.approx(1.0)                                              # capped at 0.5 ATR


# ------------------------------------------------------------------ point-in-time features
def test_atr_uses_prior_days_and_split_scale():
    rows = []
    for i, d in enumerate(pd.bdate_range("2020-01-01", periods=20).strftime("%Y-%m-%d")):
        px = 100.0 if i < 15 else 50.0              # 2:1 split on day 15
        rows.append(dict(ticker="X", date=d, open=px, high=px * 1.02, low=px * 0.98, close=px, volume=1000 if i < 15 else 2000,
                         split_factor=2.0 if i == 15 else 1.0))
    f = B.daily_features(B.load_daily_df(pd.DataFrame(rows)), 14)
    day16 = f.iloc[16]
    assert day16["atr14"] == pytest.approx(2.0, rel=0.02)       # 4% range of 100 scaled to 50 -> $2
    assert day16["avg_vol14"] == pytest.approx(2000, rel=0.01)  # volumes doubled to post-split scale
    assert np.isnan(f.iloc[13]["atr14"])                        # fewer than 14 prior days


def test_rvol_unchanged_by_future_data(tmp_path):
    make_market(str(tmp_path), n_tickers=10, n_days=20, seed=3)
    files = B.minute_files(str(tmp_path / "minutes"))
    daily = B.load_daily(str(tmp_path / "daily.csv"))
    dfe = B.daily_features(daily, 14)
    ot1 = B.add_rvol(B.opening_table(files, {5}), dfe, 14, 14)
    day = sorted(files)[16]
    # perturb every LATER day's volume and every bar after 09:34 on `day`
    for d, p in files.items():
        m = pd.read_csv(p)
        if d > day:
            m["v"] = m["v"] * 50
        elif d == day:
            m.loc[m["time"] >= "09:35", "v"] = 1
        m.to_csv(p, index=False)
    ot2 = B.add_rvol(B.opening_table(files, {5}), dfe, 14, 14)
    a = ot1[ot1["day"] == day].set_index("ticker")["rvol"]
    b = ot2[ot2["day"] == day].set_index("ticker")["rvol"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_holdout_guard(tmp_path):
    make_market(str(tmp_path), n_tickers=3, n_days=3, seed=1)
    with pytest.raises(SystemExit) as e:
        B.main(["--daily", str(tmp_path / "daily.csv"), "--minutes", str(tmp_path / "minutes"),
                "--end", "2024-06-30", "--out", str(tmp_path / "runs")])
    assert "HOLDOUT GUARD" in str(e.value)


def test_variant_that_skips_everything_does_not_crash(tmp_path):
    make_market(str(tmp_path), n_tickers=8, n_days=20, seed=5)
    out = B.main(["--daily", str(tmp_path / "daily.csv"), "--minutes", str(tmp_path / "minutes"),
                  "--start", "2019-01-23", "--end", "2019-12-31", "--nulls", "1",
                  "--variants", _write_variants(tmp_path, [{"id": "base", "knobs": {"min_avg_vol": 0}},
                                                           {"id": "none", "knobs": {"min_avg_vol": 0, "min_stop_cost_mult": 1e9}}]),
                  "--out", str(tmp_path / "runs"), "--trial-log", str(tmp_path / "trials.csv")])
    s = pd.read_csv(os.path.join(out, "summary.csv"))
    assert s.loc[s["id"] == "none", "trades"].iloc[0] == 0
    assert os.path.exists(os.path.join(out, "report.md"))


def _write_variants(tmp_path, spec):
    import json
    p = tmp_path / "v.json"
    p.write_text(json.dumps(spec))
    return str(p)


# ------------------------------------------------------------------ boss-review regressions (27.11)
def _setup(tmp_path, **kw):
    make_market(str(tmp_path), **kw)
    files = B.minute_files(str(tmp_path / "minutes"))
    daily = B.load_daily(str(tmp_path / "daily.csv"))
    dfe = B.daily_features(daily, 14)
    ot = B.add_rvol(B.opening_table(files, {5}), dfe, 14, 10)
    return files, daily, dfe, ot


def test_holdout_can_be_opened_only_once(tmp_path):
    make_market(str(tmp_path), n_tickers=4, n_days=3, seed=1, start="2024-01-02")
    args = ["--daily", str(tmp_path / "daily.csv"), "--minutes", str(tmp_path / "minutes"),
            "--start", "2024-01-01", "--end", "2025-12-31", "--open-holdout", "--nulls", "0",
            "--out", str(tmp_path / "runs"), "--trial-log", str(tmp_path / "trials.csv")]
    B.main(args)                                     # first look: allowed (no trades, history too short)
    with pytest.raises(SystemExit) as e:
        B.main(args)
    assert "opened ONCE" in str(e.value)


def test_holdout_refuses_more_than_two_variants(tmp_path):
    make_market(str(tmp_path), n_tickers=3, n_days=3, seed=1, start="2024-01-02")
    v = _write_variants(tmp_path, [{"id": f"v{i}", "knobs": {}} for i in range(4)])
    with pytest.raises(SystemExit) as e:
        B.main(["--daily", str(tmp_path / "daily.csv"), "--minutes", str(tmp_path / "minutes"), "--variants", v,
                "--start", "2024-01-01", "--end", "2025-12-31", "--open-holdout",
                "--out", str(tmp_path / "runs"), "--trial-log", str(tmp_path / "t.csv")])
    assert "base + 2" in str(e.value)


def test_data_gap_day_is_skipped_not_treated_as_half_day(tmp_path):
    files, daily, dfe, ot = _setup(tmp_path, n_tickers=30, n_days=30, seed=8)
    gap_day = sorted(files)[20]
    m = pd.read_csv(files[gap_day])
    m[m["time"] < "14:07"].to_csv(files[gap_day], index=False)        # file truncated at 14:06
    k = B.replace(B.Knobs(), min_avg_vol=0, sides="both", top_n=30)
    diag = {}
    tr = B.run_many([(B.Variant("t", k), "", 0)], files, ot, dfe, diag)[0]
    assert gap_day in diag["days_skipped_data_gap"]
    assert not (tr["day"] == gap_day).any()


def test_half_day_exits_one_bar_before_close(tmp_path):
    files, daily, dfe, ot = _setup(tmp_path, n_tickers=30, n_days=30, seed=9)
    half = sorted(files)[22]
    m = pd.read_csv(files[half])
    m[m["time"] <= "12:59"].to_csv(files[half], index=False)
    k = B.replace(B.Knobs(), min_avg_vol=0, sides="both", top_n=30)
    tr = B.run_variant(B.Variant("t", k), files, ot, dfe)
    f = tr[(tr["day"] == half) & tr["filled"]]
    assert len(f) > 0 and (f["exit_t"] <= B.hm("12:58")).all()


def test_random_names_null_only_draws_names_with_full_day_bars(tmp_path):
    files, daily, dfe, ot = _setup(tmp_path, n_tickers=30, n_days=30, seed=10)
    # strip every ticker except 5 down to its first 15 minutes (like Stage 1-only data)
    keep = {f"T{i:03d}" for i in range(5)} | {"SPY"}
    for p in files.values():
        m = pd.read_csv(p)
        m[(m["ticker"].isin(keep)) | (m["time"] < "09:45")].to_csv(p, index=False)
    k = B.replace(B.Knobs(), min_avg_vol=0, sides="both", top_n=30, rvol_min=0.0)
    diag = {}
    tr = B.run_many([(B.Variant("t", k), "random_names", 3)], files, ot, dfe, diag)[0]
    assert set(tr["ticker"]) <= keep
    assert max(diag["null_pool_sizes"]) <= 5


def test_rvol_baseline_counts_sessions_not_rows(tmp_path):
    files, daily, dfe, ot = _setup(tmp_path, n_tickers=6, n_days=40, seed=4)
    days = sorted(files)
    # remove T000's opening data on 5 sessions inside the window before day 30
    for d in days[20:25]:
        m = pd.read_csv(files[d])
        m[m["ticker"] != "T000"].to_csv(files[d], index=False)
    ot2 = B.add_rvol(B.opening_table(files, {5}), dfe, 14, 10)
    row = ot2[(ot2["ticker"] == "T000") & (ot2["day"] == days[30]) & (ot2["L"] == 5)].iloc[0]
    v = {d: ot2[(ot2["ticker"] == "T000") & (ot2["day"] == d) & (ot2["L"] == 5)]["v"] for d in days[16:30]}
    expected = np.mean([float(s.iloc[0]) for s in v.values() if len(s)])    # 9 valid of the 14 sessions
    assert len([s for s in v.values() if len(s)]) == 9
    assert np.isnan(row["rvol"])                                             # 9 < 10 valid -> no baseline
    ot3 = B.add_rvol(B.opening_table(files, {5}), dfe, 14, 9)
    row3 = ot3[(ot3["ticker"] == "T000") & (ot3["day"] == days[30]) & (ot3["L"] == 5)].iloc[0]
    assert row3["v"] / row3["base_overF"] == pytest.approx(row3["v"] / expected)


def test_entry_bar2_trigger_is_never_inside_the_or():
    k2 = B.replace(B.Knobs(), entry_bar2=True)
    r = pd.Series(dict(h=10.0, l=9.5, h2=9.8, l2=9.6))         # bar 2 stayed inside the OR
    assert B.entry_trigger(r, 1, k2) == 10.0 and B.entry_trigger(r, -1, k2) == 9.5
    r = pd.Series(dict(h=10.0, l=9.5, h2=10.3, l2=9.2))        # bar 2 extended both ways
    assert B.entry_trigger(r, 1, k2) == 10.3 and B.entry_trigger(r, -1, k2) == 9.2
    assert B.entry_trigger(r, 1, B.Knobs()) == 10.0             # base ignores bar 2
    assert np.isnan(B.entry_trigger(pd.Series(dict(h=10.0, l=9.5, h2=np.nan, l2=np.nan)), 1, k2))


def test_day_major_engine_equals_separate_runs(tmp_path):
    files, daily, dfe, ot = _setup(tmp_path, n_tickers=25, n_days=30, seed=13)
    k1 = B.replace(B.Knobs(), min_avg_vol=0, sides="both")
    k2 = B.replace(k1, stop_atr=0.25)
    both = B.run_many([(B.Variant("a", k1), "", 0), (B.Variant("b", k2), "", 0)], files, ot, dfe)
    sep1 = B.run_variant(B.Variant("a", k1), files, ot, dfe)
    sep2 = B.run_variant(B.Variant("b", k2), files, ot, dfe)
    pd.testing.assert_frame_equal(both[0], sep1)
    pd.testing.assert_frame_equal(both[1], sep2)


# ------------------------------------------------------------------ integration: no edge / planted edge
def _run(tmp_path, momentum, seed):
    make_market(str(tmp_path), n_tickers=60, n_days=60, seed=seed, momentum=momentum)
    files = B.minute_files(str(tmp_path / "minutes"))
    daily = B.load_daily(str(tmp_path / "daily.csv"))
    dfe = B.daily_features(daily, 14)
    ot = B.add_rvol(B.opening_table(files, {5}), dfe, 14, 14)
    k = B.replace(B.Knobs(), sides="both", min_avg_vol=0, top_n=10)
    tr = B.run_variant(B.Variant("t", k), files, ot, dfe)
    f = tr[tr["filled"]]
    t, se, G = B.clustered_t(f["r_gross"], f["day"])
    return f, t


def test_random_walk_has_no_gross_edge(tmp_path):
    f, t = _run(tmp_path, momentum=0.0, seed=11)
    assert len(f) > 150
    assert abs(t) < 3, f"random walk shows an edge (t={t:.2f}) -> lookahead bug?"
    assert f["r_net"].mean() < f["r_gross"].mean()          # costs always hurt


def test_planted_momentum_is_detected(tmp_path):
    f, t = _run(tmp_path, momentum=0.6, seed=12)
    assert t > 3, f"planted edge not found (t={t:.2f})"
