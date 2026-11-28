"""Tests for gate_stats.py (DSR, random-skip null, Hansen SPA, PBO via CSCV) and their wiring into b_mvb.py."""
import math
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import b_mvb as B
import gate_stats as G
from synth import make_market


# ------------------------------------------------------------------ DSR
def test_expected_max_sharpe_known_value():
    # N=100, V=1: (1-g)*z(0.99) + g*z(1-1/(100e)) = 0.4228*2.3263 + 0.5772*2.6815 ~= 2.5306
    v = G.expected_max_sharpe(100, 1.0)
    hand = (1 - G.EULER_GAMMA) * stats.norm.ppf(0.99) + G.EULER_GAMMA * stats.norm.ppf(1 - 1 / (100 * math.e))
    assert v == pytest.approx(hand) and v == pytest.approx(2.5306, abs=1e-3)
    assert G.expected_max_sharpe(1, 1.0) == 0.0 and G.expected_max_sharpe(10, float("nan")) == 0.0


def test_psr_known_value_normal_returns():
    # SR=0.1/day, SR0=0, T=251, skew 0, kurt 3: z = 0.1*sqrt(250)/sqrt(1+0.5*0.01) = 1.5772
    p = G.probabilistic_sharpe(0.1, 0.0, 251, 0.0, 3.0)
    assert p == pytest.approx(stats.norm.cdf(0.1 * math.sqrt(250) / math.sqrt(1.005)))


def test_dsr_falls_as_trials_grow():
    x = np.random.default_rng(1).normal(0.001, 0.01, 1000)        # SR ~ 0.1/day
    d1 = G.deflated_sharpe(x, 1, 0.002)["dsr"]
    d50 = G.deflated_sharpe(x, 50, 0.002)["dsr"]
    d500 = G.deflated_sharpe(x, 500, 0.002)["dsr"]
    assert d1 > d50 > d500
    assert d1 == pytest.approx(G.deflated_sharpe(x, 1, float("nan"))["dsr"])   # N=1 -> PSR vs 0


def test_dsr_noise_is_not_significant():
    x = np.random.default_rng(2).normal(0, 0.01, 1000)
    assert G.deflated_sharpe(x, 30, 0.001)["dsr"] < 0.95


def test_trial_sharpe_stats_counts_every_row_and_converts_to_daily():
    n, v = G.trial_sharpe_stats(["1.0", "", "nan", "2.0"], [3.0])
    assert n == 5
    assert v == pytest.approx(np.var(np.array([1.0, 2.0, 3.0]) / math.sqrt(252), ddof=1))


# ------------------------------------------------------------------ random-skip null
def test_random_skip_subset_is_not_significant_and_winner_filter_is():
    rng = np.random.default_rng(3)
    base = rng.normal(0.05, 1.0, 2000)
    days = np.repeat(np.arange(400), 5)
    rand = base[rng.choice(2000, 800, replace=False)]
    ps = [G.random_skip_null(base, base[rng.choice(2000, 800, replace=False)], reps=300, seed=s)["rs_p_trades"]
          for s in range(30)]
    assert 0.2 < np.mean(ps) < 0.8                                 # roughly uniform under H0
    r = G.random_skip_null(base, rand, days, 200, reps=300, seed=1)
    assert r["rs_keep_share"] == pytest.approx(0.4) and np.isfinite(r["rs_p_days"])
    smart = np.sort(base)[-800:]                                   # keeps the winners: obviously beats luck
    r = G.random_skip_null(base, smart, days, 200, reps=300, seed=1)
    assert r["rs_p_trades"] < 0.01 and r["rs_p_days"] < 0.01


def test_random_skip_undefined_for_same_size_versions():
    base = np.arange(10.0)
    assert np.isnan(G.random_skip_null(base, base + 1)["rs_p_trades"])


# ------------------------------------------------------------------ SPA
def test_stationary_bootstrap_shape_and_range():
    idx = G.stationary_bootstrap_indices(50, 7, 5.0, np.random.default_rng(0))
    assert idx.shape == (7, 50) and idx.min() >= 0 and idx.max() < 50


def test_spa_no_better_variant_gives_large_p():
    rng = np.random.default_rng(4)
    d = rng.normal(0, 1, (500, 5)) - 0.05                          # all variants slightly WORSE than base
    assert G.spa_test(d, reps=400, seed=1)["spa_p"] > 0.10


def test_spa_detects_one_truly_better_variant():
    rng = np.random.default_rng(5)
    d = rng.normal(0, 1, (500, 5))
    d[:, 3] += 0.3                                                 # t ~ 6.7
    r = G.spa_test(d, reps=400, seed=1)
    assert r["spa_p"] < 0.01 and r["spa_best"] == 3


# ------------------------------------------------------------------ PBO
def test_pbo_noise_is_about_half():
    rng = np.random.default_rng(6)
    r = G.pbo_cscv(rng.normal(0, 1, (800, 10)), n_blocks=10)
    assert r["splits"] == math.comb(10, 5) and 0.3 < r["pbo"] < 0.7


def test_pbo_true_winner_is_near_zero():
    rng = np.random.default_rng(7)
    m = rng.normal(0, 1, (800, 10))
    m[:, 2] += 0.5
    assert G.pbo_cscv(m, n_blocks=10)["pbo"] < 0.05


# ------------------------------------------------------------------ wiring into the report
def _variants(tmp_path, spec):
    import json
    p = tmp_path / "v.json"
    p.write_text(json.dumps(spec))
    return str(p)


def test_report_contains_gate_statistics_and_trial_count(tmp_path):
    make_market(str(tmp_path), n_tickers=40, n_days=50, seed=21, momentum=0.3)
    v = _variants(tmp_path, [{"id": "base", "knobs": {"min_avg_vol": 0, "sides": "both"}},
                             {"id": "f3", "knobs": {"min_avg_vol": 0, "sides": "both", "top_n": 3}},
                             {"id": "l1", "knobs": {"min_avg_vol": 0, "sides": "both", "stop_atr": 0.25}}])
    log = tmp_path / "trials.csv"
    args = ["--daily", str(tmp_path / "daily.csv"), "--minutes", str(tmp_path / "minutes"),
            "--start", "2019-01-23", "--end", "2019-12-31", "--nulls", "0", "--variants", v, "--gate-reps", "200",
            "--pbo-blocks", "6", "--out", str(tmp_path / "runs"), "--trial-log", str(log)]
    out1 = B.main(args)
    rep = open(os.path.join(out1, "report.md")).read()
    assert "Gate statistics" in rep and "Trials in the log (N for the DSR, this run included): 3**" in rep
    s = pd.read_csv(os.path.join(out1, "summary.csv")).set_index("id")
    assert {"dsr", "rs_p_trades", "rs_p_days"} <= set(s.columns)
    assert 0 <= s.loc["base", "dsr"] <= 1
    assert np.isfinite(s.loc["f3", "rs_p_trades"]) and np.isnan(s.loc["l1", "rs_p_trades"])
    assert "Hansen SPA" in rep and "PBO = " in rep and "PBO = –" not in rep
    out2 = B.main(args)                                            # a second run: N grows and DSR cannot rise
    rep2 = open(os.path.join(out2, "report.md")).read()
    assert "this run included): 6**" in rep2
    s2 = pd.read_csv(os.path.join(out2, "summary.csv")).set_index("id")
    assert s2.loc["base", "dsr"] <= s.loc["base", "dsr"] + 1e-12


def test_spa_size_under_h0_is_close_to_nominal():
    # measured at n=300: ~6% rejections at 5%, ~13% at 10% (slightly liberal; logged in the night log)
    ps = np.array([G.spa_test(np.random.default_rng(1000 + s).normal(0, 1, (300, 3)), reps=200, seed=s)["spa_p"]
                   for s in range(120)])
    assert (ps < 0.05).mean() <= 0.11


# ------------------------------------------------------------------ expected cone
def test_cone_constant_trades_is_exact():
    c = G.expected_cone([0.5] * 10, horizons=(25, 100), paths=200, seed=1)
    assert c["horizons"]["25"]["cum_R"] == pytest.approx([12.5] * 3)
    assert c["horizons"]["100"]["max_dd_R"] == pytest.approx([0.0] * 3)


def test_cone_known_shape_and_drawdown_definition():
    # +1 / -1 coin: cum R after 100 trades ~ N(0, 10) -> p5 ~ -16.4, p95 ~ +16.4; DD >= 0 and grows with N
    c = G.expected_cone([1.0, -1.0], paths=20000, seed=2)
    lo, mid, hi = c["horizons"]["100"]["cum_R"]
    assert lo == pytest.approx(-16.4, abs=1.0) and abs(mid) <= 1 and hi == pytest.approx(16.4, abs=1.0)
    dds = [c["horizons"][h]["max_dd_R"][1] for h in ("25", "50", "100", "200")]
    assert dds == sorted(dds) and dds[0] > 0
    # drawdown counts from the start: a first trade of -1 is already a 1R drawdown
    one = G.expected_cone([-1.0], horizons=(1,), paths=10, seed=0)
    assert one["horizons"]["1"]["max_dd_R"] == pytest.approx([1.0] * 3)


def test_cone_file_written_per_run(tmp_path):
    import json
    make_market(str(tmp_path), n_tickers=30, n_days=40, seed=22)
    out = B.main(["--daily", str(tmp_path / "daily.csv"), "--minutes", str(tmp_path / "minutes"),
                  "--start", "2019-01-23", "--end", "2019-12-31", "--nulls", "0", "--gate-reps", "20",
                  "--variants", _variants(tmp_path, [{"id": "b", "knobs": {"min_avg_vol": 0, "sides": "both"}}]),
                  "--out", str(tmp_path / "runs"), "--trial-log", str(tmp_path / "t.csv")])
    cone = json.load(open(os.path.join(out, "expected_cone.json")))
    c = cone["versions"]["b"]
    assert c["iid"]["paths"] == 5000 and set(c["day_block"]["horizons"]) == {"25", "50", "100", "200"}
    tr = pd.read_csv(os.path.join(out, "trades.csv"))
    tr = tr[tr["variant"] == "b"]
    assert c["iid"]["source_trades"] == c["day_block"]["source_trades"] == int(tr["filled"].sum())
    assert c["day_block"]["source_days"] == tr.loc[tr["filled"], "day"].nunique()
    assert "Expected cone for paper monitoring" in open(os.path.join(out, "report.md")).read()


def test_day_block_cone_is_wider_when_same_day_trades_move_together():
    rng = np.random.default_rng(9)
    days = np.repeat(np.arange(300), 5)
    r = np.repeat(rng.choice([1.0, -1.0], 300), 5) + rng.normal(0, 0.1, 1500)   # 5 trades/day, same sign
    iid = G.expected_cone(r, paths=3000, seed=1)["horizons"]["100"]["cum_R"]
    blk = G.expected_cone(r, paths=3000, seed=1, days=days)["horizons"]["100"]["cum_R"]
    assert (blk[2] - blk[0]) > 1.8 * (iid[2] - iid[0])                          # ~sqrt(5) wider
    one = G.expected_cone([1.0, 2.0, 3.0], horizons=(3,), paths=50, seed=0, days=[0, 0, 0])
    assert one["horizons"]["3"]["cum_R"] == pytest.approx([6.0] * 3)            # one day = one block, kept whole
