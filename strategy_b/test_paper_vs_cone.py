"""paper_vs_cone.py: paper (shadow-fill) trades placed inside the backtest's day-block cone."""
import os

import numpy as np
import pandas as pd

import paper_vs_cone as P


def _bt(n_days=400, per_day=5, mu=0.05, seed=0):
    rng = np.random.default_rng(seed)
    days = [f"2019-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n_days)]
    rows = [dict(day=d, entry_t=575 + j, r_net=rng.normal(mu, 1.0), filled=True,
                 reason="stop" if j % 2 else "close") for d in days for j in range(per_day)]
    return pd.DataFrame(rows)


def _paper(r, per_day=5, reason=None):
    p = pd.DataFrame(dict(day=[f"2026-12-{1 + i // per_day:02d}" for i in range(len(r))], r_net=r))
    if reason is not None:
        p["reason"] = reason
    return p


def test_typical_paper_sits_inside_and_extremes_are_flagged():
    bt = _bt()
    assert P.check(bt, _paper(bt["r_net"].to_numpy()[:50]))["verdict"] == "INSIDE the cone"
    assert P.check(bt, _paper(np.full(50, -1.0)))["verdict"].startswith("CHECK IMPLEMENTATION")
    assert P.check(bt, _paper(np.full(50, 1.5)))["verdict"].startswith("INVESTIGATE")


def test_numbers_and_exit_mix():
    bt = _bt()
    res = P.check(bt, _paper([1.0, -2.0, 0.5, -1.0], reason=["close", "close", "close", "stop"]))
    assert res["n"] == 4 and res["cum_R"] == -1.5 and res["max_dd_R"] == 2.5     # peak 1.0 -> trough -1.5
    assert abs(res["exit_mix_max_gap"] - 0.15) < 1e-12                           # close 75% vs 60% (j%2 over 5)
    assert res["trades_per_day_backtest"] == 5.0


def test_cli_writes_report(tmp_path):
    bt = _bt().assign(variant="B-v1__base")
    os.makedirs(tmp_path / "run")
    bt.to_csv(tmp_path / "run" / "trades.csv", index=False)
    _paper(bt["r_net"].to_numpy()[:30]).to_csv(tmp_path / "paper.csv", index=False)
    res = P.main(["--run", str(tmp_path / "run"), "--paper", str(tmp_path / "paper.csv")])
    assert res["n"] == 30 and "Paper vs expected cone" in open(tmp_path / "paper_vs_cone.md").read()
