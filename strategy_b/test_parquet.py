"""Optional Parquet storage: a backtest on the Parquet copy must equal the backtest on the CSV source."""
import os

import pandas as pd
import pytest

import b_mvb as B
from synth import make_market

pytest.importorskip("pyarrow")
import to_parquet as P  # noqa: E402


def _run(tmp_path, daily, minutes, tag):
    return B.main(["--daily", daily, "--minutes", minutes, "--start", "2019-01-23", "--end", "2019-12-31",
                   "--nulls", "2", "--stress", "--gate-reps", "30", "--out", str(tmp_path / f"runs_{tag}"),
                   "--trial-log", str(tmp_path / f"t_{tag}.csv")])


def test_parquet_backtest_equals_csv_backtest(tmp_path):
    src = tmp_path / "csv"
    make_market(str(src), n_tickers=30, n_days=40, seed=61, momentum=0.1)
    r = P.convert(str(src), str(tmp_path / "pq"))
    assert r["written"] == 41 and r["parquet_bytes"] < r["csv_bytes"]
    assert P.convert(str(src), str(tmp_path / "pq"))["skipped"] == 41          # resumable
    a = _run(tmp_path, str(src / "daily.csv"), str(src / "minutes"), "csv")
    b = _run(tmp_path, str(tmp_path / "pq" / "daily.parquet"), str(tmp_path / "pq" / "minutes"), "pq")
    ta, tb = pd.read_csv(f"{a}/trades.csv"), pd.read_csv(f"{b}/trades.csv")
    assert ta["filled"].sum() > 100
    pd.testing.assert_frame_equal(ta, tb)
    sa, sb = pd.read_csv(f"{a}/summary.csv"), pd.read_csv(f"{b}/summary.csv")
    pd.testing.assert_frame_equal(sa, sb)
    assert open(f"{a}/report.md").read().split("\n", 3)[3] == open(f"{b}/report.md").read().split("\n", 3)[3]


def test_mixed_formats_are_refused(tmp_path):
    src = tmp_path / "csv"
    make_market(str(src), n_tickers=3, n_days=3, seed=1)
    pd.read_csv(src / "minutes" / "minutes_2019-01-02.csv").to_parquet(src / "minutes" / "minutes_2019-01-07.parquet")
    with pytest.raises(SystemExit, match="both CSV and Parquet"):
        B.minute_files(str(src / "minutes"))
