"""run_dev_a.py: the whole runbook as one command, rehearsed end-to-end against the fake Alpaca (no network)."""
import os

import pandas as pd
import pytest

import b_mvb as B
import run_dev_a as RD
from fake_alpaca import FakeAlpaca
from synth import make_market
from test_fetch_b_data import client_for


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    root = tmp_path_factory.mktemp("mkt")
    make_market(str(root), n_tickers=30, n_days=45, seed=17, momentum=0.1, start="2019-01-02")
    return str(root)


def _args(tmp, *extra):
    return ["--data", str(tmp / "data"), "--runs", str(tmp / "runs"), "--trial-log", str(tmp / "trials.csv"),
            "--dl-start", "2018-12-01", "--dl-end", "2019-12-31", "--start", "2019-02-01", "--end", "2019-12-31",
            "--nulls", "2", "--gate-reps", "20", "--results", str(tmp / "results.md"), *extra]


def test_probe_no_go_stops_before_any_download(market, tmp_path):
    with pytest.raises(SystemExit) as e:
        RD.main(_args(tmp_path), client=client_for(FakeAlpaca(market)))       # the fake has no 2016 -> NO-GO
    assert e.value.code == 2
    doc = open(tmp_path / "results.md").read()
    assert "NO-GO at the probe" in doc and "NO-GO" in doc.split("## 1. Probe")[1]
    assert not os.path.exists(tmp_path / "data" / "daily.csv")


def test_full_rehearsal_with_skipped_probe(market, tmp_path):
    fake = FakeAlpaca(market)
    path = RD.main(_args(tmp_path, "--skip-probe"), client=client_for(fake))
    runs = sorted(os.listdir(tmp_path / "runs"))
    assert len(runs) == 2                                                        # first run + tick rerun
    run2 = tmp_path / "runs" / runs[1]
    assert (run2 / "gate_shortlist.md").exists() and (tmp_path / "data" / "ticks.csv").exists()
    s2 = pd.read_csv(run2 / "summary.csv")
    assert len(s2) == 15 and s2.loc[0, "ambiguous_resolved_by_ticks"] == pytest.approx(1.0)
    log = pd.read_csv(tmp_path / "trials.csv")
    assert len(log) == 2 * (15 + 2)                                              # versions + shadow + replication
    doc = open(path).read()
    for sec in ("probe skipped", "## 2. Data gate", "DATA GATE PASSED", "## 4. Sanity gate", "## 5. Base",
                "## 6. Versions vs base", "## 7. Market exposure", "## 8. Mechanical shortlist", "probe SKIPPED"):
        assert sec in doc, sec
    assert not B.TICKS                                                           # no tick state leaks out
    # only market data + the asset list were requested: nothing that could place an order
    assert {p for p, _ in fake.requests} <= {"/v2/stocks/bars", "/v1/corporate-actions", "/v2/assets",
                                             "/v2/stocks/trades"}


def test_refuses_the_holdout_and_missing_keys(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="holdout"):
        RD.main(_args(tmp_path)[:-2] + ["--results", "x.md", "--dl-end", "2024-03-01"], client=object())
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    with pytest.raises(SystemExit, match="APCA_API_KEY_ID"):
        RD.main(_args(tmp_path))
