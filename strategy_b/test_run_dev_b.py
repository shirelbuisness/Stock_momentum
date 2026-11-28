"""run_dev_b.py: DEV-B as one command, rehearsed against the fake Alpaca (no network, synthetic 2020-2021 market)."""
import json
import os

import pandas as pd
import pytest

import b_mvb as B
import fetch_b_data as F
import gate_b as GB
import run_dev_b as RB
from fake_alpaca import FakeAlpaca
from synth import make_market
from test_fetch_b_data import client_for

FROZEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variants_b.json")
PICK = ["B-v1__stop_atr=0.25", "B-v1__top_n=10"]           # a stop version + a filter


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("devb")
    mkt = str(root / "mkt")
    make_market(mkt, n_tickers=30, n_days=110, seed=23, momentum=0.1, start="2020-10-01")
    fake = FakeAlpaca(mkt)
    c = client_for(fake)
    data = str(root / "data")
    F.main(["assets", "--data", data], client=c)
    for st in ("daily", "opening", "fullday"):
        F.main([st, "--data", data, "--start", "2020-10-01", "--end", "2020-12-31"], client=c)
    F.main(["daily", "--data", data, "--start", "2021-01-01", "--end", "2021-03-31"], client=c)
    F.main(["opening", "--data", data, "--start", "2021-01-01", "--end", "2021-03-31"], client=c)
    dev_a = B.main(["--daily", os.path.join(data, "daily.csv"), "--minutes", os.path.join(data, "minutes"),
                    "--variants", FROZEN, "--start", "2020-11-02", "--end", "2020-12-31", "--nulls", "2",
                    "--stress", "--gate-reps", "20", "--out", str(root / "runs_a"),
                    "--trial-log", str(root / "trials_a.csv")])
    return dict(root=root, mkt=mkt, data=data, dev_a=dev_a)


def _args(s, tmp, *extra):
    return ["--dev-a", s["dev_a"], "--data", s["data"], "--runs", str(tmp / "runs"),
            "--trial-log", str(tmp / "trials.csv"), "--start", "2021-01-01", "--end", "2021-03-31",
            "--nulls", "2", "--gate-reps", "20", "--results", str(tmp / "results.md"), *extra]


def test_refuses_holdout_overlap_and_unsigned(setup, tmp_path):
    with pytest.raises(SystemExit, match="holdout"):
        RB.main(_args(setup, tmp_path, "--prereg-signed", "--end", "2024-01-02"), client=object())
    with pytest.raises(SystemExit, match="overlaps DEV-A"):
        RB.main(_args(setup, tmp_path, "--prereg-signed", "--start", "2020-12-01"), client=object())
    with pytest.raises(SystemExit, match="sign"):
        RB.main(_args(setup, tmp_path), client=object())
    assert not os.path.exists(tmp_path / "runs")


def test_empty_shortlist_runs_nothing(setup, tmp_path, monkeypatch):
    real = GB.shortlist
    monkeypatch.setattr(GB, "shortlist", lambda a: {**real(a), "shortlist": []})
    fake = FakeAlpaca(setup["mkt"])
    assert RB.main(_args(setup, tmp_path, "--prereg-signed"), client=client_for(fake)) is None
    assert "NOT RUN: the DEV-A shortlist is empty" in open(tmp_path / "results.md").read()
    assert not fake.requests and not os.path.exists(tmp_path / "trials.csv")      # no download, no trial


def test_subset_is_verbatim_and_frozen_file_untouched(tmp_path):
    before = open(FROZEN).read()
    ids = RB.subset_variants(FROZEN, list(reversed(PICK)), str(tmp_path / "v.json"))
    assert ids == ["B-v1__base"] + PICK                                           # frozen order, base first
    spec = {v["id"]: v for v in json.load(open(FROZEN))}
    assert json.load(open(tmp_path / "v.json")) == [spec[i] for i in ids]
    assert open(FROZEN).read() == before
    a_knobs = RB._run_knobs(_dev_a_dummy(tmp_path))
    assert RB.subset_variants(FROZEN, PICK, str(tmp_path / "x.json"), a_knobs) == ["B-v1__base"] + PICK
    a_knobs[PICK[0]]["stop_atr"] = 0.3                                           # DEV-A ran a different L1
    with pytest.raises(SystemExit, match="differ from the DEV-A"):
        RB.subset_variants(FROZEN, PICK, str(tmp_path / "y.json"), a_knobs)
    with pytest.raises(SystemExit, match="not in"):
        RB.subset_variants(FROZEN, ["B-v1__made_up"], str(tmp_path / "w.json"))


def _dev_a_dummy(tmp):
    """A config.json as b_mvb.py writes it, for the frozen versions."""
    from dataclasses import asdict
    d = tmp / "dummy_run"
    d.mkdir(exist_ok=True)
    vs = B.load_variants(FROZEN)
    json.dump(dict(args={}, variants=[dict(id=v.id, knobs=asdict(v.knobs), note=v.note) for v in vs]),
              open(d / "config.json", "w"))
    return str(d)


def test_full_rehearsal_with_a_forced_shortlist(setup, tmp_path, monkeypatch):
    real = GB.shortlist
    monkeypatch.setattr(GB, "shortlist", lambda a: {**real(a), "shortlist": list(PICK)})
    fake = FakeAlpaca(setup["mkt"])
    path = RB.main(_args(setup, tmp_path, "--prereg-signed"), client=client_for(fake))
    runs = sorted(d for d in os.listdir(tmp_path / "runs") if d.endswith("_B"))
    assert len(runs) == 2                                                         # first run + tick rerun
    run2 = tmp_path / "runs" / runs[1]
    s2 = pd.read_csv(run2 / "summary.csv")
    assert list(s2["id"]) == ["B-v1__base"] + PICK
    cfg = json.load(open(run2 / "config.json"))
    assert cfg["args"]["start"] == "2021-01-01" and cfg["args"]["ticks"]
    conf = json.load(open(run2 / "gate_confirm.json"))
    assert conf["shortlist"] == PICK and {r["id"] for r in conf["rows"]} == set(PICK)
    # the confirm verdict equals gate_b.confirm on the same two runs (nothing re-interpreted)
    again = GB.confirm(GB.load_run(setup["dev_a"]), GB.load_run(str(run2)))
    assert again["promoted"] == conf["promoted"]
    # the stop version was judged on a tick-resolved run, so the R9 check is absent, not failed
    stop_row = next(r for r in conf["rows"] if r["id"] == PICK[0])
    assert not any("R9" in c for c in stop_row["checks"])
    assert any("does not judge the whole search" in n for n in conf["notes"])     # subset PBO is labelled
    log = pd.read_csv(tmp_path / "trials.csv")
    assert len(log) == 2 * (3 + 2)                                                # 3 versions + shadow + replication
    doc = open(path).read()
    for sec in ("## 1. Data gate", "## 3. Sanity gate", "## 4. Versions (DEV-B)", "## 5. Mechanical confirmation",
                "Promoted"):
        assert sec in doc, sec
    assert not B.TICKS
    assert {p for p, _ in fake.requests} <= {"/v2/stocks/bars", "/v1/corporate-actions", "/v2/assets",
                                             "/v2/stocks/trades"}
    days = pd.read_csv(run2 / "trades.csv")["day"]
    assert days.min() >= "2021-01-01" and days.max() <= "2021-03-31"
