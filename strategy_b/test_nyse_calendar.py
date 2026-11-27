"""NYSE calendar 2015-2023 against hand-listed official dates (independent of the rules in nyse_calendar.py)."""
from datetime import date

import pandas as pd
import pytest

import b_mvb as B
import nyse_calendar as C
from synth import make_market

HOLIDAYS = {
    2015: ["01-01", "01-19", "02-16", "04-03", "05-25", "07-03", "09-07", "11-26", "12-25"],
    2016: ["01-01", "01-18", "02-15", "03-25", "05-30", "07-04", "09-05", "11-24", "12-26"],
    2017: ["01-02", "01-16", "02-20", "04-14", "05-29", "07-04", "09-04", "11-23", "12-25"],
    2018: ["01-01", "01-15", "02-19", "03-30", "05-28", "07-04", "09-03", "11-22", "12-05", "12-25"],
    2019: ["01-01", "01-21", "02-18", "04-19", "05-27", "07-04", "09-02", "11-28", "12-25"],
    2020: ["01-01", "01-20", "02-17", "04-10", "05-25", "07-03", "09-07", "11-26", "12-25"],
    2021: ["01-01", "01-18", "02-15", "04-02", "05-31", "07-05", "09-06", "11-25", "12-24"],
    2022: ["01-17", "02-21", "04-15", "05-30", "06-20", "07-04", "09-05", "11-24", "12-26"],
    2023: ["01-02", "01-16", "02-20", "04-07", "05-29", "06-19", "07-04", "09-04", "11-23", "12-25"],
}
EARLY = {
    2015: ["11-27", "12-24"], 2016: ["11-25"], 2017: ["07-03", "11-24"], 2018: ["07-03", "11-23", "12-24"],
    2019: ["07-03", "11-29", "12-24"], 2020: ["11-27", "12-24"], 2021: ["11-26"], 2022: ["11-25"],
    2023: ["07-03", "11-24"],
}
SESSIONS = {2015: 252, 2016: 252, 2017: 251, 2018: 251, 2019: 252, 2020: 253, 2021: 252, 2022: 251, 2023: 250}


@pytest.mark.parametrize("year", range(2015, 2024))
def test_holidays_match_official_list(year):
    assert sorted(d.isoformat() for d in C.holidays(year)) == [f"{year}-{x}" for x in HOLIDAYS[year]]


@pytest.mark.parametrize("year", range(2015, 2024))
def test_early_closes_match_official_list(year):
    assert sorted(d.isoformat() for d in C.early_closes(year)) == [f"{year}-{x}" for x in EARLY[year]]


def test_session_counts_per_year():
    for y, n in SESSIONS.items():
        assert len(C.trading_days(f"{y}-01-01", f"{y}-12-31")) == n, y


def test_good_friday_and_saturday_new_year_rule():
    assert C.easter(2019) == date(2019, 4, 21) and C.easter(2016) == date(2016, 3, 27)
    assert date(2021, 12, 31) not in C.holidays(2021)            # Jan 1 2022 = Saturday: not observed


def test_refuses_years_outside_coverage():
    with pytest.raises(ValueError):
        C.holidays(2024)
    assert set(C.early_close_map("2023-06-01", "2025-12-31")) == {"2023-07-03", "2023-11-24"}  # clipped


def test_committed_csv_equals_generator():
    import os
    p = os.path.join(os.path.dirname(C.__file__), "nyse_early_closes.csv")
    csv = pd.read_csv(p, dtype=str)
    assert dict(zip(csv["date"], csv["close"])) == C.early_close_map()


# ------------------------------------------------------------------ use in b_mvb.py
def test_calendar_session_close_rules():
    m_full = pd.DataFrame({"tmin": [B.hm("09:30"), B.hm("15:59")]})
    m_half = pd.DataFrame({"tmin": [B.hm("09:30"), B.hm("12:59")]})
    with B.calendar_context({"2019-11-29": "13:00"}, strict=True):
        assert B.session_close_bar("2019-11-29", m_half) == (B.hm("12:58"), "calendar")
        assert B.session_close_bar("2019-11-29", m_full) == (B.hm("12:58"), "calendar")  # calendar wins
        assert B.session_close_bar("2019-11-29", pd.DataFrame({"tmin": [B.hm("09:30"), B.hm("11:00")]}))[0] is None
        assert B.session_close_bar("2019-11-27", m_half) == (None, "data_gap")  # 12:59 end on a FULL day = gap
        assert B.session_close_bar("2019-11-27", m_full) == (B.hm("15:58"), "full")
    # outside the context: the old inference is back (no state leaks between runs)
    assert B.session_close_bar("2019-11-27", m_half) == (B.hm("12:58"), "half_inferred")


def test_main_uses_bundled_calendar_and_reports_it(tmp_path):
    # a synthetic market over Thanksgiving 2019: the day-after file is a FULL day in the synthetic data,
    # but the calendar says 13:00 close -> every trade that day exits by 12:58
    make_market(str(tmp_path), n_tickers=30, n_days=45, seed=41, start="2019-10-21")
    out = B.main(["--daily", str(tmp_path / "daily.csv"), "--minutes", str(tmp_path / "minutes"),
                  "--start", "2019-11-11", "--end", "2019-12-31", "--nulls", "0", "--gate-reps", "50",
                  "--variants", _v(tmp_path), "--out", str(tmp_path / "runs"), "--trial-log", str(tmp_path / "t.csv")])
    tr = pd.read_csv(f"{out}/trades.csv")
    f = tr[(tr["day"] == "2019-11-29") & tr["filled"]]
    assert len(f) > 0 and (f["exit_t"] <= B.hm("12:58")).all()
    rep = open(f"{out}/report.md").read()
    assert "NYSE calendar: bundled 2015-2023" in rep
    assert "Minute files on NYSE holidays: 1" in rep                   # 2019-11-28 Thanksgiving (synthetic bdays)
    assert not B.EARLY_CLOSES and not B.CAL_STATE["strict"]          # restored after main()


def _v(tmp_path):
    import json
    p = tmp_path / "v.json"
    p.write_text(json.dumps([{"id": "b", "knobs": {"min_avg_vol": 0, "sides": "both"}}]))
    return str(p)
