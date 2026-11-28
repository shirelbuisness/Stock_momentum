#!/usr/bin/env python3
"""nyse_calendar.py: NYSE full holidays and 13:00 early closes, 2015-2023 (Strategy B's DEV range).

Rule-based (NYSE Rule 7.2 holiday rules + Good Friday from the Gregorian Easter computus) plus the one special
closure in range (2018-12-05, national day of mourning for President G.H.W. Bush). The test file checks every
year against the hand-listed official dates, so a rule bug cannot hide.

Early closes (13:00 ET) in this range happen only on: July 3 (when it is a trading day), the day after
Thanksgiving, and Dec 24 (when it is a trading day).

Outside 2015-2023 the functions refuse: special closures are not encoded there (e.g. 2025-01-09, Carter), and the
holdout years must not be prepared early anyway.

CLI: python nyse_calendar.py [--out nyse_early_closes.csv]  -> CSV date,close (the format of b_mvb.py --early-closes)
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

COVERED = (2015, 2023)
SPECIAL_CLOSURES = {date(2018, 12, 5)}       # G.H.W. Bush national day of mourning (full closure)
EARLY_CLOSE_TIME = "13:00"


def _check(year: int):
    if not COVERED[0] <= year <= COVERED[1]:
        raise ValueError(f"nyse_calendar covers {COVERED[0]}-{COVERED[1]} only (asked {year})")


def easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous / Meeus-Jones-Butcher algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year, month, weekday, n):
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year, month, weekday):
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date | None:
    """Saturday -> Friday before, Sunday -> Monday after. NYSE exception: New Year's Day on a Saturday is NOT
    observed on Friday Dec 31 (handled by the caller)."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def holidays(year: int) -> set[date]:
    _check(year)
    h = set()
    ny = date(year, 1, 1)
    if ny.weekday() != 5:                    # Saturday New Year: no observance (Rule 7.2)
        h.add(_observed(ny))
    h.add(_nth_weekday(year, 1, 0, 3))       # MLK: 3rd Monday of January
    h.add(_nth_weekday(year, 2, 0, 3))       # Washington's Birthday: 3rd Monday of February
    h.add(easter(year) - timedelta(days=2))  # Good Friday
    h.add(_last_weekday(year, 5, 0))         # Memorial Day: last Monday of May
    if year >= 2022:
        h.add(_observed(date(year, 6, 19)))  # Juneteenth (NYSE from 2022)
    h.add(_observed(date(year, 7, 4)))
    h.add(_nth_weekday(year, 9, 0, 1))       # Labor Day: 1st Monday of September
    h.add(_nth_weekday(year, 11, 3, 4))      # Thanksgiving: 4th Thursday of November
    h.add(_observed(date(year, 12, 25)))
    h |= {d for d in SPECIAL_CLOSURES if d.year == year}
    return h


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in holidays(d.year)


def early_closes(year: int) -> set[date]:
    _check(year)
    out = set()
    for d in (date(year, 7, 3), _nth_weekday(year, 11, 3, 4) + timedelta(days=1), date(year, 12, 24)):
        if is_trading_day(d):
            out.add(d)
    return out


def trading_days(start: str, end: str) -> list[str]:
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    out, d = [], a
    while d <= b:
        if is_trading_day(d):
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def early_close_map(start: str = "2015-01-01", end: str = "2023-12-31") -> dict[str, str]:
    """{YYYY-MM-DD: "13:00"} for every early close in [start, end] (clipped to the covered years)."""
    y0 = max(int(start[:4]), COVERED[0])
    y1 = min(int(end[:4]), COVERED[1])
    out = {}
    for y in range(y0, y1 + 1):
        for d in early_closes(y):
            if start <= d.isoformat() <= end:
                out[d.isoformat()] = EARLY_CLOSE_TIME
    return dict(sorted(out.items()))


def holiday_set(start: str = "2015-01-01", end: str = "2023-12-31") -> set[str]:
    y0 = max(int(start[:4]), COVERED[0])
    y1 = min(int(end[:4]), COVERED[1])
    return {d.isoformat() for y in range(y0, y1 + 1) for d in holidays(y) if start <= d.isoformat() <= end}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="nyse_early_closes.csv")
    a = ap.parse_args()
    with open(a.out, "w") as f:
        f.write("date,close\n")
        for d, c in early_close_map().items():
            f.write(f"{d},{c}\n")
    print(f"wrote {a.out}")
