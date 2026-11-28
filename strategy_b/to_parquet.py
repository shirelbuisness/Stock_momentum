#!/usr/bin/env python3
"""to_parquet.py: OPTIONAL Parquet copies of Strategy B's CSV data (5-10x smaller, faster to read).

CSV stays the source of truth (the downloader writes CSV; validate reads CSV). This writes a parallel tree:
  <src>/daily.csv              -> <dst>/daily.parquet
  <src>/minutes/minutes_D.csv  -> <dst>/minutes/minutes_D.parquet
Each file is read with the SAME dtypes b_mvb.py uses for CSV and written as-is, so a backtest on the Parquet tree
gives identical results (test_parquet.py). Resumable: an existing, newer Parquet file is skipped.

Usage: python to_parquet.py --src data --dst data_parquet
       python b_mvb.py --daily data_parquet/daily.parquet --minutes data_parquet/minutes ...
Needs pyarrow (pip install pyarrow); everything else in the project works without it.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd


def convert(src: str, dst: str) -> dict:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        sys.exit("pyarrow is not installed: pip install pyarrow (Parquet is optional; CSV keeps working)")
    os.makedirs(os.path.join(dst, "minutes"), exist_ok=True)
    jobs = [(os.path.join(src, "daily.csv"), os.path.join(dst, "daily.parquet"), {"ticker": str})]
    for p in sorted(glob.glob(os.path.join(src, "minutes", "minutes_*.csv"))):
        jobs.append((p, os.path.join(dst, "minutes", os.path.basename(p)[:-4] + ".parquet"),
                     {"ticker": str, "time": str}))
    done = skipped = 0
    csv_bytes = pq_bytes = 0
    for s, d, dtype in jobs:
        if not os.path.exists(s):
            continue
        csv_bytes += os.path.getsize(s)
        if os.path.exists(d) and os.path.getmtime(d) >= os.path.getmtime(s):
            skipped += 1
        else:
            tmp = d + ".tmp"
            pd.read_csv(s, dtype=dtype).to_parquet(tmp, index=False)
            os.replace(tmp, d)                       # atomic: never a half-written file under the final name
            done += 1
        pq_bytes += os.path.getsize(d)
    print(f"parquet: {done} written, {skipped} up to date; {csv_bytes / 1e6:.1f} MB CSV -> {pq_bytes / 1e6:.1f} MB")
    return dict(written=done, skipped=skipped, csv_bytes=csv_bytes, parquet_bytes=pq_bytes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data")
    ap.add_argument("--dst", default="data_parquet")
    a = ap.parse_args()
    convert(a.src, a.dst)
