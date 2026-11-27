# Strategy B: data runbook (Alpaca free plan, SIP) → first real number
Code: `strategy_b/fetch_b_data.py` (downloader), `strategy_b/b_mvb.py` (backtester). Tests: all `strategy_b/test_*.py`, **136 pass** against an in-process fake Alpaca (`strategy_b/fake_alpaca.py`). It has never been run against the real API; step 1 below is that check. Run every command from inside `strategy_b/`.

## 0. Setup (once)
```
pip install -r requirements.txt      # from the repo root
export APCA_API_KEY_ID=...      # free Alpaca account (paper is fine)
export APCA_API_SECRET_KEY=...
python -m pytest -q             # must print "136 passed" (or 135 + 1 skipped without pyarrow) before touching the real API
```
(Windows PowerShell: `$env:APCA_API_KEY_ID="..."` instead of `export`.)

## One command (steps 1–5 below, in order)
```
python run_dev_a.py          # probe → assets → daily → opening → fullday → validate → DEV-A → ticks → rerun → shortlist
```
- Stops at a probe NO-GO (exit 2) or a failed data gate, and still writes the draft.
- Writes a DRAFT `docs/results-DEV-A.md` from the run's own files (probe output, data gate, sanity gate, base, versions, SPA/PBO, nulls, market beta, shortlist). Read it together; it is not a verdict.
- DEV-A runs use `--nulls 20 --stress --replication` (+ `--securities` when present); `--breakeven` is optional (≈ 5× runtime).
- No holdout option exists in this tool; `--dl-end` / `--end` ≥ 2024 are refused.
- Rehearsed end-to-end against the fake Alpaca in `test_run_dev_a.py`.

## 1. Probe: 5 minutes, decides GO / NO-GO
`python fetch_b_data.py probe`

It checks:
- SIP 1-min bars for 2016;
- multi-symbol paging;
- 3 delisted names (TWTR, SIVB, BBBY) with the pipeline's symbol mapping;
- one 500-symbol request;
- a split from corporate actions.

If it fails:
- **delisted names show "ONLY WITH asof"** → the pipeline will lose delisted names (survivorship). Stop and decide: accept it as "survivor-biased", or add Massive Basic inactive tickers.
- **the 500-symbol request fails** → change `batch=500` to 200 (the run takes about 2.5× longer).

## 2. Download, in order (each stage resumes where it stopped)
| Step | Command | Calls (≈) | Time at 190/min | Output |
|---|---|---|---|---|
| assets | `python fetch_b_data.py assets` | 2 | seconds | `data/securities.csv`, `README_data.md` |
| daily | `python fetch_b_data.py daily --start 2015-11-01 --end 2023-12-31` | ~2k | ~10 min | `data/daily.csv` (raw + split_factor) |
| opening | `python fetch_b_data.py opening --start 2015-11-01 --end 2023-12-31` | ~18k | ~1.6 h | `data/minutes/minutes_<day>.csv` (09:30–09:44, loose universe) |
| fullday | `python fetch_b_data.py fullday --start 2016-01-01 --end 2023-12-31` | ~12k | ~1.1 h | full-day bars: top-100 RVOL + **30 seeded random** + SPY; `selected_<day>.csv` |
| validate | `python fetch_b_data.py validate` | 0 | minutes | `data/validation.csv`; **DATA GATE PASSED** or it stops |

**Total: about 3 hours unattended, about 5–10 GB of CSV.**

- **Holdout guard:** the downloader refuses dates ≥ 2024 without `--open-holdout`. Having the data on disk is how accidental peeks happen.

## 3. First real number (DEV-A)
```
python b_mvb.py --daily data/daily.csv --minutes data/minutes --variants variants_b.json \
       --start 2016-01-01 --end 2020-12-31 --nulls 20 --stress
```
- Expected runtime: 1–3 h.
- **Read first:** the "Data diagnostics" section (data-gap days, the null pool), then the base row: mean R net, the same-bar share, and **mean R if same-bar = OHLC path**.

- **New in the report (night 27.11):** "Gate statistics" (N trials, DSR per version, random-skip p for filters, Hansen SPA, PBO), breakeven in bp as well as ticks (`--breakeven`), scratches apart from wins/losses, the NYSE calendar used (bundled by default; half days exit on the 12:58 bar; a file ending 12:59 on a full day = data gap), the short shadow (add `--securities data/securities.csv` for ETB flags) and the expected cone (`expected_cone.json`).

## 4. Resolve the ambiguous minutes with real prints (finding R9)
```
python fetch_b_data.py ticks --trades runs/<DEV-A run>/trades.csv
python b_mvb.py ... same as step 3 ... --ticks data/ticks.csv
```
- Calls: about 1 per (day, entry minute) with an ambiguous trade, a few thousand.
- The report column "of which resolved by ticks %" should be close to 100%. Whatever is not resolved stays pessimistic.
- **Decide between base and the stop variants only on the tick-resolved run.**

## 5. Shortlist, confirm, pass gate (mechanical)
```
python gate_b.py shortlist --dev-a runs/<DEV-A tick-resolved run>
python gate_b.py confirm   --dev-a runs/<DEV-A run> --dev-b runs/<DEV-B run with --stress>
# only after signing, on the ONE holdout run (needs --stress --breakeven --nulls 20 --nulls-all-versions):
python gate_b.py holdout   --run runs/<holdout run> --net-r-bar 0.05|0.08 --breakeven-ticks-bar 1|2
```
- `holdout` has no default for the two open prereg choices. `--rehearse` runs it on a DEV run as a dress rehearsal.
- New report sections (night 27.11, session 2): **B0-replication sanity gate** (`--replication`; long + short, commission only, Sharpe ≥ 2.0 over DEV), **market exposure** of the base vs SPY (beta, SPY-down days, F8 red-signal days), mean net R without the top-5 tickers.
Stop versions are only shortlisted from a `--ticks` run (R9); SPA p > 0.10 empties the shortlist (plan v3).

## 6. Optional tools
- `python to_parquet.py --src data --dst data_parquet` → then `--daily data_parquet/daily.parquet --minutes data_parquet/minutes` (identical results, smaller files; needs pyarrow).
- `python dry_run_b.py --data data --day 2019-02-22` → the 09:35 order intents for a DEV day, written to a file (no broker code).
- `python paper_vs_cone.py --run runs/<run> --paper paper_trades.csv` → paper shadow fills vs the day-block cone.

## Design decisions in the downloader (and why)
- **One symbol mapping everywhere (Alpaca default = today's ticker).** Mixing in per-day `asof` labels the same company differently across files (a boss-review finding; a test guards it).
- **Loose universe** (prior close ≥ $3, 14-day avg volume ≥ 300k) is downloaded for the opening minutes, so the RVOL baseline exists before a name enters the paper's $5 / 1M universe.
- **30 random eligible names/day get full-day bars**, so the random-names null is truly random (Round 1, R1).
- **Ticks:** canceled or corrected prints (`u` field) are dropped. The entry = the first print ≥ trigger + 1 tick; "stopped" = a later print ≤ stop within that minute.
- **Atomic writes + manifest:** a crash never leaves a half file marked done.

## Known limits (not fixed)
- **Survivorship:** Alpaca's asset list is today's view (active + inactive). Names delisted long ago may be missing. The probe measures this.
- **Ticker reuse** under today's mapping can mix two companies (Alpaca forum case "Q"). Spot-check suspicious names.
- **ETF filter is a name heuristic.** It also removes some "Trust" operating companies.
- **Disk:** CSV is large. `to_parquet.py` makes an optional Parquet copy (needs pyarrow).
- **Daily end date:** the daily stage now sends explicit NY timestamps; a date-only end could drop the end day (Dec 31 is a trading day in 2015 and 2018–2021).
