# Stock Momentum's

Intraday US-stock trading research system (paper trading first). Current focus: **Strategy B, Opening Range Breakout on "Stocks in Play"** (Zarattini, Barbon & Aziz 2024), with a loss-reduction track of 14 pre-registered versions.

**Status:** code and tests are ready (42 pass), but nothing has run on real market data yet. The next step is the Alpaca probe.

| Start here | |
|---|---|
| `CLAUDE.md` | Project context and rules (Claude Code reads it automatically) |
| `docs/strategy-B-data-runbook.md` | Exact commands: probe → download → validate → first backtest |
| `docs/strategy-B-final-plan-v3.md` | The plan: 14 versions + testing protocol |
| `docs/preregistration-B-draft.md` | Locked rules and pass gate (not signed yet) |
| `docs/build-log-strategy-B-mvb.md` | Reviews: bugs found and fixed, open issues |

```bash
cd strategy_b
pip install pandas numpy scipy matplotlib pytest
python -m pytest -q          # 42 passed
python fetch_b_data.py probe # needs APCA_API_KEY_ID / APCA_API_SECRET_KEY (free Alpaca account)
```

Never commit API keys: put them in `.env` (gitignored) or your shell environment.
