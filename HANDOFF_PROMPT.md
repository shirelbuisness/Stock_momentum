Paste this as your first message to Claude Code, from inside the stock-momentum folder:

---

Hi. This is my trading project "Stock Momentum's", moved here from claude.ai. Please:
1. Read CLAUDE.md, then docs/strategy-B-data-runbook.md and docs/build-log-strategy-B-mvb.md.
2. Load the skills in .claude/skills (trading-system-conventions, backtest-and-experiment-rigor).
3. Set up git in this folder (git init, first commit; data/ and runs/ and .env stay out of git).
4. Install what's missing (pandas numpy scipy matplotlib pytest) and run `cd strategy_b && python -m pytest -q`. Tell me the result.
5. Then help me run `python fetch_b_data.py probe` with my Alpaca keys, and explain the output to me in Hebrew.
Don't start the long downloads until we have looked at the probe together.
