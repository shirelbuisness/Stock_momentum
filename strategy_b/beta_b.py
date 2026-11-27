"""beta_b.py: market exposure of long-only Strategy B (plan v3 §5, regime/portfolio researcher).

Long-only B is net long, so red market days may hit all 20 names together. Before anything is decided about F8
(SPY filter) or about pairing B with C, measure it. Pure functions, no I/O. Used by b_mvb.py's report.

Inputs (one value per trading day of the run; days without trades count as 0 for the strategy):
  strat_ret : base 1x daily net return (equal notional 1/top_n, the same series as Sharpe 1x)
  strat_R   : base daily net R sum (equal $ risk per trade)
  spy_oc    : SPY open -> close return of the same day (raw daily bars; 16:00 close, while B exits 15:58)
  spy_sig   : sign of SPY prior close -> close of the first OR bar (exactly F8's signal), +1 / -1 / 0

Outputs:
  beta, alpha/day, corr, Newey-West t of beta (5 lags) from OLS strat_ret = a + b * spy_oc;
  mean daily R and day-level t on SPY-down vs SPY-up days, and on the worst 5% SPY days;
  F8 diagnostic: mean daily R on F8 red-signal days (sig < 0). Plan v3 §5: F8 helps only if this is negative
  after costs. The skip / half-size totals are DIAGNOSTICS on the base's own trades, never a version or a verdict.
"""
from __future__ import annotations

import math

import numpy as np


def _nw_se_beta(x: np.ndarray, resid: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett) standard error of the slope in y = a + b x."""
    n = len(x)
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    if n < 3 or sxx <= 0:
        return float("nan")
    u = xc * resid
    s = float((u ** 2).sum())
    for L in range(1, min(lags, n - 1) + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * float((u[L:] * u[:-L]).sum())
    return math.sqrt(max(s, 0.0)) / sxx


def _mean_t(x: np.ndarray) -> tuple[float, float, int]:
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = float(x.mean())
    if n < 2:
        return m, float("nan"), n
    sd = float(x.std(ddof=1))
    return m, (m / (sd / math.sqrt(n)) if sd > 0 else float("nan")), n


def market_exposure(strat_ret, strat_R, spy_oc, spy_sig=None, lags: int = 5) -> dict:
    y = np.asarray(strat_ret, dtype=float)
    R = np.asarray(strat_R, dtype=float)
    x = np.asarray(spy_oc, dtype=float)
    sig = np.asarray(spy_sig, dtype=float) if spy_sig is not None else np.full(len(x), np.nan)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(R)
    out: dict = {"days": int(ok.sum()), "days_without_spy": int((~np.isfinite(x)).sum())}
    if ok.sum() < 3:
        return out
    y, R, x, sig = y[ok], R[ok], x[ok], sig[ok]
    xc = x - x.mean()
    b = float((xc * (y - y.mean())).sum() / (xc ** 2).sum()) if (xc ** 2).sum() > 0 else float("nan")
    a = float(y.mean() - b * x.mean())
    resid = y - a - b * x
    se = _nw_se_beta(x, resid, lags)
    out.update(beta=b, alpha_day=a, beta_t_nw=b / se if se and np.isfinite(se) and se > 0 else float("nan"),
               corr=float(np.corrcoef(x, y)[0, 1]) if y.std() > 0 and x.std() > 0 else float("nan"))
    for name, mask in (("spy_down", x < 0), ("spy_up", x > 0),
                       ("spy_worst5pct", x <= np.quantile(x, 0.05))):
        m, t, n = _mean_t(R[mask])
        out[f"R_day_{name}"], out[f"t_{name}"], out[f"n_{name}"] = m, t, n
    fin = np.isfinite(sig)
    if fin.any():
        red = fin & (sig < 0)
        m, t, n = _mean_t(R[red])
        out.update(R_day_f8_red=m, t_f8_red=t, n_f8_red=n,
                   total_R=float(R.sum()),
                   total_R_skip_red=float(R[~red].sum()),
                   total_R_half_red=float(R[~red].sum() + 0.5 * R[red].sum()))
    return out
