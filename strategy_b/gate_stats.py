"""gate_stats.py: multiple-testing / overfitting statistics for the Strategy B gates (plan v3 §4, prereg rev 3).

Pure functions (numpy + scipy), no I/O, every random draw seeded. Used by b_mvb.py's report.

- deflated_sharpe(): Bailey & López de Prado (2014). Probability that the true Sharpe > the expected MAXIMUM
  Sharpe of N unskilled trials, given the observed Sharpe, sample length, skew and kurtosis. N and the
  cross-trial Sharpe variance come from experiments/trials.csv (every run ever logged).
- random_skip_null(): a filter must beat skipping the SAME number of base trades (or the same share of base
  days) at random. Skipping days removes losers by luck; this measures how often luck does as well.
- spa_test(): Hansen (2005) Superior Predictive Ability, consistent p-value, stationary bootstrap.
  H0: no variant beats the base on mean daily net R.
- pbo_cscv(): Bailey, Borwein, López de Prado & Zhu (2017) Probability of Backtest Overfitting via
  Combinatorially Symmetric Cross-Validation over the variants' daily net R.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


# ----------------------------------------------------------------------------- DSR
def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max of N Sharpe estimates] under H0 (all true Sharpes = 0), in the units of var_sr's Sharpe.
    Returns 0 when N <= 1 or the variance is unknown (then the DSR reduces to the PSR against 0)."""
    if n_trials is None or n_trials <= 1 or not np.isfinite(var_sr) or var_sr <= 0:
        return 0.0
    n = float(n_trials)
    z1 = stats.norm.ppf(1 - 1 / n)
    z2 = stats.norm.ppf(1 - 1 / (n * math.e))
    return math.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def probabilistic_sharpe(sr: float, sr0: float, t_obs: int, skew: float, kurt: float) -> float:
    """PSR: P(true SR > sr0). sr, sr0 per period (NOT annualised); kurt = Pearson kurtosis (normal = 3)."""
    if t_obs is None or t_obs < 3 or not np.isfinite(sr):
        return float("nan")
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    z = (sr - sr0) * math.sqrt(t_obs - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(z))


def deflated_sharpe(daily_returns, n_trials: int, var_sr_trials: float) -> dict:
    """DSR for one strategy. daily_returns: per-day series INCLUDING days with no trades (= 0).
    var_sr_trials: variance of the per-day (non-annualised) Sharpe across all logged trials."""
    x = np.asarray(daily_returns, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 3 or x.std(ddof=1) == 0:
        return dict(sr_day=float("nan"), sr0_day=float("nan"), dsr=float("nan"), n_trials=n_trials, T=T)
    sr = x.mean() / x.std(ddof=1)
    skew = float(stats.skew(x, bias=False)) if T > 3 else 0.0
    kurt = float(stats.kurtosis(x, fisher=False, bias=False)) if T > 4 else 3.0
    sr0 = expected_max_sharpe(n_trials, var_sr_trials)
    return dict(sr_day=float(sr), sr0_day=float(sr0), dsr=probabilistic_sharpe(sr, sr0, T, skew, kurt),
                n_trials=int(n_trials), T=T, skew=skew, kurt=kurt)


def trial_sharpe_stats(annual_sharpes, extra=()) -> tuple[int, float]:
    """N and the cross-trial variance of the PER-DAY Sharpe from the trial log's annualised sharpe_1x column
    (+ `extra` = this run's rows not yet written). N counts every row, even ones without a finite Sharpe."""
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")
    allv = [num(v) for v in list(annual_sharpes) + list(extra)]
    n = len(allv)
    s = np.array([v for v in allv if np.isfinite(v)]) / math.sqrt(252)
    var = float(s.var(ddof=1)) if len(s) >= 2 else float("nan")
    return n, var


# ----------------------------------------------------------------------------- random-skip null
def random_skip_null(base_r, var_r, base_day=None, var_days: int | None = None, reps: int = 1000,
                     seed: int = 0) -> dict:
    """Does the variant's mean net R per trade beat a random subset of the base trades of the same size?

    trade-level: draw len(var_r) of the base trades without replacement.
    day-level  : draw the same NUMBER OF DAYS as the variant traded from the base's traded days, keep all base
                 trades on those days (only when base_day and var_days are given).
    p = (1 + #{null >= variant}) / (reps + 1). Only defined when the variant keeps FEWER trades than the base
    (a filter); for same-size versions (exit rules) the null is degenerate -> NaN."""
    base_r = np.asarray(base_r, dtype=float)
    var_r = np.asarray(var_r, dtype=float)
    out = dict(rs_p_trades=float("nan"), rs_p_days=float("nan"), rs_null_mean=float("nan"),
               rs_null_p95=float("nan"), rs_keep_share=float("nan"))
    nb, nv = len(base_r), len(var_r)
    if nv == 0 or nb == 0 or nv >= nb:
        return out
    rng = np.random.default_rng(seed)
    real = var_r.mean()
    null = np.array([base_r[rng.choice(nb, nv, replace=False)].mean() for _ in range(reps)])
    out.update(rs_p_trades=float((1 + (null >= real).sum()) / (reps + 1)), rs_null_mean=float(null.mean()),
               rs_null_p95=float(np.percentile(null, 95)), rs_keep_share=nv / nb)
    if base_day is not None and var_days:
        base_day = np.asarray(base_day)
        udays = np.unique(base_day)
        if var_days < len(udays):
            idx = {d: np.nonzero(base_day == d)[0] for d in udays}
            nd = []
            for _ in range(reps):
                pick = rng.choice(len(udays), var_days, replace=False)
                sel = np.concatenate([idx[udays[i]] for i in pick])
                nd.append(base_r[sel].mean())
            nd = np.array(nd)
            out["rs_p_days"] = float((1 + (nd >= real).sum()) / (reps + 1))
    return out


# ----------------------------------------------------------------------------- Hansen SPA
def stationary_bootstrap_indices(n: int, reps: int, mean_block: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap: (reps, n) index matrix, geometric blocks, wrap-around."""
    p = 1.0 / mean_block
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    new = rng.random((reps, n)) < p
    starts = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def spa_test(diff: np.ndarray, reps: int = 1000, mean_block: float = 10.0, seed: int = 0) -> dict:
    """Hansen SPA (consistent p-value). diff: (n_days, K) = variant daily net R - base daily net R
    (positive = variant better). H0: max_k E[diff_k] <= 0, i.e. no variant beats the base."""
    d = np.asarray(diff, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    n, K = d.shape
    if n < 20 or K == 0:
        return dict(spa_p=float("nan"), spa_stat=float("nan"), spa_best=None)
    rng = np.random.default_rng(seed)
    dbar = d.mean(axis=0)
    idx = stationary_bootstrap_indices(n, reps, mean_block, rng)
    boot = np.stack([d[idx[b]].mean(axis=0) for b in range(reps)])          # (reps, K)
    omega = np.sqrt(n) * (boot - dbar).std(axis=0, ddof=0)
    omega = np.where(omega > 0, omega, np.inf)
    tstat = np.sqrt(n) * dbar / omega
    T_spa = max(0.0, float(np.max(tstat)))
    thresh = -np.sqrt(2 * np.log(np.log(n)))
    # consistent recentering (Hansen's g_c): a variant that is clearly WORSE than base (t < -sqrt(2 loglog n))
    # keeps its negative mean in the bootstrap, so it cannot inflate the null; all others are centred at 0
    mu_keep = np.where(tstat >= thresh, 0.0, dbar)
    Tb = np.max(np.sqrt(n) * (boot - dbar + mu_keep) / omega, axis=1)
    Tb = np.maximum(Tb, 0.0)
    p = float((Tb >= T_spa).mean()) if T_spa > 0 else 1.0
    return dict(spa_p=p, spa_stat=T_spa, spa_best=int(np.argmax(tstat)))


# ----------------------------------------------------------------------------- PBO via CSCV
def pbo_cscv(perf: np.ndarray, n_blocks: int = 16) -> dict:
    """perf: (n_days, N) daily net R of N candidate versions (base included). Rows split in n_blocks contiguous
    blocks; for each half/half split choose the best in-sample Sharpe, find its out-of-sample rank.
    PBO = share of splits where the in-sample winner is at or below the OOS median (logit <= 0)."""
    M = np.asarray(perf, dtype=float)
    T, N = M.shape
    if N < 2:
        return dict(pbo=float("nan"), splits=0, blocks=0)
    S = min(n_blocks, T // 2)
    S -= S % 2
    if S < 2:
        return dict(pbo=float("nan"), splits=0, blocks=0)
    blocks = np.array_split(np.arange(T), S)

    def sharpe(rows):
        x = M[rows]
        sd = x.std(axis=0, ddof=1)
        mu = x.mean(axis=0)
        return np.where(sd > 0, mu / np.where(sd > 0, sd, 1), np.where(mu > 0, np.inf, np.where(mu < 0, -np.inf, 0)))

    lam = []
    for combo in itertools.combinations(range(S), S // 2):
        is_rows = np.concatenate([blocks[i] for i in combo])
        oos_rows = np.concatenate([blocks[i] for i in range(S) if i not in combo])
        best = int(np.argmax(sharpe(is_rows)))
        oos_rank = stats.rankdata(sharpe(oos_rows))[best]                    # 1 = worst, N = best
        w = oos_rank / (N + 1)
        lam.append(math.log(w / (1 - w)))
    lam = np.array(lam)
    return dict(pbo=float((lam <= 0).mean()), splits=len(lam), blocks=S, logit_median=float(np.median(lam)))


# ----------------------------------------------------------------------------- expected cone (paper monitoring)
CONE_HORIZONS = (25, 50, 100, 200)


def expected_cone(r_net, horizons=CONE_HORIZONS, paths: int = 5000, seed: int = 0) -> dict:
    """Monte Carlo cone for paper/live monitoring (rigor skill §8): resample the run's per-trade net R with
    replacement (iid, seeded) into `paths` sequences; for each horizon (number of trades) give the 5th / 50th /
    95th percentile of cumulative net R and of the max drawdown (in R, peak-to-trough, starting from 0).
    Paper results below p5 (or drawdown above p95) -> check the implementation first; ABOVE p95 -> also check."""
    r = np.asarray(r_net, dtype=float)
    r = r[np.isfinite(r)]
    out = dict(method="iid bootstrap of per-trade net R", paths=paths, seed=seed, source_trades=int(len(r)),
               source_mean_R=float(r.mean()) if len(r) else float("nan"), horizons={})
    if len(r) == 0:
        return out
    H = max(horizons)
    rng = np.random.default_rng(seed)
    draws = r[rng.integers(0, len(r), (paths, H))]
    cum = np.cumsum(draws, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.zeros((paths, 1)), cum], axis=1), axis=1)[:, 1:]
    dd = np.maximum.accumulate(peak - cum, axis=1)
    for h in horizons:
        c, d = cum[:, h - 1], dd[:, h - 1]
        out["horizons"][str(h)] = dict(cum_R=[float(np.percentile(c, q)) for q in (5, 50, 95)],
                                       max_dd_R=[float(np.percentile(d, q)) for q in (5, 50, 95)])
    return out
