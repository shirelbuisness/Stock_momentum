"""sim_spa_size.py: measured size (false-positive rate) of gate_stats.spa_test under H0, K = 14 versions.
Run: python sim_spa_size.py 2000   (about 2 min on 2 cores). Results 27.11.2026 (2,000 sims, 500 bootstrap reps):
  iid normal  n=300  : 9.6% at 5%, 16.5% at 10%      n=1250 : 5.9% at 5%, 11.3% at 10%
  realistic   n=300  : 7.6% at 5%, 13.6% at 10%      n=1250 : 6.3% at 5%, 11.2% at 10%
("realistic" = t3 fat tails, a common day shock (corr ~0.5 between versions), AR(1) 0.1.)
Not a test: a measurement kept so it can be re-run. The DEV-A gate uses "SPA p > 0.10 -> nothing beats base".
"""
import sys, time, numpy as np
from multiprocessing import Pool
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import gate_stats as G
K = 14
def sim(args):
    design, n, s = args
    rng = np.random.default_rng(50_000 + s)
    if design == "iid_normal":
        d = rng.normal(0, 1, (n, K))
    else:   # fat tails (t3), variants share a common day shock (corr ~0.5), AR(1) 0.1 -> realistic H0
        common = rng.standard_t(3, n)[:, None]
        own = rng.standard_t(3, (n, K))
        d = 0.7 * common + 0.7 * own
        for t in range(1, n):
            d[t] += 0.1 * d[t - 1]
        d -= 0  # true mean 0 for every variant
    return G.spa_test(d, reps=500, mean_block=10.0, seed=s)["spa_p"]
if __name__ == "__main__":
    S = int(sys.argv[1])
    with Pool() as pool:
        for design in ("iid_normal", "realistic"):
            for n in (300, 1250):
                t0 = time.time()
                ps = np.array(pool.map(sim, [(design, n, s) for s in range(S)]))
                print(f"{design:11s} n={n:5d} K={K} sims={S}: rej@5% {np.mean(ps<0.05):.3f}  rej@10% {np.mean(ps<0.10):.3f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
