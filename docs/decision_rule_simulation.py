"""Decision-rule simulation for Subsection 6.6 (Evaluation Metrics).

Reproduces three sets of numbers quoted in the subsection:
  1. One-sided 5% critical values of the Spearman coefficient for M = 10, 12, 15.
  2. Table 2: share of simulated studies classified as supported (F >= 0.70),
     inconclusive (0.40 <= F < 0.70) or not supported (F < 0.40), for several
     true population Spearman correlations.
  3. Coverage and mean width of the 95% percentile bootstrap interval.

Data are bivariate normal. A population Spearman correlation rho_s corresponds
to a Pearson correlation r = 2 sin(pi * rho_s / 6). The seed is fixed, so the
output is reproducible. Requires numpy and scipy. Runtime: a few minutes.
"""
import numpy as np
from scipy import stats
rng = np.random.default_rng(20261011)

def rank(a):  # ranks along last axis (continuous data, no ties)
    return a.argsort(-1).argsort(-1).astype(float)

def spearman_rows(x, y):
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(-1, keepdims=True); ry -= ry.mean(-1, keepdims=True)
    return (rx*ry).sum(-1) / np.sqrt((rx**2).sum(-1)*(ry**2).sum(-1))

# 1. critical values under H0 (one-sided alpha=.05), Monte Carlo permutation
print("Critical values, one-sided alpha=0.05 and two-sided 0.05")
for n in (10, 12, 15):
    x = rng.standard_normal((2_000_000, n)); y = rng.standard_normal((2_000_000, n))
    r = spearman_rows(x, y)
    # smallest attainable r with P(R>=r)<=.05
    vals = np.unique(np.round(r, 6))
    one = next(v for v in vals if (r >= v-1e-9).mean() <= 0.05)
    two = next(v for v in vals if (np.abs(r) >= v-1e-9).mean() <= 0.05)
    print(n, round(one, 3), round(two, 3), "P(r>=0.70|H0)=%.4f" % (r >= 0.70).mean(), "P(r<0.40|H0)=%.3f" % (r < 0.40).mean())

# 2. operating characteristics: bivariate normal with population Spearman rho_s
def pearson_for_spearman(rs):  # rho_s = (6/pi) asin(r/2)
    return 2*np.sin(np.pi*rs/6)
print("\nOperating characteristics (bivariate normal)")
for rs in (0.0, 0.40, 0.55, 0.76, 0.90):
    r = pearson_for_spearman(rs)
    for n in (10, 12, 15):
        N = 400_000
        x = rng.standard_normal((N, n)); e = rng.standard_normal((N, n))
        y = r*x + np.sqrt(1-r*r)*e
        s = spearman_rows(x, y)
        lo, hi = np.percentile(s, [2.5, 97.5])
        print(f"rho_s={rs:.2f} n={n}: supported={np.mean(s>=0.70):.3f} inconclusive={np.mean((s>=0.40)&(s<0.70)):.3f} rejected={np.mean(s<0.40):.3f}  central95=[{lo:.2f},{hi:.2f}] median={np.median(s):.2f}")

# 3. coverage of percentile bootstrap CI for Spearman, n=12, rho_s=0.76 and 0.40
print("\nPercentile bootstrap coverage")
for rs in (0.40, 0.76):
    r = pearson_for_spearman(rs)
    for n in (10, 12, 15):
        sims, B = 3000, 2000
        cover = 0; widths = []
        for _ in range(sims):
            x = rng.standard_normal(n); y = r*x + np.sqrt(1-r*r)*rng.standard_normal(n)
            idx = rng.integers(0, n, (B, n))
            bx, by = x[idx], y[idx]
            # ties from resampling: use average ranks via scipy
            rx = stats.rankdata(bx, axis=1); ry = stats.rankdata(by, axis=1)
            rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
            den = np.sqrt((rx**2).sum(1)*(ry**2).sum(1))
            bs = (rx*ry).sum(1)/np.where(den == 0, np.nan, den)
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            cover += (lo <= rs <= hi); widths.append(hi-lo)
        print(f"rho_s={rs} n={n}: coverage={cover/sims:.3f} mean width={np.mean(widths):.2f}")
