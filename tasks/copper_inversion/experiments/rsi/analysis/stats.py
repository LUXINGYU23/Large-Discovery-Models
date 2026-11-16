"""Statistics of spec §7. Every random draw comes from a named, seeded stream."""

import hashlib
import math
from itertools import combinations

import numpy as np


def stream(seed, label):
    """Independent generator per statistic, so adding one statistic changes no other."""
    key = int(hashlib.sha256(label.encode()).hexdigest()[:16], 16)
    return np.random.default_rng(np.random.SeedSequence([seed, key]))


def pooled_sd(groups):
    """Square root of the pooled within-group variance (weights n - 1)."""
    groups = [np.asarray(g, float) for g in groups if len(g) > 1]
    df = sum(len(g) - 1 for g in groups)
    if not df:
        return None
    return math.sqrt(sum(((g - g.mean()) ** 2).sum() for g in groups) / df)


def permutation_test(x, y, n_perm, rng):
    """Two-sided Monte Carlo permutation test of mean(x) - mean(y) on run labels."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    pool, k = np.concatenate([x, y]), len(x)
    observed = abs(x.mean() - y.mean())
    hits = 0
    for start in range(0, n_perm, 5000):
        m = min(5000, n_perm - start)
        order = np.argsort(rng.random((m, len(pool))), axis=1)
        shuffled = pool[order]
        stat = np.abs(shuffled[:, :k].mean(axis=1) - shuffled[:, k:].mean(axis=1))
        hits += int((stat >= observed - 1e-12).sum())
    return (hits + 1) / (n_perm + 1)


def bootstrap_contrast(groups, coefs, n_boot, rng, chunk=2000):
    """Hierarchical bootstrap of sum_g coef_g * mean_g.

    Each group is a runs x bodies matrix. Runs are resampled within each group;
    when every group has the same bodies (more than one column), one body
    resample per draw is shared by all groups (bodies paired across arms).
    """
    groups = [np.atleast_2d(np.asarray(g, float)) for g in groups]
    bodies = groups[0].shape[1]
    paired = bodies > 1 and all(g.shape[1] == bodies for g in groups)
    out = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        weights = rng.multinomial(bodies, np.full(bodies, 1 / bodies), size=m) / bodies if paired else None
        total = np.zeros(m)
        for g, c in zip(groups, coefs):
            runs = g.shape[0]
            rw = rng.multinomial(runs, np.full(runs, 1 / runs), size=m) / runs
            total += c * (np.einsum("ir,rb,ib->i", rw, g, weights) if paired else rw @ g.mean(axis=1))
        out[start:start + m] = total
    return out


def interval(draws, level=0.95):
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
    return [float(lo), float(hi)]


def holm(pvalues):
    """Holm step-down adjusted p-values, in the input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# -- E1: stratified permutation and its inversion

def _stratum_diffs(b, a, limit, rng, draws):
    pool, k = np.concatenate([b, a]), len(b)
    n = len(pool)
    if math.comb(n, k) <= limit:
        idx = np.array(list(combinations(range(n), k)))
        mask = np.zeros((len(idx), n), bool)
        mask[np.arange(len(idx))[:, None], idx] = True
    else:
        mask = np.zeros((draws, n), bool)
        picks = np.argsort(rng.random((draws, n)), axis=1)[:, :k]
        mask[np.arange(draws)[:, None], picks] = True
    sums = mask @ pool
    return sums / k - (pool.sum() - sums) / (n - k)


def stratified_null(strata, *, exact_limit=2_000_000, draws=200_000, seed=0):
    """Observed statistic and permutation distribution of mean_s(mean b_s - mean a_s),
    permuting labels within strata. Exact when the number of relabelings allows."""
    rng = np.random.default_rng(seed)
    observed = float(np.mean([np.mean(b) - np.mean(a) for b, a in strata]))
    per = [_stratum_diffs(np.asarray(b, float), np.asarray(a, float), 5000, rng, draws) for b, a in strata]
    if math.prod(len(d) for d in per) <= exact_limit:
        total = per[0]
        for d in per[1:]:
            total = (total[:, None] + d[None, :]).ravel()
    else:
        total = sum(d[rng.integers(0, len(d), draws)] for d in per)
    return observed, total / len(strata)


def stratified_p(strata, **kw):
    observed, null = stratified_null(strata, **kw)
    return float(np.mean(np.abs(null) >= abs(observed) - 1e-12))


def shift_upper_bound(strata, alpha=0.05, iterations=60, **kw):
    """One-sided (1 - alpha) upper bound of the stratified difference b - a: the largest
    shift d of b for which the lower-tail permutation test of (b - d) vs a is not rejected."""
    def p_lower(d):
        observed, null = stratified_null([(np.asarray(b, float) - d, a) for b, a in strata], **kw)
        return float(np.mean(null <= observed + 1e-12))

    observed = float(np.mean([np.mean(b) - np.mean(a) for b, a in strata]))
    spread = max(float(np.ptp(np.concatenate([np.concatenate([b, a]) for b, a in strata]))), 1e-9)
    lo, hi = observed, observed + spread
    if p_lower(lo) < alpha:
        return lo
    while p_lower(hi) >= alpha:
        lo, hi = hi, hi + 2 * (hi - observed)
        if hi - observed > 1e6 * spread:
            return math.inf
    for _ in range(iterations):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_lower(mid) >= alpha else (lo, mid)
    return lo


def e1_verdict(diff, upper, margin):
    """spec §5 E1 rule: non-inferior if the upper bound < delta; refuted if the
    point estimate > delta; otherwise inconclusive."""
    if upper < margin:
        return "non-inferior"
    return "refuted" if diff > margin else "inconclusive"


# -- E2: stratified bootstrap of per-round differences

def stratified_bootstrap_rows(strata, n_boot, rng):
    """strata: [(planted rows x rounds, control rows x rounds)]. Returns draws x rounds of
    mean_s(mean planted_s - mean control_s), resampling rows within each stratum."""
    out = 0
    for planted, control in strata:
        part = 0
        for rows, sign in ((np.atleast_2d(np.asarray(planted, float)), 1),
                           (np.atleast_2d(np.asarray(control, float)), -1)):
            n = rows.shape[0]
            w = rng.multinomial(n, np.full(n, 1 / n), size=n_boot) / n
            part = part + sign * (w @ rows)
        out = out + part
    return out / len(strata)


def half_life(means):
    """Rounds after the first until the mean difference first falls to half its first value."""
    if not means or means[0] is None or means[0] <= 0:
        return None
    return next((t for t, v in enumerate(means) if v <= means[0] / 2), None)


# -- sample size

def mde(s, n, z=2.80):
    return z * s * math.sqrt(2 / n)


def mde_table(s, ns, bound=None, z=2.80):
    rows = []
    for n in ns:
        value = mde(s, n, z)
        row = {"n": n, "mde": value, "mde_in_s": z * math.sqrt(2 / n)}
        if bound is not None:
            row["meets_bound"] = value <= bound
        rows.append(row)
    return rows


def replicate_decision(s, bound, z=2.80):
    """spec §7 rule from the internal pilot."""
    if mde(s, 12, z) <= bound:
        return {"runs_per_arm": 12, "primary": "test"}
    if mde(s, 16, z) <= bound:
        return {"runs_per_arm": 16, "primary": "test"}
    return {"runs_per_arm": 12, "primary": "estimate_only_author_confirms"}


# -- Proposition 3

def k_star(r, h):
    """Minimizer of w(K) = r/K + 1 - (1 - exp(-hK))/(hK); None when hr >= 1 or h is 0 or inf."""
    if r is None or h is None or not math.isfinite(h) or h <= 0 or h * r >= 1:
        return None
    lo, hi = 1e-9, 1.0
    f = lambda k: 1 - math.exp(-h * k) * (1 + h * k) - h * r
    while f(hi) < 0:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def exponential_rate(times, events):
    """Right-censored exponential MLE: events / total exposure."""
    exposure = float(sum(times))
    count = int(sum(events))
    if exposure <= 0:
        return math.inf if count else None
    return count / exposure


def spearman(x, y):
    from scipy.stats import rankdata
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
