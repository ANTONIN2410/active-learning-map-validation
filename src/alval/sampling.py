"""Static sampling designs: the whole sample is drawn at once.

Each design is described by its exact variance (sampling with replacement from the pool
of points), so that they can be compared without Monte Carlo noise, and by a simulator
used to check coverage and bias.
"""

import numpy as np


def quantile_strata(score, n_strata=5):
    edges = np.quantile(score, np.linspace(0, 1, n_strata + 1)[1:-1])
    return np.searchsorted(edges, score, side="right")


def allocate(weights, n, minimum=2):
    """Integer allocation proportional to `weights`, at least `minimum` per stratum."""
    n_h = np.full(len(weights), minimum)
    rest = n - n_h.sum()
    if rest < 0:
        raise ValueError("sample too small for the number of strata")
    share = weights / weights.sum() * rest
    n_h += np.floor(share).astype(int)
    n_h[np.argsort(-(share - np.floor(share)))[: n - n_h.sum()]] += 1
    return n_h


def _strata(design, f, classes):
    """Strata and allocation weights of a stratified design."""
    if design in ("class", "class+diff"):
        strata = classes
        labels = np.unique(strata)
        weights = np.array([np.sum(strata == h) for h in labels], float)
    else:  # Neyman allocation on quintiles of the predicted error
        strata = quantile_strata(f)
        labels = np.unique(strata)
        weights = np.array([np.sum(strata == h) * np.sqrt(np.mean(f[strata == h] * (1 - f[strata == h])))
                            for h in labels])
    return strata, labels, weights


def exact_variance(design, y, f, classes, n, eps=0.10):
    """Variance of the error-rate estimate for a sample of size n.

    designs: "srs" (mean), "srs+diff" (difference estimator), "importance" (weighted
    difference, q ~ sqrt(f(1-f)) mixed with eps uniform), "class" / "class+diff"
    (strata by map class), "neyman" / "neyman+diff" (strata by quintiles of f).
    """
    N = len(y)
    if design == "srs":
        return y.var() / n
    if design == "srs+diff":
        return (y - f).var() / n
    if design == "importance":
        a = np.sqrt(f * (1 - f))
        q = (1 - eps) * a / a.sum() + eps / N
        r = y - f
        return (np.sum(r ** 2 / q) / N ** 2 - r.mean() ** 2) / n
    strata, labels, weights = _strata(design, f, classes)
    use_diff = design.endswith("+diff")
    var = 0.0
    for h, m in zip(labels, allocate(weights, n)):
        s = strata == h
        values = y[s] - f[s] if use_diff else y[s]
        var += s.mean() ** 2 * values.var() / m
    return var


def simulate(design, y, f, classes, n, n_rep, rng, eps=0.10):
    """Estimates and estimated variances of n_rep simulated samples."""
    N = len(y)
    if design in ("srs", "srs+diff"):
        idx = rng.integers(0, N, size=(n_rep, n))
        v = y[idx] - f[idx] if design == "srs+diff" else y[idx]
        base = f.mean() if design == "srs+diff" else 0.0
        return base + v.mean(1), v.var(1, ddof=1) / n
    if design == "importance":
        a = np.sqrt(f * (1 - f))
        q = (1 - eps) * a / a.sum() + eps / N
        idx = rng.choice(N, size=(n_rep, n), p=q)
        z = f.mean() + (y[idx] - f[idx]) / (N * q[idx])
        return z.mean(1), z.var(1, ddof=1) / n
    strata, labels, weights = _strata(design, f, classes)
    use_diff = design.endswith("+diff")
    est, var = np.zeros(n_rep), np.zeros(n_rep)
    for h, m in zip(labels, allocate(weights, n)):
        members = np.flatnonzero(strata == h)
        w = len(members) / N
        idx = members[rng.integers(0, len(members), size=(n_rep, m))]
        v = y[idx] - f[idx] if use_diff else y[idx].astype(float)
        est += w * ((f[members].mean() if use_diff else 0.0) + v.mean(1))
        var += w ** 2 * v.var(1, ddof=1) / m
    return est, var
