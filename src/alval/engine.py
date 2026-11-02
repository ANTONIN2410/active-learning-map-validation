"""The active learning engine: a validation sample built in waves.

Before each wave the engine predicts, at every candidate unit, the probability f that
the map is wrong. The wave is drawn with a stratified design fixed before its
observations (strata = quantiles of f, Neyman allocation). Once the observations come
back, the error model is updated and the next wave is chosen with it.

Because every wave is a known probability design, the wave-wise stratified difference
estimator is unbiased whatever the model; the waves form a martingale, which gives the
variance and the confidence interval.
"""

from dataclasses import dataclass, field

import numpy as np

from .sampling import allocate, quantile_strata

EPS = 1e-4


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def map_logistic(Z, y, prior_mean, prior_precision, start=None, n_iter=30):
    """Maximum a posteriori logistic regression with a diagonal Gaussian prior (Newton)."""
    beta = prior_mean.copy() if start is None else start.copy()
    for _ in range(n_iter):
        p = sigmoid(Z @ beta)
        grad = Z.T @ (y - p) - prior_precision * (beta - prior_mean)
        hess = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(prior_precision)
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


@dataclass
class ErrorModel:
    """Local logistic error model of one product: design matrix and Gaussian prior."""
    name: str
    Z: np.ndarray
    prior_mean: np.ndarray
    prior_precision: np.ndarray
    beta: np.ndarray = field(init=False)
    seen: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    waves: list = field(default_factory=list)  # (size, estimate, variance)

    def __post_init__(self):
        self.beta = self.prior_mean.copy()

    def predict(self):
        return np.clip(sigmoid(self.Z @ self.beta), EPS, 1 - EPS)

    def update(self, units, y):
        self.seen.extend(units)
        self.labels.extend(y)
        self.beta = map_logistic(self.Z[self.seen], np.asarray(self.labels, float),
                                 self.prior_mean, self.prior_precision, start=self.beta)


def from_prior(name, f0, classes, sd=(0.5, 0.25, 0.25), n_classes=7):
    """Recalibrate a transferred model: logit f = a + b logit f0 + gamma[class].

    The prior is centred on the transferred model (a = 0, b = 1, gamma = 0).
    """
    lf0 = logit(f0)
    cols = [np.ones_like(lf0), lf0] + [(classes == c).astype(float) for c in range(1, n_classes + 1)]
    mean = np.r_[0.0, 1.0, np.zeros(n_classes)]
    precision = 1 / np.r_[sd[0], sd[1], np.full(n_classes, sd[2])] ** 2
    return ErrorModel(name, np.column_stack(cols), mean, precision)


def cold_start(name, X, sd=0.5, base_rate=0.30, sd_intercept=1.5):
    """No prior data: logistic model on standardised features, shrunk towards a base rate.

    Until the first observations arrive, f is constant, so the first wave is uniform.
    `base_rate` only centres a wide prior on the intercept (sd 1.5 on the logit scale,
    a 95% prior range of about 2% to 89%): it fixes an order of magnitude, not the
    answer, and the estimate stays unbiased whatever the model.
    """
    X = np.asarray(X, float)
    std = X.std(0)
    Z = np.column_stack([np.ones(len(X)), (X - X.mean(0)) / np.where(std > 0, std, 1)])
    mean = np.zeros(Z.shape[1])
    mean[0] = logit(np.array([base_rate]))[0]
    precision = np.full(Z.shape[1], 1 / sd ** 2)
    precision[0] = 1 / sd_intercept ** 2
    return ErrorModel(name, Z, mean, precision)


class ActiveEngine:
    """Chooses the missions wave by wave and estimates the error rate of each product.

    Several products can share a campaign: one observation of the true class gives the
    error of every product at that unit, and the strata then follow the sum of f(1-f).
    `costs` (relative cost of a mission at each unit) divides the allocation by sqrt(cost).
    """

    def __init__(self, models, n_strata=5, costs=None, learn=True):
        self.models = models
        self.N = models[0].Z.shape[0]
        self.n_strata = n_strata
        self.costs = np.ones(self.N) if costs is None else np.asarray(costs, float)
        self.learn = learn
        self._wave = None

    def plan(self):
        f = [m.predict() for m in self.models]
        u = sum(fi * (1 - fi) for fi in f)
        score = f[0] if len(f) == 1 else u
        strata = quantile_strata(score, self.n_strata)
        members = [np.flatnonzero(strata == h) for h in np.unique(strata)]
        weights = np.array([len(m) * np.sqrt(u[m].mean() / self.costs[m].mean()) for m in members])
        return {"f": f, "members": members, "weights": weights}

    def propose(self, size, rng):
        """Draw the next wave; returns the indices of the units to visit."""
        plan = self.plan()
        plan["n_h"] = allocate(plan["weights"], size)
        plan["draws"] = [m[rng.integers(0, len(m), size=k)] for m, k in zip(plan["members"], plan["n_h"])]
        plan["units"] = np.concatenate(plan["draws"])
        self._wave = plan
        return plan["units"]

    def record(self, observations):
        """Observations of the current wave, in the order of the proposed units:
        observations[name] = 1 where the product was found wrong."""
        plan = self._wave
        size = len(plan["units"])
        for model, f in zip(self.models, plan["f"]):
            y = np.asarray(observations[model.name], float)
            est = var = 0.0
            start = 0
            for members, k, draws in zip(plan["members"], plan["n_h"], plan["draws"]):
                residual = y[start:start + k] - f[draws]
                w = len(members) / self.N
                est += w * (f[members].mean() + residual.mean())
                var += w ** 2 * residual.var(ddof=1) / k
                start += k
            model.waves.append((size, est, var))
            if self.learn:
                model.update(plan["units"].tolist(), y.tolist())
        self._wave = None

    def conditional_variance(self, truth):
        """Simulation only: exact variance of the current wave given the past, when the
        true errors of every unit are known (used for Rao-Blackwellised comparisons)."""
        plan = self._wave
        out = {}
        for model, f in zip(self.models, plan["f"]):
            y = truth[model.name]
            out[model.name] = sum((len(m) / self.N) ** 2 * (y[m] - f[m]).var() / k
                                  for m, k in zip(plan["members"], plan["n_h"]))
        return out

    def estimate(self, z=1.96):
        out = {}
        for m in self.models:
            n = sum(s for s, _, _ in m.waves)
            theta = sum(s * e for s, e, _ in m.waves) / n
            var = sum((s / n) ** 2 * v for s, _, v in m.waves)
            half = z * np.sqrt(var)
            out[m.name] = {"estimate": theta, "variance": var, "n": n, "ci": (theta - half, theta + half)}
        return out
