"""Retrospective benchmark of the engine against static designs.

For each region and product ("case"), the error models are trained on the three other
regions and the designs are evaluated on the held-out half of the target region. The
label saving at equal precision is 1 - MSE(design) / MSE(simple random sampling).

Without prior data, the engine is also compared with a classical two-phase design: a
uniform first phase, one fit of the same error model, then a single stratified second
phase. It answers the question "does the loop beat a single update?".
"""

import zlib

import numpy as np
import pandas as pd

from . import engine as E
from . import features as F
from .data import REGIONS
from .sampling import exact_variance, simulate

STATIC = ["srs+diff", "importance", "neyman", "neyman+diff", "class+diff"]
N_WAVES, N_STRATA = 5, 5
FIRST_PHASE = 0.2  # two-phase baseline: share of the sample in the uniform first phase


def seed_for(*keys, base=20260821):
    return (base + zlib.crc32("|".join(map(str, keys)).encode())) % 2**32


def wave_sizes(n, n_waves=N_WAVES):
    """n split into n_waves waves of (nearly) equal size."""
    return np.diff(np.linspace(0, n, n_waves + 1).round().astype(int))


def two_phase_sizes(n, first=FIRST_PHASE):
    """A uniform first phase, then everything else in one stratified wave."""
    n1 = int(round(first * n))
    return np.array([n1, n - n1])


def run_campaigns(make_models, truth, n, n_rep, rng, sizes=None):
    """Simulate n_rep campaigns; returns estimates, estimated variances and the
    Rao-Blackwellised variance (mean of exact wave variances) for each product."""
    sizes = wave_sizes(n) if sizes is None else np.asarray(sizes)
    names = list(truth)
    est = {k: np.zeros(n_rep) for k in names}
    var = {k: np.zeros(n_rep) for k in names}
    rb = {k: np.zeros(n_rep) for k in names}
    for r in range(n_rep):
        eng = E.ActiveEngine(make_models(), n_strata=N_STRATA)
        for size in sizes:
            units = eng.propose(int(size), rng)
            cv = eng.conditional_variance(truth)
            for k in names:
                rb[k][r] += (size / n) ** 2 * cv[k]
            eng.record({k: truth[k][units] for k in names})
        res = eng.estimate()
        for k in names:
            est[k][r], var[k][r] = res[k]["estimate"], res[k]["variance"]
    return est, var, rb


def _summary(est, var, theta):
    half = 1.96 * np.sqrt(np.maximum(var, 0))
    return {"bias": float(np.mean(est) - theta), "coverage": float(np.mean(np.abs(est - theta) <= half))}


def run_case(points, region, product, sizes=(100, 200, 500), n_rep=1000):
    train = {r: points[points["region"] == r].reset_index(drop=True) for r in REGIONS if r != region}
    test = points[(points["region"] == region) & (points["split"] == "test")].reset_index(drop=True)
    y = test[f"err_{product}"].to_numpy(float)
    classes = test[product].to_numpy()
    theta = y.mean()

    priors = {
        "transfer": F.transferred_errors(train, test, product, "opinion"),
        "transfer_no_embeddings": F.transferred_errors(train, test, product, "enriched"),
        "new_product": F.new_product_errors(train, test, product),
    }
    f_old = F.transferred_errors(train, test, product, "indicators")
    X_cold = F.enriched(test, product).to_numpy()

    rows = []
    for n in sizes:
        rng = np.random.default_rng(seed_for(region, product, n))
        v0 = exact_variance("srs", y, y, classes, n)

        def add(scenario, design, mse, est, var):
            rows.append({"region": region, "product": product, "n": n, "scenario": scenario, "design": design,
                         "theta": theta, "mse": mse, "saving": 1 - mse / v0, **_summary(est, var, theta)})

        add("all", "srs", v0, *simulate("srs", y, y, classes, n, n_rep, rng))
        add("all", "class", exact_variance("class", y, y, classes, n), *simulate("class", y, y, classes, n, n_rep, rng))
        add("all", "one_shot_indicators", exact_variance("importance", y, f_old, classes, n),
            *simulate("importance", y, f_old, classes, n, n_rep, rng))
        for scenario, f0 in priors.items():
            for design in STATIC:
                add(scenario, design, exact_variance(design, y, f0, classes, n),
                    *simulate(design, y, f0, classes, n, n_rep, rng))
            est, var, rb = run_campaigns(lambda f0=f0: [E.from_prior(product, f0, classes)], {product: y}, n,
                                         n_rep, rng)
            add(scenario, "engine", rb[product].mean(), est[product], var[product])
        est, var, rb = run_campaigns(lambda: [E.cold_start(product, X_cold)], {product: y}, n, n_rep, rng)
        add("cold_start", "engine", rb[product].mean(), est[product], var[product])
        est, var, rb = run_campaigns(lambda: [E.cold_start(product, X_cold)], {product: y}, n, n_rep, rng,
                                     sizes=two_phase_sizes(n))
        add("cold_start", "two_phase", rb[product].mean(), est[product], var[product])
    auc = {k: _auc(y, f) for k, f in {**priors, "indicators": f_old}.items()}
    return rows, {"region": region, "product": product, "theta": theta, **{f"auc_{k}": v for k, v in auc.items()}}


def _auc(y, score):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def headline(results, n=200):
    """Median label saving by scenario and design, as in the summary table."""
    r = results[results["n"] == n]
    return r.pivot_table(index="design", columns="scenario", values="saving", aggfunc="median")
