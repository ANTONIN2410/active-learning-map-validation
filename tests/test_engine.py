import numpy as np
import pytest

from alval import engine as E
from alval.sampling import allocate, exact_variance, simulate


@pytest.fixture
def population():
    rng = np.random.default_rng(1)
    n = 3000
    f = rng.beta(2, 5, n)
    y = (rng.random(n) < f).astype(float)
    classes = rng.integers(1, 8, n)
    return y, f, classes


def test_allocation_respects_minimum_and_total():
    n_h = allocate(np.array([10.0, 1.0, 0.1]), 20)
    assert n_h.sum() == 20
    assert n_h.min() >= 2


def test_exact_variance_matches_simulation(population):
    y, f, classes = population
    rng = np.random.default_rng(2)
    est, _ = simulate("neyman+diff", y, f, classes, 200, 4000, rng)
    assert est.var() == pytest.approx(exact_variance("neyman+diff", y, f, classes, 200), rel=0.1)


def test_engine_is_unbiased(population):
    """The estimate must stay unbiased whatever the model, even a useless one."""
    y, _, classes = population
    rng = np.random.default_rng(3)
    bad_prior = np.full(len(y), 0.9)
    estimates = []
    for _ in range(300):
        eng = E.ActiveEngine([E.from_prior("p", bad_prior, classes)])
        for _ in range(4):
            units = eng.propose(25, rng)
            eng.record({"p": y[units]})
        estimates.append(eng.estimate()["p"]["estimate"])
    se = np.std(estimates) / np.sqrt(len(estimates))
    assert abs(np.mean(estimates) - y.mean()) < 4 * se


def test_cold_start_first_wave_is_uniform(population):
    y, f, _ = population
    model = E.cold_start("p", np.column_stack([f, f ** 2]))
    assert np.ptp(model.predict()) == pytest.approx(0)
