import numpy as np
import pandas as pd
import pytest

from alval import glm
from alval.pca import pca


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                       "g": rng.choice(["u", "v", "w"], n)})
    eta = -0.5 + 1.0 * df["a"] + 0.0 * df["b"] + (df["g"] == "w") * 0.8
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    df["x"], df["yc"] = rng.uniform(0, 1e5, n), rng.uniform(0, 1e5, n)
    return df


def test_drop1_reads_categorical_terms(data):
    model = glm.fit_logit("y ~ a + b + C(g)", data)
    table = glm.drop1(model, data)
    assert set(table.index) == {"a", "b", "C(g)"}
    assert table.loc["C(g)", "df"] == 2


def test_terms_keep_parentheses_together(data):
    model = glm.fit_logit("y ~ I(a + b) + C(g)", data)
    assert glm._terms(model) == ("y", ["I(a + b)", "C(g)"])


def test_backward_selection_drops_the_useless_term(data):
    model = glm.backward_selection("y ~ a + b + C(g)", data, verbose=False)
    assert "b" not in glm.drop1(model, data).index


def test_simulated_residuals_are_uniform_under_the_true_model(data):
    model = glm.fit_logit("y ~ a + C(g)", data)
    assert glm.uniformity_test(glm.simulated_residuals(model)).pvalue > 0.01


def test_morans_i_finds_no_structure_in_noise(data):
    out = glm.morans_i(np.random.default_rng(1).normal(size=len(data)), data[["x", "yc"]].to_numpy(), 5000,
                       n_perm=199)
    assert out["p_value"] > 0.01


def test_pca_eigenvalues_sum_to_the_number_of_variables(data):
    res = pca(data[["a", "b", "x"]])
    assert res.eig["eigenvalue"].sum() == pytest.approx(3)
    assert np.allclose(res.var_cos2.sum(axis=1), 1)
