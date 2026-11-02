"""Binomial GLM (logistic regression) with the frequentist toolkit.

Fitting, likelihood-ratio tests of each term (the equivalent of R's drop1(test="Chisq")),
backward selection, odds ratios with confidence intervals, and simulated quantile
residuals in the spirit of the DHARMa package to check the model assumptions. The points
are spatially clustered, so two checks of independence come with it: standard errors
robust to clustering by spatial block, and Moran's I of the residuals.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def fit_logit(formula, data, groups=None):
    """Binomial GLM; with `groups`, standard errors are robust to clustering."""
    model = smf.glm(formula, data=data, family=sm.families.Binomial())
    if groups is None:
        return model.fit()
    return model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})


def _terms(model):
    """Response and additive terms of the formula; a "+" inside parentheses, as in
    C(x, Treatment(reference="a+b")) or I(x + y), does not split a term."""
    response, rhs = model.model.formula.split("~", 1)
    terms, depth, current = [], 0, ""
    for char in rhs:
        depth += (char == "(") - (char == ")")
        if char == "+" and depth == 0:
            terms.append(current.strip())
            current = ""
        else:
            current += char
    terms.append(current.strip())
    return response.strip(), [t for t in terms if t not in ("", "1")]


def drop1(model, data):
    """Likelihood-ratio test of every term, refitting the model without it."""
    rows = []
    response, terms = _terms(model)
    for term in terms:
        others = [t for t in terms if t != term] or ["1"]
        reduced = fit_logit(f"{response} ~ {' + '.join(others)}", data)
        lr = 2 * (model.llf - reduced.llf)
        df = model.df_model - reduced.df_model
        rows.append({"term": term, "df": df, "deviance_increase": lr, "AIC_without": reduced.aic,
                     "p_value": stats.chi2.sf(lr, df)})
    return pd.DataFrame(rows).set_index("term").sort_values("p_value")


def backward_selection(formula, data, alpha=0.05, verbose=True):
    """Remove the least significant term until every term is significant at `alpha`."""
    model = fit_logit(formula, data)
    while True:
        table = drop1(model, data)
        worst = table["p_value"].idxmax()
        if table.loc[worst, "p_value"] < alpha or len(table) == 1:
            return model
        if verbose:
            print(f"dropping {worst} (p = {table.loc[worst, 'p_value']:.3f})")
        response, _ = _terms(model)
        kept = [t for t in table.index if t != worst]
        model = fit_logit(f"{response} ~ {' + '.join(kept)}", data)


def odds_ratios(model, level=0.95):
    ci = model.conf_int(alpha=1 - level)
    out = pd.DataFrame({"odds_ratio": np.exp(model.params), "low": np.exp(ci[0]), "high": np.exp(ci[1]),
                        "p_value": model.pvalues})
    return out.drop(index="Intercept", errors="ignore")


def simulated_residuals(model, n_sim=1000, seed=0):
    """Randomised quantile residuals from simulations of the fitted model.

    For each observation, simulate n_sim responses from the fitted probability and take
    the (randomised) rank of the observed response among them. If the model is right,
    the residuals are uniform on [0, 1], whatever the binary nature of the data.
    """
    rng = np.random.default_rng(seed)
    p = model.fittedvalues.to_numpy()
    y = model.model.endog
    sims = rng.random((n_sim, len(p))) < p
    below = (sims < y).mean(axis=0)
    equal = (sims == y).mean(axis=0)
    return below + rng.random(len(p)) * equal


def uniformity_test(residuals):
    """Kolmogorov-Smirnov test against the uniform law (DHARMa's testUniformity)."""
    return stats.kstest(residuals, "uniform")


def morans_i(values, xy, distance, n_perm=999, seed=0):
    """Moran's I with binary weights (pairs closer than `distance`) and a one-sided
    permutation p-value for positive autocorrelation (DHARMa's testSpatialAutocorrelation)."""
    from scipy.spatial import cKDTree

    pairs = cKDTree(xy).query_pairs(distance, output_type="ndarray")
    z = np.asarray(values, float) - np.mean(values)

    def statistic(v):
        return len(v) / len(pairs) * np.sum(v[pairs[:, 0]] * v[pairs[:, 1]]) / np.sum(v ** 2)

    observed = statistic(z)
    rng = np.random.default_rng(seed)
    perm = np.array([statistic(rng.permutation(z)) for _ in range(n_perm)])
    return {"I": observed, "expected": -1 / (len(z) - 1), "p_value": (np.sum(perm >= observed) + 1) / (n_perm + 1),
            "n_pairs": len(pairs)}
