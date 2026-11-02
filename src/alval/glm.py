"""Binomial GLM (logistic regression) with the frequentist toolkit.

Fitting, likelihood-ratio tests of each term (the equivalent of R's drop1(test="Chisq")),
backward selection, odds ratios with confidence intervals, and simulated quantile
residuals in the spirit of the DHARMa package to check the model assumptions.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def fit_logit(formula, data):
    return smf.glm(formula, data=data, family=sm.families.Binomial()).fit()


def _terms(model):
    response, rhs = model.model.formula.split("~")
    return response.strip(), [t.strip() for t in rhs.split("+") if t.strip() not in ("", "1")]


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
