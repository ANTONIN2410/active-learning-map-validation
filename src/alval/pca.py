"""Normed principal component analysis with the usual interpretation aids.

Mirrors what FactoMineR::PCA returns: eigenvalues, coordinates, squared cosines
(quality of representation), contributions, and supplementary variables projected
on the axes through their correlations.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    eig: pd.DataFrame          # eigenvalue, % of variance, cumulative %
    ind_coord: pd.DataFrame
    ind_cos2: pd.DataFrame
    ind_contrib: pd.DataFrame
    var_coord: pd.DataFrame    # = correlations with the axes for a normed PCA
    var_cos2: pd.DataFrame
    var_contrib: pd.DataFrame
    sup_coord: pd.DataFrame | None = None


def pca(X, n_axes=None, supplementary=None):
    """Run a normed PCA on the columns of `X` (a DataFrame).

    `supplementary` is an optional DataFrame of quantitative variables, aligned on
    the rows of `X`, projected afterwards without taking part in the axes.
    """
    Z = (X - X.mean()) / X.std(ddof=0)
    n, p = Z.shape
    corr = (Z.T @ Z) / n
    values, vectors = np.linalg.eigh(corr.to_numpy())
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    k = n_axes or p
    axes = [f"Dim.{i + 1}" for i in range(k)]

    eig = pd.DataFrame({
        "eigenvalue": values,
        "percent": 100 * values / values.sum(),
    }, index=[f"Dim.{i + 1}" for i in range(p)])
    eig["cumulative"] = eig["percent"].cumsum()

    F = Z.to_numpy() @ vectors[:, :k]
    ind_coord = pd.DataFrame(F, index=X.index, columns=axes)
    dist2 = (Z ** 2).sum(axis=1).to_numpy()[:, None]
    ind_cos2 = pd.DataFrame(F ** 2 / dist2, index=X.index, columns=axes)
    ind_contrib = pd.DataFrame(100 * F ** 2 / (n * values[:k]), index=X.index, columns=axes)

    G = vectors[:, :k] * np.sqrt(values[:k])
    var_coord = pd.DataFrame(G, index=X.columns, columns=axes)
    var_cos2 = var_coord ** 2
    var_contrib = 100 * var_cos2 / values[:k]

    sup = None
    if supplementary is not None:
        sup = pd.DataFrame(
            {ax: [np.corrcoef(supplementary[c], ind_coord[ax])[0, 1] for c in supplementary] for ax in axes},
            index=supplementary.columns,
        )
    return PCAResult(eig, ind_coord, ind_cos2, ind_contrib, var_coord, var_cos2, var_contrib, sup)
