"""Features of the error model and the error models themselves.

Everything here is available at every pixel without a field visit: no survey metadata,
no reference label of the region being validated.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import CLASSES, INDICATORS, PRODUCTS, embedding_columns


def enriched(df, product):
    """The six indicators, plus what the product says and how the other products agree."""
    X = df[INDICATORS].copy()
    claimed = df[product]
    for c in CLASSES:
        X[f"{product}_is_{c}"] = (claimed == c).astype(float)
    for other in PRODUCTS:
        if other == product:
            continue
        X[f"agrees_{other}"] = (claimed == df[other]).astype(float)
        for c in CLASSES:
            X[f"{other}_is_{c}"] = (df[other] == c).astype(float)
    X["oso_valid_obs"] = df["oso_valid_obs"]
    return X


def generic(df, product, opinion=None):
    """Features that never name a product, so that a model can be applied to a new one."""
    others = [p for p in PRODUCTS if p != product]
    claimed = df[product].to_numpy()
    X = df[INDICATORS].copy()
    for c in CLASSES:
        X[f"claims_{c}"] = (claimed == c).astype(float)
    X["n_agree"] = sum((claimed == df[o].to_numpy()).astype(float) for o in others)
    votes = df[others].to_numpy()
    majority = np.array([np.bincount(row, minlength=8).argmax() for row in votes])
    for c in CLASSES:
        X[f"majority_{c}"] = (majority == c).astype(float)
    X["oso_valid_obs"] = df["oso_valid_obs"]
    if opinion is not None:
        rows = np.arange(len(df))
        X["opinion_claimed"] = opinion[rows, claimed - 1]
        X["opinion_others"] = np.mean([opinion[rows, df[o].to_numpy() - 1] for o in others], axis=0)
        X = _opinion_summary(X, opinion)
    return X


# ---------------------------------------------------------------- "fifth opinion"

def land_cover_classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))


def class_probabilities(train, target):
    """P(true class = c | satellite embedding) at the target points, c = 1..7."""
    emb = embedding_columns(train)
    clf = land_cover_classifier().fit(train[emb], train["lucas_class"])
    proba = np.zeros((len(target), len(CLASSES)))
    proba[:, clf.classes_ - 1] = clf.predict_proba(target[emb])
    return proba


def out_of_region_probabilities(regions):
    """Class probabilities of each region predicted by a classifier trained elsewhere."""
    return {name: class_probabilities(pd.concat([d for k, d in regions.items() if k != name]), d)
            for name, d in regions.items()}


def with_opinion(X, df, product, opinion):
    rows = np.arange(len(df))
    X = X.copy()
    X["opinion_claimed"] = opinion[rows, df[product].to_numpy() - 1]
    for other in PRODUCTS:
        if other != product:
            X[f"opinion_{other}"] = opinion[rows, df[other].to_numpy() - 1]
    return _opinion_summary(X, opinion)


def _opinion_summary(X, opinion):
    """How sure the classifier is, whatever the maps say."""
    X["opinion_max"] = opinion.max(axis=1)
    X["opinion_entropy"] = -(opinion * np.log(np.clip(opinion, 1e-9, 1))).sum(axis=1)
    return X


# ---------------------------------------------------------------- error models

def error_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def transferred_errors(train_regions, target, product, features="opinion"):
    """Error probability at the target points, learned in the other regions.

    features: "indicators", "enriched" or "opinion" (enriched + fifth opinion).
    """
    train = pd.concat(train_regions.values(), ignore_index=True)
    if features == "indicators":
        Xa, Xt = train[INDICATORS], target[INDICATORS]
    else:
        Xa = pd.concat([enriched(d, product) for d in train_regions.values()], ignore_index=True)
        Xt = enriched(target, product)
        if features == "opinion":
            oor = out_of_region_probabilities(train_regions)
            Xa = pd.concat([with_opinion(enriched(d, product), d, product, oor[k])
                            for k, d in train_regions.items()], ignore_index=True)
            Xt = with_opinion(Xt, target, product, class_probabilities(train, target))
    model = error_model().fit(Xa, train[f"err_{product}"])
    return model.predict_proba(Xt)[:, 1]


def new_product_errors(train_regions, target, product):
    """Error probability for a product never validated: one model learned on the errors of
    the three other products, with product-agnostic features."""
    oor = out_of_region_probabilities(train_regions)
    X, y = [], []
    for k, d in train_regions.items():
        for other in PRODUCTS:
            if other != product:
                X.append(generic(d, other, oor[k]))
                y.append(d[f"err_{other}"])
    model = error_model().fit(pd.concat(X, ignore_index=True), pd.concat(y, ignore_index=True))
    train = pd.concat(train_regions.values(), ignore_index=True)
    return model.predict_proba(generic(target, product, class_probabilities(train, target)))[:, 1]
