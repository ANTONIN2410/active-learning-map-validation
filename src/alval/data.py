"""Loading the LUCAS reference points and the map products evaluated on them."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CLASSES = {
    1: "built-up",
    2: "cropland",
    3: "herbaceous",
    4: "trees",
    5: "shrubs",
    6: "water",
    7: "bare soil",
}

PRODUCTS = {
    "wc": "ESA WorldCover",
    "dw": "Dynamic World",
    "oso": "Theia OSO",
    "clc": "CLC+ Backbone",
}

# Six landscape indicators available at every pixel (no field visit needed)
INDICATORS = [
    "n_classes",       # number of distinct classes given by the four products
    "heterogeneity",   # local heterogeneity of the maps around the point
    "dominant_share",  # share of the dominant class around the point
    "edge_distance",   # distance to the nearest class boundary
    "oso_confidence",  # confidence layer of Theia OSO
    "dw_entropy",      # entropy of the Dynamic World class probabilities
]

REGIONS = ["Occitanie", "Brittany", "Grand Est", "Massif Central"]


def load_points(path=None):
    """Return the 10,583 LUCAS in situ points with the four products and the features."""
    return pd.read_parquet(path or DATA_DIR / "lucas_points.parquet")


def embedding_columns(df):
    return [c for c in df.columns if c.startswith("emb_")]


def by_region(df, split=None):
    """Split the table into one frame per region, optionally keeping one half only."""
    if split is not None:
        df = df[df["split"] == split]
    return {r: df[df["region"] == r].reset_index(drop=True) for r in REGIONS}
