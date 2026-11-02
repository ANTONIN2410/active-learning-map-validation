"""Figures used in the notebooks and the README."""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

PALETTE = {"engine": "#1f4e79", "static": "#7fa7c9", "classic": "#c9c9c9", "old": "#e0a458"}


def scree(res, ax=None):
    ax = ax or plt.gca()
    ax.bar(range(1, len(res.eig) + 1), res.eig["percent"], color=PALETTE["static"])
    ax.plot(range(1, len(res.eig) + 1), res.eig["cumulative"], "o-", color=PALETTE["engine"])
    ax.set(xlabel="Axis", ylabel="% of variance", title="Eigenvalues")
    return ax


def correlation_circle(res, axes=("Dim.1", "Dim.2"), ax=None):
    ax = ax or plt.gca()
    a, b = axes
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color="grey", lw=0.8)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    for name, row in res.var_coord.iterrows():
        ax.annotate("", (row[a], row[b]), (0, 0), arrowprops=dict(arrowstyle="->", color=PALETTE["engine"]))
        ax.text(row[a] * 1.08, row[b] * 1.08, name, ha="center", fontsize=8)
    if res.sup_coord is not None:
        k = len(res.sup_coord)
        for i, (name, row) in enumerate(res.sup_coord.iterrows()):
            ax.annotate("", (row[a], row[b]), (0, 0),
                        arrowprops=dict(arrowstyle="->", color=PALETTE["old"], ls="--"))
            # supplementary arrows are often close together: spread their labels vertically
            ax.text(row[a] * 0.6, -0.3 - 0.11 * i, name, ha="center", fontsize=8, color=PALETTE["old"])
    pa, pb = res.eig.loc[a, "percent"], res.eig.loc[b, "percent"]
    ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect="equal",
           xlabel=f"{a} ({pa:.1f}%)", ylabel=f"{b} ({pb:.1f}%)", title="Correlation circle")
    ax.grid(False)
    return ax


def savings_by_scenario(table, ax=None):
    """Grouped bars of median savings (table: designs x scenarios, in %)."""
    ax = ax or plt.gca()
    scenarios = list(table.columns)
    designs = list(table.index)
    colors = [PALETTE["classic"], PALETTE["static"], PALETTE["engine"]][-len(designs):]
    width = 0.8 / len(designs)
    for i, (d, color) in enumerate(zip(designs, colors)):
        bars = ax.bar(np.arange(len(scenarios)) + i * width, table.loc[d], width, label=d, color=color,
                      edgecolor="white")
        ax.bar_label(bars, labels=[f"{v:.0f}" if v > 0 else "" for v in table.loc[d]], fontsize=8, padding=2)
    ax.set_xticks(np.arange(len(scenarios)) + 0.4 - width / 2, [s.replace("_", " ") for s in scenarios])
    ax.set_ylabel("Labels saved vs. random sampling (%)")
    ax.set_ylim(0, table.max().max() * 1.25)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    return ax
