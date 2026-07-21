"""Shared dark-mode figure theme for all research figures.

Matches the palette used in train_model.py so every figure in the paper and
README reads as one visual system. Single accent hue (indigo), reserved status
colors (rose = false positive / negative), sequential 'Purples' for matrices.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#0d1117"
PANEL = "#161b22"
INK = "#e6edf3"
INK_MUTED = "#8b949e"
GRID = "#21262d"
ACCENT = "#818cf8"      # indigo - primary series
ACCENT_2 = "#22d3ee"    # cyan - secondary series
POSITIVE = "#34d399"    # emerald - confirmed planet
NEGATIVE = "#f87171"    # rose - false positive
WARN = "#f59e0b"        # amber - thresholds / disagreement
SEQUENTIAL_CMAP = "viridis"


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def new_fig(title: str, figsize: tuple[float, float] = (7.2, 4.6)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, dpi=150, facecolor=SURFACE)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, loc="left", color=INK)
    style_axes(ax)
    return fig, ax


def legend(ax: plt.Axes, loc: str = "best") -> None:
    leg = ax.legend(loc=loc, facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9)
    leg.get_frame().set_alpha(0.9)


def save(fig: plt.Figure, path) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
