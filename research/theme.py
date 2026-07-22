"""Shared figure theme for all research figures, with a light/dark switch.

Every figure reads as one visual system: a single accent hue (indigo), reserved
status colors (emerald = planet, rose = false positive), sequential 'viridis' for
matrices. The palette matches train_model.py.

Theme is selected by the FIG_THEME environment variable:
  * FIG_THEME=dark  (default) -> GitHub-dark surface; figures land in docs/img/.
  * FIG_THEME=light          -> white surface for print/journals; figures land
                                in docs/img/light/ so the dark set is preserved.

No call site changes: modules keep calling theme.new_fig / theme.save with the
docs/img path; save() re-roots to the light subdirectory when in light mode.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEME = os.environ.get("FIG_THEME", "dark").lower()
_LIGHT = THEME == "light"

if _LIGHT:
    SURFACE = "#ffffff"      # white page
    PANEL = "#f6f8fa"        # light panel (legends)
    INK = "#1f2328"          # near-black text
    INK_MUTED = "#57606a"    # muted gray text
    GRID = "#d0d7de"         # light grid
    ACCENT = "#4f46e5"       # indigo (darker for white bg)
    ACCENT_2 = "#0891b2"     # cyan
    POSITIVE = "#059669"     # emerald
    NEGATIVE = "#dc2626"     # rose
    WARN = "#b45309"         # amber
else:
    SURFACE = "#0d1117"
    PANEL = "#161b22"
    INK = "#e6edf3"
    INK_MUTED = "#8b949e"
    GRID = "#21262d"
    ACCENT = "#818cf8"
    ACCENT_2 = "#22d3ee"
    POSITIVE = "#34d399"
    NEGATIVE = "#f87171"
    WARN = "#f59e0b"

SEQUENTIAL_CMAP = "viridis"   # perceptually uniform on both surfaces


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
    """Save a figure. In light mode, re-root docs/img/<f> -> docs/img/light/<f>."""
    path = Path(path)
    if _LIGHT and path.parent.name == "img":
        path = path.parent / "light" / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    # pad_inches gives axis labels breathing room (fixes tight-bbox clipping).
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12, facecolor=SURFACE)
    plt.close(fig)
