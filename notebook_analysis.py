"""Run the main.ipynb walkthrough as a reusable command line program.

Example: python notebook_analysis.py --target "TOI-270" --latest --refresh
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline

SURFACE, PANEL = "#0d1117", "#161b22"
INK, MUTED, GRID = "#e6edf3", "#8b949e", "#21262d"
ACCENT, RED, AMBER = "#818cf8", "#f87171", "#f59e0b"


def dark_ax(ax, title: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if title:
        ax.set_title(title, color=INK, fontweight="bold", loc="left", fontsize=11)
    ax.grid(color=GRID, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def dark_legend(ax, loc: str = "lower right") -> None:
    legend = ax.legend(loc=loc, facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9)
    legend.get_frame().set_alpha(0.9)


def save_plots(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = result["target"]
    arrays, tls = result["arrays"], result["tls"]
    period = result["features"]["period_days"]
    duration = result["features"]["duration_hrs"] / 24
    sde = result["diagnostics"]["sde"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), dpi=140, facecolor=SURFACE, sharex=True)
    axes[0].plot(arrays["time"], arrays["raw_flux"], ".", ms=1.5, color=MUTED,
                 alpha=0.5, label="Raw flux")
    axes[0].plot(arrays["time"], arrays["trend"], "-", color=RED, lw=1.2,
                 label="Wotan biweight trend")
    axes[1].plot(arrays["time"], arrays["flat_flux"], ".", ms=1.5,
                 color=ACCENT, alpha=0.5, label="Detrended flux")
    dark_ax(axes[0], f"{target} — raw flux and stellar-variability trend")
    dark_ax(axes[1])
    axes[1].set_xlabel("Time (BTJD days)")
    for ax in axes:
        ax.set_ylabel("Normalized flux")
        dark_legend(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "pipeline_detrend.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4), dpi=140, facecolor=SURFACE)
    ax.plot(tls.periods, tls.power, color=ACCENT, lw=0.9)
    ax.axvline(period, color=RED, ls="--", alpha=0.8)
    ax.axhline(pipeline.SDE_THRESHOLD, color=AMBER, ls=":", alpha=0.7)
    ax.annotate(f"P = {period:.3f} d", xy=(period * 1.06, sde * 0.94),
                color=INK, fontsize=10, fontweight="bold")
    dark_ax(ax, f"{target} — TLS periodogram")
    ax.set_xlabel("Trial period (days)")
    ax.set_ylabel("SDE")
    fig.tight_layout()
    fig.savefig(output_dir / "pipeline_periodogram.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140, facecolor=SURFACE)
    ax.plot(tls.folded_phase, tls.folded_y, ".", ms=2.5, color=MUTED,
            alpha=0.45, label="Folded flux")
    ax.plot(tls.model_folded_phase, tls.model_folded_model, color=RED, lw=2,
            label="TLS transit model")
    window = max(2.5 * duration / period, 0.02)
    ax.set_xlim(0.5 - window, 0.5 + window)
    dark_ax(ax, f"{target} — phase-folded transit (P = {period:.4f} d)")
    ax.set_xlabel("Orbital phase")
    ax.set_ylabel("Relative flux")
    dark_legend(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "pipeline_phase_fold.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="TOI-270", help="TESS target name or TIC ID")
    parser.add_argument("--max-sectors", type=int, default=pipeline.DEFAULT_MAX_SECTORS)
    parser.add_argument("--latest", action="store_true", help="select newest available sectors")
    parser.add_argument("--refresh", action="store_true", help="bypass lightcurve cache")
    parser.add_argument("--output-dir", type=Path, default=pipeline.ROOT_DIR / "analysis_output")
    args = parser.parse_args()
    if args.max_sectors < 1:
        parser.error("--max-sectors must be positive")
    logging.basicConfig(level=logging.WARNING)
    model_pkg, version = pipeline.load_model()
    if model_pkg is None:
        parser.error("No model found; run train_model.py first")
    result = pipeline.analyze(args.target, model_pkg=model_pkg,
                              max_sectors=args.max_sectors,
                              sector_order="latest" if args.latest else "earliest",
                              refresh_cache=args.refresh)
    output = args.output_dir / re.sub(r"[^A-Za-z0-9_.-]", "_", args.target)
    save_plots(result, output)
    summary = {
        "target": args.target,
        "model_version": version,
        "data": result["data"],
        "stellar": result["stellar"],
        "features": result["features"],
        "diagnostics": {k: v for k, v in result["diagnostics"].items() if k != "t0"},
        "vetting": result["vetting"],
        "prediction": result["prediction"],
        "explanation": result["explanation"],
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(f"Model {version}; {args.target}: {result['data']['n_sectors']} sector(s), "
          f"{result['data']['n_points']:,} cadences")
    for line in result["explanation"]:
        print("-", line)
    print(f"Plots and result: {output}")


if __name__ == "__main__":
    main()
