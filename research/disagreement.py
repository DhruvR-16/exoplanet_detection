"""Phase 3 - ML-physics disagreement as a triage signal (the paper's headline).

The pipeline produces two *independent* judgements of every target:
  * a calibrated ML probability (learned from Kepler feature statistics), and
  * a physics-vetting verdict (5 orthogonal, interpretable checks).

We cross-tabulate them into four quadrants and show that the *disagreement*
quadrants - where the learned model and the physics diverge - are where the
astrophysically interesting, follow-up-worthy objects concentrate. Agreement is
cheap; disagreement is information. This reframes the pipeline output from a
single score into a triage ranking for scarce follow-up resources.

Usage:
    python -m research.disagreement          # reads research/results/tess_benchmark.parquet
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from research import theme  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("disagreement")

RESULTS_DIR = ROOT / "research" / "results"
BENCHMARK_PARQUET = RESULTS_DIR / "tess_benchmark.parquet"
IMG_DIR = ROOT / "docs" / "img"
FIG_SUFFIX = ""   # e.g. "_s3" to keep multi-sector outputs separate


def set_tag(tag: str) -> None:
    """Route inputs/outputs to tag-suffixed files (keeps runs side by side)."""
    global BENCHMARK_PARQUET, FIG_SUFFIX
    suffix = f"_{tag}" if tag else ""
    BENCHMARK_PARQUET = RESULTS_DIR / f"tess_benchmark{suffix}.parquet"
    FIG_SUFFIX = suffix

QUADRANTS = {
    (1, 1): "Both agree: PLANET",
    (0, 0): "Both agree: false positive",
    (1, 0): "ML-optimistic (ML yes, physics no)",
    (0, 1): "ML-skeptical (ML no, physics yes)",
}


def load_benchmark() -> pd.DataFrame:
    if not BENCHMARK_PARQUET.exists():
        raise SystemExit(f"{BENCHMARK_PARQUET} not found - run research.tess_benchmark first.")
    df = pd.read_parquet(BENCHMARK_PARQUET)
    df = df[(df["success"] == 1) & df["probability"].notna()].copy()
    # Physics score: number of the 5 vetting checks passed (0-5).
    df["physics_score"] = (
        df["sde_pass"].astype(int) + df["odd_even_ok"].astype(int)
        + df["duration_ok"].astype(int) + df["density_ok"].astype(int)
        + (1 - df["has_secondary"].astype(int))
    )
    df["ml_planet"] = (df["probability"] >= 0.5).astype(int)
    df["physics_pass"] = df["physics_pass"].astype(int)
    df["quadrant"] = list(zip(df["ml_planet"], df["physics_pass"]))
    return df


def base_rate_analysis(df: pd.DataFrame, prevalences=(0.497, 0.25, 0.10)) -> dict:
    """Prevalence-adjusted quadrant purity.

    The benchmark is class-balanced, but purity (precision) depends on the base
    rate. We report the *prevalence-independent* class-conditional rates
    P(quadrant | planet) and P(quadrant | FP), then Bayes-adjust purity to
    several deployment prevalences:
        P(planet | q) = pi * P(q|planet) / [pi*P(q|planet) + (1-pi)*P(q|FP)]
    pi = 0.497 is the real dispositioned-TOI base rate (ExoFOP CP/KP vs FP/FA);
    lower values illustrate a blind search where planets are rarer.
    """
    n_planet = int((df["label"] == 1).sum())
    n_fp = int((df["label"] == 0).sum())
    out = {"prevalences": list(prevalences), "quadrants": []}
    for key, name in QUADRANTS.items():
        sub = df[df["quadrant"] == key]
        p_given_planet = (sub["label"] == 1).sum() / n_planet if n_planet else 0.0
        p_given_fp = (sub["label"] == 0).sum() / n_fp if n_fp else 0.0
        purities = {}
        for pi in prevalences:
            num = pi * p_given_planet
            den = num + (1 - pi) * p_given_fp
            purities[f"pi_{pi}"] = round(num / den, 3) if den > 0 else None
        out["quadrants"].append({
            "quadrant": name,
            "p_given_planet": round(float(p_given_planet), 3),
            "p_given_fp": round(float(p_given_fp), 3),
            "purity_by_prevalence": purities,
        })
    return out


def quadrant_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, name in QUADRANTS.items():
        sub = df[df["quadrant"] == key]
        n = len(sub)
        purity = float(sub["label"].mean()) if n else float("nan")   # fraction truly planets
        rows.append({
            "quadrant": name,
            "ml_planet": key[0], "physics_pass": key[1],
            "n": n,
            "planet_purity": round(purity, 3) if n else None,
            "n_true_planet": int(sub["label"].sum()) if n else 0,
            "n_true_fp": int((sub["label"] == 0).sum()) if n else 0,
            "mean_probability": round(float(sub["probability"].mean()), 3) if n else None,
        })
    return pd.DataFrame(rows)


def make_figures(df: pd.DataFrame, table: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    # 1. 2x2 quadrant panel: probability (x) vs physics-pass (y), tiles annotated
    fig, ax = plt.subplots(figsize=(7.8, 5.6), dpi=150, facecolor=theme.SURFACE)
    ax.set_facecolor(theme.SURFACE)
    ax.axvline(0.5, color=theme.GRID, lw=1)
    ax.axhline(0.5, color=theme.GRID, lw=1)
    # jitter physics_pass for visibility
    rng = np.random.default_rng(0)
    yj = df["physics_pass"] + rng.uniform(-0.18, 0.18, len(df))
    planet = df["label"] == 1
    ax.scatter(df.loc[planet, "probability"], yj[planet], s=42, color=theme.POSITIVE,
               edgecolor="white", linewidth=0.4, alpha=0.85, label="True planet")
    ax.scatter(df.loc[~planet, "probability"], yj[~planet], s=42, color=theme.NEGATIVE,
               edgecolor="white", linewidth=0.4, alpha=0.85, label="True false positive")
    # quadrant annotations
    for key, name in QUADRANTS.items():
        r = table[(table["ml_planet"] == key[0]) & (table["physics_pass"] == key[1])].iloc[0]
        x = 0.75 if key[0] else 0.25
        y = 1.32 if key[1] else -0.32
        purity = "" if r["planet_purity"] is None else f"\n{r['planet_purity']:.0%} planets"
        ax.text(x, y, f"{name}\nn={r['n']}{purity}", ha="center", va="center",
                color=theme.INK, fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=theme.PANEL, edgecolor=theme.GRID))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Physics FAIL", "Physics PASS"])
    ax.set_xlabel("Calibrated ML planet probability")
    ax.set_title("ML–physics agreement quadrants (TESS test set)", fontsize=12,
                 fontweight="bold", pad=12, loc="left", color=theme.INK)
    ax.tick_params(colors=theme.INK_MUTED, labelsize=9)
    theme.legend(ax, loc="center left")
    theme.save(fig, IMG_DIR / f"disagreement_quadrants{FIG_SUFFIX}.png")

    # 2. Probability vs physics-score (0-5), colored by truth
    fig, ax = theme.new_fig("ML probability vs physics vetting score")
    yj = df["physics_score"] + rng.uniform(-0.15, 0.15, len(df))
    ax.scatter(df.loc[planet, "probability"], yj[planet], s=40, color=theme.POSITIVE,
               edgecolor="white", linewidth=0.4, alpha=0.85, label="True planet")
    ax.scatter(df.loc[~planet, "probability"], yj[~planet], s=40, color=theme.NEGATIVE,
               edgecolor="white", linewidth=0.4, alpha=0.85, label="True false positive")
    ax.axvline(0.5, color=theme.GRID, lw=1)
    ax.set_xlabel("Calibrated ML planet probability")
    ax.set_ylabel("Physics checks passed (of 5)")
    ax.set_yticks(range(6))
    theme.legend(ax, loc="lower right")
    theme.save(fig, IMG_DIR / f"disagreement_score_scatter{FIG_SUFFIX}.png")


def main() -> None:
    df = load_benchmark()
    table = quadrant_table(df)
    logger.info("Quadrant breakdown (n=%d):\n%s", len(df), table.to_string(index=False))

    # Representative disagreement targets for the paper's discussion table.
    disagree = df[df["ml_planet"] != df["physics_pass"]].copy()
    cols = ["tic_id", "toi", "disposition", "label", "probability", "physics_score",
            "sde", "welch_p", "duration_ratio", "density_ratio", "period_days"]
    disagree = disagree[cols].sort_values("probability", ascending=False)
    disagree.to_csv(RESULTS_DIR / f"disagreement_targets{FIG_SUFFIX}.csv", index=False)
    table.to_csv(RESULTS_DIR / f"disagreement_quadrants{FIG_SUFFIX}.csv", index=False)

    # Summary: is agreement more reliable than disagreement? (the key claim)
    agree = df[df["ml_planet"] == df["physics_pass"]]
    summary = {
        "n_total": int(len(df)),
        "n_agree": int(len(agree)),
        "n_disagree": int(len(disagree)),
        "agree_planet_purity_when_both_yes": _purity(df, (1, 1)),
        "agree_fp_purity_when_both_no": _purity(df, (0, 0), want_fp=True),
        "disagree_ml_optimistic_planet_purity": _purity(df, (1, 0)),
        "disagree_ml_skeptical_planet_purity": _purity(df, (0, 1)),
        "base_rate_adjusted": base_rate_analysis(df),
    }
    (RESULTS_DIR / f"disagreement_summary{FIG_SUFFIX}.json").write_text(json.dumps(summary, indent=2))
    logger.info("Summary: %s", json.dumps(summary, indent=2))

    make_figures(df, table)
    logger.info("Wrote disagreement figures and tables to %s and %s", IMG_DIR, RESULTS_DIR)


def _purity(df: pd.DataFrame, key: tuple[int, int], want_fp: bool = False) -> float | None:
    sub = df[df["quadrant"] == key]
    if not len(sub):
        return None
    frac_planet = float(sub["label"].mean())
    return round(1 - frac_planet if want_fp else frac_planet, 3)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="", help="Read/write tag-suffixed files (e.g. 's3')")
    set_tag(ap.parse_args().tag)
    main()
