"""Phase 6b - triage efficiency: operationalizing the disagreement claim.

Follow-up time is the scarce resource. Given a budget to observe the top-k
targets, which ranking recovers the most true planets? We compare:
  * ML probability alone (the conventional output),
  * a physics-informed score that fuses the ML probability with the physics
    verdict (so physics-pass planets the ML buried get surfaced),
  * random ordering (floor), and
  * an oracle that sorts by the true label (ceiling).

If the physics-informed ranking recovers planets faster than probability alone,
that is direct, operational evidence that the independent physics verdict --- and
hence the ML-physics disagreement it encodes --- carries actionable information.

Usage:
    python -m research.triage
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
logger = logging.getLogger("triage")

RESULTS_DIR = ROOT / "research" / "results"
BENCHMARK_PARQUET = RESULTS_DIR / "tess_benchmark.parquet"
IMG_DIR = ROOT / "docs" / "img"
SEED = 42


def load() -> pd.DataFrame:
    df = pd.read_parquet(BENCHMARK_PARQUET)
    df = df[(df["success"] == 1) & df["probability"].notna()].copy()
    df["physics_score"] = (
        df["sde_pass"].astype(int) + df["odd_even_ok"].astype(int)
        + df["duration_ok"].astype(int) + df["density_ok"].astype(int)
        + (1 - df["has_secondary"].astype(int))
    )
    return df


def gain_curve(labels: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Cumulative fraction of all true planets recovered as targets are followed up."""
    total = labels.sum()
    if total == 0:
        return np.zeros(len(labels))
    return np.cumsum(labels[order]) / total


def budget_to_recall(recall: np.ndarray, frac: float) -> float:
    """Fraction of targets that must be followed up to recover `frac` of planets."""
    hit = np.where(recall >= frac)[0]
    return float((hit[0] + 1) / len(recall)) if len(hit) else 1.0


def main() -> None:
    if not BENCHMARK_PARQUET.exists():
        raise SystemExit("Run research.tess_benchmark first.")
    df = load().reset_index(drop=True)
    y = df["label"].astype(int).values
    prob = df["probability"].astype(float).values
    phys = df["physics_score"].astype(float).values / 5.0
    n, total_planets = len(y), int(y.sum())
    rng = np.random.default_rng(SEED)

    # physics-informed score: mean of calibrated ML probability and physics score.
    combined = 0.5 * prob + 0.5 * phys
    # physics-pass-tiered: all physics-pass targets (by prob) before physics-fail.
    physics_pass = df["physics_pass"].astype(int).values
    tiered = physics_pass + prob   # pass (>=1) always outranks fail (<1)

    strategies = {
        "ML probability": np.argsort(-prob, kind="stable"),
        "Physics-informed (ML+physics)": np.argsort(-combined, kind="stable"),
        "Physics-pass first, then ML": np.argsort(-tiered, kind="stable"),
        "Oracle (true label)": np.argsort(-y, kind="stable"),
    }
    curves = {name: gain_curve(y, order) for name, order in strategies.items()}
    # random floor = average of many shuffles
    rand = np.mean([gain_curve(y, rng.permutation(n)) for _ in range(500)], axis=0)
    curves["Random"] = rand

    x = np.arange(1, n + 1) / n
    summary = {"n_targets": n, "n_planets": total_planets}
    for name, recall in curves.items():
        summary[name] = {
            "gain_auc": round(float(recall.mean()), 4),           # area under gain curve
            "budget_to_50pct_planets": round(budget_to_recall(recall, 0.5), 3),
            "budget_to_80pct_planets": round(budget_to_recall(recall, 0.8), 3),
        }
    logger.info("Triage summary:\n%s", json.dumps(summary, indent=2))
    (RESULTS_DIR / "triage_summary.json").write_text(json.dumps(summary, indent=2))

    # Figure
    import matplotlib.pyplot as plt

    fig, ax = theme.new_fig("Follow-up efficiency: planets recovered vs budget", figsize=(7.6, 5.0))
    styles = {
        "Oracle (true label)": dict(color=theme.INK_MUTED, ls="--", lw=1.5),
        "Physics-pass first, then ML": dict(color=theme.POSITIVE, lw=2.5),
        "Physics-informed (ML+physics)": dict(color=theme.ACCENT_2, lw=2),
        "ML probability": dict(color=theme.ACCENT, lw=2.5),
        "Random": dict(color=theme.NEGATIVE, ls=":", lw=1.5),
    }
    for name in ["Oracle (true label)", "Physics-pass first, then ML",
                 "Physics-informed (ML+physics)", "ML probability", "Random"]:
        ax.plot(x * 100, curves[name] * 100, label=name, **styles[name])
    ax.set_xlabel("Targets followed up (% of sample)")
    ax.set_ylabel("True planets recovered (%)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 101)
    theme.legend(ax, loc="lower right")
    theme.save(fig, IMG_DIR / "triage_efficiency.png")
    logger.info("Wrote triage_efficiency.png")

    # Headline comparison
    ml = summary["ML probability"]["budget_to_80pct_planets"]
    pi = summary["Physics-informed (ML+physics)"]["budget_to_80pct_planets"]
    logger.info("To recover 80%% of planets: ML-only needs %.0f%% of targets, "
                "physics-informed needs %.0f%%.", ml * 100, pi * 100)


if __name__ == "__main__":
    main()
