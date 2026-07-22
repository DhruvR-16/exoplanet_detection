"""Phase 6a - statistical rigor: confidence intervals on the headline numbers.

Turns point estimates into defensible claims:
  * bootstrap 95% CI on the cross-mission ROC-AUC (single- and 3-sector);
  * Wilson score CIs on the four disagreement-quadrant purities;
  * paired bootstrap on the single- vs 3-sector deltas (period recovery, Brier,
    AUC) over the matched star set.

Usage:
    python -m research.statistics
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from research import disagreement as dis  # noqa: E402
from research import theme  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("statistics")

RESULTS_DIR = ROOT / "research" / "results"
IMG_DIR = ROOT / "docs" / "img"
N_BOOT = 2000
SEED = 42


def bootstrap_auc(y: np.ndarray, p: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    point = float(roc_auc_score(y, p))
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion (better than normal for small n)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def paired_delta_ci(a: pd.DataFrame, b: pd.DataFrame, metric: str,
                    n_boot: int = N_BOOT) -> dict:
    """Paired bootstrap CI for metric(b) - metric(a) over matched targets."""
    rng = np.random.default_rng(SEED)
    idx0 = np.arange(len(a))

    def score(df, idx):
        y, p = df["label"].values[idx], df["probability"].values[idx]
        if metric == "auc":
            return roc_auc_score(y, p) if len(set(y)) > 1 else np.nan
        if metric == "brier":
            return brier_score_loss(y, p)
        if metric == "period_recovery":
            return df["period_recovered"].values[idx].mean()
        raise ValueError(metric)

    point = score(b, idx0) - score(a, idx0)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        da, db = score(a, idx), score(b, idx)
        if np.isfinite(da) and np.isfinite(db):
            deltas.append(db - da)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": round(float(point), 4), "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4), "excludes_zero": bool(lo > 0 or hi < 0)}


def _load(tag: str) -> pd.DataFrame:
    dis.set_tag(tag)
    df = dis.load_benchmark()
    return df


def quadrant_purity_cis(df: pd.DataFrame) -> list[dict]:
    rows = []
    for key, name in dis.QUADRANTS.items():
        sub = df[df["quadrant"] == key]
        n = len(sub)
        # report FP-purity for the both-negative quadrant, planet-purity otherwise
        if key == (0, 0):
            k = int((sub["label"] == 0).sum())
            metric = "fp_purity"
        else:
            k = int(sub["label"].sum())
            metric = "planet_purity"
        phat, lo, hi = wilson_ci(k, n)
        rows.append({"quadrant": name, "metric": metric, "n": n,
                     "purity": round(phat, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    return rows


def make_figure(cis1: list[dict], base_rate: float) -> None:
    import matplotlib.pyplot as plt

    labels = [r["quadrant"].replace(" (", "\n(") for r in cis1]
    purities = [r["purity"] for r in cis1]
    err_lo = [r["purity"] - r["ci_low"] for r in cis1]
    err_hi = [r["ci_high"] - r["purity"] for r in cis1]
    colors = [theme.POSITIVE if "planet" in r["metric"] else theme.NEGATIVE for r in cis1]

    fig, ax = theme.new_fig("Quadrant purity with 95% Wilson CIs (single-sector)", figsize=(7.8, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, purities, yerr=[err_lo, err_hi], color=colors, width=0.6,
           error_kw={"ecolor": theme.INK, "elinewidth": 1.5, "capsize": 5})
    ax.axhline(base_rate, color=theme.WARN, ls=":", lw=1.2, alpha=0.8)
    ax.annotate(f"base rate = {base_rate:.2f}", xy=(len(labels) - 1.4, base_rate + 0.02),
                color=theme.WARN, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Purity (planet, or FP for both-negative)"); ax.set_ylim(0, 1.05)
    for i, r in enumerate(cis1):
        ax.text(i, r["ci_high"] + 0.03, f"n={r['n']}", ha="center", color=theme.INK_MUTED, fontsize=8)
    theme.save(fig, IMG_DIR / "quadrant_purity_ci.png")


def main() -> None:
    out: dict = {"n_boot": N_BOOT}

    df1 = _load("")
    y1, p1 = df1["label"].values, df1["probability"].values
    auc1, lo1, hi1 = bootstrap_auc(y1, p1)
    out["tess_auc_s1"] = {"auc": round(auc1, 4), "ci_low": round(lo1, 4), "ci_high": round(hi1, 4)}
    out["base_rate_s1"] = round(float(y1.mean()), 3)
    out["quadrant_purity_s1"] = quadrant_purity_cis(df1)
    logger.info("s1 AUC = %.3f [%.3f, %.3f]", auc1, lo1, hi1)

    try:
        df3 = _load("s3")
        y3, p3 = df3["label"].values, df3["probability"].values
        auc3, lo3, hi3 = bootstrap_auc(y3, p3)
        out["tess_auc_s3"] = {"auc": round(auc3, 4), "ci_low": round(lo3, 4), "ci_high": round(hi3, 4)}
        out["quadrant_purity_s3"] = quadrant_purity_cis(df3)
        logger.info("s3 AUC = %.3f [%.3f, %.3f]", auc3, lo3, hi3)

        # paired sector deltas on matched stars
        a = df1.set_index("tic_id"); b = df3.set_index("tic_id")
        common = a.index.intersection(b.index)
        am = a.loc[common].reset_index(); bm = b.loc[common].reset_index()
        for col in ("period_recovered",):
            for d in (am, bm):
                d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0)
        out["matched_n"] = int(len(common))
        out["sector_delta"] = {
            "period_recovery": paired_delta_ci(am, bm, "period_recovery"),
            "brier": paired_delta_ci(am, bm, "brier"),
            "auc": paired_delta_ci(am, bm, "auc"),
        }
        logger.info("sector deltas (3-sector minus 1-sector): %s",
                    json.dumps(out["sector_delta"], indent=2))
    except SystemExit:
        logger.warning("No s3 benchmark found; skipping sector-delta CIs.")

    make_figure(out["quadrant_purity_s1"], out["base_rate_s1"])
    (RESULTS_DIR / "statistics.json").write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s and quadrant_purity_ci.png", RESULTS_DIR / "statistics.json")


if __name__ == "__main__":
    main()
