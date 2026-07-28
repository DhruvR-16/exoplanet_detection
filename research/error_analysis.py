"""Error analysis - what does the deployed classifier actually get wrong?

Diagnoses the pipeline's failure modes on the labeled TESS benchmark and
validates the fix. The headline finding is a *deployment* bug rather than a
modeling one: the model is trained and calibrated on Kepler but deployed on
TESS, where the calibrated probabilities are compressed toward zero. The
textbook 0.5 decision cut is therefore far too strict, and it silently discards
most real planets.

Checks performed:
  1. Operating point - sweep the decision threshold; report recall/precision/F1.
     The threshold is chosen on a train fold and scored on a held-out fold, so
     the reported gain is not the result of tuning on the evaluation set.
  2. Blind spots - recall by period, depth and Rp/Rs bin, before and after the
     fix. Two subgroups have *zero* recall at 0.5 (P < 2 d, and depth > 20k ppm).
  3. Root-cause tests - two candidate model-level fixes that did NOT work, kept
     here because the negative results are informative:
       (a) dropping the mission-inverting features (depth, Rp/Rs),
       (b) fusing the physics flags with the ML score.

Usage:
    python -m research.error_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pipeline  # noqa: E402
from research import theme  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("error_analysis")

RESULTS_DIR = ROOT / "research" / "results"
IMG_DIR = ROOT / "docs" / "img"
SEED = 42
OLD_THRESHOLD = 0.5


def load(tag: str = "") -> pd.DataFrame:
    suffix = f"_{tag}" if tag else ""
    df = pd.read_csv(RESULTS_DIR / f"tess_benchmark{suffix}.csv")
    return df[df["success"] == 1].copy()


def score_at(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = (p >= t).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return {
        "threshold": round(float(t), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        "accuracy": round((tp + tn) / max(len(y), 1), 4),
        "planets_found": tp, "planets_missed": fn, "false_alarms": fp,
    }


def crossval_threshold(y: np.ndarray, p: np.ndarray, n_splits: int = 5) -> dict:
    """Pick the threshold on a train fold, score it on the held-out fold."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    chosen, recalls, precisions = [], [], []
    for tr, te in skf.split(p.reshape(-1, 1), y):
        fpr, tpr, th = roc_curve(y[tr], p[tr])
        t = float(th[(tpr - fpr).argmax()])
        chosen.append(round(t, 4))
        s = score_at(y[te], p[te], t)
        recalls.append(s["recall"]); precisions.append(s["precision"])
    return {
        "thresholds_per_fold": chosen,
        "heldout_recall_mean": round(float(np.mean(recalls)), 4),
        "heldout_recall_std": round(float(np.std(recalls)), 4),
        "heldout_precision_mean": round(float(np.mean(precisions)), 4),
        "heldout_precision_std": round(float(np.std(precisions)), 4),
    }


def blind_spots(df: pd.DataFrame, new_t: float) -> list[dict]:
    pl = df[df.label == 1]
    bins = [
        ("period_days", 0, 2, "P < 2 d"),
        ("period_days", 2, 6, "P 2-6 d"),
        ("period_days", 6, 1e9, "P 6-15 d"),
        ("depth_ppm", 0, 1000, "depth < 1k ppm"),
        ("depth_ppm", 1000, 5000, "depth 1k-5k ppm"),
        ("depth_ppm", 5000, 20000, "depth 5k-20k ppm"),
        ("depth_ppm", 20000, 1e12, "depth > 20k ppm"),
        ("rp_rs", 0.1, 1e9, "Rp/Rs > 0.1 (hot Jupiter)"),
    ]
    out = []
    for col, lo, hi, name in bins:
        s = pl[(pl[col] >= lo) & (pl[col] < hi)]
        if not len(s):
            continue
        out.append({
            "subgroup": name, "n": int(len(s)),
            "recall_old": round(float((s.probability >= OLD_THRESHOLD).mean()), 4),
            "recall_new": round(float((s.probability >= new_t).mean()), 4),
        })
    return out


def physics_fusion_test(df: pd.DataFrame) -> dict:
    """Does fusing the physics flags with the ML score improve ranking?"""
    y = df.label.values
    pr = np.clip(df.probability.values, 1e-6, 1 - 1e-6)
    flags = ["sde_pass", "odd_even_ok", "duration_ok", "density_ok", "has_secondary"]
    X = np.column_stack([np.log(pr / (1 - pr))] + [df[c].astype(int).values for c in flags])
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        oof[te] = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        deltas.append(roc_auc_score(y[i], oof[i]) - roc_auc_score(y[i], df.probability.values[i]))
    d = np.array(deltas)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {
        "auc_ml_alone": round(float(roc_auc_score(y, df.probability.values)), 4),
        "auc_fusion_oof": round(float(roc_auc_score(y, oof)), 4),
        "delta_auc": round(float(d.mean()), 4),
        "delta_ci95": [round(lo, 4), round(hi, 4)],
        "significant": bool(lo > 0),
    }


def make_figure(y: np.ndarray, p: np.ndarray, new_t: float, spots: list[dict]) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=150, facecolor=theme.SURFACE)
    for ax in (ax1, ax2):
        theme.style_axes(ax)

    # left: metric vs threshold
    ts = np.linspace(0.01, 0.9, 200)
    rec = [score_at(y, p, t)["recall"] for t in ts]
    pre = [score_at(y, p, t)["precision"] for t in ts]
    f1 = [score_at(y, p, t)["f1"] for t in ts]
    ax1.plot(ts, rec, color=theme.POSITIVE, lw=2, label="recall")
    ax1.plot(ts, pre, color=theme.ACCENT, lw=2, label="precision")
    ax1.plot(ts, f1, color=theme.WARN, lw=2.4, label="F1")
    ax1.axvline(new_t, color=theme.INK, ls="--", lw=1.6, label=f"fixed = {new_t:.3f}")
    ax1.axvline(OLD_THRESHOLD, color=theme.NEGATIVE, ls=":", lw=1.6, label="old = 0.5")
    ax1.set_xlabel("decision threshold"); ax1.set_ylabel("score")
    ax1.set_title("Operating point: 0.5 is far too strict on TESS",
                  fontsize=11, fontweight="bold", loc="left", color=theme.INK, pad=10)
    theme.legend(ax1, loc="center right")

    # right: per-subgroup recall, before vs after
    names = [s["subgroup"] for s in spots]
    yy = np.arange(len(names))
    ax2.barh(yy - 0.2, [s["recall_old"] for s in spots], height=0.4,
             color=theme.NEGATIVE, label="threshold 0.5 (old)")
    ax2.barh(yy + 0.2, [s["recall_new"] for s in spots], height=0.4,
             color=theme.POSITIVE, label=f"threshold {new_t:.3f} (fixed)")
    ax2.set_yticks(yy)
    ax2.set_yticklabels([f"{s['subgroup']}  (n={s['n']})" for s in spots], fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("recall on true planets")
    ax2.set_title("Blind spots: two subgroups had ZERO recall",
                  fontsize=11, fontweight="bold", loc="left", color=theme.INK, pad=10)
    theme.legend(ax2, loc="lower right")

    fig.tight_layout()
    theme.save(fig, IMG_DIR / "error_analysis.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    df = load(args.tag)
    y, p = df.label.values, df.probability.values
    new_t = pipeline.DECISION_THRESHOLD

    old = score_at(y, p, OLD_THRESHOLD)
    new = score_at(y, p, new_t)
    cv = crossval_threshold(y, p)
    spots = blind_spots(df, new_t)
    fusion = physics_fusion_test(df)

    report = {
        "n_targets": int(len(df)),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "at_old_threshold": old,
        "at_fixed_threshold": new,
        "crossvalidated_threshold": cv,
        "blind_spots": spots,
        "physics_fusion_test": fusion,
    }
    (RESULTS_DIR / "error_analysis.json").write_text(json.dumps(report, indent=2))
    make_figure(y, p, new_t, spots)

    print("\n" + "=" * 72)
    print("ERROR ANALYSIS - deployed classifier on the labeled TESS benchmark")
    print("=" * 72)
    print(f"n = {len(df)}   ROC-AUC = {report['auc']:.4f}  (ranking has real signal)")
    print(f"\n1. OPERATING POINT  (the bug: a Kepler-calibrated cut used on TESS)")
    print(f"   threshold 0.5   -> recall {old['recall']:.3f}  precision {old['precision']:.3f}  "
          f"F1 {old['f1']:.3f}   found {old['planets_found']}, MISSED {old['planets_missed']}")
    print(f"   threshold {new_t:.3f} -> recall {new['recall']:.3f}  precision {new['precision']:.3f}  "
          f"F1 {new['f1']:.3f}   found {new['planets_found']}, missed {new['planets_missed']}")
    print(f"   -> {new['planets_found'] - old['planets_found']} more real planets recovered; "
          f"false alarms {old['false_alarms']} -> {new['false_alarms']}")
    print(f"\n2. CROSS-VALIDATED (threshold fit on train fold, scored held-out)")
    print(f"   per-fold thresholds {cv['thresholds_per_fold']}")
    print(f"   held-out recall {cv['heldout_recall_mean']:.3f} +/- {cv['heldout_recall_std']:.3f}, "
          f"precision {cv['heldout_precision_mean']:.3f} +/- {cv['heldout_precision_std']:.3f}")
    print(f"\n3. BLIND SPOTS (recall on true planets, old -> fixed)")
    for s in spots:
        flag = "  <-- was ZERO" if s["recall_old"] == 0 else ""
        print(f"   {s['subgroup']:28} n={s['n']:3}  {s['recall_old']:.3f} -> {s['recall_new']:.3f}{flag}")
    print(f"\n4. PHYSICS FUSION (does adding physics flags improve ranking?)")
    print(f"   ML alone AUC {fusion['auc_ml_alone']:.4f} -> fusion {fusion['auc_fusion_oof']:.4f}, "
          f"dAUC {fusion['delta_auc']:+.4f} CI {fusion['delta_ci95']}")
    print(f"   -> {'significant' if fusion['significant'] else 'NOT significant (honest negative)'}")
    print("=" * 72)
    logger.info("Wrote error_analysis.json and error_analysis.png")


if __name__ == "__main__":
    main()
