"""In-mission TESS model: training and evaluation done honestly.

The shipped v3 model is Kepler-trained and deployed on TESS, which costs a lot
(ROC-AUC 0.96 -> 0.72). This module trains *in-mission* on the labeled TESS
benchmark instead, and - more importantly - evaluates it under a protocol that
does not flatter itself.

Protocol
--------
1. **Held-out test set, touched once.** 20% of stars are split off before any
   model selection and scored exactly once at the end. Cross-validation that
   also selects hyperparameters is optimistic even when each fold is clean, so
   the headline number comes from data no selection step ever saw.

2. **Grouped by star.** 106 stars in ExoFOP carry more than one labeled TOI and
   five carry both a planet and a false positive. Splitting by signal would put
   the same star's stellar parameters and systematics on both sides of the
   split. Groups are TIC IDs (`StratifiedGroupKFold`).

3. **Stratified by label and detection quality.** Targets whose period TLS
   actually recovered score far better (AUC 0.76) than those it did not (0.61),
   so folds that differ in detection quality differ in difficulty. Stratifying
   on the (label, period_recovered) pair keeps folds comparable.

4. **Nested selection.** Hyperparameters and the decision threshold are chosen
   inside the training folds only - the same discipline that showed the 0.117
   operating point was real rather than tuned on the evaluation set.

Metrics
-------
ROC-AUC is reported but is not sufficient on its own: it is threshold-free and
therefore hides exactly the operating-point failure that costs this pipeline
most of its recall. We also report PR-AUC (which degrades honestly under class
imbalance), recall at a fixed precision (what a follow-up programme actually
budgets against), and calibration (Brier), plus the chosen threshold.

Usage:
    python -m research.train_tess                 # 500-target benchmark
    python -m research.train_tess --tag ms        # multi-sector run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_tess")

RESULTS_DIR = ROOT / "research" / "results"
SEED = 42
TEST_FRACTION = 0.2
TARGET_PRECISION = 0.80   # follow-up programmes budget against precision, not AUC

# Scalar TLS features (the eight the shipped model uses).
BASE = ["period_days", "depth_ppm", "duration_hrs", "model_snr", "rp_rs",
        "log10_depth", "log10_period", "duration_over_period"]
# Physics diagnostics the pipeline computes but the shipped model never sees.
PHYSICS = ["sde", "welch_p", "duration_ratio", "density_ratio", "secondary_snr"]
# Folded-transit shape: the V-vs-U eclipsing-binary discriminator.
SHAPE = ["shape_vu", "flat_bottom_frac", "transit_symmetry",
         "symmetry", "shape_ratio", "depth_std"]
# Stellar context, including limb darkening (a proxy for Teff / log g).
STELLAR = ["stellar_r", "stellar_m", "ld_a", "ld_b", "radius_unc", "mass_unc"]
# Deliberately excluded: n_points / n_sectors. They encode observing strategy -
# targets prioritised for follow-up got more data - which correlates with being
# a real planet without being a property of the signal. That is selection
# leakage, and it inflates scores on this benchmark while transferring nothing.

GRIDS = [
    {"model": "rf", "n_estimators": 800, "min_samples_leaf": 1, "max_features": 0.4},
    {"model": "rf", "n_estimators": 500, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"model": "et", "n_estimators": 800, "min_samples_leaf": 1, "max_features": 0.4},
]


def build(cfg: dict):
    kw = {k: v for k, v in cfg.items() if k != "model"}
    cls = RandomForestClassifier if cfg["model"] == "rf" else ExtraTreesClassifier
    return cls(random_state=SEED, n_jobs=-1, class_weight="balanced_subsample", **kw)


def load(tag: str) -> pd.DataFrame:
    suffix = f"_{tag}" if tag else ""
    path = RESULTS_DIR / f"tess_benchmark{suffix}.csv"
    if not path.exists():
        shards = sorted(RESULTS_DIR.glob(f"tess_benchmark{suffix}_shard*.csv"))
        if not shards:
            raise SystemExit(f"No benchmark data at {path}")
        df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
        logger.info("Loaded %d shards", len(shards))
    else:
        df = pd.read_csv(path)
    df = df[df["success"] == 1].copy()
    df = df.drop_duplicates(subset="tic_id")
    df["period_recovered"] = pd.to_numeric(df["period_recovered"], errors="coerce").fillna(0)
    return df


def features_present(df: pd.DataFrame) -> list[str]:
    want = BASE + PHYSICS + SHAPE + STELLAR
    have = [f for f in want if f in df.columns]
    missing = [f for f in want if f not in df.columns]
    if missing:
        logger.warning("Missing (older benchmark run): %s", ", ".join(missing))
    return have


def strata(df: pd.DataFrame) -> np.ndarray:
    """Stratify on label AND detection quality, so folds match in difficulty."""
    return (df["label"].astype(int) * 2 + df["period_recovered"].astype(int)).values


def recall_at_precision(y, p, target: float) -> tuple[float, float]:
    """Highest recall reachable while holding precision >= target."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = prec[:-1] >= target
    if not ok.any():
        return 0.0, 1.0
    i = int(np.argmax(rec[:-1] * ok))
    return float(rec[i]), float(thr[i])


def pick_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def evaluate(y, p, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    rec_at_p, _ = recall_at_precision(y, p, TARGET_PRECISION)
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "threshold": round(thr, 4),
        "accuracy": round((tp + tn) / max(len(y), 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        f"recall_at_precision_{TARGET_PRECISION}": round(rec_at_p, 4),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


def nested_select(X, y, g, s) -> dict:
    """Choose a configuration using only the training data."""
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=SEED)
    best, best_cfg = -np.inf, GRIDS[0]
    for cfg in GRIDS:
        oof = np.zeros(len(y))
        for tr, te in inner.split(X, s, groups=g):
            oof[te] = build(cfg).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        score = average_precision_score(y, oof)   # select on PR-AUC
        logger.info("  cfg %-46s inner PR-AUC %.4f", str(cfg), score)
        if score > best:
            best, best_cfg = score, cfg
    return best_cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="", help="Benchmark tag (e.g. 'ms' for multi-sector)")
    args = ap.parse_args()

    df = load(args.tag)
    feats = features_present(df)
    X = np.nan_to_num(df[feats].values.astype(float))
    y = df["label"].astype(int).values
    g = df["tic_id"].astype(int).values
    s = strata(df)
    logger.info("n=%d (%d planets), %d features, %d unique stars",
                len(df), int(y.sum()), len(feats), len(np.unique(g)))

    # --- 1. hold out a test set, grouped by star, before any selection -------
    holdout = StratifiedGroupKFold(int(round(1 / TEST_FRACTION)), shuffle=True, random_state=SEED)
    dev_idx, test_idx = next(holdout.split(X, s, groups=g))
    logger.info("dev n=%d, held-out test n=%d (stars disjoint)", len(dev_idx), len(test_idx))
    assert not (set(g[dev_idx]) & set(g[test_idx])), "star leaked across the holdout split"

    # --- 2. select model on dev only ----------------------------------------
    logger.info("Selecting configuration on dev (nested, grouped):")
    cfg = nested_select(X[dev_idx], y[dev_idx], g[dev_idx], s[dev_idx])
    logger.info("chosen: %s", cfg)

    # --- 3. cross-validated dev estimate + threshold -------------------------
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(dev_idx))
    for tr, te in outer.split(X[dev_idx], s[dev_idx], groups=g[dev_idx]):
        oof[te] = build(cfg).fit(X[dev_idx][tr], y[dev_idx][tr]).predict_proba(X[dev_idx][te])[:, 1]
    thr = pick_threshold(y[dev_idx], oof)          # threshold from dev only
    dev_metrics = evaluate(y[dev_idx], oof, thr)

    # --- 4. fit on all dev, score the held-out test exactly once -------------
    final = build(cfg).fit(X[dev_idx], y[dev_idx])
    p_test = final.predict_proba(X[test_idx])[:, 1]
    test_metrics = evaluate(y[test_idx], p_test, thr)

    imp = sorted(zip(feats, final.feature_importances_), key=lambda t: -t[1])
    report = {
        "n_total": int(len(df)), "n_dev": int(len(dev_idx)), "n_test": int(len(test_idx)),
        "n_features": len(feats), "features": feats,
        "excluded_as_leakage": ["n_points", "n_sectors"],
        "config": cfg,
        "dev_crossval": dev_metrics,
        "heldout_test": test_metrics,
        "top_features": [{"feature": f, "importance": round(float(v), 4)} for f, v in imp[:12]],
    }
    suffix = f"_{args.tag}" if args.tag else ""
    (RESULTS_DIR / f"train_tess{suffix}.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 74)
    print(f"IN-MISSION TESS MODEL  (n={len(df)}, {len(feats)} features)")
    print("=" * 74)
    print(f"protocol: star-grouped, label+detection stratified, nested selection,")
    print(f"          held-out test scored once at threshold chosen on dev\n")
    for name, m in (("dev (cross-validated)", dev_metrics), ("HELD-OUT TEST", test_metrics)):
        print(f"{name}:")
        print(f"  ROC-AUC {m['roc_auc']:.4f}   PR-AUC {m['pr_auc']:.4f}   Brier {m['brier']:.4f}")
        print(f"  acc {m['accuracy']:.4f}  recall {m['recall']:.4f}  prec {m['precision']:.4f}  "
              f"F1 {m['f1']:.4f}  @thr {m['threshold']:.3f}")
        print(f"  recall @ precision>={TARGET_PRECISION}: "
              f"{m[f'recall_at_precision_{TARGET_PRECISION}']:.4f}")
    print("\ntop features:")
    for r in report["top_features"][:8]:
        print(f"  {r['feature']:22} {r['importance']:.4f}")
    print("=" * 74)
    logger.info("Wrote train_tess%s.json", suffix)


if __name__ == "__main__":
    main()
