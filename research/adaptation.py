"""Physics-anchored self-training - label-free cross-mission adaptation.

The idea: the five physics checks are *mission-agnostic* (they encode transit
geometry and eclipse morphology, not anything learned from Kepler), so they can
supervise adaptation to TESS without using a single true TESS label. We build
pseudo-labels from the physics verdict alone, train a classifier on TESS
features against those pseudo-labels, and only then evaluate against the real
ExoFOP labels.

This is unsupervised domain adaptation: true labels are used for evaluation
only, never for fitting. All scores are out-of-fold - the adapted model is fit
on a training fold (features + pseudo-labels) and scored on a held-out fold.

Two bars are worth distinguishing:
  * beating PHYSICS-ALONE shows the model learned something beyond the rule that
    supervised it (i.e. the features denoise the weak labels);
  * beating the KEPLER-TRAINED model would make adaptation a practical
    replacement.

The result clears the first bar and not the second, with an interpretable
subgroup structure: adaptation wins precisely in the deep-transit regime where
the Kepler prior is known to be inverted (cf. the feature-direction inversion),
and loses where the Kepler model is already well-oriented. A depth-gated hybrid
that tries to exploit this does not yield a significant overall gain under
nested cross-validation.

Usage:
    python -m research.adaptation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from research import theme  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("adaptation")

RESULTS_DIR = ROOT / "research" / "results"
IMG_DIR = ROOT / "docs" / "img"
SEED = 42
N_BOOT = 2000

FEATURES = ["period_days", "depth_ppm", "duration_hrs", "model_snr",
            "rp_rs", "log10_depth", "log10_period", "duration_over_period"]
FLAGS = ["physics_pass", "sde_pass", "odd_even_ok", "duration_ok", "density_ok", "has_secondary"]
# Pseudo-label rule. Positives need every physics check to pass *and* a strong
# detection; negatives are any physics failure. Deliberately does not use the ML
# probability (that would re-inject the Kepler prior we are trying to escape).
PSEUDO_SDE_MIN = 12.0
GATES_PPM = [5000, 10000, 20000, 30000]


def load(tag: str = "") -> pd.DataFrame:
    suffix = f"_{tag}" if tag else ""
    df = pd.read_csv(RESULTS_DIR / f"tess_benchmark{suffix}.csv")
    df = df[df["success"] == 1].copy()
    for c in FLAGS:
        df[c] = df[c].astype(int)
    return df


def pseudo_labels(df: pd.DataFrame) -> np.ndarray:
    """Weak labels from physics only. -1 marks 'no confident pseudo-label'."""
    pos = ((df.physics_pass == 1) & (df.sde > PSEUDO_SDE_MIN)).values
    neg = (df.physics_pass == 0).values
    out = np.full(len(df), -1, dtype=int)
    out[pos] = 1
    out[neg] = 0
    return out


def physics_score(df: pd.DataFrame) -> np.ndarray:
    """Ordinal physics ranking: how many of the five checks pass."""
    return (df.sde_pass + df.odd_even_ok + df.duration_ok
            + df.density_ok + (1 - df.has_secondary)).values.astype(float)


def _fit(X: np.ndarray, pseudo: np.ndarray, idx: np.ndarray) -> RandomForestClassifier | None:
    m = pseudo[idx] >= 0
    if m.sum() < 20 or len(np.unique(pseudo[idx][m])) < 2:
        return None
    clf = RandomForestClassifier(400, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    return clf.fit(X[idx][m], pseudo[idx][m])


def adapted_oof(X: np.ndarray, y: np.ndarray, pseudo: np.ndarray) -> np.ndarray:
    """Out-of-fold adapted score. Fit uses pseudo-labels only, never y."""
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        clf = _fit(X, pseudo, tr)
        if clf is not None:
            oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def gated_oof(X, y, pseudo, ml, depth) -> tuple[np.ndarray, list[int]]:
    """Depth-gated hybrid, with the gate chosen by inner CV on the train fold."""
    oof = np.zeros(len(y))
    chosen: list[int] = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=7).split(X, y):
        best, best_gate = -np.inf, GATES_PPM[0]
        for g in GATES_PPM:
            inner = np.zeros(len(tr))
            for itr, ite in StratifiedKFold(4, shuffle=True, random_state=1).split(X[tr], y[tr]):
                clf = _fit(X, pseudo, tr[itr])
                if clf is None:
                    continue
                a = clf.predict_proba(X[tr][ite])[:, 1]
                inner[ite] = np.where(depth[tr][ite] > g,
                                      rankdata(a) / len(ite),
                                      rankdata(ml[tr][ite]) / len(ite))
            s = roc_auc_score(y[tr], inner)
            if s > best:
                best, best_gate = s, g
        chosen.append(best_gate)
        clf = _fit(X, pseudo, tr)
        a = clf.predict_proba(X[te])[:, 1] if clf is not None else np.zeros(len(te))
        oof[te] = np.where(depth[te] > best_gate,
                           rankdata(a) / len(te),
                           rankdata(ml[te]) / len(te))
    return oof, chosen


def boot_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """Bootstrap CI on AUC(a) - AUC(b)."""
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.array(d)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"delta": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}


def subgroup_table(df: pd.DataFrame, ml: np.ndarray, ad: np.ndarray) -> list[dict]:
    groups = [
        ("P < 2 d", df.period_days < 2),
        ("P 2-6 d", (df.period_days >= 2) & (df.period_days < 6)),
        ("P > 6 d", df.period_days >= 6),
        ("depth > 20k ppm", df.depth_ppm > 20000),
        ("depth 5k-20k ppm", (df.depth_ppm >= 5000) & (df.depth_ppm < 20000)),
        ("depth < 5k ppm", df.depth_ppm < 5000),
        ("Rp/Rs > 0.1", df.rp_rs > 0.1),
    ]
    out = []
    for name, m in groups:
        m = m.values
        yy = df.label.values[m]
        if len(yy) < 15 or len(np.unique(yy)) < 2:
            continue
        out.append({"subgroup": name, "n": int(m.sum()),
                    "auc_kepler_ml": round(float(roc_auc_score(yy, ml[m])), 4),
                    "auc_adapted": round(float(roc_auc_score(yy, ad[m])), 4),
                    "delta": round(float(roc_auc_score(yy, ad[m]) - roc_auc_score(yy, ml[m])), 4)})
    return out


def make_figure(rows: list[dict], scores: dict) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=150, facecolor=theme.SURFACE)
    for ax in (ax1, ax2):
        theme.style_axes(ax)

    names = ["physics\nalone", "physics-anchored\nadapted", "Kepler ML\n(current)", "adapted +\nKepler blend"]
    vals = [scores["auc_physics"], scores["auc_adapted"], scores["auc_kepler_ml"], scores["auc_blend"]]
    colors = [theme.INK_MUTED, theme.ACCENT, theme.POSITIVE, theme.ACCENT_2]
    ax1.bar(names, vals, color=colors)
    ax1.axhline(0.5, color=theme.NEGATIVE, ls=":", lw=1.4)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9, color=theme.INK)
    ax1.set_ylim(0.5, 0.80)
    ax1.set_ylabel("ROC-AUC on TESS (out-of-fold)")
    ax1.set_title("Adaptation beats physics, not the Kepler model",
                  fontsize=11, fontweight="bold", loc="left", color=theme.INK, pad=10)

    rows = sorted(rows, key=lambda r: r["delta"])
    yy = np.arange(len(rows))
    cols = [theme.POSITIVE if r["delta"] > 0 else theme.NEGATIVE for r in rows]
    ax2.barh(yy, [r["delta"] for r in rows], color=cols)
    ax2.axvline(0, color=theme.INK, lw=1.2)
    ax2.set_yticks(yy)
    ax2.set_yticklabels([f"{r['subgroup']} (n={r['n']})" for r in rows], fontsize=8.5)
    ax2.set_xlabel(r"$\Delta$AUC   (adapted $-$ Kepler ML)")
    ax2.set_title("It wins where the Kepler prior is inverted",
                  fontsize=11, fontweight="bold", loc="left", color=theme.INK, pad=10)

    fig.tight_layout()
    theme.save(fig, IMG_DIR / "adaptation.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    df = load(args.tag)
    y = df.label.values
    X = np.nan_to_num(df[FEATURES].values.astype(float))
    ml = df.probability.values
    depth = df.depth_ppm.values
    pseudo = pseudo_labels(df)

    phys = physics_score(df)
    ad = adapted_oof(X, y, pseudo)
    blend = 0.5 * ad + 0.5 * ml
    gate, chosen = gated_oof(X, y, pseudo, ml, depth)

    scores = {
        "auc_kepler_ml": round(float(roc_auc_score(y, ml)), 4),
        "auc_physics": round(float(roc_auc_score(y, phys)), 4),
        "auc_adapted": round(float(roc_auc_score(y, ad)), 4),
        "auc_blend": round(float(roc_auc_score(y, blend)), 4),
        "auc_depth_gated": round(float(roc_auc_score(y, gate)), 4),
    }
    labelled = int((pseudo >= 0).sum())
    report = {
        "n_targets": int(len(df)),
        "pseudo_labels": {
            "rule": f"positive = all 5 physics checks pass AND SDE > {PSEUDO_SDE_MIN}; "
                    "negative = any physics check fails",
            "n_positive": int((pseudo == 1).sum()),
            "n_negative": int((pseudo == 0).sum()),
            "n_unlabelled": int((pseudo == -1).sum()),
            "positive_purity_vs_truth": round(float(y[pseudo == 1].mean()), 4),
            "negative_purity_vs_truth": round(float(1 - y[pseudo == 0].mean()), 4),
            "note": "purities are diagnostic only; true labels never enter fitting",
        },
        "auc": scores,
        "adapted_vs_physics": boot_delta(y, ad, phys),
        "adapted_vs_kepler_ml": boot_delta(y, ad, ml),
        "gated_vs_kepler_ml": boot_delta(y, gate, ml),
        "gates_chosen_per_fold": chosen,
        "subgroups": subgroup_table(df, ml, ad),
    }
    (RESULTS_DIR / "adaptation.json").write_text(json.dumps(report, indent=2))
    make_figure(report["subgroups"], scores)

    print("\n" + "=" * 74)
    print("PHYSICS-ANCHORED SELF-TRAINING (no true TESS labels used for fitting)")
    print("=" * 74)
    p = report["pseudo_labels"]
    print(f"pseudo-labels: {p['n_positive']} positive (purity {p['positive_purity_vs_truth']:.3f}), "
          f"{p['n_negative']} negative (purity {p['negative_purity_vs_truth']:.3f}), "
          f"{p['n_unlabelled']} unlabelled  [{labelled}/{len(df)} usable]")
    print("\nOut-of-fold ROC-AUC against TRUE labels:")
    print(f"  physics alone (the supervisor)   {scores['auc_physics']:.4f}")
    print(f"  physics-anchored adapted         {scores['auc_adapted']:.4f}")
    print(f"  Kepler-trained ML (current)      {scores['auc_kepler_ml']:.4f}")
    print(f"  adapted + Kepler blend           {scores['auc_blend']:.4f}")
    print(f"  depth-gated hybrid (nested CV)   {scores['auc_depth_gated']:.4f}")
    b1, b2, b3 = report["adapted_vs_physics"], report["adapted_vs_kepler_ml"], report["gated_vs_kepler_ml"]
    print(f"\n  adapted vs physics : {b1['delta']:+.4f} CI {b1['ci95']} -> "
          f"{'BEATS its own supervisor' if b1['significant'] else 'not significant'}")
    print(f"  adapted vs Kepler  : {b2['delta']:+.4f} CI {b2['ci95']} -> "
          f"{'significant' if b2['significant'] else 'not significant'}")
    print(f"  gated  vs Kepler   : {b3['delta']:+.4f} CI {b3['ci95']} -> "
          f"{'significant' if b3['significant'] else 'not significant'}  (gates {chosen})")
    print("\nWhere does adaptation help? (positive = adapted better)")
    for r in sorted(report["subgroups"], key=lambda r: -r["delta"]):
        mark = "  <-- inverted-prior regime" if r["delta"] > 0.05 else ""
        print(f"  {r['subgroup']:20} n={r['n']:3}  ML {r['auc_kepler_ml']:.3f} -> "
              f"adapted {r['auc_adapted']:.3f}  ({r['delta']:+.3f}){mark}")
    print("=" * 74)
    logger.info("Wrote adaptation.json and adaptation.png")


if __name__ == "__main__":
    main()
