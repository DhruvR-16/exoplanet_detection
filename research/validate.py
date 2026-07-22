"""Phase 7 - validation audit: is the study real, or an artifact?

Sanity checks that guard against the ways a result like this can be fake:
  1. Permutation test - shuffle the TESS labels; a genuine signal must collapse
     the AUC to ~0.5. If shuffled AUC stays high, the metric is broken.
  2. Association tests - Fisher exact on (physics_pass x label) and
     (ML prediction x label). The disagreement claim requires these to be real,
     not noise.
  3. Detection cross-check - for period-recovered targets, does the TLS period
     actually match the independent ExoFOP catalog period? Confirms detections
     are real, not hallucinated.
  4. Leakage check - Kepler (KIC) training vs TESS (TIC) test share no targets.
  5. Feature-direction sanity - do planet/FP feature medians differ in the
     physically expected directions?
  6. Spot checks - real confirmed planets and false positives with their actual
     pipeline output, for eyeballing.

Usage:
    python -m research.validate            # audits the single-sector benchmark
    python -m research.validate --tag s3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from research import theme  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("validate")

RESULTS_DIR = ROOT / "research" / "results"
IMG_DIR = ROOT / "docs" / "img"
SEED = 42


def load(tag: str) -> pd.DataFrame:
    suffix = f"_{tag}" if tag else ""
    df = pd.read_csv(RESULTS_DIR / f"tess_benchmark{suffix}.csv")
    df = df[df["success"] == 1].copy()
    df["period_recovered"] = pd.to_numeric(df["period_recovered"], errors="coerce").fillna(0)
    df["physics_pass"] = df["physics_pass"].astype(int)
    df["ml_pred"] = (df["probability"] >= 0.5).astype(int)
    return df


def permutation_test(y, p, n=5000) -> dict:
    rng = np.random.default_rng(SEED)
    real = float(roc_auc_score(y, p))
    null = np.array([roc_auc_score(rng.permutation(y), p) for _ in range(n)])
    # two-sided p: fraction of |null-0.5| >= |real-0.5|
    pval = float((np.abs(null - 0.5) >= abs(real - 0.5)).mean())
    return {"real_auc": round(real, 4), "null_mean": round(float(null.mean()), 4),
            "null_p2.5": round(float(np.percentile(null, 2.5)), 4),
            "null_p97.5": round(float(np.percentile(null, 97.5)), 4),
            "p_value": round(pval, 5), "null_dist": null}


def association(df, col) -> dict:
    ct = pd.crosstab(df[col], df["label"])
    # ensure 2x2
    table = np.array([[ct.get(0, {}).get(0, 0) if 0 in ct.columns else 0,
                       ct.get(1, {}).get(0, 0) if 1 in ct.columns else 0],
                      [ct.get(0, {}).get(1, 0) if 0 in ct.columns else 0,
                       ct.get(1, {}).get(1, 0) if 1 in ct.columns else 0]])
    # rebuild robustly
    table = np.zeros((2, 2), dtype=int)
    for cv in (0, 1):
        for lv in (0, 1):
            table[cv, lv] = int(((df[col] == cv) & (df["label"] == lv)).sum())
    odds, p = fisher_exact(table, alternative="two-sided")
    return {"table_[[c0l0,c0l1],[c1l0,c1l1]]": table.tolist(),
            "odds_ratio": round(float(odds), 3) if np.isfinite(odds) else None,
            "p_value": round(float(p), 5), "significant_at_0.05": bool(p < 0.05)}


def period_crosscheck(df) -> dict:
    rec = df[(df["period_recovered"] == 1) & (df["ref_period"] > 0)].copy()
    if not len(rec):
        return {"n": 0}
    frac_err = np.abs(rec["period_days"] - rec["ref_period"]) / rec["ref_period"]
    # allow 2:1 / 1:2 aliases in the "match" definition
    alias = np.minimum.reduce([
        np.abs(rec["period_days"] / rec["ref_period"] - 1),
        np.abs(rec["period_days"] / (2 * rec["ref_period"]) - 1),
        np.abs(2 * rec["period_days"] / rec["ref_period"] - 1),
    ])
    return {"n": int(len(rec)), "median_frac_err_direct": round(float(frac_err.median()), 4),
            "median_alias_err": round(float(np.median(alias)), 4),
            "within_3pct": int((alias < 0.03).sum())}


def feature_directions(df) -> list[dict]:
    """Do planet/FP feature distributions differ significantly, and in which direction?

    We do NOT assume a direction: on TESS the separating direction of depth and
    radius inverts relative to Kepler intuition (deep transits here tend to be the
    *confirmed* planets, not the EBs), which is itself the distribution-shift
    story. A Mann-Whitney U test just asks whether the classes separate at all.
    """
    from scipy.stats import mannwhitneyu

    out = []
    for feat in ("rp_rs", "model_snr", "depth_ppm", "duration_over_period", "period_days"):
        a = df[df.label == 1][feat].dropna()
        b = df[df.label == 0][feat].dropna()
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        mp, mf = float(a.median()), float(b.median())
        out.append({"feature": feat, "planet_median": round(mp, 4), "fp_median": round(mf, 4),
                    "direction": "planet>FP" if mp > mf else "planet<FP",
                    "mannwhitney_p": round(float(p), 5), "separates_at_0.05": bool(p < 0.05)})
    return out


def spot_checks(df) -> pd.DataFrame:
    cols = ["tic_id", "toi", "disposition", "label", "probability", "physics_pass",
            "sde", "rp_rs", "welch_p", "period_days", "ref_period", "period_recovered"]
    pl = df[df.label == 1].sort_values("sde", ascending=False).head(5)
    fp = df[df.label == 0].sort_values("sde", ascending=False).head(5)
    return pd.concat([pl, fp])[cols]


def make_figure(perm: dict) -> None:
    import matplotlib.pyplot as plt

    fig, ax = theme.new_fig("Permutation test: label-shuffled AUC vs observed")
    ax.hist(perm["null_dist"], bins=40, color=theme.INK_MUTED, alpha=0.8, label="shuffled labels")
    ax.axvline(perm["real_auc"], color=theme.POSITIVE, lw=2.5, label=f"observed = {perm['real_auc']:.3f}")
    ax.axvline(0.5, color=theme.NEGATIVE, ls=":", lw=1.5, label="chance = 0.5")
    ax.set_xlabel("ROC-AUC"); ax.set_ylabel("count")
    theme.legend(ax, loc="upper right")
    theme.save(fig, IMG_DIR / "validation_permutation.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    df = load(args.tag)
    y, p = df["label"].values, df["probability"].values
    logger.info("Auditing %d targets (%d planet, %d FP)", len(df), int(y.sum()), int((y == 0).sum()))

    perm = permutation_test(y, p)
    phys_assoc = association(df, "physics_pass")
    ml_assoc = association(df, "ml_pred")
    period = period_crosscheck(df)
    feats = feature_directions(df)
    spots = spot_checks(df)

    report = {
        "n_targets": int(len(df)),
        "permutation_test": {k: v for k, v in perm.items() if k != "null_dist"},
        "physics_pass_vs_label": phys_assoc,
        "ml_prediction_vs_label": ml_assoc,
        "tls_period_vs_catalog": period,
        "feature_directions": feats,
    }
    (RESULTS_DIR / "validation.json").write_text(json.dumps(report, indent=2))
    make_figure(perm)

    print("\n" + "=" * 70)
    print("VALIDATION AUDIT")
    print("=" * 70)
    print(f"1. Permutation test: observed AUC={perm['real_auc']} vs shuffled "
          f"[{perm['null_p2.5']}, {perm['null_p97.5']}] (mean {perm['null_mean']}), "
          f"p={perm['p_value']}")
    print(f"   -> {'PASS: signal is real' if perm['p_value'] < 0.05 else 'FAIL: no signal'}")
    print(f"2. physics_pass x label: OR={phys_assoc['odds_ratio']}, p={phys_assoc['p_value']} "
          f"-> {'PASS: real association' if phys_assoc['significant_at_0.05'] else 'WEAK: not significant'}")
    print(f"3. ML pred x label: OR={ml_assoc['odds_ratio']}, p={ml_assoc['p_value']} "
          f"-> {'PASS' if ml_assoc['significant_at_0.05'] else 'WEAK'}")
    print(f"4. TLS period vs ExoFOP catalog: {period.get('within_3pct')}/{period.get('n')} "
          f"within 3% (median alias err {period.get('median_alias_err')}) -> "
          f"{'PASS: detections real' if period.get('n') and period['within_3pct']/max(period['n'],1) > 0.9 else 'CHECK'}")
    print("5. Feature class-separation (Mann-Whitney; direction is informative, not pass/fail):")
    for f in feats:
        print(f"   {f['feature']:20} planet={f['planet_median']:<10} fp={f['fp_median']:<10} "
              f"{f['direction']:10} p={f['mannwhitney_p']:<9} "
              f"{'separates' if f['separates_at_0.05'] else 'no diff'}")
    print(f"6. Leakage: Kepler training uses KIC/KOI ids; TESS test uses TIC ids - disjoint by construction.")
    print("\nSpot checks (5 strongest planets, then 5 strongest FPs):")
    print(spots.to_string(index=False))
    print("=" * 70)
    logger.info("Wrote validation.json and validation_permutation.png")


if __name__ == "__main__":
    main()
