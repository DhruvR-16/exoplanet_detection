"""Phase 4a - Baselines and ablations.

Answers three reviewer questions on the same TESS benchmark:
  1. Does the RF+XGBoost ensemble beat simpler learners cross-mission?
     (SDE-threshold, logistic regression, single tree, small MLP.)
  2. Which features carry the cross-mission signal? (leave-one-group-out ablation.)
  3. Do the physics checks add orthogonal false-positive rejection?

All learners train on the Kepler KOI catalog (reusing train_model.build_dataset)
and are evaluated on the real TESS benchmark (research.tess_benchmark output).

Usage:
    python -m research.baselines
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
import train_model  # noqa: E402
from pipeline import FEATURE_NAMES  # noqa: E402
from research import theme  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("baselines")

RESULTS_DIR = ROOT / "research" / "results"
BENCHMARK_PARQUET = RESULTS_DIR / "tess_benchmark.parquet"
KOI_CSV = ROOT / "data" / "koi_cumulative.csv"
IMG_DIR = ROOT / "docs" / "img"
SEED = 42

# Feature groups for leave-one-group-out ablation.
FEATURE_GROUPS = {
    "shape (Rp/Rs, dur/P)": ["rp_rs", "duration_over_period"],
    "signal (SNR)": ["model_snr"],
    "depth (depth_ppm, log10_depth)": ["depth_ppm", "log10_depth"],
    "period (period_days, log10_period)": ["period_days", "log10_period"],
    "duration (duration_hrs)": ["duration_hrs"],
}


def _metrics(y, p) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    if len(set(y)) < 2:
        return {"auc": None, "ap": None, "brier": round(float(brier_score_loss(y, p)), 4)}
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "ap": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
    }


def load_tess() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_parquet(BENCHMARK_PARQUET)
    df = df[(df["success"] == 1) & df["probability"].notna()].copy()
    X = df[FEATURE_NAMES].astype(float).values
    y = df["label"].astype(int).values
    return X, y, df


def compare_models(Xk, yk, Xt, yt) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    scaler = StandardScaler().fit(Xk)
    Xk_s, Xt_s = scaler.transform(Xk), scaler.transform(Xt)

    rows = []
    # 0. SDE-threshold baseline (no learning): probability = normalized SNR feature.
    snr_idx = FEATURE_NAMES.index("model_snr")
    p_sde = 1 / (1 + np.exp(-(Xt[:, snr_idx] - 10) / 5))   # logistic on SNR around 10
    rows.append({"model": "SDE/SNR threshold (no ML)", **_metrics(yt, p_sde)})

    learners = {
        "Logistic regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision tree (depth 4)": DecisionTreeClassifier(max_depth=4, class_weight="balanced",
                                                          random_state=SEED),
        "MLP (32,16)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=SEED),
        "RF+XGB ensemble (ours)": train_model.build_ensemble(yk),
    }
    for name, model in learners.items():
        model.fit(Xk_s, yk)
        p = model.predict_proba(Xt_s)[:, 1]
        rows.append({"model": name, **_metrics(yt, p)})
    return pd.DataFrame(rows)


def feature_ablation(Xk_df, yk, Xt_df, yt) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler

    def fit_eval(cols):
        scaler = StandardScaler().fit(Xk_df[cols])
        model = train_model.build_ensemble(yk)
        model.fit(scaler.transform(Xk_df[cols]), yk)
        p = model.predict_proba(scaler.transform(Xt_df[cols]))[:, 1]
        return _metrics(yt, p)["auc"]

    full_auc = fit_eval(FEATURE_NAMES)
    rows = [{"ablation": "full model", "features_used": len(FEATURE_NAMES),
             "tess_auc": full_auc, "delta_auc": 0.0}]
    for group, cols in FEATURE_GROUPS.items():
        kept = [c for c in FEATURE_NAMES if c not in cols]
        auc = fit_eval(kept)
        rows.append({"ablation": f"drop {group}", "features_used": len(kept),
                     "tess_auc": auc, "delta_auc": round((auc or 0) - (full_auc or 0), 4)})
    return pd.DataFrame(rows)


def physics_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """False-positive rejection contributed by each physics check on the benchmark."""
    fp = df[df["label"] == 0]
    checks = {
        "SDE >= 7": fp["sde_pass"] == 0,
        "odd/even": fp["odd_even_ok"] == 0,
        "duration": fp["duration_ok"] == 0,
        "density": fp["density_ok"] == 0,
        "no secondary": fp["has_secondary"] == 1,
    }
    n_fp = len(fp)
    rows = [{"check": name, "fp_flagged": int(mask.sum()),
             "fp_flag_rate": round(float(mask.mean()), 3) if n_fp else None}
            for name, mask in checks.items()]
    any_flag = np.zeros(n_fp, dtype=bool)
    for mask in checks.values():
        any_flag |= mask.values
    rows.append({"check": "ANY physics check", "fp_flagged": int(any_flag.sum()),
                 "fp_flag_rate": round(float(any_flag.mean()), 3) if n_fp else None})
    return pd.DataFrame(rows)


def make_figure(models: pd.DataFrame) -> None:
    m = models.dropna(subset=["auc"]).sort_values("auc")
    fig, ax = theme.new_fig("Cross-mission AUC on TESS — model comparison", figsize=(7.4, 4.4))
    colors = [theme.ACCENT if "ours" in n else theme.INK_MUTED for n in m["model"]]
    ax.barh(m["model"], m["auc"], color=colors, height=0.6)
    ax.axvline(0.5, color=theme.WARN, ls=":", lw=1, alpha=0.7)
    for i, v in enumerate(m["auc"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", color=theme.INK, fontsize=9)
    ax.set_xlim(0.4, max(0.75, m["auc"].max() + 0.05))
    ax.set_xlabel("ROC-AUC on TESS test set")
    theme.save(fig, IMG_DIR / "baselines_auc.png")


def main() -> None:
    if not BENCHMARK_PARQUET.exists():
        raise SystemExit("Run research.tess_benchmark first.")
    Xk_df, yk_s = train_model.build_dataset(KOI_CSV)
    yk = yk_s.values
    Xt, yt, tdf = load_tess()
    Xt_df = tdf[FEATURE_NAMES].astype(float).reset_index(drop=True)
    logger.info("KOI train: %d | TESS test: %d (%d planet, %d FP)",
                len(yk), len(yt), int(yt.sum()), int((yt == 0).sum()))

    models = compare_models(Xk_df.values, yk, Xt, yt)
    # Fold in the 1D-CNN result if it has been computed (research.cnn_baseline).
    cnn_path = RESULTS_DIR / "cnn_baseline.json"
    if cnn_path.exists():
        c = json.loads(cnn_path.read_text())
        models = pd.concat([models, pd.DataFrame([{
            "model": "1D-CNN (folded views)", "auc": c["tess_auc"],
            "ap": c["tess_ap"], "brier": c["tess_brier"]}])], ignore_index=True)
    logger.info("Model comparison:\n%s", models.to_string(index=False))
    models.to_csv(RESULTS_DIR / "baselines_models.csv", index=False)

    fabl = feature_ablation(Xk_df, yk, Xt_df, yt)
    logger.info("Feature ablation:\n%s", fabl.to_string(index=False))
    fabl.to_csv(RESULTS_DIR / "baselines_feature_ablation.csv", index=False)

    pabl = physics_ablation(tdf)
    logger.info("Physics ablation (FP rejection):\n%s", pabl.to_string(index=False))
    pabl.to_csv(RESULTS_DIR / "baselines_physics_ablation.csv", index=False)

    make_figure(models)
    (RESULTS_DIR / "baselines_summary.json").write_text(json.dumps({
        "models": models.to_dict("records"),
        "feature_ablation": fabl.to_dict("records"),
        "physics_ablation": pabl.to_dict("records"),
    }, indent=2))
    logger.info("Wrote baseline tables + figure.")


if __name__ == "__main__":
    main()
