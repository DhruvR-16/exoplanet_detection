"""Train the exoplanet candidate classifier (model v3).

Data: NASA Exoplanet Archive KOI cumulative catalog (downloaded via the TAP
API and cached under data/). Labels are real dispositions - CONFIRMED planets
are positives, FALSE POSITIVEs are negatives, CANDIDATEs are excluded.

Model: soft-voting ensemble (RandomForest + XGBoost) with Platt-scaled
probability calibration on a held-out calibration split. Features use the
same units the TLS pipeline produces at inference time (see pipeline.py).

Outputs:
    model/exoplanet_model_v3.pkl   - model package (model, scaler, feature names)
    docs/metrics.json              - held-out test metrics
    docs/img/*.png                 - training result figures for the README

Usage:
    python train_model.py [--refresh-data]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pipeline import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "koi_cumulative.csv"
MODEL_PATH = ROOT / "model" / "exoplanet_model_v3.pkl"
METRICS_PATH = ROOT / "docs" / "metrics.json"
IMG_DIR = ROOT / "docs" / "img"

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
TAP_QUERY = (
    "select kepoi_name,koi_disposition,koi_period,koi_depth,koi_duration,"
    "koi_model_snr,koi_ror from cumulative"
)

SEED = 42

# Figure theme (GitHub-dark surface; single accent hue, statuses reserved)
SURFACE = "#0d1117"
PANEL = "#161b22"
INK = "#e6edf3"
INK_MUTED = "#8b949e"
GRID = "#21262d"
ACCENT = "#818cf8"       # indigo - primary series
ACCENT_2 = "#22d3ee"     # cyan - second series where needed
NEGATIVE = "#f87171"     # rose - 'false positive' class only


def download_koi_catalog(dest: Path) -> None:
    """Fetch the KOI cumulative table (with dispositions) from the Exoplanet Archive."""
    logger.info("Downloading KOI cumulative catalog from NASA Exoplanet Archive...")
    resp = requests.get(
        TAP_URL, params={"query": TAP_QUERY, "format": "csv"}, timeout=120
    )
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    logger.info("Saved catalog to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


def build_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Turn the KOI catalog into (features, labels) in pipeline units."""
    df = pd.read_csv(csv_path, comment="#")
    df = df.dropna(subset=["koi_disposition", "koi_period", "koi_depth", "koi_duration", "koi_model_snr"])
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()

    # Physical sanity filters
    df = df[(df["koi_period"] > 0) & (df["koi_depth"] > 0) & (df["koi_duration"] > 0) & (df["koi_model_snr"] > 0)]
    df = df[df["koi_depth"] < 1e6]  # a >100% "transit" is a catalog artifact

    rp_rs = df["koi_ror"].fillna(np.sqrt(df["koi_depth"] * 1e-6))
    duration_days = df["koi_duration"] / 24.0

    features = pd.DataFrame(
        {
            "period_days": df["koi_period"],
            "depth_ppm": df["koi_depth"],
            "duration_hrs": df["koi_duration"],
            "model_snr": df["koi_model_snr"],
            "rp_rs": rp_rs,
            "log10_depth": np.log10(df["koi_depth"].clip(lower=1.0)),
            "log10_period": np.log10(df["koi_period"].clip(lower=1e-3)),
            "duration_over_period": duration_days / df["koi_period"],
        }
    )[FEATURE_NAMES]

    labels = (df["koi_disposition"] == "CONFIRMED").astype(int)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    labels = labels.loc[features.index]
    return features, labels


def build_ensemble(y_fit: np.ndarray) -> VotingClassifier:
    scale_pos_weight = float((y_fit == 0).sum()) / max(int((y_fit == 1).sum()), 1)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=-1,
    )
    return VotingClassifier(estimators=[("rf", rf), ("xgb", xgb)], voting="soft")


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.title.set_color(INK)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def _new_fig(title: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150, facecolor=SURFACE)
    # loc="left" stores the heading in _left_title, so color must be set here
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, loc="left", color=INK)
    _style_axes(ax)
    return fig, ax


def _save(fig: plt.Figure, name: str) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    path = IMG_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    logger.info("Wrote %s", path)


def make_figures(
    model: CalibratedClassifierCV,
    X_test_s: np.ndarray,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    metrics: dict,
) -> None:
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = _new_fig("ROC curve — held-out test set")
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle="--", alpha=0.6)
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2)
    ax.annotate(
        f"AUC = {metrics['roc_auc']:.3f}", xy=(0.62, 0.12), color=INK, fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    _save(fig, "roc_curve.png")

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = _new_fig("Precision–recall curve — held-out test set")
    ax.plot(recall, precision, color=ACCENT_2, linewidth=2)
    base_rate = float(np.mean(y_test))
    ax.axhline(base_rate, color=INK_MUTED, linewidth=1, linestyle="--", alpha=0.6)
    ax.annotate(f"AP = {metrics['pr_auc']:.3f}", xy=(0.05, 0.15), color=INK, fontsize=11, fontweight="bold")
    ax.annotate(f"baseline = {base_rate:.2f}", xy=(0.05, base_rate + 0.03), color=INK_MUTED, fontsize=9)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    _save(fig, "pr_curve.png")

    # Confusion matrix (single-hue sequential, direct labels)
    cm = confusion_matrix(y_test, (y_prob >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=150, facecolor=SURFACE)
    disp = ConfusionMatrixDisplay(cm, display_labels=["False Positive", "Confirmed Planet"])
    disp.plot(ax=ax, cmap="Purples", colorbar=False, values_format="d", text_kw={"fontsize": 13})
    _style_axes(ax)
    ax.grid(visible=False)
    ax.set_title(
        "Confusion matrix — held-out test set",
        fontsize=12, fontweight="bold", pad=12, loc="left", color=INK,
    )
    _save(fig, "confusion_matrix.png")

    # Reliability (calibration) diagram
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
    fig, ax = _new_fig("Probability calibration — held-out test set")
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle="--", alpha=0.6)
    ax.plot(mean_pred, frac_pos, color=ACCENT, linewidth=2, marker="o", markersize=5)
    ax.annotate(f"Brier = {metrics['brier']:.3f}", xy=(0.62, 0.10), color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted planet probability")
    ax.set_ylabel("Observed planet fraction")
    _save(fig, "calibration_curve.png")

    # Permutation feature importance (model-agnostic, computed on the test set)
    logger.info("Computing permutation importance...")
    imp = permutation_importance(
        model, X_test_s, y_test, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=-1
    )
    order = np.argsort(imp.importances_mean)
    fig, ax = _new_fig("Permutation importance (ΔAUC) — held-out test set")
    names = [FEATURE_NAMES[i] for i in order]
    ax.barh(names, imp.importances_mean[order], xerr=imp.importances_std[order],
            color=ACCENT, height=0.55, error_kw={"ecolor": INK_MUTED, "elinewidth": 1})
    ax.set_xlabel("Mean decrease in ROC-AUC when shuffled")
    ax.grid(axis="y", visible=False)
    _save(fig, "feature_importance.png")

    # Predicted-probability distributions per class
    fig, ax = _new_fig("Predicted probability by true class — held-out test set")
    bins = np.linspace(0, 1, 41)
    ax.hist(y_prob[y_test == 0], bins=bins, color=NEGATIVE, alpha=0.75, label="False positives")
    ax.hist(y_prob[y_test == 1], bins=bins, color=ACCENT, alpha=0.75, label="Confirmed planets")
    ax.set_xlabel("Predicted planet probability")
    ax.set_ylabel("Count")
    legend = ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9)
    legend.get_frame().set_alpha(0.9)
    _save(fig, "score_distribution.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-data", action="store_true", help="Re-download the KOI catalog")
    args = parser.parse_args()

    if args.refresh_data or not DATA_PATH.exists():
        download_koi_catalog(DATA_PATH)

    X_df, y = build_dataset(DATA_PATH)
    X, y = X_df.values, y.values
    logger.info(
        "Dataset: %d samples (%d confirmed planets, %d false positives)",
        len(y), int(y.sum()), int((y == 0).sum()),
    )

    # test held out for final metrics; calibration split held out from fitting
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=SEED, stratify=y_trainval
    )

    scaler = StandardScaler().fit(X_fit)
    X_fit_s, X_cal_s, X_test_s = (scaler.transform(a) for a in (X_fit, X_cal, X_test))

    logger.info("Cross-validating ensemble (5-fold stratified, ROC-AUC)...")
    cv_pipeline = make_pipeline(StandardScaler(), build_ensemble(y_trainval))
    cv_scores = cross_val_score(
        cv_pipeline, X_trainval, y_trainval,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc", n_jobs=1,
    )
    logger.info("CV ROC-AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    logger.info("Training RandomForest + XGBoost soft-voting ensemble...")
    ensemble = build_ensemble(y_fit)
    ensemble.fit(X_fit_s, y_fit)

    logger.info("Calibrating probabilities (Platt scaling on held-out split)...")
    calibrated = CalibratedClassifierCV(FrozenEstimator(ensemble), method="sigmoid")
    calibrated.fit(X_cal_s, y_cal)

    y_prob = calibrated.predict_proba(X_test_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "n_samples": int(len(y)),
        "n_train": int(len(y_fit)),
        "n_calibration": int(len(y_cal)),
        "n_test": int(len(y_test)),
        "class_balance": {"confirmed": int(y.sum()), "false_positive": int((y == 0).sum())},
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_test, y_prob)), 4),
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "training_data": "NASA Exoplanet Archive KOI cumulative catalog",
    }

    print("\n" + classification_report(y_test, y_pred, target_names=["False Positive", "Confirmed Planet"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Brier: {metrics['brier']:.4f}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote %s", METRICS_PATH)

    make_figures(calibrated, X_test_s, y_test, y_prob, metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    package = {
        "model": calibrated,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "version": "v3",
        "metrics": metrics,
        "description": (
            "Calibrated RandomForest+XGBoost soft-voting ensemble trained on the "
            "NASA Exoplanet Archive KOI cumulative catalog (CONFIRMED vs FALSE POSITIVE)"
        ),
    }
    joblib.dump(package, MODEL_PATH, compress=3)
    logger.info("Model saved to %s (%.1f MB)", MODEL_PATH, MODEL_PATH.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
