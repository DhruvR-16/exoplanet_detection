"""Phase 1 - Real TESS benchmark.

Runs the Kepler-trained pipeline on a labeled set of TESS Objects of Interest
(ExoFOP TFOPWG dispositions: CP/KP = planet, FP/FA = false positive) to measure
true cross-mission deployment performance - the number that actually matters,
as opposed to the in-mission Kepler cross-validation score.

The run is resumable: each target's result is appended to a CSV as it finishes,
and already-processed TICs are skipped on restart (downloads + TLS are slow).

Usage:
    python -m research.tess_benchmark --limit 20        # smoke test
    python -m research.tess_benchmark --limit 300        # full benchmark
    python -m research.tess_benchmark --figures-only     # just redraw figures
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pipeline  # noqa: E402
from research import theme  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("tess_benchmark")

TOI_TABLE = ROOT / "data" / "exofop_toi.csv"
TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?output=csv"
RESULTS_DIR = ROOT / "research" / "results"
BENCHMARK_CSV = RESULTS_DIR / "tess_benchmark.csv"
BENCHMARK_PARQUET = RESULTS_DIR / "tess_benchmark.parquet"
METRICS_JSON = RESULTS_DIR / "tess_metrics.json"
IMG_DIR = ROOT / "docs" / "img"
FIG_SUFFIX = ""   # e.g. "_s3" to keep multi-sector figures separate from single-sector


def set_output_tag(tag: str) -> None:
    """Route all outputs to tag-suffixed files (keeps runs side by side)."""
    global BENCHMARK_CSV, BENCHMARK_PARQUET, METRICS_JSON, FIG_SUFFIX
    suffix = f"_{tag}" if tag else ""
    BENCHMARK_CSV = RESULTS_DIR / f"tess_benchmark{suffix}.csv"
    BENCHMARK_PARQUET = RESULTS_DIR / f"tess_benchmark{suffix}.parquet"
    METRICS_JSON = RESULTS_DIR / f"tess_metrics{suffix}.json"
    FIG_SUFFIX = suffix

POSITIVE_DISP = {"CP", "KP"}   # confirmed / known planet
NEGATIVE_DISP = {"FP", "FA"}   # false positive / false alarm
SEED = 42
# Single sector per target keeps the benchmark tractable on a laptop (download +
# TLS run in ~15-40s instead of tens of minutes for long multi-sector baselines).
# A TOI transit is, by construction, detectable within one sector.
BENCHMARK_MAX_SECTORS = 1

# Folded-transit shape metrics. Recorded for every target but deliberately not
# part of pipeline.FEATURE_NAMES: the shipped v3 model was trained on the eight
# scalar features and would break if the vector changed underneath it.
SHAPE_FIELDS = ("shape_vu", "flat_bottom_frac", "transit_symmetry",
                "symmetry", "shape_ratio", "depth_std")

# Stellar context already returned by the TIC lookup but previously discarded.
# Limb-darkening coefficients are functions of Teff/log g, so they encode
# stellar type; the uncertainties flag poorly characterized hosts.
STELLAR_FIELDS = ("ld_a", "ld_b", "radius_unc", "mass_unc")

# Columns written per target (schema is fixed so resume can append safely).
FIELDS = (
    ["tic_id", "toi", "disposition", "label", "success", "error",
     "ref_period", "probability", "prediction", "physics_pass",
     "sde", "sde_pass", "welch_p", "odd_even_ok", "duration_ok", "density_ok",
     "has_secondary", "duration_ratio", "density_ratio", "secondary_snr",
     "stellar_r", "stellar_m", "n_sectors", "n_points", "period_recovered"]
    + list(pipeline.FEATURE_NAMES)
    + list(SHAPE_FIELDS)
    + list(STELLAR_FIELDS)
)


def download_toi_table(dest: Path = TOI_TABLE) -> None:
    import requests

    logger.info("Downloading ExoFOP TOI table...")
    resp = requests.get(TOI_URL, timeout=120)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    logger.info("Saved %s", dest)


def load_labeled_targets(limit: int, seed: int = SEED) -> pd.DataFrame:
    """Return a class-balanced sample of labeled TOIs (one row per TIC)."""
    if not TOI_TABLE.exists():
        download_toi_table()
    df = pd.read_csv(TOI_TABLE)
    disp = df["TFOPWG Disposition"]
    df = df[disp.isin(POSITIVE_DISP | NEGATIVE_DISP)].copy()
    df["label"] = disp.isin(POSITIVE_DISP).astype(int)
    # One entry per star (a TIC with multiple TOIs is one lightcurve to analyze).
    df = df.sort_values("TOI").drop_duplicates("TIC ID", keep="first")

    n_each = max(1, limit // 2)
    rng = np.random.default_rng(seed)
    pos = df[df["label"] == 1].sample(min(n_each, (df["label"] == 1).sum()), random_state=seed)
    neg = df[df["label"] == 0].sample(min(n_each, (df["label"] == 0).sum()), random_state=seed)
    sample = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return sample[["TIC ID", "TOI", "TFOPWG Disposition", "label", "Period (days)"]]


def _blank_row(tic_id: int, toi, disposition: str, label: int) -> dict:
    row = {f: "" for f in FIELDS}
    row.update(tic_id=tic_id, toi=toi, disposition=disposition, label=label)
    return row


def analyze_one(tic_id: int, toi, disposition: str, label: int, ref_period: float,
                model_pkg: dict) -> dict:
    """Run the full pipeline on one TIC and flatten the result into a CSV row."""
    row = _blank_row(tic_id, toi, disposition, label)
    row["ref_period"] = ref_period
    try:
        res = pipeline.analyze(f"TIC {tic_id}", model_pkg=model_pkg,
                               max_sectors=BENCHMARK_MAX_SECTORS)
    except Exception as exc:  # data-fetch or TLS failure; record and move on
        row["success"] = 0
        row["error"] = str(exc)[:200]
        return row

    feats, diag, vet, pred = res["features"], res["diagnostics"], res["vetting"], res["prediction"]
    physics_pass = int(
        diag["sde_pass"] and vet["odd_even_ok"] and vet["duration_ok"]
        and vet["density_ok"] and not vet["has_secondary"]
    )
    period_recovered = ""
    if ref_period and np.isfinite(ref_period) and ref_period > 0:
        p = feats["period_days"]
        # recovered if within 3% of the true period or a 2:1 / 1:2 alias
        ratios = [p / ref_period, p / (2 * ref_period), (2 * p) / ref_period]
        period_recovered = int(any(abs(r - 1) < 0.03 for r in ratios))

    row.update(
        success=1, error="",
        probability=pred["probability"], prediction=pred["prediction"], physics_pass=physics_pass,
        sde=diag["sde"], sde_pass=int(diag["sde_pass"]), welch_p=diag["welch_p"],
        odd_even_ok=int(vet["odd_even_ok"]), duration_ok=int(vet["duration_ok"]),
        density_ok=int(vet["density_ok"]), has_secondary=int(vet["has_secondary"]),
        duration_ratio=vet["duration_ratio"], density_ratio=vet["density_ratio"],
        secondary_snr=vet["secondary_snr"], stellar_r=res["stellar"]["radius"],
        stellar_m=res["stellar"]["mass"], n_sectors=res["data"]["n_sectors"],
        n_points=res["data"]["n_points"], period_recovered=period_recovered,
    )
    row.update({k: feats[k] for k in pipeline.FEATURE_NAMES})
    # Folded-transit shape: the V-vs-U discriminator plus the ingress/egress
    # diagnostics the pipeline already computes but never surfaced.
    row.update(pipeline.transit_shape_features(res["tls"]))
    row.update(symmetry=diag["symmetry"], shape_ratio=diag["shape_ratio"],
               depth_std=diag["depth_std"])
    row.update({k: res["stellar"].get(k, 0.0) for k in STELLAR_FIELDS})
    return row


def run_benchmark(limit: int, max_sectors: int = BENCHMARK_MAX_SECTORS) -> None:
    global BENCHMARK_MAX_SECTORS
    BENCHMARK_MAX_SECTORS = max_sectors
    model_pkg, version = pipeline.load_model()
    if model_pkg is None:
        raise SystemExit("No model found - run `python train_model.py` first.")
    logger.info("Loaded model %s", version)

    targets = load_labeled_targets(limit)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if BENCHMARK_CSV.exists():
        prev = pd.read_csv(BENCHMARK_CSV)
        done = set(prev["tic_id"].astype(int))
        logger.info("Resuming: %d targets already processed", len(done))

    new_file = not BENCHMARK_CSV.exists()
    with BENCHMARK_CSV.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, t in enumerate(targets.itertuples(index=False), 1):
            tic_id = int(t[0])
            if tic_id in done:
                continue
            t0 = time.time()
            row = analyze_one(tic_id, t[1], t[2], int(t[3]), float(t[4]) if pd.notna(t[4]) else 0.0,
                              model_pkg)
            writer.writerow(row)
            fh.flush()
            status = "ok" if row["success"] else f"FAIL ({row['error'][:40]})"
            logger.info("[%d/%d] TIC %d (%s, label=%d) -> %s  %.0fs",
                        i, len(targets), tic_id, row["disposition"], row["label"],
                        status, time.time() - t0)

    finalize()


def finalize() -> None:
    """Convert the CSV log to parquet and regenerate figures."""
    if not BENCHMARK_CSV.exists():
        logger.warning("No benchmark CSV yet.")
        return
    df = pd.read_csv(BENCHMARK_CSV)
    df.to_parquet(BENCHMARK_PARQUET, index=False)
    logger.info("Wrote %s (%d rows)", BENCHMARK_PARQUET, len(df))
    make_figures(df)


def make_figures(df: pd.DataFrame) -> None:
    from sklearn.metrics import (
        ConfusionMatrixDisplay, average_precision_score, brier_score_loss,
        confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve,
    )
    from sklearn.calibration import calibration_curve

    ok = df[(df["success"] == 1) & df["probability"].notna()].copy()
    if len(ok) < 10:
        logger.warning("Only %d successful targets - skipping figures (need >=10).", len(ok))
        return
    y = ok["label"].astype(int).values
    p = ok["probability"].astype(float).values
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    logger.info("Figures on %d targets (%d planet, %d FP)", len(ok), n_pos, n_neg)

    auc = roc_auc_score(y, p) if n_pos and n_neg else float("nan")
    ap = average_precision_score(y, p) if n_pos and n_neg else float("nan")
    brier = brier_score_loss(y, p)

    # ROC
    if n_pos and n_neg:
        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = theme.new_fig("Cross-mission ROC — Kepler-trained, TESS test set")
        ax.plot([0, 1], [0, 1], color=theme.INK_MUTED, ls="--", lw=1, alpha=0.6)
        ax.plot(fpr, tpr, color=theme.ACCENT, lw=2)
        ax.annotate(f"TESS AUC = {auc:.3f}\n(Kepler CV = 0.964)", xy=(0.5, 0.12),
                    color=theme.INK, fontsize=10, fontweight="bold")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        theme.save(fig, IMG_DIR / f"tess_roc{FIG_SUFFIX}.png")

        prec, rec, _ = precision_recall_curve(y, p)
        fig, ax = theme.new_fig("Cross-mission PR — TESS test set")
        ax.plot(rec, prec, color=theme.ACCENT_2, lw=2)
        ax.axhline(n_pos / len(y), color=theme.INK_MUTED, ls="--", lw=1, alpha=0.6)
        ax.annotate(f"AP = {ap:.3f}", xy=(0.05, 0.15), color=theme.INK, fontsize=11, fontweight="bold")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.05)
        theme.save(fig, IMG_DIR / f"tess_pr{FIG_SUFFIX}.png")

    # Confusion matrix at 0.5
    cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
    fig, ax = theme.new_fig("TESS confusion matrix (threshold 0.5)", figsize=(5.4, 4.6))
    ConfusionMatrixDisplay(cm, display_labels=["False Pos", "Planet"]).plot(
        ax=ax, cmap="Purples", colorbar=False, values_format="d", text_kw={"fontsize": 13})
    theme.style_axes(ax); ax.grid(visible=False)
    ax.set_title("TESS confusion matrix (threshold 0.5)", fontsize=12, fontweight="bold",
                 pad=12, loc="left", color=theme.INK)
    theme.save(fig, IMG_DIR / f"tess_confusion{FIG_SUFFIX}.png")

    # Calibration
    n_bins = min(10, max(3, len(ok) // 15))
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    fig, ax = theme.new_fig("Probability calibration on TESS")
    ax.plot([0, 1], [0, 1], color=theme.INK_MUTED, ls="--", lw=1, alpha=0.6)
    ax.plot(mean_pred, frac_pos, color=theme.ACCENT, lw=2, marker="o", markersize=5)
    ax.annotate(f"Brier = {brier:.3f}", xy=(0.6, 0.1), color=theme.INK, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted planet probability"); ax.set_ylabel("Observed planet fraction")
    theme.save(fig, IMG_DIR / f"tess_calibration{FIG_SUFFIX}.png")

    metrics = {
        "n_targets_attempted": int(len(df)),
        "n_success": int(len(ok)),
        "n_planet": n_pos, "n_false_positive": n_neg,
        "tess_roc_auc": round(float(auc), 4) if np.isfinite(auc) else None,
        "tess_pr_auc": round(float(ap), 4) if np.isfinite(ap) else None,
        "tess_brier": round(float(brier), 4),
        "kepler_cv_roc_auc": 0.964,
    }
    import json
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    logger.info("TESS metrics: %s", metrics)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=20, help="Number of targets (class-balanced)")
    ap.add_argument("--max-sectors", type=int, default=BENCHMARK_MAX_SECTORS,
                    help="Sectors per target (1 = fast; more = fairer detection, slower)")
    ap.add_argument("--tag", default="", help="Suffix for output files (e.g. 's3' keeps runs separate)")
    ap.add_argument("--refresh-toi", action="store_true", help="Re-download the ExoFOP TOI table")
    ap.add_argument("--figures-only", action="store_true", help="Only regenerate figures from the CSV")
    args = ap.parse_args()

    set_output_tag(args.tag)
    if args.refresh_toi:
        download_toi_table()
    if args.figures_only:
        finalize()
    else:
        run_benchmark(args.limit, max_sectors=args.max_sectors)


if __name__ == "__main__":
    main()
