"""Phase 4b - 1D-CNN shape baseline.

A convolutional baseline that classifies from the *shape* of the phase-folded
transit rather than scalar features. To stay reproducible on a laptop (no bulk
Kepler lightcurve downloads), folded "local views" are generated parametrically
with batman from each object's catalog/TLS parameters plus realistic noise, so
the CNN sees ingress/egress slope and curvature - information a limb-darkened
planet transit and a V-shaped grazing eclipse express differently.

This is a controlled shape-vs-scalar comparison, not a raw-pixel detector; a
full AstroNet-style CNN on real light curves is documented as future work. The
CNN trains on Kepler KOI parameters and is evaluated on the real TESS benchmark
(both views generated identically), matching the cross-mission protocol used
everywhere else in the study.

Usage:
    python -m research.cnn_baseline --epochs 40
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Avoid a macOS OpenMP/MKL duplicate-runtime segfault when torch shares the
# process with numpy/scipy BLAS threads (must be set before torch imports).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import batman
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_model  # noqa: E402
from research import theme  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("cnn")

RESULTS_DIR = ROOT / "research" / "results"
BENCHMARK_PARQUET = RESULTS_DIR / "tess_benchmark.parquet"
KOI_CSV = ROOT / "data" / "koi_cumulative.csv"
IMG_DIR = ROOT / "docs" / "img"
N_BINS = 201          # points in the local phase-folded view
PHASE_WINDOW = 2.5    # +/- this many transit durations around mid-transit
SEED = 42


def folded_view(period: float, duration_hrs: float, depth_ppm: float, rp_rs: float,
                snr: float, rng: np.random.Generator) -> np.ndarray:
    """Parametric limb-darkened local view (length N_BINS), normalized to [~0,1]."""
    duration_days = max(duration_hrs / 24.0, 1e-3)
    depth_frac = max(depth_ppm * 1e-6, 1e-6)
    rp_rs = float(np.clip(rp_rs if rp_rs > 0 else np.sqrt(depth_frac), 1e-3, 0.6))
    a_over_r = max(period / (np.pi * duration_days), 1.5)
    b = rng.uniform(0, 0.7)
    inc = np.degrees(np.arccos(np.clip(b / a_over_r, 0, 0.999)))

    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a = 0.0, period, rp_rs, a_over_r
    params.inc, params.ecc, params.w = inc, 0.0, 90.0
    params.u, params.limb_dark = [0.4, 0.3], "quadratic"

    half = PHASE_WINDOW * duration_days
    t = np.linspace(-half, half, N_BINS)
    flux = batman.TransitModel(params, t).light_curve(params)

    # noise so the in-transit SNR ~ catalog value
    sigma = depth_frac / max(snr, 1.0) * np.sqrt(N_BINS / 4)
    flux = flux + rng.normal(0, max(sigma, 1e-6), N_BINS)
    # normalize each view to unit depth range (shape, not absolute depth)
    lo, hi = flux.min(), flux.max()
    return (flux - lo) / (hi - lo + 1e-9)


def build_views(df: pd.DataFrame, cols: dict, rng: np.random.Generator) -> np.ndarray:
    views = np.empty((len(df), N_BINS), dtype=np.float32)
    for i, (_, r) in enumerate(df.iterrows()):
        views[i] = folded_view(r[cols["period"]], r[cols["duration"]], r[cols["depth"]],
                               r[cols["rp_rs"]], r[cols["snr"]], rng)
    return views


def koi_frame() -> pd.DataFrame:
    df = pd.read_csv(KOI_CSV, comment="#")
    df = df.dropna(subset=["koi_disposition", "koi_period", "koi_depth", "koi_duration", "koi_model_snr"])
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    df = df[(df["koi_period"] > 0) & (df["koi_depth"] > 0) & (df["koi_duration"] > 0)
            & (df["koi_model_snr"] > 0) & (df["koi_depth"] < 1e6)]
    df["rp_rs"] = df["koi_ror"].fillna(np.sqrt(df["koi_depth"] * 1e-6))
    df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int)
    return df


def train_cnn(Xtr, ytr, epochs: int, device):
    import torch
    from torch import nn

    class TransitCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 16), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(16, 1))

        def forward(self, x):
            return self.head(self.net(x))

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    model = TransitCNN().to(device)
    Xt = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1).to(device)
    yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1).to(device)
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    model.train()
    n = len(Xt)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for k in range(0, n, 128):
            idx = perm[k:k + 128]
            opt.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0:
            logger.info("  epoch %d/%d  loss=%.4f", epoch + 1, epochs, total / n)
    return model


def main() -> None:
    import torch
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    if not BENCHMARK_PARQUET.exists():
        raise SystemExit("Run research.tess_benchmark first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)

    koi = koi_frame()
    logger.info("Generating %d Kepler training views...", len(koi))
    Xk = build_views(koi, {"period": "koi_period", "duration": "koi_duration",
                           "depth": "koi_depth", "rp_rs": "rp_rs", "snr": "koi_model_snr"}, rng)
    yk = koi["label"].values

    tdf = pd.read_parquet(BENCHMARK_PARQUET)
    tdf = tdf[(tdf["success"] == 1) & tdf["probability"].notna()].copy()
    Xt = build_views(tdf, {"period": "period_days", "duration": "duration_hrs",
                          "depth": "depth_ppm", "rp_rs": "rp_rs", "snr": "model_snr"}, rng)
    yt = tdf["label"].astype(int).values

    logger.info("Training 1D-CNN (%d epochs, device=%s)...", args.epochs, device)
    model = train_cnn(Xk, yk, args.epochs, device)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xt, dtype=torch.float32).unsqueeze(1).to(device))
        p = torch.sigmoid(logits).cpu().numpy().ravel()

    result = {
        "model": "1D-CNN on parametric folded views",
        "n_train": int(len(yk)), "n_test": int(len(yt)),
        "tess_auc": round(float(roc_auc_score(yt, p)), 4) if len(set(yt)) > 1 else None,
        "tess_ap": round(float(average_precision_score(yt, p)), 4) if len(set(yt)) > 1 else None,
        "tess_brier": round(float(brier_score_loss(yt, p)), 4),
        "epochs": args.epochs,
    }
    (RESULTS_DIR / "cnn_baseline.json").write_text(json.dumps(result, indent=2))
    logger.info("CNN result: %s", result)

    # A representative training view figure (planet vs FP shape)
    fig, ax = theme.new_fig("Parametric folded views the 1D-CNN sees")
    ax.plot(np.linspace(-1, 1, N_BINS), Xk[np.where(yk == 1)[0][0]], color=theme.POSITIVE,
            lw=2, label="Confirmed planet (U-shaped)")
    ax.plot(np.linspace(-1, 1, N_BINS), Xk[np.where(yk == 0)[0][0]], color=theme.NEGATIVE,
            lw=2, label="False positive")
    ax.set_xlabel("Normalized phase (transit-centered)"); ax.set_ylabel("Normalized flux")
    theme.legend(ax, loc="lower right")
    theme.save(fig, IMG_DIR / "cnn_views.png")


if __name__ == "__main__":
    main()
