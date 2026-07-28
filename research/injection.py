"""Phase 2 - Injection-recovery completeness characterization.

Injects synthetic Mandel-Agol transits (batman) into real, detrended TESS
lightcurves and runs the full detection pipeline to measure detection
completeness as a function of orbital period, planet radius and SNR. This is
the survey-grade characterization that turns an accuracy number into a
statement of what the pipeline can and cannot find.

Method (per realization):
  1. Take a real TESS host lightcurve (real cadence, gaps and noise).
  2. Detrend it to a flat baseline so the injected transit is the only signal.
  3. Multiply in a batman transit at known (period, Rp, impact, t0).
  4. Run the exact pipeline: gap-aware binning -> TLS -> features -> classifier.
  5. Record TLS recovery (right period & SDE>7) and classifier recovery.

Usage:
    python -m research.injection --smoke            # ~30 injections, 1 host
    python -m research.injection --full             # full grid, several hosts
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import batman
import lightkurve as lk
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pipeline  # noqa: E402
from research import theme  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("injection")

RESULTS_DIR = ROOT / "research" / "results"
INJECTION_CSV = RESULTS_DIR / "injection.csv"
INJECTION_PARQUET = RESULTS_DIR / "injection.parquet"
IMG_DIR = ROOT / "docs" / "img"

R_EARTH_R_SUN = 0.009168   # Earth radius in solar radii
SEED = 42
QUIET_SDE_MAX = 8.0        # a host is "quiet" (usable) if its pre-injection TLS SDE < this
CACHE_DIR = ROOT / "lc_cache"

# A single fallback host for the smoke test (screening disabled there for speed).
SMOKE_HOST = "TIC 307210830"

FIELDS = [
    "host", "stellar_r", "stellar_m", "inj_period", "inj_rp_earth", "inj_rp_rs",
    "inj_depth_ppm", "inj_impact", "inj_snr", "n_transits",
    "rec_period", "rec_sde", "rec_depth_ppm", "tls_recovered", "clf_probability",
    "clf_recovered", "error",
]


def expected_snr(depth_frac: float, flux: np.ndarray, period: float, duration_days: float,
                 baseline_days: float) -> float:
    """Total transit SNR = depth / (per-point scatter) * sqrt(points in transit)."""
    scatter = float(np.nanstd(flux))
    if scatter <= 0:
        return 0.0
    cadence = baseline_days / max(len(flux), 1)
    n_transits = max(baseline_days / period, 1.0)
    pts_in_transit = max((duration_days / cadence) * n_transits, 1.0)
    return depth_frac / scatter * np.sqrt(pts_in_transit)


def inject_transit(time: np.ndarray, flux_flat: np.ndarray, r_star: float, m_star: float,
                   period: float, rp_earth: float, impact: float, t0: float) -> tuple[np.ndarray, dict]:
    """Multiply a batman transit into a flat (detrended) lightcurve."""
    rp_rs = (rp_earth * R_EARTH_R_SUN) / max(r_star, 1e-3)
    a_over_r = pipeline.A_OVER_R_COEFF * (m_star ** (1 / 3)) * (period ** (2 / 3)) / max(r_star, 1e-3)
    a_over_r = max(a_over_r, 1.5)
    inc = np.degrees(np.arccos(np.clip(impact / a_over_r, 0, 0.999)))

    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp_rs
    params.a = a_over_r
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.4, 0.3]           # quadratic limb darkening (Sun-like default)
    params.limb_dark = "quadratic"

    model = batman.TransitModel(params, time)
    transit = model.light_curve(params)
    depth_frac = float(1.0 - transit.min())
    meta = {"rp_rs": rp_rs, "depth_ppm": depth_frac * 1e6, "a_over_r": a_over_r, "inc": inc}
    return flux_flat * transit, meta


def _flatten_lc(lc: lk.LightCurve, label: str) -> tuple[np.ndarray, np.ndarray, dict] | None:
    stellar = pipeline.get_stellar_params(label, dict(lc.meta))
    try:
        time_arr, _raw, flat, _trend = pipeline.detrend(lc)
    except Exception as exc:
        logger.warning("Detrend failed for %s: %s", label, exc)
        return None
    if len(time_arr) < 500:
        return None
    flat = flat / np.nanmedian(flat)   # baseline at 1.0 with real residual noise
    return time_arr, flat, stellar


def load_flat_host(host: str) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Fetch a host by name, detrend to a flat baseline, return (time, flat, stellar)."""
    lc, info = pipeline.load_lightcurve(host, max_sectors=1)
    if lc is None:
        logger.warning("Host %s unavailable: %s", host, info["message"])
        return None
    return _flatten_lc(lc, host)


def load_flat_from_cache(fits_path: Path) -> tuple[str, np.ndarray, np.ndarray, dict] | None:
    """Load a flat baseline directly from a cached FITS file (no network)."""
    try:
        lc = lk.read(str(fits_path)).remove_nans().normalize()
    except Exception as exc:
        logger.warning("Could not read %s: %s", fits_path.name, exc)
        return None
    tic = lc.meta.get("TICID")
    label = f"TIC {tic}" if tic else fits_path.stem
    flat = _flatten_lc(lc, label)
    if flat is None:
        return None
    return label, flat[0], flat[1], flat[2]


def screen_quiet_hosts(n_wanted: int) -> list[tuple[str, np.ndarray, np.ndarray, dict]]:
    """Screen cached lightcurves and return up to n_wanted transit-free baselines.

    A host is accepted only if a TLS run on its detrended flux finds no
    significant signal (SDE < QUIET_SDE_MAX), guaranteeing the injected transit
    is the only transit present. Reuses already-downloaded data (fast).
    """
    accepted: list[tuple[str, np.ndarray, np.ndarray, dict]] = []
    fits_files = sorted(CACHE_DIR.glob("*.fits"))
    rng = np.random.default_rng(SEED)
    rng.shuffle(fits_files)
    logger.info("Screening %d cached lightcurves for %d quiet hosts...", len(fits_files), n_wanted)
    for path in fits_files:
        loaded = load_flat_from_cache(path)
        if loaded is None:
            continue
        label, time_arr, flat, stellar = loaded
        try:
            t_bin, f_bin = pipeline.gap_aware_bin(time_arr, flat)
            tls = pipeline.run_tls(t_bin, f_bin, stellar=stellar)
            sde = pipeline.to_scalar(tls.SDE)
        except Exception as exc:
            logger.warning("Screening TLS failed for %s: %s", label, exc)
            continue
        verdict = "QUIET" if sde < QUIET_SDE_MAX else "has signal"
        logger.info("  %s: SDE=%.1f (%s)", label, sde, verdict)
        if sde < QUIET_SDE_MAX:
            accepted.append((label, time_arr, flat, stellar))
            if len(accepted) >= n_wanted:
                break
    logger.info("Accepted %d quiet hosts", len(accepted))
    return accepted


def recover(time_arr: np.ndarray, injected_flux: np.ndarray, stellar: dict,
            inj_period: float) -> dict:
    """Run the pipeline on injected data and test recovery of the injected period."""
    t_bin, f_bin = pipeline.gap_aware_bin(time_arr, injected_flux)
    tls = pipeline.run_tls(t_bin, f_bin, stellar=stellar)
    p = pipeline.to_scalar(tls.period)
    sde = pipeline.to_scalar(tls.SDE)
    ratios = [p / inj_period, p / (2 * inj_period), (2 * p) / inj_period]
    tls_ok = bool(any(abs(r - 1) < 0.03 for r in ratios) and sde >= pipeline.SDE_THRESHOLD)

    feats = pipeline.extract_features(tls)
    model_pkg, _ = _MODEL
    prob = pipeline.predict(model_pkg, feats)["probability"] if model_pkg else float("nan")
    return {
        "rec_period": p, "rec_sde": sde, "rec_depth_ppm": feats["depth_ppm"],
        "tls_recovered": int(tls_ok), "clf_probability": prob,
        # Score against the same TESS-calibrated operating point the pipeline
        # deploys with; a hardcoded 0.5 would understate completeness.
        "clf_recovered": int(tls_ok and prob >= pipeline.DECISION_THRESHOLD),
    }


_MODEL: tuple = (None, None)


def build_grid(smoke: bool) -> tuple[np.ndarray, np.ndarray, list[float]]:
    if smoke:
        periods = np.array([1.5, 4.0, 8.0])
        radii = np.array([2.0, 4.0, 12.0])
        impacts = [0.2]
    else:
        periods = np.geomspace(0.7, 12.0, 8)
        radii = np.array([0.8, 1.5, 2.5, 4.0, 6.0, 9.0, 13.0])
        impacts = [0.1, 0.5]
    return periods, radii, impacts


def run(smoke: bool, n_hosts: int) -> None:
    global _MODEL
    _MODEL = pipeline.load_model()
    if _MODEL[0] is None:
        raise SystemExit("No model found - run `python train_model.py` first.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    periods, radii, impacts = build_grid(smoke)
    rng = np.random.default_rng(SEED)

    if smoke:
        loaded = load_flat_host(SMOKE_HOST)
        hosts = [(SMOKE_HOST, *loaded)] if loaded else []
    else:
        hosts = screen_quiet_hosts(n_hosts)
    if not hosts:
        raise SystemExit("No usable injection hosts found.")

    done = set()
    if INJECTION_CSV.exists():
        prev = pd.read_csv(INJECTION_CSV)
        done = {(r.host, round(r.inj_period, 4), round(r.inj_rp_earth, 3), round(r.inj_impact, 3))
                for r in prev.itertuples()}
        logger.info("Resuming: %d injections already done", len(done))

    new_file = not INJECTION_CSV.exists()
    total = len(hosts) * len(periods) * len(radii) * len(impacts)
    n = 0
    with INJECTION_CSV.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for host, time_arr, flat, stellar in hosts:
            baseline = float(time_arr[-1] - time_arr[0])
            for period in periods:
                for rp in radii:
                    for b in impacts:
                        n += 1
                        key = (host, round(float(period), 4), round(float(rp), 3), round(float(b), 3))
                        if key in done:
                            continue
                        t0 = float(time_arr[0]) + rng.uniform(0, period)
                        row = {f: "" for f in FIELDS}
                        row.update(host=host, stellar_r=stellar["radius"], stellar_m=stellar["mass"],
                                   inj_period=float(period), inj_rp_earth=float(rp), inj_impact=float(b))
                        try:
                            injected, meta = inject_transit(time_arr, flat, stellar["radius"],
                                                            stellar["mass"], period, rp, b, t0)
                            snr = expected_snr(meta["depth_ppm"] * 1e-6, flat, period,
                                               pipeline.transit_duration_estimate(period, meta["a_over_r"]),
                                               baseline)
                            n_tr = max(int(baseline / period), 1)
                            row.update(inj_rp_rs=meta["rp_rs"], inj_depth_ppm=meta["depth_ppm"],
                                       inj_snr=snr, n_transits=n_tr)
                            t1 = time.time()
                            row.update(recover(time_arr, injected, stellar, period))
                            logger.info("[%d/%d] %s P=%.2f Rp=%.1f b=%.1f -> TLS=%d CLF=%d "
                                        "(SDE=%.1f) %.0fs", n, total, host, period, rp, b,
                                        row["tls_recovered"], row["clf_recovered"], row["rec_sde"],
                                        time.time() - t1)
                        except Exception as exc:
                            row["error"] = str(exc)[:150]
                            logger.warning("[%d/%d] %s P=%.2f Rp=%.1f FAILED: %s",
                                           n, total, host, period, rp, row["error"][:60])
                        writer.writerow(row)
                        fh.flush()
    finalize()


def finalize() -> None:
    if not INJECTION_CSV.exists():
        logger.warning("No injection CSV yet.")
        return
    df = pd.read_csv(INJECTION_CSV)
    df.to_parquet(INJECTION_PARQUET, index=False)
    logger.info("Wrote %s (%d injections)", INJECTION_PARQUET, len(df))
    make_figures(df)


def make_figures(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    ok = df[df["error"].isna() | (df["error"] == "")].copy()
    if len(ok) < 4:
        logger.warning("Too few injections (%d) for figures.", len(ok))
        return

    # Completeness heatmap: recovery fraction over (period x radius)
    for metric, fname, title in [
        ("tls_recovered", "injection_completeness_tls.png", "Detection completeness (TLS) — injection–recovery"),
        ("clf_recovered", "injection_completeness_clf.png", "Detection completeness (TLS + classifier)"),
    ]:
        grid = ok.groupby(["inj_rp_earth", "inj_period"])[metric].mean().unstack()
        if grid.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=150, facecolor=theme.SURFACE)
        im = ax.imshow(grid.values, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels([f"{p:.1f}" for p in grid.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels([f"{r:.1f}" for r in grid.index])
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                            color="white" if v < 0.6 else "black", fontsize=8)
        ax.set_xlabel("Injected period (days)")
        ax.set_ylabel("Injected planet radius (R⊕)")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12, loc="left", color=theme.INK)
        ax.tick_params(colors=theme.INK_MUTED, labelsize=9)
        cbar = fig.colorbar(im, ax=ax); cbar.set_label("Recovery fraction", color=theme.INK_MUTED)
        cbar.ax.yaxis.set_tick_params(color=theme.INK_MUTED)
        plt.setp(plt.getp(cbar.ax, "yticklabels"), color=theme.INK_MUTED)
        theme.save(fig, IMG_DIR / fname)

    # Detection efficiency vs SNR (sigmoid-like completeness curve)
    ok = ok[ok["inj_snr"] > 0]
    if len(ok) >= 8:
        bins = np.geomspace(max(ok["inj_snr"].min(), 1), ok["inj_snr"].max() + 1, 10)
        centers = np.sqrt(bins[:-1] * bins[1:])
        idx = np.digitize(ok["inj_snr"], bins) - 1
        eff = [ok["tls_recovered"][idx == k].mean() if (idx == k).any() else np.nan
               for k in range(len(centers))]
        fig, ax = theme.new_fig("Detection efficiency vs injected SNR")
        ax.plot(centers, eff, color=theme.ACCENT, lw=2, marker="o", markersize=5)
        ax.axhline(0.5, color=theme.WARN, ls=":", lw=1, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("Injected transit SNR"); ax.set_ylabel("Recovery fraction"); ax.set_ylim(-0.05, 1.05)
        theme.save(fig, IMG_DIR / "injection_efficiency_snr.png")

    import json
    summary = {
        "n_injections": int(len(df)),
        "n_valid": int(len(df[(df["error"].isna()) | (df["error"] == "")])),
        "overall_tls_recovery": round(float(ok["tls_recovered"].mean()), 3) if len(ok) else None,
        "overall_clf_recovery": round(float(ok["clf_recovered"].mean()), 3) if len(ok) else None,
        "hosts": sorted(df["host"].unique().tolist()),
    }
    (RESULTS_DIR / "injection_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Injection summary: %s", summary)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Small grid, single host (fast sanity run)")
    ap.add_argument("--full", action="store_true", help="Full grid over screened quiet hosts")
    ap.add_argument("--n-hosts", type=int, default=3, help="Number of quiet hosts to screen for (full run)")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    if args.figures_only:
        finalize()
        return
    run(smoke=not args.full, n_hosts=args.n_hosts)


if __name__ == "__main__":
    main()
