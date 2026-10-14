"""Screen recent unresolved TOIs and search their residual TESS light curves.

This is a triage tool. A residual transit is an unverified signal, not a planet
discovery; follow-up requires independent photometry, centroid checks and vetting.

Example: python hunt_planets.py --limit 3 --max-sectors 1
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import pipeline

CATALOG_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?output=csv"
BTJD_OFFSET = 2457000.0
OUTPUT = pipeline.ROOT_DIR / "analysis_output" / "planet_hunt.json"
logger = logging.getLogger("planet_hunt")


def load_catalog(path: Path | None) -> tuple[pd.DataFrame, str]:
    if path is None:
        response = requests.get(CATALOG_URL, timeout=120)
        response.raise_for_status()
        df = pd.read_csv(io.BytesIO(response.content), low_memory=False)
        source = CATALOG_URL
    else:
        df = pd.read_csv(path, low_memory=False)
        source = str(path.resolve())
    needed = {"TIC ID", "TOI", "TFOPWG Disposition", "Period (days)",
              "Epoch (BJD)", "Duration (hours)", "TESS Mag",
              "Date TOI Alerted (UTC)", "Date Modified"}
    if not needed.issubset(df.columns):
        raise ValueError(f"TOI table lacks columns: {sorted(needed - set(df.columns))}")
    return df, source


def select_targets(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    candidates = df[df["TFOPWG Disposition"].isin(["PC", "APC"])].copy()
    for column in ("TOI", "Period (days)", "TESS Mag", "TIC ID"):
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    candidates["alerted_at"] = pd.to_datetime(
        candidates["Date TOI Alerted (UTC)"], errors="coerce")
    candidates = candidates[
        candidates["TIC ID"].notna()
        & candidates["Period (days)"].between(0.5, 15)
        & candidates["TESS Mag"].le(12)
    ]
    return (candidates.sort_values(["alerted_at", "TOI"], ascending=False)
            .drop_duplicates("TIC ID").head(limit))


def match_catalog(period: float, known: pd.DataFrame) -> list[str]:
    matches = []
    for _, item in known.iterrows():
        ref = pd.to_numeric(item["Period (days)"], errors="coerce")
        if not np.isfinite(ref) or ref <= 0:
            continue
        if any(abs(period / (ref * factor) - 1) < 0.03 for factor in (0.5, 1, 2)):
            matches.append(str(item["TOI"]))
    return matches


def mask_catalog_transits(time: np.ndarray, known: pd.DataFrame) -> np.ndarray:
    keep = np.ones(len(time), dtype=bool)
    for _, item in known.iterrows():
        period = pd.to_numeric(item["Period (days)"], errors="coerce")
        epoch = pd.to_numeric(item["Epoch (BJD)"], errors="coerce")
        duration = pd.to_numeric(item["Duration (hours)"], errors="coerce")
        if not (np.isfinite(period) and period > 0 and np.isfinite(epoch)
                and np.isfinite(duration) and duration > 0):
            continue
        phase_days = ((time - (epoch - BTJD_OFFSET) + period / 2) % period) - period / 2
        keep &= np.abs(phase_days) > 1.5 * duration / 24
    return keep


def observed_transit_count(time: np.ndarray, period: float, t0: float,
                           duration: float) -> int:
    cycle = np.rint((time - t0) / period).astype(int)
    distance = np.abs(time - t0 - cycle * period)
    return int(np.unique(cycle[distance < duration / 2]).size)


def residual_search(result: dict, known: pd.DataFrame, model_pkg: dict) -> dict:
    time = result["arrays"]["time"]
    flux = result["arrays"]["flat_flux"]
    keep = mask_catalog_transits(time, known)
    if keep.sum() < 500 or keep.mean() < 0.5:
        return {"status": "insufficient_unmasked_data", "unmasked_fraction": float(keep.mean())}
    t_bin, f_bin = pipeline.gap_aware_bin(time[keep], flux[keep])
    tls = pipeline.run_tls(t_bin, f_bin, stellar=result["stellar"])
    features = pipeline.extract_features(tls)
    period = features["period_days"]
    duration = pipeline.transit_duration(tls)
    t0 = pipeline.to_scalar(tls.T0)
    sde = pipeline.to_scalar(tls.SDE)
    depth_fraction = features["depth_ppm"] * 1e-6
    depth_diff, _, _, welch_p = pipeline.odd_even_test(
        time[keep], flux[keep], period, duration, t0)
    odd_even_ok = bool(welch_p >= pipeline.WELCH_P_THRESHOLD or
                       depth_fraction > 0 and depth_diff / depth_fraction < pipeline.OE_REL_DIFF_THRESHOLD)
    physics = pipeline.check_transit_physics(
        period, duration, result["stellar"]["radius"], result["stellar"]["mass"])
    secondary = pipeline.check_secondary_eclipse(
        time[keep], flux[keep], period, duration, t0, primary_depth_frac=depth_fraction)
    prediction = pipeline.predict(model_pkg, features)
    matches = match_catalog(period, known)
    n_transits = observed_transit_count(time[keep], period, t0, duration)
    passed = bool(sde >= pipeline.SDE_THRESHOLD and n_transits >= 3 and odd_even_ok
                  and physics["duration_ok"] and physics["density_ok"]
                  and not secondary["has_secondary"]
                  and prediction["probability"] >= pipeline.DECISION_THRESHOLD)
    return {
        "status": "unlisted_signal_for_followup" if passed and not matches else
                  "catalog_period_recovered" if matches else "no_vetted_unlisted_signal",
        "period_days": period,
        "depth_ppm": features["depth_ppm"],
        "duration_hrs": features["duration_hrs"],
        "sde": sde,
        "observed_transits": n_transits,
        "planet_probability": prediction["probability"],
        "odd_even_ok": odd_even_ok,
        "duration_ok": bool(physics["duration_ok"]),
        "density_ok": bool(physics["density_ok"]),
        "has_secondary": bool(secondary["has_secondary"]),
        "matching_tois": matches,
        "unmasked_fraction": float(keep.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3, help="number of recent candidate hosts")
    parser.add_argument("--max-sectors", type=int, default=1)
    parser.add_argument("--catalog", type=Path, help="use a previously downloaded ExoFOP CSV")
    parser.add_argument("--use-cache", action="store_true", help="reuse local light curves")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.limit < 1 or args.max_sectors < 1:
        parser.error("--limit and --max-sectors must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    catalog, source = load_catalog(args.catalog)
    targets = select_targets(catalog, args.limit)
    model_pkg, model_version = pipeline.load_model()
    if model_pkg is None:
        parser.error("No model found; run train_model.py first")
    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_source": source,
        "catalog_upstream": CATALOG_URL,
        "catalog_rows": len(catalog),
        "latest_toi": float(pd.to_numeric(catalog["TOI"], errors="coerce").max()),
        "latest_catalog_modification": str(pd.to_datetime(
            catalog["Date Modified"], errors="coerce").max()),
        "model_version": model_version,
        "sector_order": "latest",
        "refresh_lightcurves": not args.use_cache,
        "targets": [],
    }
    for _, item in targets.iterrows():
        tic = int(item["TIC ID"])
        known = catalog[pd.to_numeric(catalog["TIC ID"], errors="coerce") == tic]
        row = {"tic_id": tic, "target": f"TIC {tic}", "catalog_toi": str(item["TOI"]),
               "catalog_disposition": item["TFOPWG Disposition"],
               "catalog_alert_date": str(item["Date TOI Alerted (UTC)"]),
               "catalog_period_days": float(item["Period (days)"])}
        logger.info("Analyzing %s (TOI %s)", row["target"], row["catalog_toi"])
        try:
            result = pipeline.analyze(row["target"], model_pkg=model_pkg,
                                      max_sectors=args.max_sectors, sector_order="latest",
                                      refresh_cache=not args.use_cache)
            period = result["features"]["period_days"]
            row["data"] = result["data"]
            row["primary"] = {
                "period_days": period,
                "depth_ppm": result["features"]["depth_ppm"],
                "sde": result["diagnostics"]["sde"],
                "planet_probability": result["prediction"]["probability"],
                "matching_tois": match_catalog(period, known),
                "verdict": result["explanation"][-1],
            }
            row["residual"] = residual_search(result, known, model_pkg)
            logger.info("Primary %.4f d; residual %s", period, row["residual"]["status"])
        except Exception as exc:
            logger.exception("Analysis failed for %s", row["target"])
            row["error"] = str(exc)
        report["targets"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
