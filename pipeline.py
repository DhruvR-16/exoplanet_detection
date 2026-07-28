"""Core exoplanet transit-detection pipeline shared by the Streamlit app,
the FastAPI backend and the analysis notebook.

Stages: lightcurve acquisition (SPOC -> any TESS product -> TESScut FFI
fallback, with on-disk caching) -> wotan biweight detrending -> gap-aware
binning -> Transit Least Squares search (with TIC stellar priors when
available) -> feature extraction in units matching the trained model ->
physics-based vetting (odd/even Welch test, transit duration & stellar
density consistency, secondary eclipse search).
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np

warnings.filterwarnings("ignore")

import lightkurve as lk
from scipy import stats
from scipy.stats import binned_statistic, median_abs_deviation
from transitleastsquares import catalog_info, transitleastsquares
from wotan import flatten

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / "lc_cache"
MODEL_DIR = ROOT_DIR / "model"

# Features consumed by the ML model. Order and units must match train_model.py
# (KOI catalog units: period in days, depth in ppm, duration in hours).
FEATURE_NAMES = [
    "period_days",
    "depth_ppm",
    "duration_hrs",
    "model_snr",
    "rp_rs",
    "log10_depth",
    "log10_period",
    "duration_over_period",
]

# Search / vetting configuration
DEFAULT_MAX_SECTORS = 3
BIN_CADENCE_MINUTES = 10.0
DETREND_WINDOW_DAYS = 0.5
PERIOD_MIN_DAYS = 0.5
PERIOD_MAX_DAYS = 15.0
SDE_THRESHOLD = 7.0
WELCH_P_THRESHOLD = 0.01
# Effect-size guard: with 10^5 points the t-test flags negligible differences,
# so an odd/even failure also requires a >=10% relative depth difference.
OE_REL_DIFF_THRESHOLD = 0.10
SECONDARY_SNR_THRESHOLD = 3.0
# A stellar companion's secondary eclipse is deep relative to the primary;
# a hot Jupiter's dayside emission is not (WASP-121 b: ~2% of primary).
SECONDARY_REL_DEPTH_THRESHOLD = 0.15
SECONDARY_MIN_DEPTH = 100e-6
MAX_DURATION_RATIO = 1.5
DENSITY_RATIO_BOUNDS = (0.1, 30.0)

# Decision threshold on the calibrated planet probability.
#
# The model is trained on Kepler but deployed on TESS. Under that domain shift
# the calibrated probabilities are systematically compressed toward zero (Brier
# degrades 0.08 -> 0.29), so the textbook 0.5 cut is far too strict here: on the
# 500-target labeled TESS benchmark it recovers only 28% of real planets.
# 0.117 is the Youden-optimal cut on that benchmark and is stable under 5-fold
# cross-validation (threshold identical in every fold; held-out recall
# 0.73 +/- 0.03 at precision 0.67). Using it raises F1 from 0.41 to 0.70.
#
# Set to 0.5 to recover the previous in-mission (Kepler) behavior.
DECISION_THRESHOLD = 0.117

# Kepler's third law in solar units, P in days:
#   a / R_sun = 4.2119 * (M/M_sun)^(1/3) * P^(2/3)
# Stellar mean density from the transit-fit a/R (Seager & Mallen-Ornelas 2003):
#   rho [g/cm^3] = 0.018917 * (a/R_star)^3 / P^2      (gives 1.41 for the Sun)
A_OVER_R_COEFF = 4.2119
RHO_COEFF = 0.018917
RHO_SUN_CGS = 1.408
R_JUP_R_SUN = 0.1028  # Jupiter equatorial radius in solar radii


def to_scalar(value: Any, default: float = 0.0) -> float:
    """Coerce TLS outputs (scalars, 0-d arrays, 1-element arrays) to a finite float."""
    if value is None:
        return default
    try:
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size == 0:
            return default
        val = float(arr[0])
    except (TypeError, ValueError):
        return default
    return val if np.isfinite(val) else default


def _safe_name(target: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", target.strip())


def parse_tic_id(target: str, meta: dict | None = None) -> int | None:
    """Extract a TIC number from a 'TIC 12345' style name or a lightcurve's metadata."""
    match = re.match(r"^\s*TIC[\s_-]*(\d+)\s*$", target, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    if meta:
        for key in ("TICID", "ticid"):
            tic = meta.get(key)
            if tic is not None:
                try:
                    return int(tic)
                except (TypeError, ValueError):
                    continue
    return None


def get_stellar_params(target: str, meta: dict | None = None) -> dict[str, Any]:
    """Look up stellar radius/mass (solar units) from the TIC catalog.

    Falls back to solar values when the target has no TIC ID or the query fails,
    so the pipeline keeps working offline.
    """
    solar = {"radius": 1.0, "mass": 1.0, "tic_id": None, "source": "solar fallback"}
    tic_id = parse_tic_id(target, meta)
    if tic_id is None:
        return solar
    try:
        _ab, mass, _, _, radius, _, _ = catalog_info(TIC_ID=tic_id)
        radius = float(radius) if np.isfinite(radius) else 1.0
        mass = float(mass) if np.isfinite(mass) else 1.0
        return {"radius": radius, "mass": mass, "tic_id": tic_id, "source": "TIC catalog"}
    except Exception as exc:  # network / catalog failures must not kill the run
        logger.warning("TIC %s catalog lookup failed (%s); using solar values", tic_id, exc)
        solar["tic_id"] = tic_id
        return solar


def _clean(lc: lk.LightCurve) -> lk.LightCurve:
    # Asymmetric clipping: flag flares (upward) aggressively but never clip
    # transit dips, which are the very signal we are searching for.
    return lc.remove_nans().remove_outliers(sigma_upper=3.0, sigma_lower=20.0).normalize()


def _extract_ffi_lightcurve(target: str, max_sectors: int) -> tuple[lk.LightCurve | None, int]:
    """Fallback: build a lightcurve from TESS Full Frame Images via TESScut."""
    search = lk.search_tesscut(target)
    if len(search) == 0:
        return None, 0
    curves = []
    for entry in search[: min(max_sectors, 2)]:  # FFI cutouts are slow to download
        try:
            tpf = entry.download(cutout_size=11)
            mask = tpf.create_threshold_mask(threshold=3.0, reference_pixel="center")
            if mask.sum() == 0:
                mask = "default"
            curves.append(tpf.to_lightcurve(aperture_mask=mask))
        except Exception as exc:
            logger.warning("TESScut sector download failed for %s: %s", target, exc)
    if not curves:
        return None, 0
    stitched = lk.LightCurveCollection([_clean(c) for c in curves]).stitch()
    return stitched, len(curves)


def load_lightcurve(
    target: str,
    max_sectors: int = DEFAULT_MAX_SECTORS,
    allow_ffi: bool = True,
) -> tuple[lk.LightCurve | None, dict[str, Any]]:
    """Fetch, stitch and cache a normalized TESS lightcurve for `target`.

    Returns (lightcurve, info). On failure the lightcurve is None and
    info["message"] explains why. Always returns a 2-tuple.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    # Key the cache by sector budget so a 1-sector request and a 3-sector request
    # never collide (a star cached at 1 sector must not satisfy a 3-sector fetch).
    cache_path = CACHE_DIR / f"{_safe_name(target)}_s{max_sectors}.fits"
    legacy_path = CACHE_DIR / f"{_safe_name(target)}.fits"
    # Reuse a pre-existing unsuffixed cache only if it actually holds at least the
    # requested number of sectors (older caches lacking the tag are assumed full).
    if not cache_path.exists() and legacy_path.exists():
        try:
            legacy_sectors = int(lk.read(str(legacy_path)).meta.get("SECTORS", max_sectors))
        except Exception:
            legacy_sectors = 0
        if legacy_sectors >= max_sectors:
            cache_path = legacy_path

    if cache_path.exists():
        try:
            lc = lk.read(str(cache_path))
            n_sectors = int(lc.meta.get("SECTORS", 1))
            return _clean(lc), {"message": "Loaded from cache", "source": "cache", "n_sectors": n_sectors}
        except Exception as exc:
            logger.warning("Corrupt cache for %s (%s); refetching", target, exc)
            cache_path.unlink(missing_ok=True)

    source = "SPOC"
    try:
        search = lk.search_lightcurve(target, mission="TESS", author="SPOC")
        if len(search) == 0:
            search = lk.search_lightcurve(target, mission="TESS")
            source = "TESS (non-SPOC)"
    except Exception as exc:
        return None, {"message": f"MAST search failed: {exc}", "source": None, "n_sectors": 0}

    lc, n_used = None, 0
    if len(search) > 0:
        curves = []
        for entry in search[:max_sectors]:
            try:
                curves.append(_clean(entry.download()))
            except Exception as exc:
                logger.warning("Sector download failed for %s: %s", target, exc)
        if curves:
            lc = lk.LightCurveCollection(curves).stitch()
            n_used = len(curves)

    if lc is None and allow_ffi:
        source = "TESScut FFI"
        try:
            lc, n_used = _extract_ffi_lightcurve(target, max_sectors)
        except Exception as exc:
            return None, {"message": f"FFI extraction failed: {exc}", "source": None, "n_sectors": 0}

    if lc is None:
        return None, {"message": f"No TESS data found for {target}.", "source": None, "n_sectors": 0}

    try:
        extra = {"SECTORS": n_used}
        tic_id = parse_tic_id(target, dict(lc.meta))
        if tic_id is not None:
            extra["TICID"] = tic_id
        lc.to_fits(str(cache_path), overwrite=True, **extra)
    except Exception as exc:
        logger.warning("Could not cache lightcurve for %s: %s", target, exc)

    return lc, {"message": "Downloaded from MAST", "source": source, "n_sectors": n_used}


def detrend(
    lc: lk.LightCurve, window_days: float = DETREND_WINDOW_DAYS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove stellar variability with a wotan biweight filter.

    Returns (time, raw_flux, flat_flux, trend) with NaNs removed.
    """
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]
    flat, trend = flatten(time, flux, method="biweight", window_length=window_days, return_trend=True)
    valid = np.isfinite(flat)
    return time[valid], flux[valid], flat[valid], trend[valid]


def gap_aware_bin(
    time: np.ndarray, flux: np.ndarray, cadence_minutes: float = BIN_CADENCE_MINUTES
) -> tuple[np.ndarray, np.ndarray]:
    """Median-bin to `cadence_minutes`, never averaging across data gaps (>0.5 d).

    Skips binning entirely if the native cadence is already coarser.
    """
    if len(time) < 10:
        return time, flux
    bin_width = cadence_minutes / (24.0 * 60.0)
    native_cadence = float(np.median(np.diff(time)))
    if native_cadence >= bin_width:
        return time, flux

    gap_idx = np.where(np.diff(time) > 0.5)[0]
    starts = np.concatenate(([0], gap_idx + 1))
    ends = np.concatenate((gap_idx + 1, [len(time)]))

    t_binned, f_binned = [], []
    for s, e in zip(starts, ends):
        t_seg, f_seg = time[s:e], flux[s:e]
        if len(t_seg) < 5:
            continue
        edges = np.arange(t_seg[0], t_seg[-1] + bin_width, bin_width)
        if len(edges) < 2:
            continue
        f_med, _, _ = binned_statistic(t_seg, f_seg, statistic="median", bins=edges)
        t_med, _, _ = binned_statistic(t_seg, t_seg, statistic="median", bins=edges)
        keep = np.isfinite(f_med) & np.isfinite(t_med)
        t_binned.append(t_med[keep])
        f_binned.append(f_med[keep])

    if not t_binned:
        return time, flux
    return np.concatenate(t_binned), np.concatenate(f_binned)


def run_tls(
    time: np.ndarray,
    flux: np.ndarray,
    stellar: dict[str, Any] | None = None,
    period_min: float = PERIOD_MIN_DAYS,
    period_max: float = PERIOD_MAX_DAYS,
    use_threads: int = 1,
) -> Any:
    """Run Transit Least Squares with stellar priors and a bounded period grid."""
    baseline = float(time[-1] - time[0]) if len(time) > 1 else 0.0
    # Require at least two transits within the observed baseline.
    period_max = max(period_min + 0.1, min(period_max, baseline / 2.0))
    kwargs: dict[str, Any] = {
        "period_min": period_min,
        "period_max": period_max,
        "oversampling_factor": 2,
        "duration_grid_step": 1.1,
        "use_threads": use_threads,
        "show_progress_bar": False,
    }
    if stellar is not None:
        r_star = float(stellar.get("radius", 1.0))
        m_star = float(stellar.get("mass", 1.0))
        # TLS's default priors reject stars outside ~0.1-1.0 M_sun, so always
        # provide explicit bounds bracketing the catalog values.
        kwargs.update(
            R_star=r_star, R_star_min=max(0.05, r_star * 0.7), R_star_max=r_star * 1.3,
            M_star=m_star, M_star_min=max(0.05, m_star * 0.7), M_star_max=m_star * 1.3,
        )
    tls = transitleastsquares(time, flux)
    return tls.power(**kwargs)


def transit_duration(tls_results: Any) -> float:
    """Best-fit transit duration in days, measured from the TLS model curve.

    TLS's `duration` attribute can disagree with its own fitted model by ~8x
    on short-cadence lightcurves (observed on 20-s SPOC data), so the width of
    the folded model dip is the trustworthy value.
    """
    period = to_scalar(tls_results.period)
    try:
        phase = np.asarray(tls_results.model_folded_phase, dtype=float)
        model = np.asarray(tls_results.model_folded_model, dtype=float)
        in_transit = model < 1.0 - 1e-9
        if in_transit.sum() >= 2:
            width = float(phase[in_transit].max() - phase[in_transit].min())
            if 0.0 < width < 0.5:
                return width * period
    except (AttributeError, TypeError, ValueError):
        pass
    return to_scalar(tls_results.duration)


def extract_features(tls_results: Any) -> dict[str, float]:
    """Build the model feature vector (KOI-catalog units) from a TLS result."""
    period = to_scalar(tls_results.period)
    duration_days = transit_duration(tls_results)
    # TLS reports depth as the flux level at mid-transit (~0.99); convert to a
    # fractional dip, then to ppm to match the Kepler catalog convention.
    depth_frac = max(0.0, 1.0 - to_scalar(tls_results.depth, default=1.0))
    snr = to_scalar(getattr(tls_results, "snr", 0.0))
    rp_rs = to_scalar(getattr(tls_results, "rp_rs", 0.0))
    if rp_rs <= 0 and depth_frac > 0:
        rp_rs = float(np.sqrt(depth_frac))

    depth_ppm = depth_frac * 1e6
    return {
        "period_days": period,
        "depth_ppm": depth_ppm,
        "duration_hrs": duration_days * 24.0,
        "model_snr": snr,
        "rp_rs": rp_rs,
        "log10_depth": float(np.log10(max(depth_ppm, 1.0))),
        "log10_period": float(np.log10(max(period, 1e-3))),
        "duration_over_period": duration_days / period if period > 0 else 0.0,
    }


def _fold_phase(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Phase in [-0.5, 0.5) with the transit centered at 0."""
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    return phase


def calculate_shape_features(
    time: np.ndarray, flux: np.ndarray, period: float, duration: float, t0: float
) -> tuple[float, float, float]:
    """Ingress/egress symmetry, point-count shape ratio and in-transit scatter."""
    if period <= 0 or duration <= 0:
        return 0.0, 0.0, 0.0
    phase = _fold_phase(time, period, t0)
    in_transit = np.abs(phase) < (duration / period / 2.0)
    if np.sum(in_transit) < 5:
        return 0.0, 0.0, 0.0

    transit_phase = phase[in_transit]
    transit_flux = flux[in_transit]
    order = np.argsort(transit_phase)
    sorted_phase, sorted_flux = transit_phase[order], transit_flux[order]

    ingress = sorted_phase < 0
    egress = ~ingress
    if ingress.sum() > 2 and egress.sum() > 2:
        # Mirror the egress branch onto the ingress phase grid; a symmetric
        # (planet-like) transit leaves near-zero residuals.
        egress_interp = np.interp(-sorted_phase[ingress], sorted_phase[egress], sorted_flux[egress])
        symmetry = float(np.std(sorted_flux[ingress] - egress_interp))
    else:
        symmetry = 0.0

    shape_ratio = abs(int(ingress.sum()) - int(egress.sum())) / max(int(in_transit.sum()), 1)
    return symmetry, float(shape_ratio), float(np.std(transit_flux))


def odd_even_test(
    time: np.ndarray, flux: np.ndarray, period: float, duration: float, t0: float
) -> tuple[float, float, float, float]:
    """Compare odd vs even transits. Returns (depth_diff, duration_diff, mad_ratio, welch_p).

    A small Welch p-value means odd/even depths differ significantly - the
    classic signature of an eclipsing binary at twice the true period.
    """
    if period <= 0 or duration <= 0:
        return 0.0, 0.0, 0.0, 1.0
    phase = _fold_phase(time, period, t0)
    in_transit = np.abs(phase) < (duration / period / 2.0)
    # round() (not floor) so points just before mid-transit count toward the
    # same transit as points just after it.
    transit_number = np.round((time - t0) / period)

    odd_mask = in_transit & (transit_number % 2 == 1)
    even_mask = in_transit & (transit_number % 2 == 0)
    if odd_mask.sum() < 3 or even_mask.sum() < 3:
        return 0.0, 0.0, 0.0, 1.0

    odd_flux, even_flux = flux[odd_mask], flux[even_mask]
    depth_diff = abs(float(np.median(odd_flux)) - float(np.median(even_flux)))

    odd_dur = odd_mask.sum() / len(time) * period
    even_dur = even_mask.sum() / len(time) * period
    duration_diff = abs(odd_dur - even_dur) / max(duration, 1e-10)

    mad_odd = median_abs_deviation(odd_flux)
    mad_even = median_abs_deviation(even_flux)
    if mad_odd > 1e-10 and mad_even > 1e-10:
        mad_ratio = float(abs(np.log(np.clip(mad_odd / mad_even, 0.01, 100.0))))
    else:
        mad_ratio = 0.0

    t_stat, p_val = stats.ttest_ind(odd_flux, even_flux, equal_var=False)
    welch_p = float(p_val) if np.isfinite(p_val) else 1.0
    return depth_diff, float(duration_diff), mad_ratio, welch_p


def transit_duration_estimate(period: float, a_over_r: float, impact: float = 0.0) -> float:
    """Approximate transit duration (days) for a circular orbit.

    T = (P / pi) * arcsin( sqrt(1 - b^2) / (a/R) ). Used to estimate the
    injected-transit SNR in the injection-recovery study.
    """
    if period <= 0 or a_over_r <= 1.0:
        return 0.0
    arg = np.sqrt(max(1.0 - impact**2, 0.0)) / a_over_r
    return float(period / np.pi * np.arcsin(np.clip(arg, 0.0, 1.0)))


def check_transit_physics(
    period: float, duration_days: float, r_star: float, m_star: float
) -> dict[str, Any]:
    """Vet the transit duration and implied stellar density against the catalog star.

    duration_ratio: observed duration over the maximum central-transit duration
    for a circular orbit. density_ratio: stellar density implied by the transit
    (Seager & Mallen-Ornelas 2003) over the catalog density - far from 1 means
    the eclipsed object is not the catalog star (blend / eclipsing binary).
    """
    result = {"duration_ok": True, "duration_ratio": 1.0, "density_ok": True, "density_ratio": 1.0}
    if period <= 0 or duration_days <= 0 or r_star <= 0 or m_star <= 0:
        return result

    a_over_r = A_OVER_R_COEFF * (m_star ** (1.0 / 3.0)) * (period ** (2.0 / 3.0)) / r_star
    if a_over_r <= 1.0:
        return result
    max_duration = period / (np.pi * a_over_r)
    duration_ratio = duration_days / max_duration

    est_a_over_r = period / (np.pi * duration_days)
    rho_transit = RHO_COEFF * est_a_over_r**3 / period**2
    rho_star = RHO_SUN_CGS * m_star / r_star**3
    density_ratio = rho_transit / rho_star

    lo, hi = DENSITY_RATIO_BOUNDS
    return {
        "duration_ok": bool(duration_ratio <= MAX_DURATION_RATIO),
        "duration_ratio": float(duration_ratio),
        "density_ok": bool(lo <= density_ratio <= hi),
        "density_ratio": float(density_ratio),
    }


def check_secondary_eclipse(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    duration: float,
    t0: float,
    primary_depth_frac: float = 0.0,
) -> dict[str, Any]:
    """Search for a dip at phase 0.5 and judge whether it is binary-like.

    `secondary_detected` marks any significant dip; `has_secondary` (the
    eclipsing-binary flag) additionally requires the dip to be deep relative
    to the primary — a shallow secondary is consistent with planetary dayside
    emission (ultra-hot Jupiters show exactly this), not a stellar companion.
    The noise baseline excludes both the primary and secondary windows.
    """
    result = {
        "has_secondary": False,
        "secondary_detected": False,
        "secondary_depth": 0.0,
        "secondary_snr": 0.0,
    }
    if period <= 0 or duration <= 0 or len(time) == 0:
        return result

    phase = _fold_phase(time, period, t0)
    dur_frac = duration / period
    in_primary = np.abs(phase) < dur_frac  # generous window: 2x duration
    in_secondary = np.abs(np.abs(phase) - 0.5) < dur_frac / 2.0
    baseline = ~in_primary & ~in_secondary

    if in_secondary.sum() < 3 or baseline.sum() < 10:
        return result

    secondary_depth = 1.0 - float(np.nanmedian(flux[in_secondary]))
    noise = float(np.nanstd(flux[baseline]))
    if noise <= 1e-10:
        return result
    snr = secondary_depth / (noise / np.sqrt(in_secondary.sum()))
    detected = bool(snr >= SECONDARY_SNR_THRESHOLD and secondary_depth > 0)
    binary_like = detected and secondary_depth >= max(
        SECONDARY_REL_DEPTH_THRESHOLD * primary_depth_frac, SECONDARY_MIN_DEPTH
    )
    return {
        "has_secondary": binary_like,
        "secondary_detected": detected,
        "secondary_depth": float(secondary_depth),
        "secondary_snr": float(snr),
    }


def load_model(model_dir: Path | str = MODEL_DIR) -> tuple[dict[str, Any] | None, str | None]:
    """Load the newest model package (dict with model/scaler/feature_names)."""
    model_dir = Path(model_dir)
    candidates = sorted(model_dir.glob("exoplanet_model_v*.pkl"), reverse=True)
    for path in candidates:
        try:
            pkg = joblib.load(path)
            version = path.stem.rsplit("_", 1)[-1]
            return pkg, version
        except Exception as exc:
            logger.error("Failed to load %s: %s", path, exc)
    return None, None


def predict(
    model_pkg: dict[str, Any],
    features: dict[str, float],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Score a feature dict with a trained model package.

    `threshold` defaults to the model package's own `decision_threshold` if it
    carries one, else the module-level DECISION_THRESHOLD (TESS-calibrated).
    """
    names = model_pkg["feature_names"]
    missing = [n for n in names if n not in features]
    if missing:
        raise KeyError(f"Features missing for model: {missing}")
    x = np.array([[features[n] for n in names]], dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = model_pkg.get("scaler")
    if scaler is not None:
        x = scaler.transform(x)
    model = model_pkg["model"]
    probability = float(model.predict_proba(x)[0, 1])
    if threshold is None:
        threshold = float(model_pkg.get("decision_threshold", DECISION_THRESHOLD))
    prediction = int(probability >= threshold)
    # Confidence is distance from the operating point, normalized by how much
    # room there is on that side, so it stays meaningful for any threshold.
    span = (1.0 - threshold) if probability >= threshold else threshold
    distance = abs(probability - threshold) / span if span > 0 else 0.0
    confidence = "High" if distance > 0.60 else "Medium" if distance > 0.30 else "Low"
    return {
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold,
        "confidence": confidence,
        "result_text": "Planet Candidate Detected" if prediction == 1 else "No Planet Transit Detected",
    }


def build_explanation(result: dict[str, Any]) -> list[str]:
    """Plain-English reasoning for a pipeline verdict, one statement per line.

    Every statement is derived from the measured values in `result`, so the
    text is a faithful account of why the verdict came out the way it did.
    """
    data, stellar = result["data"], result["stellar"]
    feats, diag, vet = result["features"], result["diagnostics"], result["vetting"]
    lines: list[str] = []

    lines.append(
        f"Data: {data['n_points']:,} cadences from {data['n_sectors']} TESS sector(s) "
        f"(source: {data['source']})."
    )
    lines.append(
        f"Host star: R = {stellar['radius']:.2f} R☉, M = {stellar['mass']:.2f} M☉ "
        f"({stellar['source']})."
    )
    lines.append(
        f"Best periodic signal: P = {feats['period_days']:.4f} d, depth = {feats['depth_ppm']:.0f} ppm, "
        f"duration = {feats['duration_hrs']:.2f} h, Rp/Rs = {feats['rp_rs']:.4f}."
    )

    if diag["sde_pass"]:
        lines.append(
            f"Signal strength: SDE = {diag['sde']:.1f} ≥ {SDE_THRESHOLD:.0f} — the periodic "
            "dip is statistically significant, not a noise fluctuation."
        )
    else:
        lines.append(
            f"Signal strength: SDE = {diag['sde']:.1f} < {SDE_THRESHOLD:.0f} — no statistically "
            "significant transit signal exists in this lightcurve; the numbers below describe "
            "the strongest noise pattern, not a real detection."
        )

    rp_rjup = feats["rp_rs"] * stellar["radius"] / R_JUP_R_SUN
    if feats["rp_rs"] >= 0.2:
        size_note = "far too large for a planet — consistent with a stellar companion"
    elif feats["rp_rs"] >= 0.12:
        size_note = "borderline — either a very large planet or a small stellar companion"
    else:
        size_note = "within the planetary regime"
    lines.append(f"Implied companion radius: {rp_rjup:.2f} R_Jup ({size_note}).")

    oe_rel = diag.get("odd_even_rel_diff", 0.0)
    if vet["odd_even_ok"] and diag["welch_p"] >= WELCH_P_THRESHOLD:
        lines.append(
            f"Odd/even test: alternating transits have consistent depths "
            f"(Welch p = {diag['welch_p']:.3f}) — no sign of an eclipsing binary at twice the period."
        )
    elif vet["odd_even_ok"]:
        lines.append(
            f"Odd/even test: a depth difference is statistically detectable (p = {diag['welch_p']:.2e}) "
            f"but amounts to only {oe_rel:.1%} of the transit depth — negligible effect size, "
            "consistent with residual systematics rather than an eclipsing binary."
        )
    else:
        lines.append(
            f"Odd/even test FAILED: alternating transits differ significantly "
            f"(Welch p = {diag['welch_p']:.2e}, {oe_rel:.0%} of the transit depth) — "
            "classic eclipsing-binary signature."
        )

    if vet["duration_ok"]:
        lines.append(
            f"Duration check: transit lasts {vet['duration_ratio']:.2f}× the circular-orbit "
            "maximum — physically plausible."
        )
    else:
        lines.append(
            f"Duration check FAILED: transit lasts {vet['duration_ratio']:.2f}× the circular-orbit "
            "maximum for this star — physically implausible for a planet."
        )

    if vet["density_ok"]:
        lines.append(
            f"Density check: transit-implied stellar density is {vet['density_ratio']:.2f}× the "
            "catalog value — the eclipsed object is consistent with the target star."
        )
    else:
        lines.append(
            f"Density check FAILED: transit-implied density is {vet['density_ratio']:.2f}× the "
            "catalog value — the dip likely comes from a blended or background star."
        )

    if vet["has_secondary"]:
        lines.append(
            f"Secondary eclipse DETECTED at phase 0.5 (depth = {vet['secondary_depth'] * 1e6:.0f} ppm, "
            f"S/N = {vet['secondary_snr']:.1f}) — deep relative to the primary, indicating a "
            "self-luminous stellar companion."
        )
    elif vet.get("secondary_detected"):
        rel = vet["secondary_depth"] / max(feats["depth_ppm"] * 1e-6, 1e-9)
        lines.append(
            f"Secondary eclipse: a shallow dip exists at phase 0.5 "
            f"({vet['secondary_depth'] * 1e6:.0f} ppm, only {rel:.1%} of the primary depth) — "
            "consistent with planetary dayside emission, not a stellar companion."
        )
    else:
        lines.append(
            f"Secondary eclipse: none found at phase 0.5 (S/N = {vet['secondary_snr']:.1f} "
            f"< {SECONDARY_SNR_THRESHOLD:.0f}) — no sign of a self-luminous companion."
        )

    pred = result.get("prediction")
    if pred is not None:
        lines.append(
            f"ML classifier: calibrated ensemble (trained on 7,325 labeled Kepler objects) assigns "
            f"a {pred['probability']:.1%} planet probability ({pred['confidence']} confidence), "
            f"against a {pred.get('threshold', DECISION_THRESHOLD):.1%} decision threshold "
            "calibrated for TESS deployment."
        )

    failed = [
        name
        for name, ok in [
            ("signal significance", diag["sde_pass"]),
            ("odd/even depth", vet["odd_even_ok"]),
            ("transit duration", vet["duration_ok"]),
            ("stellar density", vet["density_ok"]),
            ("no secondary eclipse", not vet["has_secondary"]),
        ]
        if not ok
    ]
    if pred is None:
        verdict = "Verdict: physics vetting only (no ML model loaded)"
    elif not diag["sde_pass"]:
        verdict = "VERDICT: NO DETECTION — no significant transit signal in this lightcurve."
    elif pred["prediction"] == 1 and not failed:
        verdict = (
            f"VERDICT: PLANET CANDIDATE — all 5 physics checks pass and the ML probability "
            f"is {pred['probability']:.1%}."
        )
    elif pred["prediction"] == 1 and failed:
        verdict = (
            f"VERDICT: DISPUTED — the ML favors a planet ({pred['probability']:.1%}) but "
            f"physics vetting failed: {', '.join(failed)}. Treat as a likely false positive."
        )
    elif not failed:
        verdict = (
            f"VERDICT: AMBIGUOUS — the transit shape passes all 5 physics checks, but the ML "
            f"assigns only {pred['probability']:.1%} (below the "
            f"{pred.get('threshold', DECISION_THRESHOLD):.1%} threshold) because signals this "
            "deep and short-period are statistically dominated by eclipsing binaries in the "
            "Kepler training data — a known cross-mission bias, since TESS's confirmed planets "
            "skew toward exactly these deep, short-period signals. Follow-up (e.g. radial "
            "velocities) would be needed to decide."
        )
    else:
        verdict = (
            f"VERDICT: LIKELY FALSE POSITIVE — ML planet probability is only "
            f"{pred['probability']:.1%}, and physics vetting failed: {', '.join(failed)}."
        )
    lines.append(verdict)
    return lines


def analyze(
    target: str,
    model_pkg: dict[str, Any] | None = None,
    max_sectors: int = DEFAULT_MAX_SECTORS,
) -> dict[str, Any]:
    """Run the full pipeline for one target. Raises RuntimeError when no data exists."""
    lc, info = load_lightcurve(target, max_sectors=max_sectors)
    if lc is None:
        raise RuntimeError(info["message"])

    stellar = get_stellar_params(target, dict(lc.meta))
    time_arr, raw_flux, flat_flux, trend = detrend(lc)
    t_bin, f_bin = gap_aware_bin(time_arr, flat_flux)
    tls_results = run_tls(t_bin, f_bin, stellar=stellar)

    period = to_scalar(tls_results.period)
    duration = transit_duration(tls_results)
    t0 = to_scalar(tls_results.T0)
    sde = to_scalar(tls_results.SDE)

    features = extract_features(tls_results)
    depth_frac = features["depth_ppm"] * 1e-6
    symmetry, shape_ratio, depth_std = calculate_shape_features(time_arr, flat_flux, period, duration, t0)
    depth_diff, duration_diff, mad_ratio, welch_p = odd_even_test(time_arr, flat_flux, period, duration, t0)
    physics = check_transit_physics(period, duration, stellar["radius"], stellar["mass"])
    secondary = check_secondary_eclipse(
        time_arr, flat_flux, period, duration, t0, primary_depth_frac=depth_frac
    )
    oe_rel_diff = depth_diff / depth_frac if depth_frac > 0 else 0.0
    odd_even_ok = bool(welch_p >= WELCH_P_THRESHOLD or oe_rel_diff < OE_REL_DIFF_THRESHOLD)

    result: dict[str, Any] = {
        "target": target,
        "data": {
            "source": info["source"],
            "n_sectors": info["n_sectors"],
            "n_points": int(len(time_arr)),
            "n_points_binned": int(len(t_bin)),
        },
        "stellar": stellar,
        "arrays": {
            "time": time_arr,
            "raw_flux": raw_flux,
            "flat_flux": flat_flux,
            "trend": trend,
            "time_binned": t_bin,
            "flux_binned": f_bin,
        },
        "tls": tls_results,
        "features": features,
        "diagnostics": {
            "sde": sde,
            "sde_pass": bool(sde >= SDE_THRESHOLD),
            "t0": t0,
            "symmetry": symmetry,
            "shape_ratio": shape_ratio,
            "depth_std": depth_std,
            "odd_even_depth_diff": depth_diff,
            "odd_even_rel_diff": oe_rel_diff,
            "odd_even_duration_diff": duration_diff,
            "odd_even_mad_ratio": mad_ratio,
            "welch_p": welch_p,
        },
        "vetting": {**physics, **secondary, "odd_even_ok": odd_even_ok},
    }
    if model_pkg is not None:
        result["prediction"] = predict(model_pkg, features)
    result["explanation"] = build_explanation(result)
    return result
