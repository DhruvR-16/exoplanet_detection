import os, warnings, logging, random, time, hashlib, json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import lightkurve as lk
from astroquery.mast import Catalogs
from wotan import flatten, slide_clip
from transitleastsquares import transitleastsquares, cleaned_array, catalog_info

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ExoPipelineV9")

R_JUP_R_SUN = 0.10271
R_EARTH_R_SUN = 0.00916

@dataclass
class PipelineConfig:
    tic_id: str = "TIC 307210830"
    cadence: str = "long"
    author: str = "SPOC"
    quality_bitmask: str = "hardest"
    sigma_clip: float = 3.0
    min_sector_cadences: int = 200
    max_sectors: int = 4                # Limit baseline to speed up TLS (Issue #1, #16)
    period_min: float = 0.5
    period_max: float = 15.0
    tls_oversampling: int = 2
    tls_duration_grid_step: float = 1.1
    tls_max_passes: int = 3
    transit_depth_min: float = 10e-6
    bin_cadence_minutes: float = 10.0
    wotan_method: str = "biweight"
    wotan_window_default: float = 0.5
    max_rp_rjup: float = 2.5
    centroid_threshold_px: float = 0.4
    secondary_eclipse_threshold: float = 0.5
    odd_even_sigma_threshold: float = 3.0
    duration_tolerance: float = 0.5
    min_transits: int = 2
    sde_threshold: float = 7.0
    use_threads: int = 1
    max_runtime_minutes: float = 30.0
    random_seed: int = 42
    pipeline_version: str = "9.0"

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

def fetch_stellar_params(tic_id):
    tic_num = tic_id.replace("TIC ", "").strip()
    try:
        res = Catalogs.query_criteria(catalog="TIC", ID=tic_num, objType="STAR")
        def sf(v, fb):
            try:
                v2 = float(v)
                return v2 if np.isfinite(v2) else fb
            except: return fb
        stellar = {
            "radius_sun": sf(res['rad'][0], 1.0) if res['rad'][0] else 1.0,
            "mass_sun": sf(res['mass'][0], 1.0) if res['mass'][0] else 1.0,
            "teff_k": sf(res['Teff'][0], 5778.0) if res['Teff'][0] else 5778.0,
            "logg": sf(res['logg'][0], 4.44) if res['logg'][0] else 4.44,
            "source": "TIC"
        }
    except Exception as e:
        stellar = {"radius_sun": 1.0, "mass_sun": 1.0, "teff_k": 5778.0, "logg": 4.44, "source": "default"}
    try:
        ci = catalog_info(TIC_ID=int(tic_num))
        if len(ci) == 7:
            stellar["limb_dark"] = list(ci[0])
            stellar["mass_sun"] = float(ci[1]) if np.isfinite(ci[1]) else stellar["mass_sun"]
            stellar["radius_sun_tls"] = float(ci[4]) if np.isfinite(ci[4]) else stellar["radius_sun"]
        elif len(ci) >= 3:
            stellar["limb_dark"] = list(ci[0])
    except:
        stellar["limb_dark"] = [0.4, 0.3]
    return stellar

def fetch_lightcurves(tic_id, cfg):
    sr = lk.search_lightcurve(tic_id, mission="TESS", cadence=cfg.cadence, author=cfg.author)
    if not sr: sr = lk.search_lightcurve(tic_id, mission="TESS", cadence="long", author="QLP")
    if not sr: raise RuntimeError(f"No TESS data for {tic_id}")
    
    if hasattr(cfg, 'max_sectors') and cfg.max_sectors > 0:
        sr = sr[:cfg.max_sectors]
        logger.info(f"Limiting to first {cfg.max_sectors} sectors to bound TLS search space.")
    
    return sr.download_all(quality_bitmask=cfg.quality_bitmask, cache=True)

def preprocess_sectors(lc_coll, cfg):
    clean_sectors = []
    crowdsap_vals = []
    for i, lc in enumerate(lc_coll):
        if len(lc) < cfg.min_sector_cadences: continue
        lc = lc.remove_nans().remove_outliers(sigma=cfg.sigma_clip, maxiters=5)
        if len(lc) < 100: continue
        f = lc.flux.value
        med = np.nanmedian(f)
        good = np.abs(f - med) < 5.0 * np.nanstd(f)
        lc = lc[good]
        if len(lc) < 100: continue
        clean_sectors.append(lc.normalize())
        try:
            cs = getattr(lc, 'meta', {}).get('CROWDSAP', None)
            if cs is not None: crowdsap_vals.append(float(cs))
        except: pass
    if not clean_sectors: raise RuntimeError("No sectors survived quality filtering!")
    stitched = lk.LightCurveCollection(clean_sectors).stitch()
    idx = np.argsort(stitched.time.value)
    stitched = stitched[idx].normalize()
    f = stitched.flux.value
    med, std = np.nanmedian(f), np.nanstd(f)
    good = (f > med - 4*std) & (f < med + 4*std) & np.isfinite(f)
    stitched = stitched[good].normalize()
    crowdsap = float(np.nanmedian(crowdsap_vals)) if crowdsap_vals else 1.0
    return stitched, crowdsap

def gap_aware_bin(time, flux, cadence_min=10.0):
    bin_width = cadence_min / (24.0 * 60.0)
    dt = np.diff(time)
    gap_idx = np.where(dt > 0.5)[0]
    segments = []
    start = 0
    for gi in gap_idx:
        segments.append((start, gi + 1))
        start = gi + 1
    segments.append((start, len(time)))
    t_binned, f_binned = [], []
    for s_start, s_end in segments:
        t_seg, f_seg = time[s_start:s_end], flux[s_start:s_end]
        if len(t_seg) < 3: continue
        bins = np.arange(t_seg[0], t_seg[-1], bin_width)
        for j in range(len(bins) - 1):
            m = (t_seg >= bins[j]) & (t_seg < bins[j+1])
            if m.sum() > 0:
                t_binned.append(np.median(t_seg[m]))
                f_binned.append(np.median(f_seg[m]))
    return np.array(t_binned), np.array(f_binned)

def build_transit_mask(time, period, t0, duration_h=4.0):
    dur_days = duration_h / 24.0
    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    return np.abs(phase) < (dur_days / period)

def adaptive_detrend(time, flux, cfg):
    flat_flux, trend = flatten(time, flux, window_length=cfg.wotan_window_default,
        method=cfg.wotan_method, return_trend=True, break_tolerance=0.5, robust=True)
    flat_flux = slide_clip(time, flat_flux, window_length=1.0,
        low=cfg.sigma_clip, high=cfg.sigma_clip, method="mad", center="median")
    return flat_flux, trend

def compute_noise_metrics(time, flux):
    valid = np.isfinite(flux)
    t, f = time[valid], flux[valid]
    rms = float(np.std(f))
    return {"rms": rms, "cdpp_ppm": rms * 1e6, "red_noise_factor": 1.0}

def run_tls_search(time, flux, stellar, cfg):
    t_c, f_c = cleaned_array(time, flux)
    baseline = float(t_c[-1] - t_c[0])
    p_max = min(cfg.period_max, baseline / 2.0)
    R_s = stellar.get("radius_sun", 1.0)
    M_s = stellar.get("mass_sun", 1.0)
    ab = stellar.get("limb_dark", [0.4, 0.3])
    logger.info(f"Searching {len(t_c)} points with TLS. Period range: [{cfg.period_min:.1f}, {p_max:.1f}]d")
    tls = transitleastsquares(t_c, f_c)
    result = tls.power(
        R_star=float(R_s), M_star=float(M_s), u=list(ab),
        period_min=cfg.period_min, period_max=p_max,
        oversampling_factor=cfg.tls_oversampling,
        duration_grid_step=cfg.tls_duration_grid_step,
        transit_depth_min=cfg.transit_depth_min,
        use_threads=cfg.use_threads, show_progress_bar=True)
    return result

def is_harmonic(p1, p2, tol=0.01):
    for n in [0.5, 2.0, 1/3, 3.0, 0.25, 4.0]:
        if abs(p2 - p1 * n) / p1 < tol: return True
    return False

def multi_pass_tls(time, flux, stellar, cfg, noise_metrics):
    candidates = []
    t_work, f_work = time.copy(), flux.copy()
    found_periods = []
    for pass_num in range(cfg.tls_max_passes):
        logger.info(f"TLS pass {pass_num+1}/{cfg.tls_max_passes} ...")
        try:
            tls_res = run_tls_search(t_work, f_work, stellar, cfg)
        except Exception as e:
            logger.error(f"TLS pass {pass_num+1} failed: {e}")
            break
        sde = float(tls_res.SDE) if hasattr(tls_res, 'SDE') else 0.0
        if sde < cfg.sde_threshold: break
        candidates.append(tls_res)
        found_periods.append(tls_res.period)
        if hasattr(tls_res, 'duration') and tls_res.duration:
            mask = build_transit_mask(t_work, tls_res.period, tls_res.T0, tls_res.duration * 24)
            f_work = np.where(mask, np.nanmedian(f_work), f_work)
    return candidates

def run_pipeline(tic_id, cfg=None):
    if cfg is None: cfg = PipelineConfig(tic_id=tic_id)
    stellar = fetch_stellar_params(cfg.tic_id)
    lc_coll = fetch_lightcurves(cfg.tic_id, cfg)
    lc_s, crowdsap = preprocess_sectors(lc_coll, cfg)
    t_raw, f_raw = lc_s.time.value, lc_s.flux.value
    f_flat, trend = adaptive_detrend(t_raw, f_raw, cfg)
    valid = np.isfinite(f_flat)
    t_det, f_det = t_raw[valid], f_flat[valid]
    noise = compute_noise_metrics(t_det, f_det)
    
    # Proper binning step
    if cfg.bin_cadence_minutes > 0:
        t_bin, f_bin = gap_aware_bin(t_det, f_det, cadence_min=cfg.bin_cadence_minutes)
        logger.info(f"Binned {len(t_det)} -> {len(t_bin)} points ({cfg.bin_cadence_minutes}min cadence)")
    else:
        t_bin, f_bin = t_det, f_det

    candidates = multi_pass_tls(t_bin, f_bin, stellar, cfg, noise)
    print(f"\n{'='*40}")
    print(f"Number of candidates: {len(candidates)}")
    if candidates:
        print(f"Top candidate Period: {candidates[0].period:.5f}d, SDE: {candidates[0].SDE:.2f}")
    print(f"{'='*40}")

if __name__ == "__main__":
    run_pipeline("TIC 307210830")
