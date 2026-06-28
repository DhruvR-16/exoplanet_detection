import os
import time
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Exoplanet Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to import necessary astrophysical libraries
try:
    import lightkurve as lk
    from wotan import flatten
    from transitleastsquares import transitleastsquares
    from scipy.stats import median_abs_deviation
except ImportError as e:
    print(f"Missing required astrophysics libraries: {e}")

class AnalysisRequest(BaseModel):
    target_star: str

CACHE_DIR = "lc_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Load available models
models = {}
feature_names = None

def load_models():
    global feature_names
    # Try to load v1
    try:
        if os.path.exists('../model/exoplanet_model_v1.pkl'):
            pkg_v1 = joblib.load('../model/exoplanet_model_v1.pkl')
            models['Legacy Model (v1)'] = pkg_v1
            feature_names = pkg_v1['feature_names']
    except Exception as e:
        print(f"Error loading v1 model: {e}")
        
    # Try to load v2
    try:
        if os.path.exists('../model/exoplanet_model_v2.pkl'):
            pkg_v2 = joblib.load('../model/exoplanet_model_v2.pkl')
            models['Advanced Model (v2)'] = pkg_v2
            feature_names = pkg_v2['feature_names']
    except Exception as e:
        print(f"Error loading v2 model: {e}")

load_models()

def load_lightcurve(target):
    safe_name = target.replace(" ", "_").replace("-", "_")
    path = os.path.join(CACHE_DIR, safe_name + ".fits")

    if os.path.exists(path):
        try:
            lc = lk.read(path)
            return lc.remove_nans().remove_outliers().normalize(), "Success"
        except:
            pass

    search_results = None
    try:
        search_results = lk.search_lightcurve(target, mission='TESS', author='SPOC')
        if len(search_results) == 0:
            search_results = lk.search_lightcurve(target, mission='TESS')
    except Exception as e:
        return None, str(e)
    
    if search_results is None or len(search_results) == 0:
        return None, f"No data found for {target} in TESS archive."
    
    for attempt in range(3):
        try:
            lc = search_results[0].download()
            lc.to_fits(path, overwrite=True)
            return lc.remove_nans().remove_outliers().normalize(), "Success"
        except Exception as e:
            if attempt == 2:
                return None, f"Download attempts failed: {e}"
            time.sleep(1)
            
    return None, "Download failed."

def detrend(lc):
    time = lc.time.value
    flux = lc.flux.value
    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]
    flat_flux, trend = flatten(time, flux, method='biweight', window_length=0.5, return_trend=True)
    return time, flux, flat_flux, trend

def detect_tls(time, flux):
    tls_model = transitleastsquares(time, flux)
    results = tls_model.power(oversampling_factor=2, duration_grid_step=1.05)
    return results

def calculate_shape_features(time, flux, period, duration, t0):
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0 
    in_transit = np.abs(phase) < (duration / period / 2)
    
    if np.sum(in_transit) < 5:
        return 0.0, 0.0, 0.0
    
    transit_flux = flux[in_transit]
    transit_phase = phase[in_transit]
    
    # Sort by phase to order from ingress to egress
    sort_idx = np.argsort(transit_phase)
    sorted_flux = transit_flux[sort_idx]
    sorted_phase = transit_phase[sort_idx]
    
    # Split ingress and egress based on phase sign (< 0 vs >= 0)
    ingress_mask = sorted_phase < 0
    egress_mask = sorted_phase >= 0
    
    ingress_flux = sorted_flux[ingress_mask]
    egress_flux = sorted_flux[egress_mask]
    
    # Interpolate egress flux onto ingress phase grid for direct symmetry subtraction
    if len(ingress_flux) > 2 and len(egress_flux) > 2:
        egress_interp = np.interp(-sorted_phase[ingress_mask], sorted_phase[egress_mask], egress_flux)
        symmetry = float(np.std(ingress_flux - egress_interp))
    else:
        symmetry = 0.0
        
    ingress_points = np.sum(ingress_mask)
    egress_points = np.sum(egress_mask)
    shape_ratio = abs(ingress_points - egress_points) / max(ingress_points + egress_points, 1)
    
    depth_std = float(np.std(transit_flux))
    return symmetry, shape_ratio, depth_std


def odd_even_test(time, flux, period, duration, t0):
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    in_transit = np.abs(phase) < (duration / period / 2)
    transit_number = np.floor((time - t0) / period)
    
    odd_mask = in_transit & (transit_number % 2 == 1)
    even_mask = in_transit & (transit_number % 2 == 0)
    
    if np.sum(odd_mask) < 3 or np.sum(even_mask) < 3:
        return 0.0, 0.0, 0.0, 1.0
    
    odd_flux = flux[odd_mask]
    even_flux = flux[even_mask]
    
    odd_depth = 1 - np.median(odd_flux)
    even_depth = 1 - np.median(even_flux)
    depth_diff = abs(odd_depth - even_depth)
    
    odd_duration = np.sum(odd_mask) / len(time) * period
    even_duration = np.sum(even_mask) / len(time) * period
    duration_diff = abs(odd_duration - even_duration) / max(duration, 1e-10)
    
    mad_odd = median_abs_deviation(odd_flux)
    mad_even = median_abs_deviation(even_flux)
    
    if mad_even > 1e-10 and mad_odd > 1e-10:
        mad_ratio = mad_odd / mad_even
        mad_ratio = np.clip(mad_ratio, 0.01, 100)
        mad_ratio = abs(np.log(mad_ratio))
    else:
        mad_ratio = 0.0
        
    # Run Welch's t-test to check if odd/even transit depth distributions differ significantly
    try:
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(odd_flux, even_flux, equal_var=False)
        welch_p = float(p_val) if np.isfinite(p_val) else 1.0
    except:
        welch_p = 1.0
        
    return depth_diff, duration_diff, mad_ratio, welch_p


def check_multi_sector(target):
    try:
        search_results = lk.search_lightcurve(target, mission='TESS')
        return len(search_results) if search_results is not None else 0
    except:
        return 0

def to_scalar(value):
    if value is None: return 0.0
    val = float(value[0]) if isinstance(value, (list, np.ndarray)) and len(value) > 0 else float(value)
    return val if np.isfinite(val) else 0.0

def check_transit_physics(period, duration_days, stellar_radius, stellar_mass):
    if not stellar_radius or not stellar_mass or not period or duration_days <= 0:
        return True, 1.0, 1.0, True
    try:
        r_star = float(stellar_radius[0]) if isinstance(stellar_radius, (list, np.ndarray)) else float(stellar_radius)
        m_star = float(stellar_mass[0]) if isinstance(stellar_mass, (list, np.ndarray)) else float(stellar_mass)
        
        # 1. Semi-major axis over stellar radius
        a_over_R = 4.2649 * (m_star**(1/3)) * (period**(2/3)) / r_star
        
        # 2. Maximum circular duration
        max_duration = period / (np.pi * a_over_R)
        duration_ratio = duration_days / max_duration
        duration_ok = duration_ratio <= 1.5
        
        # 3. Density calculation: rho_transit = 1.41 * (a/R)^3 / P^2  (in g/cm^3)
        # Estimated a/R from duration: a/R ~ P / (pi * duration)
        est_a_over_R = period / (np.pi * duration_days)
        rho_transit = 1.41 * (est_a_over_R**3) / (period**2)
        
        # Catalog density: rho_star = 1.41 * M_star / R_star^3 (in g/cm^3)
        rho_star = 1.41 * m_star / (r_star**3)
        
        density_ratio = rho_transit / rho_star
        # Vetting constraint: density ratio should be within [0.02, 50.0]
        density_ok = 0.02 <= density_ratio <= 50.0
        
        return bool(duration_ok), float(duration_ratio), float(density_ratio), bool(density_ok)
    except:
        return True, 1.0, 1.0, True

def check_secondary_eclipse(time, flux, period, duration, t0):
    t_secondary = t0 + 0.5 * period
    phase = ((time - t_secondary) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    
    # Check within the transit duration
    in_secondary = np.abs(phase) < (duration / period / 2)
    out_secondary = ~in_secondary
    
    if np.sum(in_secondary) < 3 or np.sum(out_secondary) < 10:
        return False, 0.0, 0.0
    
    secondary_flux = flux[in_secondary]
    out_flux = flux[out_secondary]
    
    # Estimate depth at phase 0.5 relative to out-of-transit baseline
    secondary_depth = 1.0 - np.nanmedian(secondary_flux)
    noise_std = np.nanstd(out_flux)
    
    # S/N of secondary eclipse dip
    se_snr = secondary_depth / (noise_std / np.sqrt(len(secondary_flux))) if noise_std > 1e-10 else 0.0
    
    # Secondary eclipse is classified if S/N exceeds 3.0 and depth is positive
    has_secondary = se_snr >= 3.0 and secondary_depth > 0.0
    return bool(has_secondary), float(secondary_depth), float(se_snr)


@app.post("/api/analyze")
async def analyze_target(req: AnalysisRequest):
    if not models:
        raise HTTPException(status_code=500, detail="No models loaded. Ensure model files exist in ../model/")

    target_star = req.target_star
    lc, msg = load_lightcurve(target_star)
    if lc is None:
        raise HTTPException(status_code=400, detail=f"Data Fetch Error: {msg}")

    time_arr, raw_flux, flat_flux, trend = detrend(lc)
    num_sectors = check_multi_sector(target_star)
    
    tls_results = detect_tls(time_arr, flat_flux)
    
    period = to_scalar(tls_results.period)
    duration = to_scalar(tls_results.duration)
    depth = to_scalar(tls_results.depth)
    snr = to_scalar(tls_results.SDE)
    t0 = to_scalar(tls_results.T0)
    sde_pass = 1 if snr >= 7.0 else 0
    rp_rs = to_scalar(tls_results.rp_rs if hasattr(tls_results, 'rp_rs') else 0)
    snr_pink = to_scalar(tls_results.snr_pink_per_transit if hasattr(tls_results, 'snr_pink_per_transit') else snr)
    odd_even_mismatch = to_scalar(tls_results.odd_even_mismatch if hasattr(tls_results, 'odd_even_mismatch') else 0)
    
    symmetry, shape_ratio, depth_std = calculate_shape_features(time_arr, flat_flux, period, duration, t0)
    depth_diff, duration_diff, mad_ratio, welch_p = odd_even_test(time_arr, flat_flux, period, duration, t0)
    
    raw_features = [
        period, depth, duration, snr, sde_pass, rp_rs, snr_pink, odd_even_mismatch,
        to_scalar(symmetry), to_scalar(shape_ratio), to_scalar(depth_std),
        to_scalar(depth_diff), to_scalar(duration_diff), to_scalar(mad_ratio),
        num_sectors, len(time_arr)
    ]
    
    features = [f if np.isfinite(f) else 0.0 for f in raw_features]
    features_array = np.array(features).reshape(1, -1)
    
    # Generate predictions from ALL models side-by-side
    model_predictions = []
    
    for name, pkg in models.items():
        if 'scaler' in pkg:
            scaler = pkg['scaler']
            features_model = scaler.transform(features_array)
        else:
            features_model = features_array
            
        model = pkg['model']
        prediction = int(model.predict(features_model)[0])
        probability = float(model.predict_proba(features_model)[0][1])
        
        confidence = "Low"
        if probability > 0.85 or probability < 0.15:
            confidence = "High"
        elif probability > 0.70 or probability < 0.30:
            confidence = "Medium"
            
        model_predictions.append({
            "model_name": name,
            "prediction": prediction,
            "probability": probability,
            "confidence": confidence,
            "result_text": "Planet Candidate Detected" if prediction == 1 else "No Planet Transit Detected"
        })

    # Physics-based vetting check for circular orbit maximum duration and stellar density
    r_star = getattr(tls_results, 'R_star', 1.0)
    m_star = getattr(tls_results, 'M_star', 1.0)
    duration_ok, duration_ratio, density_ratio, density_ok = check_transit_physics(period, duration, r_star, m_star)

    # Physics-based vetting check for secondary eclipse at phase 0.5
    has_secondary, secondary_depth, secondary_snr = check_secondary_eclipse(time_arr, flat_flux, period, duration, t0)

    # Prepare limited plot data (downsample for web transfer)
    downsample = max(1, len(time_arr) // 1000)
    plot_data = {
        "time": time_arr[::downsample].tolist(),
        "raw_flux": raw_flux[::downsample].tolist(),
        "flat_flux": flat_flux[::downsample].tolist(),
        "trend": trend[::downsample].tolist(),
    }
    
    return {
        "target": target_star,
        "predictions": model_predictions,
        "features": dict(zip(feature_names, features)),
        "welch_p": welch_p,
        "duration_ok": duration_ok,
        "duration_ratio": duration_ratio,
        "density_ok": density_ok,
        "density_ratio": density_ratio,
        "has_secondary": has_secondary,
        "secondary_depth": secondary_depth,
        "secondary_snr": secondary_snr,
        "stellar_r": float(r_star[0]) if isinstance(r_star, (list, np.ndarray)) else float(r_star),
        "stellar_m": float(m_star[0]) if isinstance(m_star, (list, np.ndarray)) else float(m_star),
        "plot_data": plot_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)