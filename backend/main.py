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
        return 0, 0, 0
    
    transit_flux = flux[in_transit]
    transit_phase = phase[in_transit]
    
    mid_idx = len(transit_flux) // 2
    first_half = transit_flux[:mid_idx]
    second_half = transit_flux[mid_idx:]
    symmetry = np.std(first_half - second_half[::-1][:len(first_half)]) if len(first_half) > 0 else 0
    
    sorted_indices = np.argsort(transit_flux)
    ingress_points = np.sum(transit_phase < 0)
    egress_points = np.sum(transit_phase > 0)
    shape_ratio = abs(ingress_points - egress_points) / max(ingress_points + egress_points, 1)
    
    depth_std = np.std(transit_flux)
    return symmetry, shape_ratio, depth_std

def odd_even_test(time, flux, period, duration, t0):
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    in_transit = np.abs(phase) < (duration / period / 2)
    transit_number = np.floor((time - t0) / period)
    
    odd_mask = in_transit & (transit_number % 2 == 1)
    even_mask = in_transit & (transit_number % 2 == 0)
    
    if np.sum(odd_mask) < 3 or np.sum(even_mask) < 3:
        return 0, 0, 0
    
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
        mad_ratio = 0
    return depth_diff, duration_diff, mad_ratio

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
    depth_diff, duration_diff, mad_ratio = odd_even_test(time_arr, flat_flux, period, duration, t0)
    
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
        "plot_data": plot_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)