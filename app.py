import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import time

# --- Layout and Styling ---
st.set_page_config(
    page_title="Exoplanet Detection AI",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #1e2532;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4B8BBE;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #fff;
    }
    .metric-label {
        font-size: 14px;
        color: #a0aebc;
    }
</style>
""", unsafe_allow_html=True)

# Try to import necessary astrophysical libraries
try:
    import lightkurve as lk
    from wotan import flatten
    from transitleastsquares import transitleastsquares
    from scipy.stats import median_abs_deviation
    import torch
except ImportError as e:
    st.error(f"Missing required astrophysics libraries: {e}")
    st.info("Please run: `pip install lightkurve wotan transitleastsquares torch xgboost scikit-learn pandas`")
    st.stop()


# --- Title & Sidebar ---
st.title("🪐 Exoplanet Detection AI")
st.markdown("Enter a star system's designation to fetch real TESS (Transiting Exoplanet Survey Satellite) data, extract features, and predict the likelihood of an exoplanet transit using a machine learning ensemble.")

st.sidebar.header("Controls")
st.sidebar.markdown("""
### Example Targets:
- **TOI-270** (Known planet host)
- **TIC 38846515** (Known planet host)
- **Ross 176** (No known transits)
- **TIC 307210830** (Test case)
""")

target_star = st.sidebar.text_input("Enter Target Name", value="TOI-270")
analyze_button = st.sidebar.button("Analyze System 🚀", type="primary")


# --- Pipeline Functions ---
CACHE_DIR = "lc_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

@st.cache_resource
def load_ml_model():
    model_path_v2 = 'model/exoplanet_model_v2.pkl'
    model_path_v1 = 'model/exoplanet_model_v1.pkl'
    if os.path.exists(model_path_v2):
        return joblib.load(model_path_v2), "v2"
    elif os.path.exists(model_path_v1):
        return joblib.load(model_path_v1), "v1"
    return None, None

def load_lightcurve(target):
    safe_name = target.replace(" ", "_").replace("-", "_")
    path = os.path.join(CACHE_DIR, safe_name + ".fits")

    if os.path.exists(path):
        try:
            lc = lk.read(path)
            return lc.remove_nans().remove_outliers().normalize()
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


# --- Main Application Logic ---
if analyze_button and target_star:
    st.markdown("---")
    
    # Check for model first
    model_pkg, model_version = load_ml_model()
    if model_pkg is None:
        st.error("Model not found! Please ensure either `exoplanet_model_v2.pkl` or `exoplanet_model_v1.pkl` is inside the `model/` folder.")
        st.stop()
        
    if model_version == "v1":
        st.warning("⚠️ Using legacy V1 model. Run all cells in `main.ipynb` to generate the improved V2 model with higher accuracy.")
        
    model = model_pkg['model']
    feature_names = model_pkg['feature_names']
    
    progress_bar = st.progress(0, text="Fetching Lightcurve data...")
    
    # 1. Load Data
    with st.spinner(f"Querying MAST archive for {target_star}..."):
        lc, msg = load_lightcurve(target_star)
        if lc is None:
            st.error(f"Data Fetch Error: {msg}")
            st.stop()
            
    progress_bar.progress(20, text="Detrending Lightcurve...")
    
    # 2. Detrend
    with st.spinner("Removing stellar variability and detrending..."):
        time_arr, raw_flux, flat_flux, trend = detrend(lc)
        num_sectors = check_multi_sector(target_star)
        
    progress_bar.progress(40, text="Running Transit Least Squares (TLS)...")
    
    # 3. TLS
    with st.spinner("Searching for periodic transit dips... This may take up to 30 seconds."):
        tls_results = detect_tls(time_arr, flat_flux)
        
    progress_bar.progress(70, text="Extracting Physical and Statistical Features...")
    
    # 4. Feature Extraction
    def to_scalar(value):
        if value is None: return 0.0
        val = float(value[0]) if isinstance(value, (list, np.ndarray)) and len(value) > 0 else float(value)
        return val if np.isfinite(val) else 0.0
        
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
    
    progress_bar.progress(90, text="Evaluating Machine Learning Model...")
    
    # 5. ML Prediction
    features_array = np.array(features).reshape(1, -1)
    
    if model_version == "v2":
        scaler = model_pkg['scaler']
        features_model = scaler.transform(features_array)
    else:
        features_model = features_array
        
    prediction = model.predict(features_model)[0]
    probability = model.predict_proba(features_model)[0][1]
    
    result = "Planet Candidate Detected" if prediction == 1 else "No Planet Transit Detected"
    
    if probability > 0.85 or probability < 0.15:
        confidence = "High"
    elif probability > 0.70 or probability < 0.30:
        confidence = "Medium"
    else:
        confidence = "Low"
        
    progress_bar.progress(100, text="Complete!")
    time.sleep(0.5)
    progress_bar.empty()
    
    # --- UI RESULTS DISPLAY ---
    
    st.header("Prediction Results")
    
    # Hero metric
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {'#4CAF50' if prediction == 1 else '#F44336'};">
            <div class="metric-label">Model Classification</div>
            <div class="metric-value">{result}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #FFC107;">
            <div class="metric-label">Planet Probability</div>
            <div class="metric-value">{probability:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #2196F3;">
            <div class="metric-label">Model Confidence</div>
            <div class="metric-value">{confidence}</div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("---")
    st.subheader("Lightcurve Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Raw & Detrended Flux", "Phase Folded Transit", "TLS Periodogram"])
    
    # Colors suitable for dark mode
    line_col = '#4B8BBE'
    transit_col = 'red'
    
    with tab1:
        fig1, ax1 = plt.subplots(figsize=(12, 4), facecolor='#0e1117')
        ax1.set_facecolor('#0e1117')
        ax1.plot(time_arr, raw_flux, '.', color='gray', alpha=0.3, label='Raw Flux')
        ax1.plot(time_arr, trend, '-', color='red', alpha=0.8, linewidth=1, label='Trend')
        ax1.plot(time_arr, flat_flux - 0.02, '.', color=line_col, alpha=0.5, label='Detrended (-0.02 offset)')
        ax1.set_xlabel('Time (Days)', color='white')
        ax1.set_ylabel('Normalized Flux', color='white')
        ax1.tick_params(colors='white')
        ax1.legend(loc='upper right', facecolor='#1e2532', labelcolor='white')
        st.pyplot(fig1)
        
    with tab2:
        if hasattr(tls_results, 'folded_phase'):
            fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='#0e1117')
            ax2.set_facecolor('#0e1117')
            
            phase = tls_results.folded_phase
            flux_folded = tls_results.folded_y
            
            # Plot binned or raw
            ax2.plot(phase, flux_folded, '.', color='gray', alpha=0.3, zorder=1)
            
            # Overlay TLS model fit
            if hasattr(tls_results, 'model_folded_phase'):
                ax2.plot(tls_results.model_folded_phase, tls_results.model_folded_model, color=transit_col, linewidth=2, zorder=2)
                
            ax2.set_xlim([0.45, 0.55])  # Zoom in on transit
            ax2.set_xlabel('Phase', color='white')
            ax2.set_ylabel('Relative Flux', color='white')
            ax2.set_title(f'Phase Folded (Period: {period:.2f} Days)', color='white')
            ax2.tick_params(colors='white')
            st.pyplot(fig2)
        else:
            st.info("No valid transit signal to fold.")
            
    with tab3:
        if hasattr(tls_results, 'periods'):
            fig3, ax3 = plt.subplots(figsize=(10, 4), facecolor='#0e1117')
            ax3.set_facecolor('#0e1117')
            ax3.plot(tls_results.periods, tls_results.power, color=line_col)
            if period > 0:
                ax3.axvline(period, color='red', linestyle='--', alpha=0.8)
            ax3.set_xlabel('Trial Period (Days)', color='white')
            ax3.set_ylabel('SDE (Signal Detection Efficiency)', color='white')
            ax3.tick_params(colors='white')
            st.pyplot(fig3)
        else:
            st.info("Periodogram unavailable.")

            
    st.markdown("---")
    st.subheader("Physical & Analytical Features")
    
    f_df = pd.DataFrame({
        "Feature Name": feature_names,
        "Extracted Value": features
    })
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.dataframe(f_df.iloc[:8], use_container_width=True, hide_index=True)
    with col_f2:
        st.dataframe(f_df.iloc[8:], use_container_width=True, hide_index=True)
        
    
elif analyze_button and not target_star:
    st.warning("Please enter a target star name.")

