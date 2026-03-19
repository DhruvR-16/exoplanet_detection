# 🪐 Exoplanet Detection Pipeline & Web Dashboard

An end-to-end Machine Learning pipeline and interactive web application designed to automatically detect exoplanet transit signatures from raw TESS (Transiting Exoplanet Survey Satellite) lightcurve data. 

The system leverages a hybrid physical-statistical approach combined with an Ensemble Machine Learning model (Random Forest + XGBoost) to predict whether a given star system hosts an exoplanet.

---

## ✨ Key Features

*   **Live TESS Data Fetching:** Automatically queries the MAST archive using `lightkurve` to download SPOC or custom lightcurves for any TESS target.
*   **Stellar Detrending:** Removes stellar variability, flares, and long-term instrumental trends using the `wotan` biweight method.
*   **Transit Least Squares (TLS):** Uses optimized TLS to detect periodic, U-shaped planetary transit dips, even those buried in noise.
*   **Advanced Feature Extraction:** Computes 16 critical physical and statistical features (e.g., Transit Depth, Signal-to-Noise Ratio, Ingress/Egress shape ratio, Odd-Even mismatch).
*   **Ensemble ML Model:** A calibrated Random Forest and XGBoost voting classifier that provides a probability score and confidence rating.
*   **Interactive Web Dashboard:** A sleek, dark-mode Streamlit UI that visualizes raw data, phase-folded transits, and periodograms in real-time.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed. The project relies on specific astrophysics and machine learning libraries.

```bash
# Install the required dependencies
pip install streamlit matplotlib lightkurve wotan transitleastsquares torch xgboost scikit-learn pandas joblib
```

### 2. Running the Web UI
The easiest way to interact with the project is through the Streamlit Web Dashboard. Let the model do the heavy lifting for you!

```bash
# Start the Streamlit server
streamlit run app.py
```
*The dashboard will automatically open in your browser at `http://localhost:8501`.*

**Try these example targets in the UI:**
*   `TOI-270` (Known multi-planet system)
*   `TIC 38846515` (Known planet host)
*   `Ross 176` (Star with no known transits)

### 3. Training the Model (Optional)
If you want to re-train the model from scratch, gather new training data, or tweak the XGBoost hyperparameters, you can run the provided Jupyter Notebook.

```bash
# Open the main pipeline notebook
jupyter notebook main.ipynb
```
*Note: Running the full notebook can take 30+ minutes as it downloads and processes lightcurves for hundreds of control and planet stars.*

---

## 📁 Project Structure

```text
expoplanet_detection/
├── app.py                     # Streamlit web application & prediction pipeline
├── main.ipynb                 # Full training pipeline, data collection, and evaluation
├── model/                     
│   ├── exoplanet_model_v1.pkl # Legacy Random Forest model
│   └── exoplanet_model_v2.pkl # Modern Calibrated Ensemble (RF + XGBoost)
├── docs/                      
│   ├── pipeline_improvements.md # Detailed breakdown of recent ML updates
│   ├── web_ui_guide.md          # User manual for the dashboard
│   └── future_roadmap.md        # Architectural/Astrophysical plans for v3.0
├── lc_cache/                  # Local cache of downloaded TESS .fits files
├── data/                      # Raw CSV datasets and metadata logs
└── predictions.csv            # Automatically updated log of all UI predictions
```

---

## 🧠 How the Pipeline Works

1.  **Ingestion:** The user provides a Target ID (e.g., `TIC 307210830`). The pipeline securely downloads the flux data.
2.  **Cleaning:** `wotan` applies a rolling biweight filter to flatten the lightcurve, isolating quick transit dips.
3.  **Search:** `transitleastsquares` scans thousands of trial periods and durations to find the strongest periodic signal.
4.  **Extraction:** Features such as transit symmetry, odd-even depth differences, and pink-noise-aware SNRs are mathematically quantified.
5.  **Scaling & Prediction:** Features are normalized using `StandardScaler` and passed into the calibrated Ensemble Model.
6.  **Results:** The app outputs the prediction, probability percentage, and generates diagnostic plots for human verification.

---

## 🔮 Future Development
We have an extensive roadmap planned to transition this from a strong prototype into a professional-grade astronomical tool. Planned features include **Multi-Sector Stitching**, **Gaia DR3 Astrometric filtering**, and **1D-CNN deep learning** models. 

