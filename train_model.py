"""
Quick model training script using the Kepler DR25 TCE catalog and NASA Exoplanet Archive.
Produces model/exoplanet_model_v2.pkl with a calibrated RF + XGBoost ensemble.
"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import os

# Feature names used throughout the pipeline
FEATURE_NAMES = [
    'Period', 'Depth', 'Duration', 'SNR', 'SDE_Pass', 'Rp/Rs',
    'SNR_Pink', 'Odd_Even_Mismatch', 'Symmetry', 'Shape_Ratio',
    'Depth_Std', 'Depth_Diff', 'Duration_Diff', 'MAD_Ratio',
    'Num_Sectors', 'Num_Points'
]

print("Loading Kepler DR25 TCE catalog...")
try:
    tce = pd.read_csv("data/q1_q17_dr25_tce_2026.01.27_07.29.56.csv")
    print(f"Loaded TCE catalog: {len(tce)} rows, columns: {list(tce.columns[:20])}")
except Exception as e:
    print(f"Error loading TCE: {e}")
    tce = None

print("\nLoading NASA Exoplanet Archive (confirmed planets)...")
try:
    ps = pd.read_csv("data/PS_2026.02.02_22.30.54.csv")
    print(f"Loaded PS catalog: {len(ps)} rows")
except Exception as e:
    print(f"Error loading PS: {e}")
    ps = None

# ── Build training data from Kepler DR25 TCE ──────────────────────────────────
if tce is not None:
    print("\nBuilding training features from Kepler DR25 TCE catalog...")
    print("Available columns:", list(tce.columns[:40]))
    
    # Map Kepler DR25 columns to our feature space
    col_map = {}
    cols = tce.columns.tolist()
    
    for c in cols:
        c_lo = c.lower()
        if 'period' in c_lo and 'err' not in c_lo: col_map['Period'] = c
        if 'depth' in c_lo and 'err' not in c_lo and 'second' not in c_lo and 'centroid' not in c_lo: col_map.setdefault('Depth', c)
        if 'duration' in c_lo and 'err' not in c_lo: col_map.setdefault('Duration', c)
        if 'snr' in c_lo and 'err' not in c_lo: col_map.setdefault('SNR', c)
        if 'ratio' in c_lo and 'rp' in c_lo: col_map.setdefault('Rp/Rs', c)
    
    print("\nColumn mapping found:", col_map)
    
    rows = []
    for _, row in tce.iterrows():
        try:
            period = float(row.get(col_map.get('Period', 'tce_period'), 1.0))
            depth = float(row.get(col_map.get('Depth', 'tce_depth'), 0.01))
            duration = float(row.get(col_map.get('Duration', 'tce_duration'), 0.1))
            snr = float(row.get(col_map.get('SNR', 'tce_model_snr'), 10.0))
            rp_rs = float(row.get(col_map.get('Rp/Rs', 'tce_ror'), 0.05))
            
            # Get disposition label from av_training_set column
            disp = str(row.get('av_training_set', 'UNK')).strip().upper()
            if disp in ['PC', 'AFP']:
                label = 1
            elif disp in ['FP', 'NTP', 'INV']:
                label = 0
            else:
                # Use SNR heuristic as fallback when disposition is unknown
                label = 1 if snr >= 10 and rp_rs < 0.15 else 0
            
            # Check columns exist and extract
            if not all(np.isfinite([period, depth, duration, snr, rp_rs])):
                continue
                
            sde_pass = 1 if snr >= 7.0 else 0
            snr_pink = snr * 0.85 + np.random.normal(0, 0.5)
            odd_even_mismatch = abs(np.random.normal(0, 0.02))
            symmetry = abs(np.random.normal(0, 0.001))
            shape_ratio = abs(np.random.normal(0.1, 0.05))
            depth_std = depth * 0.05
            depth_diff = abs(np.random.normal(0, depth * 0.1))
            duration_diff = abs(np.random.normal(0.1, 0.05))
            mad_ratio = abs(np.random.normal(0, 0.1))
            num_sectors = 2
            num_points = 2000

            rows.append([period, depth, duration, snr, sde_pass, rp_rs, snr_pink,
                         odd_even_mismatch, symmetry, shape_ratio, depth_std,
                         depth_diff, duration_diff, mad_ratio, num_sectors, num_points, label])
        except Exception:
            continue
    
    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ['Label'])
    print(f"\nBuilt {len(df)} training samples: {df['Label'].value_counts().to_dict()}")
else:
    # Fallback: generate synthetic training data
    print("\nNo TCE data found, generating synthetic training data...")
    np.random.seed(42)
    n = 5000
    # Planet candidates (label=1): short period, high SNR, small Rp/Rs
    n_pos = n // 2
    pos = pd.DataFrame({
        'Period': np.random.uniform(0.5, 15, n_pos),
        'Depth': np.random.uniform(0.001, 0.05, n_pos),
        'Duration': np.random.uniform(0.05, 0.5, n_pos),
        'SNR': np.random.uniform(7, 50, n_pos),
        'SDE_Pass': 1,
        'Rp/Rs': np.random.uniform(0.01, 0.1, n_pos),
        'SNR_Pink': np.random.uniform(6, 45, n_pos),
        'Odd_Even_Mismatch': np.abs(np.random.normal(0, 0.01, n_pos)),
        'Symmetry': np.abs(np.random.normal(0, 0.001, n_pos)),
        'Shape_Ratio': np.abs(np.random.normal(0.05, 0.03, n_pos)),
        'Depth_Std': np.random.uniform(0.0001, 0.003, n_pos),
        'Depth_Diff': np.abs(np.random.normal(0, 0.001, n_pos)),
        'Duration_Diff': np.abs(np.random.normal(0.05, 0.03, n_pos)),
        'MAD_Ratio': np.abs(np.random.normal(0.05, 0.05, n_pos)),
        'Num_Sectors': np.random.randint(1, 5, n_pos),
        'Num_Points': np.random.randint(500, 5000, n_pos),
        'Label': 1
    })
    # False positives (label=0): inconsistent depth, high odd-even mismatch
    n_neg = n - n_pos
    neg = pd.DataFrame({
        'Period': np.random.uniform(0.5, 15, n_neg),
        'Depth': np.random.uniform(0.001, 0.1, n_neg),
        'Duration': np.random.uniform(0.05, 1.0, n_neg),
        'SNR': np.random.uniform(5, 30, n_neg),
        'SDE_Pass': np.random.choice([0, 1], n_neg),
        'Rp/Rs': np.random.uniform(0.1, 0.3, n_neg),
        'SNR_Pink': np.random.uniform(4, 25, n_neg),
        'Odd_Even_Mismatch': np.abs(np.random.normal(0.05, 0.05, n_neg)),
        'Symmetry': np.abs(np.random.normal(0.01, 0.01, n_neg)),
        'Shape_Ratio': np.abs(np.random.normal(0.3, 0.2, n_neg)),
        'Depth_Std': np.random.uniform(0.003, 0.02, n_neg),
        'Depth_Diff': np.abs(np.random.normal(0.01, 0.01, n_neg)),
        'Duration_Diff': np.abs(np.random.normal(0.4, 0.3, n_neg)),
        'MAD_Ratio': np.abs(np.random.normal(0.3, 0.3, n_neg)),
        'Num_Sectors': np.random.randint(1, 5, n_neg),
        'Num_Points': np.random.randint(500, 5000, n_neg),
        'Label': 0
    })
    df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=42)
    print(f"Generated {len(df)} synthetic samples: {df['Label'].value_counts().to_dict()}")

# ── Clean and split ────────────────────────────────────────────────────────────
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[(df[FEATURE_NAMES] < 1e6).all(axis=1)]
print(f"\nAfter cleaning: {len(df)} samples, label distribution: {df['Label'].value_counts().to_dict()}")

X = df[FEATURE_NAMES].values
y = df['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ── Train models ───────────────────────────────────────────────────────────────
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                            class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)

print("Training XGBoost...")
xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                    eval_metric='logloss', random_state=42, scale_pos_weight=(y_train==0).sum()/(y_train==1).sum())
xgb.fit(X_train_s, y_train)

print("Building calibrated voting ensemble...")
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft', weights=[1, 1]
)
ensemble.fit(X_train_s, y_train)

# Calibrate using cross-validation on the training set
calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=5)
calibrated.fit(X_train_s, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
print("\n── Evaluation Results ──")
y_pred = calibrated.predict(X_test_s)
y_prob = calibrated.predict_proba(X_test_s)[:, 1]
print(classification_report(y_test, y_pred, target_names=['False Positive', 'Planet Candidate']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
pkg = {
    'model': calibrated,
    'scaler': scaler,
    'feature_names': FEATURE_NAMES,
    'version': 'v2',
    'description': 'Calibrated RF+XGBoost ensemble trained on Kepler DR25 TCE / synthetic data'
}
joblib.dump(pkg, 'model/exoplanet_model_v2.pkl')
print("\n✅ Model saved to model/exoplanet_model_v2.pkl")
print(f"   Features: {FEATURE_NAMES}")
