import joblib
try:
    model = joblib.load('model/exoplanet_model_v1.pkl')
    print("Model loaded successfully with joblib!")
    if hasattr(model, 'feature_names_in_'):
        print("Features:", model.feature_names_in_)
    if hasattr(model, 'score'): print("has score function")
except Exception as e:
    print(f"Error: {e}")
