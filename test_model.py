import pickle
with open('model/exoplanet_model_v1.pkl', 'rb') as f:
    model = pickle.load(f)
print(type(model))
if hasattr(model, 'score'): print("has score")
if isinstance(model, dict): print(model.keys())
