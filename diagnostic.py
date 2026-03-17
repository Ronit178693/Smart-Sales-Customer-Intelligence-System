import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

packages = ["streamlit", "pandas", "numpy", "joblib", "plotly", "sklearn", "scipy"]
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg} is installed")
    except ImportError as e:
        print(f"❌ {pkg} is NOT installed: {e}")

# Try loading models as in app.py
print("\n--- Testing Model Loading ---")
import joblib
paths = {
    'scaler': 'pkl/scaler.pkl',
    'pca': 'pkl/pca.pkl',
    'kmeans': 'pkl/kmeans_model.pkl',
    'classification': 'pkl/Classification_Model.pkl',
    'regression': 'pkl/Regression_Model.pkl',
    'encoder': 'pkl/gender_encoder.pkl',
}

for name, path in paths.items():
    if os.path.exists(path):
        try:
            joblib.load(path)
            print(f"✅ {name} loaded successfully from {path}")
        except Exception as e:
            print(f"❌ Failed to load {name} from {path}: {e}")
    else:
        print(f"❓ {name} file NOT FOUND at {path}")

# Check dataset
print("\n--- Testing Dataset ---")
import pandas as pd
path = "Dataset/customer_data.csv"
if os.path.exists(path):
    try:
        df = pd.read_csv(path)
        print(f"✅ Dataset loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
else:
    print(f"❓ Dataset NOT FOUND at {path}")
