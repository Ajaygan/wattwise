import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

os.makedirs("models", exist_ok=True)

# Seed for reproducibility
np.random.seed(42)
n_samples = 1000

# Synthetic telemetry + weather features matching pipeline expectations
data = {
    "hour": np.random.randint(0, 24, size=n_samples),
    "day_of_week": np.random.randint(0, 7, size=n_samples),
    "is_academic": np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8]),
    "temperature": np.random.uniform(24.0, 38.0, size=n_samples),
    "humidity": np.random.uniform(50.0, 90.0, size=n_samples),
    "lag_15m": np.random.uniform(200.0, 750.0, size=n_samples),
    "lag_30m": np.random.uniform(200.0, 750.0, size=n_samples),
    "lag_60m": np.random.uniform(200.0, 750.0, size=n_samples),
}

df = pd.DataFrame(data)

# Baseline formula for target kVA
target_kva = (
    df["lag_15m"] * 0.5 + 
    df["lag_30m"] * 0.3 + 
    df["temperature"] * 4.0 + 
    df["is_academic"] * 50.0 + 
    np.random.normal(0, 15, n_samples)
)

X = df[["hour", "day_of_week", "is_academic", "temperature", "humidity", "lag_15m", "lag_30m", "lag_60m"]]
y = target_kva

# Train XGBoost Model
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
model.fit(X, y)

# Save artifact
with open("models/demand_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model artifact trained and exported to models/demand_model.pkl")