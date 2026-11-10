from pathlib import Path
import csv
import json
import math
import random
from datetime import datetime, timedelta

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PIPELINE = ROOT / "pipeline"

DATA.mkdir(exist_ok=True)
PIPELINE.mkdir(exist_ok=True)

random.seed(42)

# Generate seven days of 15-minute telemetry.
start = datetime(2026, 11, 2, 8, 0)
rows = []
demand = []

for i in range(673):
    ts = start + timedelta(minutes=15 * i)
    hour = ts.hour
    weekday = ts.weekday()

    academic = int(
        weekday < 5 and 7 <= hour < 20
    )

    value = (
        275
        + 48 * math.sin((hour - 8) / 24 * 2 * math.pi)
        + (38 if 9 <= hour <= 17 and academic else 0)
        + random.uniform(-9, 9)
    )

    value = max(220, min(420, value))
    demand.append(value)

for i, value in enumerate(demand):
    ts = start + timedelta(minutes=15 * i)
    hour = ts.hour
    weekday = ts.weekday()

    temperature = max(
        24, min(38, 30 + 4 * math.sin(hour / 24 * 2 * math.pi)
                + random.uniform(-0.5, 0.5))
    )

    humidity = max(
        50, min(90, 75 + random.uniform(-5, 5))
    )

    rows.append({
        "timestamp": ts.isoformat(),
        "demand_kva": round(value, 2),
        "hour": hour,
        "day_of_week": weekday,
        "is_academic": int(weekday < 5 and 7 <= hour < 20),
        "temperature": round(temperature, 2),
        "humidity": round(humidity, 2),
        "lag_15m": round(demand[max(0, i-1)], 2),
        "lag_30m": round(demand[max(0, i-2)], 2),
        "lag_60m": round(demand[max(0, i-4)], 2),
    })

with (DATA / "telemetry.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Dashboard data fixture.
dashboard = {
    "meta": {
        "site": "CIT-SS-01",
        "mode": "SYNTHETIC_DEMO",
        "interval_minutes": 15
    },
    "overview": {
        "current_demand_kva": 278.7,
        "sanctioned_demand_kva": 800,
        "threshold_percent": 85,
        "threshold_kva": 680,
        "headroom_kva": 401.3,
        "trend": "STABLE"
    },
    "energy_monitoring": {
        "voltage": {
            "L1": 415.0, "L2": 415.4, "L3": 414.9
        },
        "current": {
            "L1": 389.8, "L2": 382.0, "L3": 385.9
        },
        "active_power_kw": 267.8,
        "apparent_power_kva": 278.7,
        "reactive_power_kvar": 71.1,
        "power_factor": 0.961
    },
    "forecasting": {
        "points": [
            {"minutes": 0, "demand_kva": 278.7},
            {"minutes": 15, "demand_kva": 309.9},
            {"minutes": 30, "demand_kva": 335.2},
            {"minutes": 60, "demand_kva": 386.4}
        ],
        "risk": "LOW"
    },
    "peak_shield": {
        "threshold_kva": 680,
        "state": "SAFE",
        "tiers": [1, 2, 3, 4]
    },
    "mitigation": {
        "active": 2,
        "pending_approval": 2,
        "actions": []
    },
    "weather": {
        "temperature_c": 28.5,
        "humidity_percent": 75,
        "wind_speed_kmh": 10.4
    },
    "historical": {
        "daily_peaks_kva": [412, 438, 421, 456, 389, 402, 387],
        "threshold_crossings": 0
    },
    "settings": {
        "telemetry_interval": 15,
        "device_control_enabled": False,
        "mode": "SIMULATION_ONLY"
    }
}

(DATA / "dashboard_state.json").write_text(
    json.dumps(dashboard, indent=2),
    encoding="utf-8"
)

# Pipeline feature construction.
(PIPELINE / "__init__.py").write_text("")

(PIPELINE / "features.py").write_text('''
import pandas as pd

FEATURE_COLUMNS = [
    "hour", "day_of_week", "is_academic",
    "temperature", "humidity",
    "lag_15m", "lag_30m", "lag_60m"
]

def build_features(row):
    values = {
        name: row[name]
        for name in FEATURE_COLUMNS
    }

    assert 0 <= values["hour"] <= 23
    assert 0 <= values["day_of_week"] <= 6
    assert values["is_academic"] in (0, 1)
    assert 24 <= values["temperature"] <= 38
    assert 50 <= values["humidity"] <= 90

    for name in ("lag_15m", "lag_30m", "lag_60m"):
        assert 200 <= values[name] <= 750

    return pd.DataFrame(
        [[values[name] for name in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )
'''.strip())

(PIPELINE / "model.py").write_text('''
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parents[1] / "models" / "demand_model.pkl"

def load_model():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)

def predict(model, features):
    return float(model.predict(features)[0])
'''.strip())

(ROOT / "requirements.txt").write_text(
    "pandas\\nnumpy\\nxgboost\\nrequests\\npython-dotenv\\n"
)

(ROOT / "README.md").write_text("""
# CIT Campus Energy Project

All bundled telemetry and dashboard values are synthetic demo data.

Run:
    pip install -r requirements.txt

The existing XGBoost model must be placed at:
    models/demand_model.pkl

Data:
    data/telemetry.csv
    data/dashboard_state.json

No physical equipment is controlled.
""".strip())

print("CIT Energy Project created successfully!")
print(f"Telemetry rows: {len(rows)}")
print(f"Project location: {ROOT}")