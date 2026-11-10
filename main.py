import os
import pickle
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="CIT Smart Campus Energy Management API",
    description="Backend service housing the XGBoost model, Peak Demand Shield logic, and Operator Overrides.",
    version="2.0.0"
)

# CORS enabled for React.js frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML Model
MODEL_PATH = "models/demand_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None

# Grid Constants
SANCTIONED_LIMIT_KVA = 800.0
SHIELD_THRESHOLD_KVA = 680.0  # 85% limit

# In-Memory Operator Override State
operator_state = {
    "manual_override": False,
    "override_status": "NORMAL",
    "operator_notes": ""
}

# --- Schemas ---

class TelemetryInput(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    is_academic: int = Field(..., ge=0, le=1, description="1 if academic day, 0 otherwise")
    temperature: float = Field(30.0, description="Temperature in °C")
    humidity: float = Field(70.0, description="Humidity in %")
    lag_15m: float = Field(..., ge=0.0, le=1000.0, description="Demand 15m ago (kVA)")
    lag_30m: float = Field(..., ge=0.0, le=1000.0, description="Demand 30m ago (kVA)")
    lag_60m: float = Field(..., ge=0.0, le=1000.0, description="Demand 60m ago (kVA)")

class MitigationAction(BaseModel):
    priority: int
    load_name: str
    action: str
    estimated_relief_kva: float
    status: str

class PredictionResponse(BaseModel):
    predicted_demand_kva: float
    sanctioned_limit_kva: float
    capacity_utilization_pct: float
    shield_threshold_kva: float
    shield_active: bool
    simulation_mode: bool
    recommended_mitigations: List[MitigationAction]

class OperatorOverrideInput(BaseModel):
    action: str = Field(..., description="'FORCE_SHIELD', 'SUPPRESS_SHIELD', or 'RESET'")
    notes: Optional[str] = None

# --- Routes ---

@app.get("/")
def get_status():
    return {
        "status": "online",
        "system": "CIT Energy Backend",
        "model_loaded": model is not None,
        "sanctioned_limit_kva": SANCTIONED_LIMIT_KVA,
        "shield_threshold_kva": SHIELD_THRESHOLD_KVA,
        "operator_state": operator_state
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_demand(data: TelemetryInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found. Run train_dummy_model.py first.")

    input_df = pd.DataFrame([{
        "hour": data.hour,
        "day_of_week": data.day_of_week,
        "is_academic": data.is_academic,
        "temperature": data.temperature,
        "humidity": data.humidity,
        "lag_15m": data.lag_15m,
        "lag_30m": data.lag_30m,
        "lag_60m": data.lag_60m
    }])

    predicted_kva = round(float(model.predict(input_df)[0]), 2)
    utilization_pct = round((predicted_kva / SANCTIONED_LIMIT_KVA) * 100, 2)
    calculated_trigger = predicted_kva >= SHIELD_THRESHOLD_KVA

    if operator_state["manual_override"]:
        if operator_state["override_status"] == "FORCED_SHIELD":
            shield_active = True
        elif operator_state["override_status"] == "SUPPRESSED":
            shield_active = False
        else:
            shield_active = calculated_trigger
    else:
        shield_active = calculated_trigger

    mitigations = []
    if shield_active:
        mitigations = [
            MitigationAction(
                priority=1,
                load_name="Hostel Common Water Heaters & Pumps",
                action="Stagger operation & turn off non-critical blocks",
                estimated_relief_kva=45.0,
                status="SIMULATED_PENDING"
            ),
            MitigationAction(
                priority=2,
                load_name="Academic Block B HVAC System",
                action="Set central chillers to Eco-Thermostat Mode (+2°C)",
                estimated_relief_kva=35.0,
                status="SIMULATED_PENDING"
            ),
            MitigationAction(
                priority=3,
                load_name="Campus Outdoor & Decorative Lighting",
                action="Dim lighting array to 30% capacity",
                estimated_relief_kva=15.0,
                status="SIMULATED_PENDING"
            )
        ]

    return PredictionResponse(
        predicted_demand_kva=predicted_kva,
        sanctioned_limit_kva=SANCTIONED_LIMIT_KVA,
        capacity_utilization_pct=utilization_pct,
        shield_threshold_kva=SHIELD_THRESHOLD_KVA,
        shield_active=shield_active,
        simulation_mode=True,
        recommended_mitigations=mitigations
    )

@app.post("/operator/override")
def set_operator_override(override: OperatorOverrideInput):
    if override.action == "FORCE_SHIELD":
        operator_state["manual_override"] = True
        operator_state["override_status"] = "FORCED_SHIELD"
    elif override.action == "SUPPRESS_SHIELD":
        operator_state["manual_override"] = True
        operator_state["override_status"] = "SUPPRESSED"
    elif override.action == "RESET":
        operator_state["manual_override"] = False
        operator_state["override_status"] = "NORMAL"
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'FORCE_SHIELD', 'SUPPRESS_SHIELD', or 'RESET'.")

    operator_state["operator_notes"] = override.notes or ""
    return {"message": "Operator state updated", "operator_state": operator_state}