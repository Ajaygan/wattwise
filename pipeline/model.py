import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parents[1] / "models" / "demand_model.pkl"

def load_model():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)

def predict(model, features):
    return float(model.predict(features)[0])