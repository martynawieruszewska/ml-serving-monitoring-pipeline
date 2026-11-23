from fastapi import FastAPI
import joblib
import pandas as pd
from src.schemas import CustomerFeatures
from src.config import MODEL_PATH

THRESHOLD = 0.5

model = joblib.load(MODEL_PATH)
app = FastAPI(title="ml-serving-monitoring-pipeline")

@app.post("/predict")
def predict(features: CustomerFeatures):
    df = pd.DataFrame([features.model_dump()])
    proba = model.predict_proba(df)[0, 1]
    return {
        "churn_probability": round(float(proba), 4),
        "churn": bool(proba >= THRESHOLD),
    }