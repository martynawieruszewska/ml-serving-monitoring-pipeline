from fastapi import FastAPI
import joblib
import pandas as pd
from src.schemas import CustomerFeatures
from src.config import (
    MODEL_PATH,
    THRESHOLD
)
from src.db import (
    init_db,
    log_prediction
)

model = joblib.load(MODEL_PATH)
app = FastAPI(title="ml-serving-monitoring-pipeline")
init_db()

@app.post("/predict")
def predict(features: CustomerFeatures):
    data = features.model_dump()
    df = pd.DataFrame([data])
    proba = float(model.predict_proba(df)[0, 1])
    churn = proba >= THRESHOLD
    log_prediction(data, proba, churn)
    return {"churn_probability": round(proba, 4), "churn": churn}