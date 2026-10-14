from os import mkdir

import pandas as pd
from sklearn.model_selection import train_test_split
from typing_inspection.typing_objects import target
import json
import joblib
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import (
    RAW_DATA_PATH,
    SIMULATION_DIR,
    PRODUCTION_DATA_PATH,
    REPORTS_DIR,
    METRICS_PATH,
    MODELS_DIR,
    MODEL_PATH
)

from src.preprocessing import (
    clean_data,
    build_pipeline
)

from src.config import (
    ID_COLUMN,
    TARGET,
    PRODUCTION_SIZE,
    TEST_SIZE,
    RANDOM_STATE,
)

df = pd.read_csv(RAW_DATA_PATH)

df_model, df_production = train_test_split(df, test_size=PRODUCTION_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE)

SIMULATION_DIR.mkdir(parents=True, exist_ok=True)
df_production.to_csv(PRODUCTION_DATA_PATH, index=False)

X = df_model.drop(columns=[ID_COLUMN, TARGET])
y = df_model[TARGET].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

pipeline = build_pipeline()
pipeline.fit(X_train, y_train)

y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

metrics = {
    "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1": round(f1_score(y_test, y_pred), 4),
}
print(metrics)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
    f.write("\n")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print(f"Metrics saved to {METRICS_PATH}")
print(f"Model saved to {MODEL_PATH}")
