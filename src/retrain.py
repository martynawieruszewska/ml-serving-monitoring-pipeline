import joblib
import pandas as pd

from src.config import (
    TRAINING_DATA_PATH,
    PRODUCTION_DATA_PATH,
    ID_COLUMN,
    TARGET,
    MODEL_PATH
)
from src.preprocessing import (
    build_pipeline,
)
from src.drift import (
    check_drift,
)

def load_combined_data() -> pd.DataFrame:
    training = pd.read_csv(TRAINING_DATA_PATH)
    production = pd.read_csv(PRODUCTION_DATA_PATH)

    return pd.concat([training, production], ignore_index=True).drop_duplicates(subset=[ID_COLUMN], keep="last")

def retrain():
    results = check_drift()

    if not any(results.values()):
        print("No drift detected")
        return

    df = load_combined_data()

    X = df.drop(columns=[ID_COLUMN, TARGET])
    y = df[TARGET].map({"Yes": 1, "No": 0})

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    joblib.dump(pipeline, MODEL_PATH)
    df.to_csv(TRAINING_DATA_PATH, index=False)
    print("Model retrained and saved")

if __name__ == "__main__":
    retrain()




