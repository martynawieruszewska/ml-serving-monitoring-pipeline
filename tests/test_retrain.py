import joblib
import numpy as np
import pandas as pd
import pytest

from src import db, drift, retrain
from src.config import CATEGORICAL_FEATURES, ID_COLUMN, NUMERIC_FEATURES, TARGET


CATEGORY_VALUES = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "Two year", "One year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}


def make_customers(rng, ids):
    n = len(ids)
    tenure = rng.integers(0, 73, size=n)
    monthly = rng.uniform(20, 120, size=n)
    df = pd.DataFrame({ID_COLUMN: ids})
    for feature in CATEGORICAL_FEATURES:
        df[feature] = rng.choice(CATEGORY_VALUES[feature], size=n)
    df["SeniorCitizen"] = rng.integers(0, 2, size=n)
    df["tenure"] = tenure
    df["MonthlyCharges"] = monthly
    df["TotalCharges"] = tenure * monthly
    df[TARGET] = rng.choice(["Yes", "No"], size=n)
    return df


@pytest.fixture
def paths(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    training_path = tmp_path / "training_data.csv"
    production_path = tmp_path / "production_traffic.csv"
    model_path = tmp_path / "model.pkl"

    rng = np.random.default_rng(42)
    training = make_customers(rng, [f"T{i}" for i in range(200)])
    # 10 production customers overlap with training to exercise deduplication
    production = make_customers(rng, [f"T{i}" for i in range(10)] + [f"P{i}" for i in range(50)])
    training.to_csv(training_path, index=False)
    production.to_csv(production_path, index=False)

    monkeypatch.setattr(db, "DB_PATH", db_path)
    monkeypatch.setattr(drift, "DB_PATH", db_path)
    monkeypatch.setattr(drift, "TRAINING_DATA_PATH", training_path)
    monkeypatch.setattr(retrain, "TRAINING_DATA_PATH", training_path)
    monkeypatch.setattr(retrain, "PRODUCTION_DATA_PATH", production_path)
    monkeypatch.setattr(retrain, "MODEL_PATH", model_path)

    db.init_db()
    return {
        "training": training_path,
        "production": production_path,
        "model": model_path,
        "training_df": training,
        "production_df": production,
    }


def log_traffic(df, monthly_factor=1.0):
    for row in df.itertuples(index=False):
        db.log_prediction(
            {
                "tenure": int(row.tenure),
                "MonthlyCharges": float(row.MonthlyCharges) * monthly_factor,
                "TotalCharges": float(row.TotalCharges),
            },
            0.5,
            False,
        )


def expected_ids(paths):
    return set(paths["training_df"][ID_COLUMN]) | set(paths["production_df"][ID_COLUMN])


def test_no_drift_leaves_model_and_training_data_untouched(paths):
    # Checks that with traffic identical to training data, retrain() neither creates the model nor rewrites the training CSV.
    log_traffic(paths["training_df"])
    training_before = paths["training"].read_bytes()

    retrain.retrain()

    assert not paths["model"].exists()
    assert paths["training"].read_bytes() == training_before


def test_drift_saves_model_and_overwrites_training_data_without_duplicates(paths):
    # Checks that shifted MonthlyCharges triggers retraining, which saves a working model and writes the deduplicated union to the training CSV.
    log_traffic(paths["training_df"], monthly_factor=2.0)

    retrain.retrain()

    assert paths["model"].exists()
    model = joblib.load(paths["model"])
    features = paths["production_df"].drop(columns=[ID_COLUMN, TARGET])
    assert model.predict_proba(features).shape == (len(features), 2)

    saved = pd.read_csv(paths["training"])
    assert not saved[ID_COLUMN].duplicated().any()
    assert set(saved[ID_COLUMN]) == expected_ids(paths)
    assert len(saved) == len(expected_ids(paths))


def test_load_combined_data_after_retrain_does_not_duplicate_rows(paths):
    # Checks that merging production data again after a retrain (when it is already in the training CSV) adds no duplicate rows.
    log_traffic(paths["training_df"], monthly_factor=2.0)
    retrain.retrain()
    saved = pd.read_csv(paths["training"])

    combined = retrain.load_combined_data()

    assert not combined[ID_COLUMN].duplicated().any()
    assert len(combined) == len(saved) == len(expected_ids(paths))
    assert set(NUMERIC_FEATURES) <= set(combined.columns)
