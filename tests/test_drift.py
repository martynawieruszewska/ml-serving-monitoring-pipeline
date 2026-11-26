import numpy as np
import pandas as pd
import pytest

from src import db, drift


@pytest.fixture
def rng(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    training_path = tmp_path / "training_data.csv"

    rng = np.random.default_rng(42)
    tenure = rng.integers(0, 73, size=1000)
    monthly = rng.uniform(20, 120, size=1000)
    reference = pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly,
    })
    reference.to_csv(training_path, index=False)

    monkeypatch.setattr(db, "DB_PATH", db_path)
    monkeypatch.setattr(drift, "DB_PATH", db_path)
    monkeypatch.setattr(drift, "TRAINING_DATA_PATH", training_path)

    db.init_db()
    return rng


def log_traffic(tenure, monthly):
    for t, m in zip(tenure, monthly):
        db.log_prediction(
            {"tenure": int(t), "MonthlyCharges": float(m), "TotalCharges": float(t * m)},
            0.5,
            False,
        )


def test_no_drift_when_traffic_matches_training(rng):
    tenure = rng.integers(0, 73, size=500)
    monthly = rng.uniform(20, 120, size=500)
    log_traffic(tenure, monthly)

    results = drift.check_drift()

    assert not results["tenure"]
    assert not results["MonthlyCharges"]
    assert not results["TotalCharges"]


def test_drift_detected_when_monthly_charges_shift(rng):
    tenure = rng.integers(0, 73, size=500)
    monthly = rng.uniform(20, 120, size=500) * 2
    log_traffic(tenure, monthly)

    results = drift.check_drift()

    assert results["MonthlyCharges"]