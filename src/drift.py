import json
import sqlite3

import pandas as pd
from scipy.stats import ks_2samp

from src.config import DB_PATH, DRIFT_P_VALUE_THRESHOLD, NUMERIC_FEATURES, TRAINING_DATA_PATH
from src.preprocessing import (
    clean_data,
)


def load_logged_inputs() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT
                input_json
            FROM predictions
        """).fetchall()

    records = [json.loads(row[0]) for row in rows]
    return pd.DataFrame(records)


def check_drift() -> dict:
    reference = clean_data(pd.read_csv(TRAINING_DATA_PATH))
    logged = load_logged_inputs()

    results = {}

    for feature in NUMERIC_FEATURES:
        _, p_value = ks_2samp(reference[feature], logged[feature])
        drift = bool(p_value < DRIFT_P_VALUE_THRESHOLD)
        results[feature] = drift

        if drift:
            print(f"drift detected in feature {feature} (p={p_value:.4f})")
        else:
            print(f"no drift in feature {feature} (p={p_value:.4f})")

    return results
