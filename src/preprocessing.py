import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    BINARY_FEATURES,
    RANDOM_STATE,
)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = df.replace({"No internet service": "No", "No phone service": "No"})

    return df

def build_pipeline() -> Pipeline:
    encoder = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("clean", FunctionTransformer(clean_data)),
            ("encode", encoder),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )