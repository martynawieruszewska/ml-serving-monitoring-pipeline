from os import mkdir

import pandas as pd
from sklearn.model_selection import train_test_split
from typing_inspection.typing_objects import target

from src.config import (
    RAW_DATA_PATH,
    SIMULATION_DIR,
    PRODUCTION_DATA_PATH,
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

print(X_train.shape)
print(X_test.shape)

print(y_train.mean())
print(y_test.mean())

pipeline = build_pipeline()
print(pipeline)
