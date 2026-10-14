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
)

from src.config import (
    TARGET,
    PRODUCTION_SIZE,
    RANDOM_STATE,
)

df = pd.read_csv(RAW_DATA_PATH)

df_model, df_production = train_test_split(df, test_size=PRODUCTION_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE)

SIMULATION_DIR.mkdir(parents=True, exist_ok=True)
df_production.to_csv(PRODUCTION_DATA_PATH, index=False)

print(df_model.shape)
print(df_model[TARGET].value_counts(normalize=True))
print(df_production.shape)
print(df_production[TARGET].value_counts(normalize=True))