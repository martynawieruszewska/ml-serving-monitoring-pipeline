import pandas as pd

from src.config import (
    RAW_DATA_PATH,
)

from src.preprocessing import (
    clean_data,
)


df = pd.read_csv(RAW_DATA_PATH)
df_clean = clean_data(df)

print(df_clean.shape)
print(df_clean.dtypes)
print(df_clean["TotalCharges"].isna().sum())
print(df_clean["OnlineSecurity"].value_counts())