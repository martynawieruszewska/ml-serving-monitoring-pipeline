from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"

REPORTS_DIR = ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"

SIMULATION_DIR = ROOT / "data" / "simulation"
PRODUCTION_DATA_PATH = SIMULATION_DIR / "production_traffic.csv"
TRAINING_DATA_PATH = ROOT / "data" / "training_data.csv"

TARGET = "Churn"
ID_COLUMN = "customerID"
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES = ["SeniorCitizen"]
RANDOM_STATE = 42
PRODUCTION_SIZE = 0.2
TEST_SIZE = 0.25

DB_PATH = ROOT / "data" / "predictions.db"

THRESHOLD = 0.5
DRIFT_P_VALUE_THRESHOLD = 0.05
