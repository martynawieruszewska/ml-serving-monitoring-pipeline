import json
import sqlite3
from datetime import datetime, timezone

from src.config import DB_PATH


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT not null,
                input_json TEXT not null,
                churn_probability REAL not null,
                churn INTEGER not null
            )
        """)


def log_prediction(input_data: dict, churn_probability: float, churn: bool) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO predictions (timestamp, input_json, churn_probability, churn) VALUES (?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                json.dumps(input_data),
                churn_probability,
                churn,
            ),
        )
