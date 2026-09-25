import os
import sqlite3

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("ETD_DB_PATH", os.path.join(BASE_DIR, "database", "electricity.db"))

USAGE_COLUMNS = "id, meter_id, timestamp, voltage, consumption, is_theft, theft_type"


def get_connection(db_path=None):
    return sqlite3.connect(db_path or DB_PATH)


def load_usage(conn, meter_id=None):
    """Load readings ordered by meter and time, optionally for a single meter."""
    query = f"SELECT {USAGE_COLUMNS} FROM electricity_usage"
    params = ()
    if meter_id is not None:
        query += " WHERE meter_id = ?"
        params = (meter_id,)
    query += " ORDER BY meter_id, timestamp"

    df = pd.read_sql_query(query, conn, params=params)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
