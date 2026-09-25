CREATE TABLE IF NOT EXISTS electricity_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meter_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    voltage REAL,
    consumption REAL,
    is_theft INTEGER NOT NULL DEFAULT 0,
    theft_type TEXT
);

CREATE INDEX IF NOT EXISTS idx_usage_meter_time ON electricity_usage (meter_id, timestamp);
