import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

from database.db import get_connection

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
START = datetime(2026, 8, 1)

# name: (probability, consumption multiplier range, duration range in hours)
THEFT_TYPES = {
    "meter_bypass": (0.40, (0.25, 0.55), (24, 72)),
    "abnormal_surge": (0.30, (2.5, 4.0), (2, 6)),
    "meter_stall": (0.30, (0.0, 0.05), (12, 48)),
}


def _daily_profile(hour_of_day, kind):
    h = hour_of_day.astype(float)
    if kind == "commercial":
        return 0.25 + 1.0 * np.exp(-(((h - 13.5) / 4.0) ** 2))
    morning = 0.5 * np.exp(-(((h - 7.5) / 1.8) ** 2))
    evening = 1.0 * np.exp(-(((h - 20.0) / 2.3) ** 2))
    return 0.45 + morning + evening


def generate_readings(n_meters=10, days=30, seed=42, theft_meter_fraction=0.4):
    """
    Simulate hourly smart-meter data with a daily/weekly load shape, then inject
    labelled theft episodes into a subset of meters.

    Theft is mostly a *drop* in recorded consumption (bypass, stalled meter), with
    occasional surges, so it is not just "big number = fraud".
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(START, periods=days * 24, freq="h")
    hour_of_day = timestamps.hour.to_numpy()
    is_weekend = timestamps.dayofweek.to_numpy() >= 5
    n = len(timestamps)

    n_theft_meters = max(1, round(n_meters * theft_meter_fraction))
    theft_meters = set(rng.choice(n_meters, size=n_theft_meters, replace=False).tolist())

    type_names = list(THEFT_TYPES)
    type_probs = [THEFT_TYPES[t][0] for t in type_names]

    frames = []
    for m in range(n_meters):
        kind = "commercial" if m % 4 == 3 else "residential"
        base = rng.uniform(3.0, 6.0) if kind == "commercial" else rng.uniform(0.8, 1.6)
        weekend_factor = np.where(is_weekend, 0.5 if kind == "commercial" else 1.15, 1.0)

        expected = base * _daily_profile(hour_of_day, kind) * weekend_factor
        consumption = expected * rng.lognormal(0.0, 0.10, n)
        voltage = 230.0 + rng.normal(0.0, 2.5, n) - 1.5 * (consumption / base - 1.0)

        is_theft = np.zeros(n, dtype=int)
        theft_type = np.full(n, None, dtype=object)

        if m in theft_meters:
            # One episode per equal slice of the timeline (after a clean 3-day
            # warm-up) so theft appears in both the training and hold-out periods.
            n_episodes = int(rng.integers(3, 5))
            edges = np.linspace(72, n, n_episodes + 1).astype(int)
            for e in range(n_episodes):
                name = type_names[int(rng.choice(len(type_names), p=type_probs))]
                _, (f_lo, f_hi), (d_lo, d_hi) = THEFT_TYPES[name]
                duration = int(rng.integers(d_lo, d_hi + 1))
                start = int(rng.integers(edges[e], edges[e + 1] - duration))
                consumption[start : start + duration] *= rng.uniform(f_lo, f_hi)
                is_theft[start : start + duration] = 1
                theft_type[start : start + duration] = name

        frames.append(
            pd.DataFrame(
                {
                    "meter_id": f"M{m:02d}",
                    "timestamp": timestamps,
                    "voltage": voltage.round(2),
                    "consumption": consumption.round(4),
                    "is_theft": is_theft,
                    "theft_type": theft_type,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def initialize_database(db_path=None, seed=42, n_meters=10, days=30):
    df = generate_readings(n_meters=n_meters, days=days, seed=seed)

    out = df.copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out = out.astype(object).where(out.notna(), None)
    rows = list(out.itertuples(index=False, name=None))

    conn = get_connection(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS electricity_usage")
        with open(SCHEMA_PATH, "r") as f:
            conn.executescript(f.read())
        conn.executemany(
            """
            INSERT INTO electricity_usage
                (meter_id, timestamp, voltage, consumption, is_theft, theft_type)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows],
        )
        conn.commit()
    finally:
        conn.close()

    print(
        f"Database initialized: {len(df)} readings, {n_meters} meters, {days} days, "
        f"{df['is_theft'].mean():.1%} theft rows (seed={seed})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the synthetic usage database.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--meters", type=int, default=10)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    initialize_database(seed=args.seed, n_meters=args.meters, days=args.days)
