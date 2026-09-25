from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

from database.db import get_connection, load_usage
from model.config import MODEL_PATH
from model.features import SEASONAL_DAYS, build_features

HISTORY_ROWS = 24 * (SEASONAL_DAYS + 1)

RESPONSE_COLUMNS = [
    "id",
    "meter_id",
    "timestamp",
    "voltage",
    "consumption",
    "expected_consumption",
    "seasonal_ratio",
    "zscore",
    "anomaly",
    "anomaly_score",
    "is_theft",
    "theft_type",
]


class UnknownMeterError(KeyError):
    pass


class FraudDetector:
    """Scores readings with the saved model. Labels (is_theft) are never used for scoring."""

    def __init__(self, model=None, feature_columns=None, connection_factory=get_connection, model_path=MODEL_PATH):
        if model is None:
            bundle = joblib.load(model_path)
            model = bundle["model"]
            feature_columns = bundle["features"]
            self.model_name = bundle.get("model_name", type(model).__name__)
        else:
            self.model_name = type(model).__name__
        self.model = model
        self.feature_columns = list(feature_columns)
        self._connect = connection_factory

    def _load(self, meter_id=None):
        conn = self._connect()
        try:
            return load_usage(conn, meter_id)
        finally:
            conn.close()

    def _score(self, features):
        x = features[self.feature_columns]
        features["anomaly"] = self.model.predict(x)  # -1 = anomaly, 1 = normal
        features["anomaly_score"] = -self.model.score_samples(x)  # higher = more anomalous
        return features

    def detect(self, meter_id=None, limit=None):
        df = self._load(meter_id)
        if df.empty:
            return df
        scored = self._score(build_features(df))
        scored = scored[RESPONSE_COLUMNS]
        return scored.tail(limit) if limit else scored

    def fraud_cases(self, meter_id=None, limit=None):
        scored = self.detect(meter_id)
        if scored.empty:
            return scored
        flagged = scored[scored["anomaly"] == -1].sort_values("anomaly_score", ascending=False)
        return flagged.head(limit) if limit else flagged

    def meter_summary(self):
        scored = self.detect()
        if scored.empty:
            return []
        summary = scored.groupby("meter_id").agg(
            readings=("id", "count"),
            flagged=("anomaly", lambda s: int((s == -1).sum())),
            mean_consumption=("consumption", "mean"),
        )
        summary["flag_rate"] = summary["flagged"] / summary["readings"]
        return summary.round(4).reset_index().to_dict(orient="records")

    def predict_single(self, consumption, voltage, meter_id=None, timestamp=None):
        """
        With a meter_id the reading is appended to that meter's recent history and
        goes through exactly the same feature code as batch detection. Without one
        there is no history, so the behavioural features take neutral values.
        """
        if meter_id is None:
            row = pd.DataFrame([{"log_seasonal_ratio": 0.0, "zscore": 0.0, "seasonal_ratio": 1.0}])
            context = "no_history"
        else:
            history = self._load(meter_id)
            if history.empty:
                raise UnknownMeterError(meter_id)
            history = history.tail(HISTORY_ROWS)
            ts = pd.Timestamp(timestamp) if timestamp else history["timestamp"].max() + timedelta(hours=1)
            new = pd.DataFrame(
                [
                    {
                        "meter_id": meter_id,
                        "timestamp": ts,
                        "voltage": voltage,
                        "consumption": consumption,
                        "is_theft": 0,
                        "_new": True,
                    }
                ]
            )
            frame = pd.concat([history.assign(_new=False), new], ignore_index=True)
            features = build_features(frame)
            row = features[features["_new"]].iloc[[-1]]
            context = "history" if bool(row["has_history"].iloc[0]) else "no_history"

        x = row[self.feature_columns]
        prediction = int(self.model.predict(x)[0])
        score = float(-self.model.score_samples(x)[0])
        return {
            "consumption": float(consumption),
            "voltage": float(voltage),
            "meter_id": meter_id,
            "prediction": prediction,
            "is_fraud": prediction == -1,
            "anomaly_score": round(score, 6),
            "context": context,
            "features": {c: round(float(row[c].iloc[0]), 4) for c in ("seasonal_ratio", "zscore")},
        }


def to_records(df):
    """DataFrame -> JSON-safe list of dicts (no NaN/NaT/numpy types)."""
    if df is None or len(df) == 0:
        return []
    clean = df.copy()
    clean["timestamp"] = clean["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    clean = clean.replace([np.inf, -np.inf], np.nan).astype(object)
    clean = clean.where(clean.notna(), None)
    return clean.to_dict(orient="records")
