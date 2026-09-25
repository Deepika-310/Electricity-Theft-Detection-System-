import json
import os
from functools import lru_cache
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query

from backend.schemas import PredictionResponse, UsageInput
from backend.services import FraudDetector, UnknownMeterError, to_records
from model.config import METRICS_PATH

app = FastAPI(
    title="Electricity Theft Detection API",
    description="Unsupervised anomaly detection on smart-meter readings.",
    version="2.0.0",
)


@lru_cache(maxsize=1)
def _load_detector():
    return FraudDetector()


def get_detector():
    try:
        return _load_detector()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run: python -m database.init_db && python -m model.train",
        )


@app.get("/")
def home():
    return {"message": "Electricity Theft Detection API Running"}


@app.get("/detect")
def detect(
    meter_id: Optional[str] = None,
    limit: Optional[int] = Query(None, ge=1),
    detector: FraudDetector = Depends(get_detector),
):
    """Every reading with its anomaly label (-1 = anomaly, 1 = normal) and score."""
    return to_records(detector.detect(meter_id, limit))


@app.get("/fraud-cases")
def get_fraud_cases(
    meter_id: Optional[str] = None,
    limit: Optional[int] = Query(None, ge=1),
    detector: FraudDetector = Depends(get_detector),
):
    """Only readings the model flags as anomalous (label -1), most suspicious first."""
    return to_records(detector.fraud_cases(meter_id, limit))


@app.get("/meters")
def meters(detector: FraudDetector = Depends(get_detector)):
    """Per-meter reading counts and flag rates."""
    return detector.meter_summary()


@app.post("/predict", response_model=PredictionResponse)
def predict(data: UsageInput, detector: FraudDetector = Depends(get_detector)):
    try:
        return detector.predict_single(data.consumption, data.voltage, data.meter_id, data.timestamp)
    except UnknownMeterError:
        raise HTTPException(status_code=404, detail=f"Unknown meter_id: {data.meter_id}")


@app.get("/metrics")
def metrics():
    """Hold-out evaluation results written by `python -m model.evaluate`."""
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="No metrics yet. Run: python -m model.evaluate")
    with open(METRICS_PATH) as f:
        return json.load(f)
