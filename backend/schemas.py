from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UsageInput(BaseModel):
    consumption: float = Field(..., ge=0, le=10_000, description="Energy consumed in the reading interval (kWh)")
    voltage: float = Field(..., gt=0, le=1_000, description="Supply voltage (V)")
    meter_id: Optional[str] = Field(
        None, description="If given, the reading is scored against that meter's own history"
    )
    timestamp: Optional[datetime] = Field(
        None, description="When the reading was taken; defaults to one hour after the meter's last reading"
    )


class PredictionResponse(BaseModel):
    consumption: float
    voltage: float
    meter_id: Optional[str] = None
    prediction: int = Field(..., description="-1 = anomaly (theft suspected), 1 = normal")
    is_fraud: bool
    anomaly_score: float = Field(..., description="Higher means more anomalous")
    context: str = Field(..., description="'history' if scored against the meter's history, else 'no_history'")
    features: dict
