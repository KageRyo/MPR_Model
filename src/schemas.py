from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ModelType = Literal["direct_wqi5", "mpr", "rf", "xgboost", "lightgbm", "svm", "lr"]


class PredictionRequest(BaseModel):
    DO: float = Field(..., description="Dissolved oxygen saturation percentage.")
    BOD: float = Field(..., description="Biochemical oxygen demand in mg/L.")
    NH3N: float = Field(..., description="Ammonia nitrogen in mg/L.")
    EC: float = Field(..., description="Electrical conductivity in umho/cm.")
    SS: float = Field(..., description="Suspended solids in mg/L.")
    model_type: ModelType = "direct_wqi5"


class PredictionResponse(BaseModel):
    score: float
    category: str
    rating_range: str
    model_type: str
    latency_ms: float
    assessment: dict[str, str]
    warnings: list[str]


class HealthResponse(BaseModel):
    status: str
    message: str
    default_model: str
