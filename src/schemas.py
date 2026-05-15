from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import ModelType


class PredictionRequest(BaseModel):
    DO: float = Field(..., description="Dissolved oxygen saturation percentage.")
    BOD: float = Field(..., description="Biochemical oxygen demand in mg/L.")
    NH3N: float = Field(..., description="Ammonia nitrogen in mg/L.")
    EC: float = Field(..., description="Electrical conductivity in umho/cm.")
    SS: float = Field(..., description="Suspended solids in mg/L.")
    model_type: ModelType = ModelType.DIRECT_WQI5


class PredictionResponse(BaseModel):
    score: float
    category: str
    rating_range: str
    model_type: ModelType
    latency_ms: float
    assessment: dict[str, str]
    warnings: list[str]


class HealthResponse(BaseModel):
    status: str
    message: str
    default_model: ModelType
