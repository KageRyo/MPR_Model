from __future__ import annotations

import os

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .enums import ModelTypeEnum
from .schemas import AssessmentRequestSchema, AssessmentResponseSchema, HealthResponseSchema
from .services import WaterQualityService

service = WaterQualityService()


app = FastAPI(
    title="WQSurrogateModels API",
    version="2.1.0",
    description=(
        "WQI5-based current-state water quality assessment backend. "
        "Supports a direct WQI5 baseline and surrogate regression models.\n\n"
        "Primary contract is under /api/v2/* . Legacy endpoints at root level are "
        "retained for backward compatibility with WaterMirror and are marked deprecated."
    ),
)

# CORS configuration for WaterMirror frontend
cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponseSchema)
async def read_root() -> HealthResponseSchema:
    return HealthResponseSchema(
        status="ok",
        message="WQSurrogateModels assessment backend is reachable.",
        default_model=service.settings.default_model,
    )


# -----------------------------------------------------------------------------
# New primary API v2 (recommended)
# -----------------------------------------------------------------------------


@app.get("/api/v2/health", response_model=HealthResponseSchema, tags=["v2"])
async def health_v2() -> HealthResponseSchema:
    return HealthResponseSchema(
        status="ok",
        message="WQSurrogateModels v2 is healthy.",
        default_model=service.settings.default_model,
    )


@app.get("/api/v2/models", tags=["v2"])
async def list_models_v2() -> dict:
    return {"models": service.list_models(), "default_model": service.settings.default_model}


@app.get("/api/v2/percentile", tags=["v2"])
async def percentile_v2(score: float) -> dict:
    return {"percentile": service.calculate_percentile(score)}


@app.get("/api/v2/categories", tags=["v2"])
async def categories_v2() -> dict:
    return {"data": service.category_distribution()}


@app.post("/api/v2/assessment", response_model=AssessmentResponseSchema, tags=["v2"])
async def assess(request: AssessmentRequestSchema) -> AssessmentResponseSchema:
    return service.assess_single(request)


@app.post("/api/v2/assessment/csv/summary", response_model=AssessmentResponseSchema, tags=["v2"])
async def assess_csv_summary(
    file: UploadFile = File(...),
    model_type: ModelTypeEnum | None = Form(default=None),
) -> AssessmentResponseSchema:
    return service.assess_csv_summary(file, model_type=model_type)


@app.post("/api/v2/assessment/csv/rows", tags=["v2"])
async def assess_csv_rows(
    file: UploadFile = File(...),
    model_type: ModelTypeEnum | None = Form(default=None),
) -> dict:
    return service.assess_csv_rows(file, model_type=model_type)


# -----------------------------------------------------------------------------
# Deprecated compatibility endpoints (kept to avoid breaking WaterMirror / legacy clients)
# These will continue to work but are no longer the primary contract.
# Prefer the /api/v2/* equivalents above.
# -----------------------------------------------------------------------------


@app.get("/status", response_model=HealthResponseSchema, deprecated=True)
async def status() -> HealthResponseSchema:
    """Deprecated: Use GET /api/v2/health instead."""
    return HealthResponseSchema(
        status="ok",
        message="Service healthy. (deprecated endpoint)",
        default_model=service.settings.default_model,
    )


@app.get("/models", deprecated=True)
async def list_models() -> dict:
    """Deprecated: Use GET /api/v2/models instead."""
    return {"models": service.list_models(), "default_model": service.settings.default_model}


@app.get("/percentile", deprecated=True)
async def percentile(score: float) -> dict:
    """Deprecated: Use GET /api/v2/percentile instead."""
    return {"percentile": service.calculate_percentile(score)}


@app.get("/categories", deprecated=True)
async def categories() -> dict:
    """Deprecated: Use GET /api/v2/categories instead."""
    return {"data": service.category_distribution()}


@app.post("/predict", response_model=AssessmentResponseSchema, deprecated=True)
async def predict(request: AssessmentRequestSchema) -> AssessmentResponseSchema:
    """Deprecated: Use POST /api/v2/assessment instead."""
    return service.assess_single(request)


@app.post("/score/total/", response_model=AssessmentResponseSchema, deprecated=True)
async def predict_total(
    file: UploadFile = File(...),
    model_type: ModelTypeEnum | None = Form(default=None),
) -> AssessmentResponseSchema:
    """Deprecated: Use POST /api/v2/assessment/csv/summary instead."""
    return service.assess_csv_summary(file, model_type=model_type)


@app.post("/score/all/", deprecated=True)
async def predict_all(
    file: UploadFile = File(...),
    model_type: ModelTypeEnum | None = Form(default=None),
) -> dict:
    """Deprecated: Use POST /api/v2/assessment/csv/rows instead."""
    return service.assess_csv_rows(file, model_type=model_type)
