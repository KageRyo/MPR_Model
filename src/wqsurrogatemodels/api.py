from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .enums import ModelTypeEnum
from .schemas import (
    AssessmentRequestSchema,
    AssessmentResponseSchema,
    ErrorResponseSchema,
    HealthResponseSchema,
    ReadinessResponseSchema,
)
from .services import RuntimeConfigurationError, WaterQualityService
from .errors import ApplicationError, ErrorCode

service = WaterQualityService()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        service.validate_startup()
    except RuntimeConfigurationError:
        logger.exception("WQSurrogateModels runtime validation failed during startup")
        raise
    yield


app = FastAPI(
    title="WQSurrogateModels API",
    version="2.1.0",
    description=(
        "WQI5-based current-state water quality assessment backend. "
        "Supports a direct WQI5 baseline and surrogate regression models.\n\n"
        "Primary contract is under /api/v2/* . Legacy endpoints at root level are "
        "retained for backward compatibility with WaterMirror and are marked deprecated."
    ),
    lifespan=lifespan,
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


def error_response(status_code: int, code: ErrorCode, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponseSchema(error={"code": code.value, "message": message}).model_dump(),
    )


@app.exception_handler(ApplicationError)
async def application_error_handler(_: Request, exc: ApplicationError) -> JSONResponse:
    return error_response(exc.status_code, exc.code, exc.message)


@app.exception_handler(RuntimeConfigurationError)
async def runtime_configuration_error_handler(_: Request, exc: RuntimeConfigurationError) -> JSONResponse:
    logger.error("Runtime configuration error: %s", exc)
    return error_response(503, ErrorCode.INVALID_CONFIGURATION, "The backend configuration is invalid.")


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    logger.info("Request validation failed: %s errors", len(exc.errors()))
    return error_response(
        422,
        ErrorCode.INVALID_ASSESSMENT_INPUT,
        "The assessment request is invalid.",
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled application error", exc_info=exc)
    return error_response(500, ErrorCode.INTERNAL_ERROR, "An unexpected server error occurred.")


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


@app.get("/api/v2/ready", response_model=ReadinessResponseSchema, tags=["v2"])
async def ready_v2(response: Response) -> ReadinessResponseSchema:
    """Report assessment dependency status separately from process health."""
    readiness = ReadinessResponseSchema(**service.readiness())
    if readiness.status == "not_ready":
        response.status_code = 503
    return readiness


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
