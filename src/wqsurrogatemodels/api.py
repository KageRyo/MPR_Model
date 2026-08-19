from __future__ import annotations

import logging
import time
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
from .version import __version__
from .errors import ApplicationError, ErrorCode
from .observability import (
    REQUEST_ID_HEADER,
    bind_request_id,
    log_event,
    reset_request_id,
    route_template,
)

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
    version=__version__,
    description=(
        "WQI5-based current-state water quality assessment backend. "
        "Supports a direct WQI5 baseline and surrogate regression models.\n\n"
        "Primary contract is under /api/v2/* . Legacy endpoints at root level are "
        "retained for backward compatibility with WaterMirror and are marked deprecated."
    ),
    lifespan=lifespan,
)

# CORS configuration for WaterMirror frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=service.settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id, token = bind_request_id()
    start = time.perf_counter()
    response: Response | None = None
    try:
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        log_event(
            "api_request",
            method=request.method,
            route=route_template(request),
            status_code=response.status_code if response is not None else 500,
            latency_ms=latency_ms,
            model_type=getattr(request.state, "model_type", None),
        )
        reset_request_id(token)


def error_response(status_code: int, code: ErrorCode, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponseSchema(error={"code": code.value, "message": message}).model_dump(),
    )


def log_api_error(request: Request, status_code: int, code: ErrorCode) -> None:
    log_event(
        "api_error",
        method=request.method,
        route=route_template(request),
        status_code=status_code,
        error_code=code.value,
    )


@app.exception_handler(ApplicationError)
async def application_error_handler(request: Request, exc: ApplicationError) -> JSONResponse:
    log_api_error(request, exc.status_code, exc.code)
    return error_response(exc.status_code, exc.code, exc.message)


@app.exception_handler(RuntimeConfigurationError)
async def runtime_configuration_error_handler(request: Request, exc: RuntimeConfigurationError) -> JSONResponse:
    log_api_error(request, 503, ErrorCode.INVALID_CONFIGURATION)
    return error_response(503, ErrorCode.INVALID_CONFIGURATION, "The backend configuration is invalid.")


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    log_api_error(request, 422, ErrorCode.INVALID_ASSESSMENT_INPUT)
    return error_response(
        422,
        ErrorCode.INVALID_ASSESSMENT_INPUT,
        "The assessment request is invalid.",
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log_api_error(request, 500, ErrorCode.INTERNAL_ERROR)
    logger.exception("Unhandled application error", exc_info=exc)
    return error_response(500, ErrorCode.INTERNAL_ERROR, "An unexpected server error occurred.")


@app.get("/", response_model=HealthResponseSchema)
async def read_root() -> HealthResponseSchema:
    return HealthResponseSchema(
        status="ok",
        message="WQSurrogateModels assessment backend is reachable.",
        default_model=service.settings.default_model,
        version=__version__,
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
        version=__version__,
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
async def assess(payload: AssessmentRequestSchema, request: Request) -> AssessmentResponseSchema:
    request.state.model_type = payload.model_type.value
    return service.assess_single(payload)


@app.post("/api/v2/assessment/csv/summary", response_model=AssessmentResponseSchema, tags=["v2"])
async def assess_csv_summary(
    request: Request,
    file: UploadFile = File(...),
    model_type: ModelTypeEnum | None = Form(default=None),
) -> AssessmentResponseSchema:
    request.state.model_type = (model_type or service.settings.default_model).value
    return service.assess_csv_summary(file, model_type=model_type)


@app.post("/api/v2/assessment/csv/rows", tags=["v2"])
async def assess_csv_rows(
    request: Request,
    file: UploadFile = File(...),
    model_type: ModelTypeEnum | None = Form(default=None),
) -> dict:
    request.state.model_type = (model_type or service.settings.default_model).value
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
        version=__version__,
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
async def predict(payload: AssessmentRequestSchema, request: Request) -> AssessmentResponseSchema:
    """Deprecated: Use POST /api/v2/assessment instead."""
    request.state.model_type = payload.model_type.value
    return service.assess_single(payload)


@app.post("/score/total/", response_model=AssessmentResponseSchema, deprecated=True)
async def predict_total(
    request: Request,
    file: UploadFile = File(...),
    model_type: ModelTypeEnum | None = Form(default=None),
) -> AssessmentResponseSchema:
    """Deprecated: Use POST /api/v2/assessment/csv/summary instead."""
    request.state.model_type = (model_type or service.settings.default_model).value
    return service.assess_csv_summary(file, model_type=model_type)


@app.post("/score/all/", deprecated=True)
async def predict_all(
    request: Request,
    file: UploadFile = File(...),
    model_type: ModelTypeEnum | None = Form(default=None),
) -> dict:
    """Deprecated: Use POST /api/v2/assessment/csv/rows instead."""
    request.state.model_type = (model_type or service.settings.default_model).value
    return service.assess_csv_rows(file, model_type=model_type)
