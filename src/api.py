from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile

from .schemas import HealthResponse, PredictionRequest, PredictionResponse
from .services import WaterQualityService

service = WaterQualityService()


app = FastAPI(
    title="WQSurrogateModels API",
    version="2.0.0",
    description=(
        "WQI5-based current-state water quality assessment backend. "
        "Supports a direct WQI5 baseline and surrogate regression models."
    ),
)


@app.get("/", response_model=HealthResponse)
async def read_root() -> HealthResponse:
    return HealthResponse(
        status="ok",
        message="WQSurrogateModels assessment backend is reachable.",
        default_model=service.settings.default_model,
    )


@app.get("/status", response_model=HealthResponse)
async def status() -> HealthResponse:
    return HealthResponse(
        status="ok",
        message="Service healthy.",
        default_model=service.settings.default_model,
    )


@app.get("/models")
async def list_models() -> dict:
    return {"models": service.list_models(), "default_model": service.settings.default_model}


@app.get("/percentile")
async def percentile(score: float) -> dict:
    return {"percentile": service.calculate_percentile(score)}


@app.get("/categories")
async def categories() -> dict:
    return {"data": service.category_distribution()}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    return service.predict_single(request)


@app.post("/score/total/", response_model=PredictionResponse)
async def predict_total(
    file: UploadFile = File(...),
    model_type: str | None = Form(default=None),
) -> PredictionResponse:
    return service.predict_csv_mean(file, model_type=model_type)


@app.post("/score/all/")
async def predict_all(
    file: UploadFile = File(...),
    model_type: str | None = Form(default=None),
) -> dict:
    return service.predict_csv_all(file, model_type=model_type)
