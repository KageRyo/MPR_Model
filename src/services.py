from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from fastapi import HTTPException, UploadFile

from .enums import ModelType
from .schemas import AssessmentRequest, AssessmentResponse
from .settings import FEATURE_COLUMNS, MODEL_DIR_NAMES, Settings
from .wqi import assess_indicator_quality, categorize_score, direct_wqi5_score


@dataclass
class ModelMetadata:
    model_type: ModelType
    available: bool
    artifact_path: str | None


class WaterQualityService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._dataset: pd.DataFrame | None = None
        self._scores: pd.Series | None = None
        self._models: dict[str, object] = {}

    def preload(self) -> None:
        _ = self.dataset

    @property
    def dataset(self) -> pd.DataFrame:
        if self._dataset is None:
            try:
                self._dataset = pd.read_csv(self.settings.dataset_path)
                self._scores = self._dataset["Score"]
            except FileNotFoundError as exc:
                raise HTTPException(status_code=500, detail=f"Dataset not found: {exc}") from exc
        return self._dataset

    def _validate_record(self, record: dict[str, float]) -> list[str]:
        warnings: list[str] = []
        ranges = {
            "DO": (0, 150),
            "BOD": (0, 200),
            "NH3N": (0, 50),
            "EC": (0, 50000),
            "SS": (0, 5000),
        }
        for key in FEATURE_COLUMNS:
            value = float(record[key])
            lower, upper = ranges[key]
            if not lower <= value <= upper:
                warnings.append(f"{key}={value} is outside the recommended range [{lower}, {upper}].")
        return warnings

    def _artifact_candidates(self, model_type: ModelType) -> list[Path]:
        directory_name = MODEL_DIR_NAMES[model_type]
        directory = self.settings.model_dir / directory_name
        if not directory.exists():
            return []
        return sorted(directory.glob("*.pkl"))

    def _pick_artifact(self, model_type: ModelType) -> Path | None:
        candidates = self._artifact_candidates(model_type)
        if not candidates:
            return None
        preferred = [path for path in candidates if self.settings.preferred_artifact_size in path.name]
        if preferred:
            return preferred[-1]
        return candidates[-1]

    def _load_model(self, model_type: ModelType):
        if model_type not in MODEL_DIR_NAMES:
            raise HTTPException(status_code=400, detail=f"Unsupported model_type: {model_type}")
        if model_type not in self._models:
            artifact = self._pick_artifact(model_type)
            if artifact is None:
                raise HTTPException(status_code=503, detail=f"No model artifact available for {model_type}")
            self._models[model_type] = joblib.load(artifact)
        return self._models[model_type]

    def list_models(self) -> list[dict]:
        models: list[dict] = [
            {
                "model_type": ModelType.DIRECT_WQI5,
                "available": True,
                "artifact_path": None,
            },
        ]
        for model_type in MODEL_DIR_NAMES:
            artifact = self._pick_artifact(model_type)
            models.append(
                {
                    "model_type": model_type,
                    "available": artifact is not None,
                    "artifact_path": str(artifact.relative_to(self.settings.project_root)) if artifact else None,
                }
            )
        return models

    def calculate_percentile(self, score: float) -> float:
        if self._scores is None:
            _ = self.dataset
        return round(float((self._scores <= score).mean() * 100), 3)

    def category_distribution(self) -> list[dict]:
        distribution: list[dict] = []
        counts = {}
        for score in self.dataset["Score"]:
            label, _ = categorize_score(score)
            counts[label] = counts.get(label, 0) + 1
        for label in ["Excellent", "Good", "Fair", "Poor", "Bad", "Terrible"]:
            distribution.append({"category": label, "rating": counts.get(label, 0)})
        return distribution

    def _build_response(self, score: float, record: dict[str, float], model_type: ModelType, latency_ms: float) -> AssessmentResponse:
        category, rating_range = categorize_score(score)
        assessment = {column: assess_indicator_quality(column, float(record[column])) for column in FEATURE_COLUMNS}
        warnings = self._validate_record(record)
        return AssessmentResponse(
            score=round(float(score), 3),
            category=category,
            rating_range=rating_range,
            model_type=model_type,
            latency_ms=round(latency_ms, 3),
            assessment=assessment,
            warnings=warnings,
        )

    def assess_single(self, request: AssessmentRequest) -> AssessmentResponse:
        record = request.model_dump()
        model_type: ModelType = record.pop("model_type")
        start = time.perf_counter()
        if model_type == ModelType.DIRECT_WQI5:
            score = direct_wqi5_score(
                do=record["DO"],
                bod=record["BOD"],
                nh3n=record["NH3N"],
                ec=record["EC"],
                ss=record["SS"],
            )
        else:
            model = self._load_model(model_type)
            frame = pd.DataFrame([record], columns=FEATURE_COLUMNS)
            score = float(model.predict(frame)[0])
        latency_ms = (time.perf_counter() - start) * 1000
        return self._build_response(score, record, model_type, latency_ms)

    def _load_csv(self, upload_file: UploadFile) -> pd.DataFrame:
        content = upload_file.file.read()
        try:
            frame = pd.read_csv(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid CSV payload: {exc}") from exc
        missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_columns)}",
            )
        return frame

    def assess_csv_summary(self, upload_file: UploadFile, model_type: ModelType | None = None) -> AssessmentResponse:
        frame = self._load_csv(upload_file)
        model_name = model_type or self.settings.default_model
        start = time.perf_counter()
        if model_name == ModelType.DIRECT_WQI5:
            predictions = frame[FEATURE_COLUMNS].apply(
                lambda row: direct_wqi5_score(
                    do=row["DO"],
                    bod=row["BOD"],
                    nh3n=row["NH3N"],
                    ec=row["EC"],
                    ss=row["SS"],
                ),
                axis=1,
            )
        else:
            model = self._load_model(model_name)
            predictions = pd.Series(model.predict(frame[FEATURE_COLUMNS]))
        latency_ms = (time.perf_counter() - start) * 1000
        score = float(predictions.mean())
        representative_record = frame[FEATURE_COLUMNS].mean().to_dict()
        return self._build_response(score, representative_record, model_name, latency_ms)

    def assess_csv_rows(self, upload_file: UploadFile, model_type: ModelType | None = None) -> dict:
        frame = self._load_csv(upload_file)
        model_name = model_type or self.settings.default_model
        start = time.perf_counter()
        if model_name == ModelType.DIRECT_WQI5:
            predictions = frame[FEATURE_COLUMNS].apply(
                lambda row: direct_wqi5_score(
                    do=row["DO"],
                    bod=row["BOD"],
                    nh3n=row["NH3N"],
                    ec=row["EC"],
                    ss=row["SS"],
                ),
                axis=1,
            )
        else:
            model = self._load_model(model_name)
            predictions = pd.Series(model.predict(frame[FEATURE_COLUMNS]))
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "scores": [round(float(value), 3) for value in predictions],
            "model_type": str(model_name),   # Ensure string for WaterMirror compatibility
            "latency_ms": round(latency_ms, 3),
        }
