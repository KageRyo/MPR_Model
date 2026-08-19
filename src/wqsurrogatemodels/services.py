from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from fastapi import UploadFile

from .artifacts import (
    ArtifactValidationError,
    ProductionModelManifest,
    load_production_manifest,
    validate_artifact,
)
from .enums import ModelTypeEnum
from .errors import ApplicationError, ErrorCode
from .schemas import AssessmentRequestSchema, AssessmentResponseSchema
from .settings import FEATURE_COLUMNS, MODEL_DIR_NAMES, Settings
from .wqi import assess_indicator_quality, categorize_score, direct_wqi5_score


class RuntimeConfigurationError(ValueError):
    """Raised when a required runtime dependency cannot serve assessments safely."""


@dataclass
class ModelMetadata:
    model_type: ModelTypeEnum
    available: bool
    artifact_path: str | None


class WaterQualityService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._dataset: pd.DataFrame | None = None
        self._scores: pd.Series | None = None
        self._models: dict[str, object] = {}
        self._production_manifest: ProductionModelManifest | None = None

    def preload(self) -> None:
        _ = self.dataset

    def _dataset_is_available(self) -> bool:
        try:
            columns = pd.read_csv(self.settings.dataset_path, nrows=0).columns
        except (FileNotFoundError, OSError, pd.errors.ParserError):
            return False
        return "Score" in columns

    def _validate_default_model(self) -> ModelTypeEnum:
        default_model = self.settings.default_model
        if not isinstance(default_model, ModelTypeEnum):
            supported = ", ".join(model.value for model in ModelTypeEnum)
            raise RuntimeConfigurationError(f"DEFAULT_MODEL must be one of: {supported}.")
        return default_model

    def _startup_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.settings.project_root.is_dir():
            errors.append("PROJECT_ROOT must be an existing directory.")

        try:
            default_model = self._validate_default_model()
        except RuntimeConfigurationError as exc:
            errors.append(str(exc))
            return errors

        if self.settings.model_dir.exists() and not self.settings.model_dir.is_dir():
            errors.append("MODEL_DIR must be a directory when it exists.")
        if default_model != ModelTypeEnum.DIRECT_WQI5 and not self.settings.model_dir.is_dir():
            errors.append("MODEL_DIR is required when a surrogate model is the default.")

        manifest_error: ArtifactValidationError | None = None
        if self.settings.production_manifest_path.exists():
            try:
                _ = self.production_manifest
            except ArtifactValidationError as exc:
                manifest_error = exc
        elif default_model != ModelTypeEnum.DIRECT_WQI5:
            manifest_error = ArtifactValidationError("Production model manifest is missing.")
        if manifest_error is not None:
            errors.append(str(manifest_error))

        if manifest_error is None and default_model != ModelTypeEnum.DIRECT_WQI5:
            try:
                _ = self._pick_artifact(default_model)
            except ArtifactValidationError as exc:
                errors.append(str(exc))

        if self.settings.require_dataset_for_readiness and not self._dataset_is_available():
            errors.append("Configured dataset is unavailable or does not contain the Score column.")
        return errors

    def validate_startup(self) -> None:
        """Fail startup only for dependencies configured as required."""
        errors = self._startup_errors()
        if errors:
            raise RuntimeConfigurationError(" ".join(errors))

    def model_availability(self) -> list[dict]:
        availability = [
            {
                "model_type": ModelTypeEnum.DIRECT_WQI5,
                "available": True,
            }
        ]
        for model_type in MODEL_DIR_NAMES:
            try:
                _ = self._pick_artifact(model_type)
                available = True
            except ArtifactValidationError:
                available = False
            availability.append({"model_type": model_type, "available": available})
        return availability

    def readiness(self) -> dict:
        """Return safe dependency status without exposing local filesystem details."""
        errors = self._startup_errors()
        is_ready = not errors
        return {
            "status": "ready" if is_ready else "not_ready",
            "message": (
                "WQSurrogateModels v2 is ready to serve assessments."
                if is_ready
                else "WQSurrogateModels v2 is not ready: " + " ".join(errors)
            ),
            "default_model": self.settings.default_model,
            "dataset_available": self._dataset_is_available(),
            "dataset_required": self.settings.require_dataset_for_readiness,
            "models": self.model_availability(),
        }

    @property
    def dataset(self) -> pd.DataFrame:
        if self._dataset is None:
            try:
                self._dataset = pd.read_csv(self.settings.dataset_path)
                self._scores = self._dataset["Score"]
            except (FileNotFoundError, OSError, pd.errors.ParserError, KeyError) as exc:
                raise ApplicationError(
                    status_code=503,
                    code=ErrorCode.DATASET_UNAVAILABLE,
                    message="The required dataset is unavailable.",
                ) from exc
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

    @property
    def production_manifest(self) -> ProductionModelManifest:
        if self._production_manifest is None:
            self._production_manifest = load_production_manifest(self.settings.production_manifest_path)
        return self._production_manifest

    def _pick_artifact(self, model_type: ModelTypeEnum) -> Path:
        return validate_artifact(
            self.settings.project_root,
            self.production_manifest.artifact_for(model_type),
        )

    def _load_model(self, model_type: ModelTypeEnum):
        if model_type not in MODEL_DIR_NAMES:
            raise ApplicationError(
                status_code=400,
                code=ErrorCode.INVALID_ASSESSMENT_INPUT,
                message="The requested model type is not supported.",
            )
        if model_type not in self._models:
            try:
                artifact = self._pick_artifact(model_type)
            except ArtifactValidationError as exc:
                raise ApplicationError(
                    status_code=503,
                    code=ErrorCode.MODEL_UNAVAILABLE,
                    message="The selected surrogate model is unavailable.",
                ) from exc
            try:
                self._models[model_type] = joblib.load(artifact)
            except Exception as exc:
                raise ApplicationError(
                    status_code=503,
                    code=ErrorCode.MODEL_UNAVAILABLE,
                    message="The selected surrogate model is unavailable.",
                ) from exc
        return self._models[model_type]

    def list_models(self) -> list[dict]:
        models: list[dict] = [
            {
                "model_type": ModelTypeEnum.DIRECT_WQI5,
                "available": True,
                "artifact_path": None,
            },
        ]
        for model_type in MODEL_DIR_NAMES:
            try:
                artifact = self._pick_artifact(model_type)
            except ArtifactValidationError:
                artifact = None
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

    def _build_response(self, score: float, record: dict[str, float], model_type: ModelTypeEnum, latency_ms: float) -> AssessmentResponseSchema:
        category, rating_range = categorize_score(score)
        assessment = {column: assess_indicator_quality(column, float(record[column])) for column in FEATURE_COLUMNS}
        warnings = self._validate_record(record)
        return AssessmentResponseSchema(
            score=round(float(score), 3),
            category=category,
            rating_range=rating_range,
            model_type=model_type,
            latency_ms=round(latency_ms, 3),
            assessment=assessment,
            warnings=warnings,
        )

    def assess_single(self, request: AssessmentRequestSchema) -> AssessmentResponseSchema:
        record = request.model_dump()
        model_type: ModelTypeEnum = record.pop("model_type")
        start = time.perf_counter()
        if model_type == ModelTypeEnum.DIRECT_WQI5:
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
            raise ApplicationError(
                status_code=400,
                code=ErrorCode.INVALID_CSV,
                message="The CSV upload is invalid.",
            ) from exc
        missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ApplicationError(
                status_code=400,
                code=ErrorCode.INVALID_CSV,
                message="The CSV upload must include all required measurement columns.",
            )
        try:
            frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ApplicationError(
                status_code=400,
                code=ErrorCode.INVALID_CSV,
                message="The CSV upload must contain numeric measurement values.",
            ) from exc
        return frame

    def assess_csv_summary(self, upload_file: UploadFile, model_type: ModelTypeEnum | None = None) -> AssessmentResponseSchema:
        frame = self._load_csv(upload_file)
        model_name = model_type or self.settings.default_model
        start = time.perf_counter()
        if model_name == ModelTypeEnum.DIRECT_WQI5:
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

    def assess_csv_rows(self, upload_file: UploadFile, model_type: ModelTypeEnum | None = None) -> dict:
        frame = self._load_csv(upload_file)
        model_name = model_type or self.settings.default_model
        start = time.perf_counter()
        if model_name == ModelTypeEnum.DIRECT_WQI5:
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
            "model_type": model_name.value,
            "latency_ms": round(latency_ms, 3),
        }
