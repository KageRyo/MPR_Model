from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .enums import ModelTypeEnum
from .settings import FEATURE_COLUMNS


class ArtifactValidationError(ValueError):
    """Raised when production-model metadata or a local artifact is invalid."""


@dataclass(frozen=True)
class ProductionArtifact:
    model_type: ModelTypeEnum
    version: str
    relative_path: Path
    sha256: str
    training_seed: int
    feature_columns: tuple[str, ...]
    evaluation: dict[str, str]
    metrics: dict[str, float]
    runtime_compatibility: dict[str, str]


@dataclass(frozen=True)
class ProductionModelManifest:
    manifest_version: int
    required_feature_columns: tuple[str, ...]
    artifacts: dict[ModelTypeEnum, ProductionArtifact]

    def artifact_for(self, model_type: ModelTypeEnum) -> ProductionArtifact:
        try:
            return self.artifacts[model_type]
        except KeyError as exc:
            raise ArtifactValidationError(
                f"Production manifest does not define an artifact for {model_type.value}."
            ) from exc


def _manifest_error(message: str) -> ArtifactValidationError:
    return ArtifactValidationError(f"Invalid production model manifest: {message}")


def _as_string_mapping(value: object, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise _manifest_error(f"{field_name} must be a non-empty object.")
    if not all(isinstance(key, str) and isinstance(item, str) and item for key, item in value.items()):
        raise _manifest_error(f"{field_name} must contain non-empty string values.")
    return value


def _as_metrics(value: object) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise _manifest_error("metrics must be a non-empty object.")
    if not all(isinstance(key, str) and isinstance(item, (int, float)) for key, item in value.items()):
        raise _manifest_error("metrics must contain numeric values.")
    return {key: float(item) for key, item in value.items()}


def _parse_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise _manifest_error("production_artifact must be a non-empty relative path.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise _manifest_error("production_artifact must stay within the project root.")
    return path


def _parse_artifact(entry: object) -> ProductionArtifact:
    if not isinstance(entry, dict):
        raise _manifest_error("every artifact entry must be an object.")
    try:
        model_type = ModelTypeEnum(entry["model_type"])
    except (KeyError, ValueError) as exc:
        raise _manifest_error("artifact model_type must be a supported surrogate model.") from exc
    if model_type == ModelTypeEnum.DIRECT_WQI5:
        raise _manifest_error("direct_wqi5 must not have a serialized production artifact.")

    version = entry.get("version")
    if not isinstance(version, str) or not version:
        raise _manifest_error(f"{model_type.value} version must be a non-empty string.")

    sha256 = entry.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256.lower()):
        raise _manifest_error(f"{model_type.value} sha256 must be a 64-character hexadecimal digest.")

    training_seed = entry.get("training_seed")
    if not isinstance(training_seed, int):
        raise _manifest_error(f"{model_type.value} training_seed must be an integer.")

    feature_columns = entry.get("feature_columns")
    if not isinstance(feature_columns, list) or feature_columns != FEATURE_COLUMNS:
        raise _manifest_error(
            f"{model_type.value} feature_columns must be exactly {', '.join(FEATURE_COLUMNS)}."
        )

    return ProductionArtifact(
        model_type=model_type,
        version=version,
        relative_path=_parse_relative_path(entry.get("production_artifact")),
        sha256=sha256.lower(),
        training_seed=training_seed,
        feature_columns=tuple(feature_columns),
        evaluation=_as_string_mapping(entry.get("evaluation"), f"{model_type.value} evaluation"),
        metrics=_as_metrics(entry.get("metrics")),
        runtime_compatibility=_as_string_mapping(
            entry.get("runtime_compatibility"), f"{model_type.value} runtime_compatibility"
        ),
    )


def load_production_manifest(manifest_path: Path) -> ProductionModelManifest:
    """Load and validate metadata only; this never loads a serialized model."""
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactValidationError("Production model manifest is missing.") from exc
    except json.JSONDecodeError as exc:
        raise _manifest_error("file is not valid JSON.") from exc

    if not isinstance(raw_manifest, dict):
        raise _manifest_error("top-level value must be an object.")
    if raw_manifest.get("manifest_version") != 1:
        raise _manifest_error("manifest_version must be 1.")
    if raw_manifest.get("required_feature_columns") != FEATURE_COLUMNS:
        raise _manifest_error(
            f"required_feature_columns must be exactly {', '.join(FEATURE_COLUMNS)}."
        )

    raw_artifacts = raw_manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise _manifest_error("artifacts must be an array.")
    artifacts = [_parse_artifact(entry) for entry in raw_artifacts]
    artifact_by_type = {artifact.model_type: artifact for artifact in artifacts}
    expected_model_types = {
        ModelTypeEnum.LR,
        ModelTypeEnum.MPR,
        ModelTypeEnum.SVM,
        ModelTypeEnum.RF,
        ModelTypeEnum.XGBOOST,
        ModelTypeEnum.LIGHTGBM,
    }
    if set(artifact_by_type) != expected_model_types or len(artifact_by_type) != len(artifacts):
        raise _manifest_error("artifacts must define each supported surrogate model exactly once.")

    return ProductionModelManifest(
        manifest_version=1,
        required_feature_columns=tuple(FEATURE_COLUMNS),
        artifacts=artifact_by_type,
    )


def artifact_path(project_root: Path, artifact: ProductionArtifact) -> Path:
    """Resolve an artifact without allowing a manifest path to escape its root."""
    root = project_root.resolve()
    path = (root / artifact.relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ArtifactValidationError(
            f"Production artifact path for {artifact.model_type.value} is outside the project root."
        ) from exc
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact(project_root: Path, artifact: ProductionArtifact) -> Path:
    path = artifact_path(project_root, artifact)
    if not path.is_file():
        raise ArtifactValidationError(f"Production artifact for {artifact.model_type.value} is missing.")
    if sha256_file(path) != artifact.sha256:
        raise ArtifactValidationError(
            f"Production artifact checksum does not match the manifest for {artifact.model_type.value}."
        )
    return path


def validate_configured_artifacts(
    project_root: Path,
    manifest: ProductionModelManifest,
    model_types: Iterable[ModelTypeEnum],
) -> dict[ModelTypeEnum, Path]:
    """Validate only model families selected for use, keeping optional ones optional."""
    validated: dict[ModelTypeEnum, Path] = {}
    for model_type in model_types:
        if model_type == ModelTypeEnum.DIRECT_WQI5:
            continue
        validated[model_type] = validate_artifact(project_root, manifest.artifact_for(model_type))
    return validated
