from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from wqsurrogatemodels.artifacts import (
    ArtifactValidationError,
    load_production_manifest,
    validate_artifact,
    validate_configured_artifacts,
)
from wqsurrogatemodels.enums import ModelTypeEnum
from wqsurrogatemodels.services import WaterQualityService
from wqsurrogatemodels.settings import Settings


MODEL_PATHS = {
    ModelTypeEnum.LR: "models/LR/test-lr.pkl",
    ModelTypeEnum.MPR: "models/MPR/test-mpr.pkl",
    ModelTypeEnum.SVM: "models/SVM/test-svm.pkl",
    ModelTypeEnum.RF: "models/RF/test-rf.pkl",
    ModelTypeEnum.XGBOOST: "models/XGBoost/test-xgboost.pkl",
    ModelTypeEnum.LIGHTGBM: "models/LightGBM/test-lightgbm.pkl",
}


def write_manifest(project_root: Path) -> Path:
    artifacts = []
    for model_type, relative_path in MODEL_PATHS.items():
        artifact_path = project_root / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        contents = f"safe test artifact for {model_type.value}".encode()
        artifact_path.write_bytes(contents)
        artifacts.append(
            {
                "model_type": model_type.value,
                "version": "test-1",
                "production_artifact": relative_path,
                "sha256": hashlib.sha256(contents).hexdigest(),
                "training_seed": 7,
                "feature_columns": ["DO", "BOD", "NH3N", "EC", "SS"],
                "evaluation": {"source": "test", "experiment": "contract"},
                "runtime_compatibility": {"scikit_learn": "1.5.2"},
                "metrics": {"mae": 0.25, "r2": 0.99},
            }
        )
    manifest_path = project_root / "models" / "production_model_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "required_feature_columns": ["DO", "BOD", "NH3N", "EC", "SS"],
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_manifest_loads_exactly_one_artifact_for_each_surrogate(tmp_path: Path):
    manifest_path = write_manifest(tmp_path)

    manifest = load_production_manifest(manifest_path)

    assert manifest.manifest_version == 1
    assert set(manifest.artifacts) == set(MODEL_PATHS)
    assert manifest.artifact_for(ModelTypeEnum.LR).feature_columns == ("DO", "BOD", "NH3N", "EC", "SS")


def test_artifact_validation_accepts_matching_checksum(tmp_path: Path):
    manifest = load_production_manifest(write_manifest(tmp_path))

    artifact_path = validate_artifact(tmp_path, manifest.artifact_for(ModelTypeEnum.LR))

    assert artifact_path == tmp_path / MODEL_PATHS[ModelTypeEnum.LR]


def test_artifact_validation_reports_missing_artifact_without_local_path(tmp_path: Path):
    manifest = load_production_manifest(write_manifest(tmp_path))
    (tmp_path / MODEL_PATHS[ModelTypeEnum.LR]).unlink()

    with pytest.raises(ArtifactValidationError, match="Production artifact for lr is missing") as exc_info:
        validate_artifact(tmp_path, manifest.artifact_for(ModelTypeEnum.LR))

    assert str(tmp_path) not in str(exc_info.value)


def test_artifact_validation_reports_checksum_mismatch(tmp_path: Path):
    manifest = load_production_manifest(write_manifest(tmp_path))
    (tmp_path / MODEL_PATHS[ModelTypeEnum.LR]).write_bytes(b"changed")

    with pytest.raises(ArtifactValidationError, match="checksum does not match"):
        validate_artifact(tmp_path, manifest.artifact_for(ModelTypeEnum.LR))


def test_malformed_manifest_is_rejected_with_a_clear_error(tmp_path: Path):
    manifest_path = write_manifest(tmp_path)
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del raw_manifest["artifacts"][0]["sha256"]
    manifest_path.write_text(json.dumps(raw_manifest), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="sha256"):
        load_production_manifest(manifest_path)


def test_only_configured_surrogate_models_require_local_artifacts(tmp_path: Path):
    manifest = load_production_manifest(write_manifest(tmp_path))
    (tmp_path / MODEL_PATHS[ModelTypeEnum.LR]).unlink()

    assert validate_configured_artifacts(tmp_path, manifest, [ModelTypeEnum.DIRECT_WQI5]) == {}
    with pytest.raises(ArtifactValidationError, match="Production artifact for lr is missing"):
        validate_configured_artifacts(tmp_path, manifest, [ModelTypeEnum.LR])


def test_service_uses_the_manifest_selected_artifact(tmp_path: Path):
    manifest_path = write_manifest(tmp_path)
    service = WaterQualityService(
        Settings(
            project_root=tmp_path,
            model_dir=tmp_path / "models",
            production_manifest_path=manifest_path,
        )
    )

    assert service._pick_artifact(ModelTypeEnum.LR) == tmp_path / MODEL_PATHS[ModelTypeEnum.LR]
