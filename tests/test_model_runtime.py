from __future__ import annotations

import json
import warnings
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import Mock

import pytest

from wqsurrogatemodels import model_runtime, xgboost_artifacts
from wqsurrogatemodels.artifacts import ProductionArtifact, sha256_file
from wqsurrogatemodels.enums import ModelTypeEnum
from wqsurrogatemodels.errors import ApplicationError, ErrorCode
from wqsurrogatemodels.services import WaterQualityService
from wqsurrogatemodels.settings import FEATURE_COLUMNS, Settings


PROJECT_MANIFEST = Path(__file__).parents[1] / "models" / "production_model_manifest.json"


def runtime_artifact(model_type: ModelTypeEnum, requirements: dict[str, str]) -> ProductionArtifact:
    return ProductionArtifact(
        model_type=model_type,
        version="test",
        relative_path=Path("models/test.pkl"),
        sha256="0" * 64,
        training_seed=0,
        feature_columns=tuple(FEATURE_COLUMNS),
        evaluation={"source": "test"},
        metrics={"mae": 0.0},
        runtime_compatibility=requirements,
    )


def write_manifest_with_runtime(
    tmp_path: Path,
    *,
    model_type: ModelTypeEnum,
    requirements: dict[str, str],
) -> tuple[Path, Path]:
    manifest = json.loads(PROJECT_MANIFEST.read_text(encoding="utf-8"))
    artifact_path = tmp_path / "models" / model_type.value / "model.pkl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"test artifact")

    selected = next(item for item in manifest["artifacts"] if item["model_type"] == model_type.value)
    selected["production_artifact"] = artifact_path.relative_to(tmp_path).as_posix()
    selected["sha256"] = sha256_file(artifact_path)
    selected["runtime_compatibility"] = requirements

    manifest_path = tmp_path / "models" / "production_model_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, artifact_path


def test_runtime_validation_identifies_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch):
    def missing_xgboost(distribution: str) -> str:
        if distribution == "xgboost":
            raise PackageNotFoundError
        return "1.5.3"

    monkeypatch.setattr(model_runtime, "distribution_version", missing_xgboost)

    with pytest.raises(model_runtime.ArtifactRuntimeCompatibilityError, match="xgboost==2.1.4 is not installed") as exc_info:
        model_runtime.validate_artifact_runtime(
            runtime_artifact(ModelTypeEnum.XGBOOST, {"xgboost": "2.1.4"}),
            artifact_path=Path("model.pkl"),
        )

    assert "uv sync --locked --extra xgboost" in str(exc_info.value)


def test_runtime_validation_identifies_version_mismatch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(model_runtime, "distribution_version", lambda _: "4.7.0")

    with pytest.raises(model_runtime.ArtifactRuntimeCompatibilityError, match="installed lightgbm==4.7.0"):
        model_runtime.validate_artifact_runtime(
            runtime_artifact(ModelTypeEnum.LIGHTGBM, {"lightgbm": "4.6.0"}),
            artifact_path=Path("model.pkl"),
        )


def test_service_rejects_incompatible_runtime_before_loading_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    manifest_path, _ = write_manifest_with_runtime(
        tmp_path,
        model_type=ModelTypeEnum.RF,
        requirements={"scikit_learn": "1.5.2"},
    )
    service = WaterQualityService(
        Settings(
            project_root=tmp_path,
            model_dir=tmp_path / "models",
            production_manifest_path=manifest_path,
        )
    )
    monkeypatch.setattr(model_runtime, "distribution_version", lambda _: "1.6.0")
    monkeypatch.setattr("wqsurrogatemodels.services.joblib.load", lambda _: pytest.fail("must not load"))

    with pytest.raises(ApplicationError) as exc_info:
        service._load_model(ModelTypeEnum.RF)

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == ErrorCode.MODEL_UNAVAILABLE
    assert exc_info.value.message == "The selected surrogate model is unavailable."


def test_service_intercepts_known_xgboost_legacy_pickle_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, recwarn: pytest.WarningsRecorder
):
    manifest_path, _ = write_manifest_with_runtime(
        tmp_path,
        model_type=ModelTypeEnum.XGBOOST,
        requirements={"scikit_learn": "1.5.2"},
    )
    service = WaterQualityService(
        Settings(
            project_root=tmp_path,
            model_dir=tmp_path / "models",
            production_manifest_path=manifest_path,
        )
    )
    loaded_model = object()
    migration_notice = Mock()

    def legacy_loader(_: Path):
        warnings.warn(
            f"{xgboost_artifacts.LEGACY_PICKLE_WARNING_FRAGMENT}; export the model first.",
            UserWarning,
        )
        return loaded_model

    monkeypatch.setattr(model_runtime, "distribution_version", lambda _: "1.5.2")
    monkeypatch.setattr(xgboost_artifacts.joblib, "load", legacy_loader)
    monkeypatch.setattr("wqsurrogatemodels.services.logger.warning", migration_notice)

    assert service._load_model(ModelTypeEnum.XGBOOST) is loaded_model
    assert not [warning for warning in recwarn if xgboost_artifacts.LEGACY_PICKLE_WARNING_FRAGMENT in str(warning.message)]
    migration_notice.assert_called_once()
    assert "rehydrate_xgboost_artifact.py" in migration_notice.call_args.args[0]
