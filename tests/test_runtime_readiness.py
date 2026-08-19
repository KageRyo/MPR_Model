from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import pytest

from wqsurrogatemodels import api
from wqsurrogatemodels.enums import ModelTypeEnum
from wqsurrogatemodels.services import RuntimeConfigurationError, WaterQualityService
from wqsurrogatemodels.settings import Settings


PROJECT_MANIFEST = Path(__file__).parents[1] / "models" / "production_model_manifest.json"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def make_settings(
    tmp_path: Path,
    *,
    default_model: ModelTypeEnum | str = ModelTypeEnum.DIRECT_WQI5,
    require_dataset: bool = False,
    manifest_contents: str | None = None,
) -> Settings:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    manifest_path = model_dir / "production_model_manifest.json"
    if manifest_contents is None:
        shutil.copyfile(PROJECT_MANIFEST, manifest_path)
    else:
        manifest_path.write_text(manifest_contents, encoding="utf-8")
    return Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        model_dir=model_dir,
        production_manifest_path=manifest_path,
        default_model=default_model,
        dataset_path=tmp_path / "data" / "scores.csv",
        require_dataset_for_readiness=require_dataset,
    )


def test_direct_wqi5_is_ready_without_optional_dataset_or_surrogate_artifacts(tmp_path: Path):
    service = WaterQualityService(make_settings(tmp_path))

    service.validate_startup()
    readiness = service.readiness()

    assert readiness["status"] == "ready"
    assert readiness["dataset_available"] is False
    assert readiness["dataset_required"] is False
    assert readiness["models"][0] == {"model_type": ModelTypeEnum.DIRECT_WQI5, "available": True}
    assert all(model["available"] is False for model in readiness["models"][1:])


def test_readiness_requires_a_configured_dataset_and_valid_score_column(tmp_path: Path):
    settings = make_settings(tmp_path, require_dataset=True)
    service = WaterQualityService(settings)

    with pytest.raises(RuntimeConfigurationError, match="Configured dataset is unavailable"):
        service.validate_startup()

    readiness = service.readiness()
    assert readiness["status"] == "not_ready"
    assert readiness["dataset_available"] is False
    assert str(tmp_path) not in readiness["message"]


def test_readiness_accepts_required_dataset_when_score_column_exists(tmp_path: Path):
    settings = make_settings(tmp_path, require_dataset=True)
    settings.dataset_path.parent.mkdir()
    settings.dataset_path.write_text("Score\n89.542\n", encoding="utf-8")
    service = WaterQualityService(settings)

    service.validate_startup()

    assert service.readiness()["status"] == "ready"


def test_invalid_default_model_fails_startup_with_actionable_message(tmp_path: Path):
    service = WaterQualityService(make_settings(tmp_path, default_model="not-a-model"))

    with pytest.raises(RuntimeConfigurationError, match="DEFAULT_MODEL must be one of"):
        service.validate_startup()


def test_missing_default_surrogate_artifact_fails_startup(tmp_path: Path):
    service = WaterQualityService(make_settings(tmp_path, default_model=ModelTypeEnum.LR))

    with pytest.raises(RuntimeConfigurationError, match="Production artifact for lr is missing"):
        service.validate_startup()


def test_malformed_manifest_fails_startup_even_with_direct_default(tmp_path: Path):
    service = WaterQualityService(make_settings(tmp_path, manifest_contents="{}"))

    with pytest.raises(RuntimeConfigurationError, match="Invalid production model manifest"):
        service.validate_startup()


@pytest.mark.anyio
async def test_api_ready_returns_safe_503_for_missing_required_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service = WaterQualityService(make_settings(tmp_path, require_dataset=True))
    monkeypatch.setattr(api, "service", service)
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/v2/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["dataset_available"] is False
    assert payload["dataset_required"] is True
    assert str(tmp_path) not in response.text


@pytest.mark.anyio
async def test_lifespan_rejects_invalid_required_runtime_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service = WaterQualityService(make_settings(tmp_path, default_model=ModelTypeEnum.LR))
    monkeypatch.setattr(api, "service", service)

    with pytest.raises(RuntimeConfigurationError, match="Production artifact for lr is missing"):
        async with api.lifespan(api.app):
            pass
