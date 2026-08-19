from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from wqsurrogatemodels import api
from wqsurrogatemodels.enums import ModelTypeEnum
from wqsurrogatemodels.services import RuntimeConfigurationError, WaterQualityService
from wqsurrogatemodels.settings import Settings


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client


def assert_error(response: httpx.Response, status_code: int, code: str, message: str) -> None:
    assert response.status_code == status_code
    assert response.json() == {"error": {"code": code, "message": message}}
    assert "detail" not in response.json()


@pytest.mark.anyio
async def test_invalid_assessment_input_uses_a_stable_error_shape(client: httpx.AsyncClient):
    response = await client.post("/api/v2/assessment", json={"DO": "not-a-number"})

    assert_error(response, 422, "invalid_assessment_input", "The assessment request is invalid.")


@pytest.mark.anyio
async def test_invalid_csv_uses_a_safe_error_without_parser_details(client: httpx.AsyncClient):
    response = await client.post(
        "/api/v2/assessment/csv/summary",
        files={"file": ("invalid.csv", b"DO,BOD,NH3N,EC,SS\nnot,numeric,values,at,all\n", "text/csv")},
    )

    assert_error(response, 400, "invalid_csv", "The CSV upload must contain numeric measurement values.")


@pytest.mark.anyio
async def test_unavailable_model_uses_a_stable_error_shape(client: httpx.AsyncClient):
    response = await client.post(
        "/api/v2/assessment",
        json={"DO": 96.2, "BOD": 1.5, "NH3N": 0.22, "EC": 171, "SS": 2.6, "model_type": "lr"},
    )

    assert_error(response, 503, "model_unavailable", "The selected surrogate model is unavailable.")


@pytest.mark.anyio
async def test_unavailable_dataset_uses_a_safe_error_without_local_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service = WaterQualityService(
        Settings(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            model_dir=tmp_path / "models",
            production_manifest_path=tmp_path / "models" / "production_model_manifest.json",
            default_model=ModelTypeEnum.DIRECT_WQI5,
            dataset_path=tmp_path / "data" / "missing.csv",
        )
    )
    monkeypatch.setattr(api, "service", service)
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/v2/percentile?score=82.5")

    assert_error(response, 503, "dataset_unavailable", "The required dataset is unavailable.")
    assert str(tmp_path) not in response.text


@pytest.mark.anyio
async def test_runtime_configuration_error_uses_a_stable_error_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(api.service, "calculate_percentile", lambda _: (_ for _ in ()).throw(RuntimeConfigurationError("bad")))
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/v2/percentile?score=82.5")

    assert_error(response, 503, "invalid_configuration", "The backend configuration is invalid.")
