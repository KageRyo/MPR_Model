from __future__ import annotations

import json
import logging
import uuid

import httpx
import pytest

from wqsurrogatemodels import api
from wqsurrogatemodels.observability import JsonLogFormatter, REQUEST_LOGGER_NAME


class CapturedRecords(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def request_logs():
    logger = logging.getLogger(REQUEST_LOGGER_NAME)
    handler = CapturedRecords()
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client


def records_for(records: list[logging.LogRecord], event: str) -> list[logging.LogRecord]:
    return [record for record in records if record.getMessage() == event]


@pytest.mark.anyio
async def test_every_request_gets_a_request_id_and_structured_log(
    client: httpx.AsyncClient, request_logs: list[logging.LogRecord]
):
    response = await client.get("/api/v2/health")

    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    assert str(uuid.UUID(request_id)) == request_id
    request_record = records_for(request_logs, "api_request")[-1]
    assert request_record.request_id == request_id
    assert request_record.method == "GET"
    assert request_record.route == "/api/v2/health"
    assert request_record.status_code == 200
    assert request_record.latency_ms >= 0
    assert not hasattr(request_record, "model_type") or request_record.model_type is None


@pytest.mark.anyio
async def test_assessment_log_records_selected_model_without_request_payload(
    client: httpx.AsyncClient, request_logs: list[logging.LogRecord]
):
    response = await client.post(
        "/api/v2/assessment",
        json={"DO": 96.2, "BOD": 1.5, "NH3N": 0.22, "EC": 171, "SS": 2.6, "model_type": "direct_wqi5"},
    )

    assert response.status_code == 200
    request_record = records_for(request_logs, "api_request")[-1]
    assert request_record.model_type == "direct_wqi5"
    assert "96.2" not in str(request_record.__dict__)


@pytest.mark.anyio
async def test_error_logs_share_the_response_request_id(
    client: httpx.AsyncClient, request_logs: list[logging.LogRecord]
):
    response = await client.post("/api/v2/assessment", json={})

    assert response.status_code == 422
    request_id = response.headers["X-Request-ID"]
    error_record = records_for(request_logs, "api_error")[-1]
    request_record = records_for(request_logs, "api_request")[-1]
    assert error_record.request_id == request_id
    assert error_record.error_code == "invalid_assessment_input"
    assert request_record.request_id == request_id
    assert request_record.status_code == 422


def test_json_formatter_emits_bounded_stdout_fields():
    record = logging.LogRecord(REQUEST_LOGGER_NAME, logging.INFO, __file__, 1, "api_request", (), None)
    record.request_id = "request-id"
    record.method = "POST"
    record.route = "/api/v2/assessment"
    record.status_code = 200
    record.latency_ms = 1.25
    record.model_type = "direct_wqi5"

    payload = json.loads(JsonLogFormatter().format(record))

    assert payload == {
        "event": "api_request",
        "latency_ms": 1.25,
        "level": "info",
        "method": "POST",
        "model_type": "direct_wqi5",
        "request_id": "request-id",
        "route": "/api/v2/assessment",
        "status_code": 200,
        "timestamp": payload["timestamp"],
    }
