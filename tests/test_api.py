from __future__ import annotations

import httpx
import pytest

from src.api import app
from src.wqi import categorize_score, direct_wqi5_score


def test_direct_wqi5_formula_returns_bounded_score():
    score = direct_wqi5_score(do=96.2, bod=1.5, nh3n=0.22, ec=171, ss=2.6)
    assert 0 <= score <= 100


def test_score_category_mapping():
    assert categorize_score(82.5) == ("Good", "70 < WQI5 ≤ 85")


@pytest.mark.anyio
async def test_status_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["default_model"]


@pytest.mark.anyio
async def test_models_endpoint_lists_direct_baseline():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/models")
    assert response.status_code == 200
    payload = response.json()
    assert any(model["model_type"] == "direct_wqi5" for model in payload["models"])


@pytest.mark.anyio
async def test_predict_endpoint_with_direct_baseline():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/predict",
            json={
                "DO": 96.2,
                "BOD": 1.5,
                "NH3N": 0.22,
                "EC": 171,
                "SS": 2.6,
                "model_type": "direct_wqi5",
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_type"] == "direct_wqi5"
    assert payload["category"] in {"Excellent", "Good", "Fair", "Poor", "Bad", "Terrible"}


@pytest.mark.anyio
async def test_legacy_csv_endpoint_accepts_model_type_form():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/score/total/",
            data={"model_type": "direct_wqi5"},
            files={"file": ("sample.csv", b"DO,BOD,NH3N,EC,SS\n96.2,1.5,0.22,171,2.6\n", "text/csv")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_type"] == "direct_wqi5"
    assert "score" in payload
