"""Start the API and exercise the public direct-WQI5 smoke path."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, payload: dict | None = None) -> tuple[int, dict]:
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(url, data=body, headers={"Content-Type": "application/json"} if body else {})
    with urlopen(request, timeout=2) as response:
        return response.status, json.loads(response.read())


def wait_for_health(base_url: str) -> dict:
    for _ in range(30):
        try:
            status, payload = request_json(f"{base_url}/api/v2/health")
            if status == 200:
                return payload
        except (URLError, ConnectionError):
            time.sleep(0.2)
    raise RuntimeError("API did not become healthy within six seconds.")


def main() -> None:
    port = free_port()
    with tempfile.TemporaryDirectory() as project_root:
        environment = {
            **os.environ,
            "PROJECT_ROOT": project_root,
            "DEFAULT_MODEL": "direct_wqi5",
            "API_HOST": "127.0.0.1",
            "API_PORT": str(port),
            "REQUIRE_DATASET_FOR_READINESS": "false",
        }
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "wqsurrogatemodels.api:app", "--host", "127.0.0.1", "--port", str(port)],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            health = wait_for_health(base_url)
            assert health["status"] == "ok"
            assert health["version"]
            readiness_status, readiness = request_json(f"{base_url}/api/v2/ready")
            assert readiness_status == 200
            assert readiness["status"] == "ready"
            assessment_status, assessment = request_json(
                f"{base_url}/api/v2/assessment",
                {"DO": 96.2, "BOD": 1.5, "NH3N": 0.22, "EC": 171, "SS": 2.6, "model_type": "direct_wqi5"},
            )
            assert assessment_status == 200
            assert assessment["model_type"] == "direct_wqi5"
            assert 0 <= assessment["score"] <= 100
        finally:
            process.terminate()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
