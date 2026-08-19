from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).parents[1]


def test_dockerfile_copies_only_runtime_code_and_public_manifest():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY . ." not in dockerfile
    assert "COPY models/production_model_manifest.json" in dockerfile
    assert "COPY data" not in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "PYTHONUNBUFFERED=1" in dockerfile
    assert "ARG INSTALL_MODEL_EXTRAS=false" in dockerfile


def test_compose_uses_read_only_local_artifact_and_data_mounts():
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["wq-surrogate-models"]

    assert service["restart"] == "unless-stopped"
    assert service["build"]["args"] == {"INSTALL_MODEL_EXTRAS": "${INSTALL_MODEL_EXTRAS:-false}"}
    assert service["ports"] == ["${HOST_PORT:-8001}:8001"]
    assert service["env_file"] == [".env"]
    assert service["volumes"] == [
        {
            "type": "bind",
            "source": "${LOCAL_MODEL_DIR:-./models}",
            "target": "/app/models",
            "read_only": True,
        },
        {
            "type": "bind",
            "source": "${LOCAL_DATA_DIR:-./data}",
            "target": "/app/data",
            "read_only": True,
        },
    ]


def test_example_environment_documents_compose_mounts_and_port():
    environment = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "LOCAL_MODEL_DIR=./models" in environment
    assert "LOCAL_DATA_DIR=./data" in environment
    assert "HOST_PORT=8001" in environment
    assert "INSTALL_MODEL_EXTRAS=false" in environment
