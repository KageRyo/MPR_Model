from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from wqsurrogatemodels.settings import Settings, SettingsConfigurationError, parse_bool_env


def test_parse_bool_env_truthy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTO_PORT", "true")
    assert parse_bool_env("AUTO_PORT") is True


def test_parse_bool_env_falsey(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTO_PORT", "false")
    assert parse_bool_env("AUTO_PORT") is False


def test_parse_bool_env_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTO_PORT", "not-a-boolean")

    with pytest.raises(SettingsConfigurationError, match="AUTO_PORT must be a boolean"):
        parse_bool_env("AUTO_PORT")


def test_settings_from_environment_parses_supported_runtime_values(tmp_path: Path):
    settings = Settings.from_environment(
        {
            "PROJECT_ROOT": str(tmp_path),
            "MODEL_DIR": "local-models",
            "PRODUCTION_MODEL_MANIFEST": "manifest.json",
            "DEFAULT_MODEL": "direct_wqi5",
            "API_HOST": "127.0.0.1",
            "API_PORT": "9010",
            "AUTO_PORT": "yes",
            "DATASET_FILE": "scores.csv",
            "REQUIRE_DATASET_FOR_READINESS": "false",
            "MODEL_ARTIFACT_SIZE": "2500",
            "REQUEST_TIMEOUT_MS": "4500",
            "CORS_ALLOW_ORIGINS": "http://localhost:8081,https://app.example.com",
        }
    )

    assert settings.project_root == tmp_path
    assert settings.model_dir == tmp_path / "local-models"
    assert settings.production_manifest_path == tmp_path / "local-models" / "manifest.json"
    assert settings.dataset_path == tmp_path / "data" / "scores.csv"
    assert settings.api_port == 9010
    assert settings.auto_port is True
    assert settings.request_timeout_ms == 4500
    assert settings.cors_allow_origins == ("http://localhost:8081", "https://app.example.com")


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("DEFAULT_MODEL", "unknown", "DEFAULT_MODEL must be one of"),
        ("API_PORT", "not-a-port", "API_PORT must be an integer"),
        ("REQUEST_TIMEOUT_MS", "0", "REQUEST_TIMEOUT_MS must be at least 1"),
        ("CORS_ALLOW_ORIGINS", " , ", "CORS_ALLOW_ORIGINS must include"),
        ("MODEL_DIR", "", "MODEL_DIR must not be empty"),
    ],
)
def test_settings_from_environment_rejects_invalid_values(name: str, value: str, message: str):
    with pytest.raises(SettingsConfigurationError, match=message):
        Settings.from_environment({name: value})


def test_resolve_port_uses_configured_port_when_available(monkeypatch: pytest.MonkeyPatch):
    settings = Settings(api_host="127.0.0.1", api_port=8001, auto_port=False)
    monkeypatch.setattr(main, "is_port_available", lambda host, port: True)
    assert main.resolve_port(settings) == 8001


def test_resolve_port_fails_fast_when_auto_port_disabled(monkeypatch: pytest.MonkeyPatch):
    settings = Settings(api_host="127.0.0.1", api_port=8001, auto_port=False)
    monkeypatch.setattr(main, "is_port_available", lambda host, port: False)

    with pytest.raises(SystemExit) as exc_info:
        main.resolve_port(settings)

    assert exc_info.value.code == 1


def test_resolve_port_finds_next_available_port(monkeypatch: pytest.MonkeyPatch):
    settings = Settings(api_host="127.0.0.1", api_port=8001, auto_port=True)

    def fake_is_port_available(host: str, port: int) -> bool:
        return port == 8003

    monkeypatch.setattr(main, "is_port_available", fake_is_port_available)
    assert main.resolve_port(settings) == 8003


def test_resolve_port_fails_when_no_port_found(monkeypatch: pytest.MonkeyPatch):
    settings = Settings(api_host="127.0.0.1", api_port=8001, auto_port=True)
    monkeypatch.setattr(main, "is_port_available", lambda host, port: False)

    with pytest.raises(SystemExit) as exc_info:
        main.resolve_port(settings, max_attempts=2)

    assert exc_info.value.code == 1
