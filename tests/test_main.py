from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from src.settings import Settings, parse_bool_env


def test_parse_bool_env_truthy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTO_PORT", "true")
    assert parse_bool_env("AUTO_PORT") is True


def test_parse_bool_env_falsey(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTO_PORT", "false")
    assert parse_bool_env("AUTO_PORT") is False


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
