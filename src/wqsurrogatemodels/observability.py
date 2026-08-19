from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from fastapi import Request


REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_LOGGER_NAME = "wqsurrogatemodels.request"
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


class JsonLogFormatter(logging.Formatter):
    """Format bounded application fields for stdout/container log collection."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        for field in (
            "request_id",
            "method",
            "route",
            "status_code",
            "latency_ms",
            "model_type",
            "error_code",
        ):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def configure_request_logger() -> logging.Logger:
    logger = logging.getLogger(REQUEST_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(getattr(handler, "_wq_structured", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler._wq_structured = True  # type: ignore[attr-defined]
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)
    return logger


request_logger = configure_request_logger()


def bind_request_id() -> tuple[str, Any]:
    request_id = str(uuid.uuid4())
    return request_id, _request_id.set(request_id)


def reset_request_id(token: Any) -> None:
    _request_id.reset(token)


def route_template(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


def log_event(event: str, **fields: Any) -> None:
    request_id = _request_id.get()
    if request_id is not None:
        fields["request_id"] = request_id
    request_logger.info(event, extra=fields)
