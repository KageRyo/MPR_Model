from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    INVALID_ASSESSMENT_INPUT = "invalid_assessment_input"
    INVALID_CSV = "invalid_csv"
    MODEL_UNAVAILABLE = "model_unavailable"
    DATASET_UNAVAILABLE = "dataset_unavailable"
    INVALID_CONFIGURATION = "invalid_configuration"
    INTERNAL_ERROR = "internal_error"


class ApplicationError(Exception):
    """A client-safe application failure with a stable HTTP error code."""

    def __init__(self, status_code: int, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
