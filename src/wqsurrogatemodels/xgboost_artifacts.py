from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .settings import FEATURE_COLUMNS


LEGACY_PICKLE_WARNING_FRAGMENT = "If you are loading a serialized model"


class XGBoostArtifactRehydrationError(RuntimeError):
    """Raised when an XGBoost artifact cannot be safely rehydrated."""


@dataclass(frozen=True)
class XGBoostArtifactRehydrationResult:
    source: Path
    target: Path
    source_legacy_warning_detected: bool
    verification_rows: int


def load_xgboost_pickle(path: Path) -> tuple[Any, bool]:
    """Load an XGBoost pickle without emitting its verbose legacy warning.

    Only XGBoost's known serialized-Booster warning is intercepted. Other
    warnings preserve their usual handling. Callers can record the returned
    flag and offer a rehydration path instead of hiding the compatibility risk.
    """
    legacy_warning_detected = False
    with warnings.catch_warnings():
        original_showwarning = warnings.showwarning

        def showwarning(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: Any | None = None,
            line: str | None = None,
        ) -> None:
            nonlocal legacy_warning_detected
            if issubclass(category, UserWarning) and LEGACY_PICKLE_WARNING_FRAGMENT in str(message):
                legacy_warning_detected = True
                return
            original_showwarning(message, category, filename, lineno, file, line)

        warnings.showwarning = showwarning
        warnings.simplefilter("always", UserWarning)
        model = joblib.load(path)
    return model, legacy_warning_detected


def default_verification_frame() -> pd.DataFrame:
    """Return representative complete-input records for a local migration check."""
    return pd.DataFrame(
        [
            {"DO": 7.2, "BOD": 2.1, "NH3N": 0.3, "EC": 450.0, "SS": 12.0},
            {"DO": 96.2, "BOD": 1.5, "NH3N": 0.22, "EC": 171.0, "SS": 2.6},
            {"DO": 55.0, "BOD": 8.4, "NH3N": 1.1, "EC": 900.0, "SS": 48.0},
        ],
        columns=FEATURE_COLUMNS,
    )


def rehydrate_xgboost_pickle(
    source: Path,
    target: Path,
    *,
    verification_frame: pd.DataFrame | None = None,
) -> XGBoostArtifactRehydrationResult:
    """Create a new, warning-free XGBoost pickle without overwriting its source.

    The candidate is reloaded with the pinned runtime and checked for prediction
    parity before it is moved into place. It is intentionally not promoted into
    the production manifest; that requires a separate review and artifact
    checksum update.
    """
    source = source.resolve()
    target = target.resolve()
    if source == target:
        raise XGBoostArtifactRehydrationError("The target must differ from the source artifact.")
    if not source.is_file():
        raise XGBoostArtifactRehydrationError(f"Source artifact does not exist: {source}")
    if target.exists():
        raise XGBoostArtifactRehydrationError(f"Refusing to overwrite existing target: {target}")

    frame = verification_frame.copy() if verification_frame is not None else default_verification_frame()
    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise XGBoostArtifactRehydrationError(
            f"Verification records are missing required columns: {', '.join(missing_columns)}"
        )
    frame = frame[FEATURE_COLUMNS]

    source_model, source_legacy_warning_detected = load_xgboost_pickle(source)
    source_predictions = np.asarray(source_model.predict(frame), dtype=float)

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        joblib.dump(source_model, temporary_path)
        rehydrated_model, target_legacy_warning_detected = load_xgboost_pickle(temporary_path)
        if target_legacy_warning_detected:
            raise XGBoostArtifactRehydrationError(
                "The rehydrated candidate still triggers XGBoost's legacy pickle warning."
            )
        target_predictions = np.asarray(rehydrated_model.predict(frame), dtype=float)
        np.testing.assert_allclose(source_predictions, target_predictions, rtol=0.0, atol=1e-12)
        os.replace(temporary_path, target)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        if isinstance(exc, XGBoostArtifactRehydrationError):
            raise
        raise XGBoostArtifactRehydrationError(
            f"Could not rehydrate {source} into {target}: {exc}"
        ) from exc

    return XGBoostArtifactRehydrationResult(
        source=source,
        target=target,
        source_legacy_warning_detected=source_legacy_warning_detected,
        verification_rows=len(frame),
    )
