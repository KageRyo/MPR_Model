from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    model_dir: Path = project_root / os.getenv("MODEL_DIR", "models")
    default_model: str = os.getenv("DEFAULT_MODEL", "direct_wqi5")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8001"))
    dataset_path: Path = data_dir / os.getenv("DATASET_FILE", "dataV1.csv")
    preferred_artifact_size: str = os.getenv("MODEL_ARTIFACT_SIZE", "50000")
    request_timeout_ms: int = int(os.getenv("REQUEST_TIMEOUT_MS", "10000"))


FEATURE_COLUMNS = ["DO", "BOD", "NH3N", "EC", "SS"]

MODEL_DIR_NAMES = {
    "lr": "LR",
    "mpr": "MPR",
    "svm": "SVM",
    "rf": "RF",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

CATEGORY_BANDS = [
    ("Excellent", 85.0, 100.0, "85 < WQI5 ≤ 100"),
    ("Good", 70.0, 85.0, "70 < WQI5 ≤ 85"),
    ("Fair", 50.0, 70.0, "50 < WQI5 ≤ 70"),
    ("Poor", 30.0, 50.0, "30 < WQI5 ≤ 50"),
    ("Bad", 15.0, 30.0, "15 < WQI5 ≤ 30"),
    ("Terrible", 0.0, 15.0, "0 ≤ WQI5 ≤ 15"),
]
