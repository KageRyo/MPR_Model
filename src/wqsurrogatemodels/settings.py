from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

from .enums import ModelTypeEnum

load_dotenv()


class SettingsConfigurationError(ValueError):
    """Raised when a supported environment variable has an invalid value."""


def parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise SettingsConfigurationError(f"{name} must be a boolean value.")


def parse_integer_env(name: str, default: int, *, minimum: int = 0, env: Mapping[str, str] | None = None) -> int:
    value = (env or os.environ).get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SettingsConfigurationError(f"{name} must be an integer.") from exc
    if parsed < minimum:
        raise SettingsConfigurationError(f"{name} must be at least {minimum}.")
    return parsed


def parse_model_type_env(env: Mapping[str, str]) -> ModelTypeEnum:
    value = env.get("DEFAULT_MODEL", ModelTypeEnum.DIRECT_WQI5.value)
    try:
        return ModelTypeEnum(value)
    except ValueError as exc:
        supported = ", ".join(model.value for model in ModelTypeEnum)
        raise SettingsConfigurationError(f"DEFAULT_MODEL must be one of: {supported}.") from exc


def parse_cors_origins_env(env: Mapping[str, str]) -> tuple[str, ...]:
    value = env.get("CORS_ALLOW_ORIGINS", "*")
    origins = tuple(origin.strip() for origin in value.split(",") if origin.strip())
    if not origins:
        raise SettingsConfigurationError("CORS_ALLOW_ORIGINS must include at least one origin.")
    return origins


def resolve_env_path(value: str, base_directory: Path, name: str) -> Path:
    if not value.strip():
        raise SettingsConfigurationError(f"{name} must not be empty.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_directory / path).resolve()


@dataclass(frozen=True)
class Settings:
    project_root: Path = field(default_factory=lambda: Path.cwd().resolve())
    data_dir: Path | None = None
    model_dir: Path | None = None
    production_manifest_path: Path | None = None
    default_model: ModelTypeEnum | str = ModelTypeEnum.DIRECT_WQI5
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    auto_port: bool = False
    dataset_path: Path | None = None
    require_dataset_for_readiness: bool = False
    preferred_artifact_size: str = "50000"
    request_timeout_ms: int = 10000
    cors_allow_origins: tuple[str, ...] = ("*",)

    def __post_init__(self) -> None:
        root = self.project_root.expanduser().resolve()
        data_dir = (self.data_dir or root / "data").expanduser().resolve()
        model_dir = (self.model_dir or root / "models").expanduser().resolve()
        manifest_path = (self.production_manifest_path or model_dir / "production_model_manifest.json").expanduser().resolve()
        dataset_path = (self.dataset_path or data_dir / "dataV1.csv").expanduser().resolve()
        object.__setattr__(self, "project_root", root)
        object.__setattr__(self, "data_dir", data_dir)
        object.__setattr__(self, "model_dir", model_dir)
        object.__setattr__(self, "production_manifest_path", manifest_path)
        object.__setattr__(self, "dataset_path", dataset_path)

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if env is None else env
        root = resolve_env_path(values.get("PROJECT_ROOT", str(Path.cwd())), Path.cwd(), "PROJECT_ROOT")
        data_dir = root / "data"
        model_dir = resolve_env_path(values.get("MODEL_DIR", "models"), root, "MODEL_DIR")
        manifest_path = resolve_env_path(
            values.get("PRODUCTION_MODEL_MANIFEST", "production_model_manifest.json"),
            model_dir,
            "PRODUCTION_MODEL_MANIFEST",
        )
        dataset_path = resolve_env_path(values.get("DATASET_FILE", "dataV1.csv"), data_dir, "DATASET_FILE")
        return cls(
            project_root=root,
            data_dir=data_dir,
            model_dir=model_dir,
            production_manifest_path=manifest_path,
            default_model=parse_model_type_env(values),
            api_host=values.get("API_HOST", "0.0.0.0"),
            api_port=parse_integer_env("API_PORT", 8001, minimum=1, env=values),
            auto_port=parse_bool_env("AUTO_PORT", default=False) if env is None else _parse_bool_value(values, "AUTO_PORT", False),
            dataset_path=dataset_path,
            require_dataset_for_readiness=(
                parse_bool_env("REQUIRE_DATASET_FOR_READINESS", default=False)
                if env is None
                else _parse_bool_value(values, "REQUIRE_DATASET_FOR_READINESS", False)
            ),
            preferred_artifact_size=values.get("MODEL_ARTIFACT_SIZE", "50000"),
            request_timeout_ms=parse_integer_env("REQUEST_TIMEOUT_MS", 10000, minimum=1, env=values),
            cors_allow_origins=parse_cors_origins_env(values),
        )


def _parse_bool_value(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise SettingsConfigurationError(f"{name} must be a boolean value.")


FEATURE_COLUMNS = ["DO", "BOD", "NH3N", "EC", "SS"]

MODEL_DIR_NAMES: dict[ModelTypeEnum, str] = {
    ModelTypeEnum.LR: "LR",
    ModelTypeEnum.MPR: "MPR",
    ModelTypeEnum.SVM: "SVM",
    ModelTypeEnum.RF: "RF",
    ModelTypeEnum.XGBOOST: "XGBoost",
    ModelTypeEnum.LIGHTGBM: "LightGBM",
}

CATEGORY_BANDS = [
    ("Excellent", 85.0, 100.0, "85 < WQI5 ≤ 100"),
    ("Good", 70.0, 85.0, "70 < WQI5 ≤ 85"),
    ("Fair", 50.0, 70.0, "50 < WQI5 ≤ 70"),
    ("Poor", 30.0, 50.0, "30 < WQI5 ≤ 50"),
    ("Bad", 15.0, 30.0, "15 < WQI5 ≤ 30"),
    ("Terrible", 0.0, 15.0, "0 ≤ WQI5 ≤ 15"),
]
