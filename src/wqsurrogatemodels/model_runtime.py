from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path

from .artifacts import ProductionArtifact


class ArtifactRuntimeCompatibilityError(RuntimeError):
    """Raised when an installed runtime cannot safely load an artifact."""


_DISTRIBUTION_NAMES = {
    "scikit_learn": "scikit-learn",
}

_EXTRA_BY_RUNTIME_NAME = {
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
}

_PYTHON_REQUIREMENT = re.compile(r"^(>=|==)(\d+)\.(\d+)(?:\.(\d+))?$")
_EXACT_PACKAGE_VERSION = re.compile(r"^\d+(?:\.\d+)+$")


def _artifact_label(artifact: ProductionArtifact) -> str:
    return f"the {artifact.model_type.value} production artifact"


def _installation_hint(runtime_name: str, distribution: str, required_version: str) -> str:
    extra = _EXTRA_BY_RUNTIME_NAME.get(runtime_name)
    if extra:
        return (
            f"Install the matching optional dependency with `uv sync --locked --extra {extra}` "
            f"or `pip install \"wqsurrogatemodels[{extra}]\"`."
        )
    return f"Install `{distribution}=={required_version}` from the project's locked environment."


def _validate_python_requirement(artifact: ProductionArtifact, requirement: str) -> None:
    match = _PYTHON_REQUIREMENT.fullmatch(requirement)
    if match is None:
        raise ArtifactRuntimeCompatibilityError(
            f"{_artifact_label(artifact)} declares unsupported Python requirement {requirement!r}."
        )

    operator, major, minor, patch = match.groups()
    required = (int(major), int(minor), int(patch or 0))
    installed = sys.version_info[:3]
    valid = installed >= required if operator == ">=" else installed == required
    if not valid:
        installed_text = ".".join(str(part) for part in installed)
        raise ArtifactRuntimeCompatibilityError(
            f"{_artifact_label(artifact)} requires Python {requirement}, but the runtime is {installed_text}."
        )


def validate_artifact_runtime(artifact: ProductionArtifact, *, artifact_path: Path) -> None:
    """Require the exact runtime versions recorded with a production artifact.

    ``artifact_path`` deliberately remains an argument so callers validate the
    file selected from the manifest. It is not exposed in exceptions, keeping
    readiness responses free of local filesystem paths.
    """
    del artifact_path
    for runtime_name, requirement in artifact.runtime_compatibility.items():
        if runtime_name == "python":
            _validate_python_requirement(artifact, requirement)
            continue

        if not _EXACT_PACKAGE_VERSION.fullmatch(requirement):
            raise ArtifactRuntimeCompatibilityError(
                f"{_artifact_label(artifact)} declares a non-exact {runtime_name} version {requirement!r}."
            )

        distribution = _DISTRIBUTION_NAMES.get(runtime_name, runtime_name)
        try:
            installed_version = distribution_version(distribution)
        except PackageNotFoundError as exc:
            raise ArtifactRuntimeCompatibilityError(
                f"Cannot load {_artifact_label(artifact)}: {distribution}=={requirement} is not installed. "
                f"{_installation_hint(runtime_name, distribution, requirement)}"
            ) from exc
        if installed_version != requirement:
            raise ArtifactRuntimeCompatibilityError(
                f"Cannot load {_artifact_label(artifact)}: installed {distribution}=={installed_version}, "
                f"but it requires {distribution}=={requirement}. "
                f"{_installation_hint(runtime_name, distribution, requirement)}"
            )
