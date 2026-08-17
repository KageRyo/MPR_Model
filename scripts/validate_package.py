from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


FORBIDDEN_DIRECTORIES = {"configs", "data", "docs", "models", "results", "scripts", "statistics", "tests"}
FORBIDDEN_SUFFIXES = (".csv", ".joblib", ".pkl", ".png", ".xlsx")


def archive_names(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return archive.getnames()
    raise ValueError(f"Unsupported distribution archive: {path}")


def validate_names(path: Path, names: list[str]) -> None:
    for name in names:
        parts = set(Path(name).parts)
        if parts & FORBIDDEN_DIRECTORIES:
            raise SystemExit(f"Forbidden project material included in {path.name}: {name}")
        if name.lower().endswith(FORBIDDEN_SUFFIXES):
            raise SystemExit(f"Forbidden artifact included in {path.name}: {name}")
        allowed_sdist_metadata = path.name.endswith(".tar.gz") and (
            name.endswith(".egg-info") or name.endswith(".egg-info/SOURCES.txt")
        )
        if ".egg-info" in name and not allowed_sdist_metadata:
            raise SystemExit(f"Build metadata directory included in {path.name}: {name}")

    if path.suffix == ".whl":
        if not any(name.startswith("wqsurrogatemodels/") for name in names):
            raise SystemExit(f"Runtime package is missing from {path.name}")
        if any(name.startswith("src/") for name in names):
            raise SystemExit(f"Source-layout paths leaked into wheel: {path.name}")
    elif not any("/src/wqsurrogatemodels/" in f"/{name}" for name in names):
        raise SystemExit(f"Runtime package is missing from {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the public distribution boundary.")
    parser.add_argument("dist_dir", type=Path, nargs="?", default=Path("dist"))
    args = parser.parse_args()

    wheels = sorted(args.dist_dir.glob("*.whl"))
    sdists = sorted(args.dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("Expected exactly one wheel and one source distribution.")

    for archive in [*wheels, *sdists]:
        validate_names(archive, archive_names(archive))
        print(f"Validated {archive.name}")


if __name__ == "__main__":
    main()
