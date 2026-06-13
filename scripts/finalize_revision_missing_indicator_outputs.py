from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_revision_missing_indicator_experiments import (  # noqa: E402
    FULL_DROPOUT,
    FULL_REFERENCE,
    REDUCED_RETRAINING,
    TWO_STAGE,
    append_stress_rows,
    bootstrap_ci_from_metric_rows,
    extract_external_set,
    paired_tests_from_metric_rows,
    stress_frames,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite-derived", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_can_write(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists() and path.stat().st_size > 0]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Derived output already exists: {joined}. Pass --overwrite-derived to replace it.")


def resolve_config_path(args_config: str | None, manifest: dict[str, Any]) -> Path:
    config_value = args_config or manifest["config_path"]
    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return config_path


def load_external_frame(config: dict[str, Any]) -> pd.DataFrame:
    subset = pd.read_csv(PROJECT_ROOT / config["dataset_50000"])
    full = pd.read_csv(PROJECT_ROOT / config["full_dataset"])
    return extract_external_set(
        full,
        subset,
        config.get("expected_external_rows"),
        config.get("external_max_rows"),
    )


def recompute_stress_rows(output_dir: Path, config: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    stress_config = config.get("stress_test", {})
    if not bool(stress_config.get("enabled", True)):
        return []

    external = load_external_frame(config)
    stress_inputs = stress_frames(external, stress_config.get("scenarios", {}))
    rows: list[dict[str, Any]] = []
    for seed in manifest["seeds"]:
        for model_type in manifest["models"]:
            full_model_path = output_dir / "models" / f"seed_{seed}" / "full_model" / f"{model_type}.joblib"
            reduced_model_path = output_dir / "models" / f"seed_{seed}" / REDUCED_RETRAINING / f"{model_type}.joblib"
            two_stage_model_path = output_dir / "models" / f"seed_{seed}" / TWO_STAGE / f"{model_type}.joblib"
            full_bundle = joblib.load(full_model_path)
            reduced_bundle = joblib.load(reduced_model_path)
            two_stage_bundle = joblib.load(two_stage_model_path)
            bundles = {
                FULL_REFERENCE: full_bundle,
                FULL_DROPOUT: full_bundle,
                REDUCED_RETRAINING: reduced_bundle,
                TWO_STAGE: two_stage_bundle,
            }
            for experiment, bundle in bundles.items():
                for scenario_name, scenario_frame in stress_inputs.items():
                    append_stress_rows(
                        rows,
                        experiment=experiment,
                        seed=int(seed),
                        model_type=model_type,
                        baseline_frame=external,
                        scenario_name=scenario_name,
                        scenario_frame=scenario_frame,
                        model_bundle=bundle,
                    )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = resolve_config_path(args.config, manifest)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    output_paths = [
        output_dir / "stats" / "bootstrap_ci.csv",
        output_dir / "stats" / "paired_error_tests.csv",
        output_dir / "stress_tests" / "stress_summary.csv",
    ]
    ensure_can_write(output_paths, args.overwrite_derived)

    metric_rows = read_csv_rows(output_dir / "metrics" / "metrics_by_seed.csv")
    n_bootstrap = int(config.get("n_bootstrap", 200))
    write_csv(output_dir / "stats" / "bootstrap_ci.csv", bootstrap_ci_from_metric_rows(metric_rows, n_bootstrap))
    write_csv(output_dir / "stats" / "paired_error_tests.csv", paired_tests_from_metric_rows(metric_rows))
    write_csv(output_dir / "stress_tests" / "stress_summary.csv", recompute_stress_rows(output_dir, config, manifest))
    logger.success(f"finalized derived outputs in: {output_dir}")


if __name__ == "__main__":
    main()
