from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from pydantic import BaseModel, Field

logger.remove()
logger.add(sys.stderr, level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_missing_indicator_robustness_experiments import (  # noqa: E402
    COMPLETE_SET,
    FULL_REFERENCE,
    INDICATOR_RECONSTRUCTION,
    INFERENCE_DROPOUT,
    REDUCED_RETRAINING,
    category_rank,
    experiment_name,
    predict_bundle,
    score_to_category,
)
from run_revision_missing_indicator_experiments import (  # noqa: E402
    clipped_features,
    extract_external_set,
    output_json,
    write_csv,
)
from src.settings import FEATURE_COLUMNS  # noqa: E402


class Stress107ScenarioSchema(BaseModel):
    DO_decrease_factors: dict[str, float] = Field(default_factory=dict)
    increase_indicators: dict[str, dict[str, float]] = Field(default_factory=dict)
    multipliers: dict[str, dict[str, float]] = Field(default_factory=dict)


class Stress107ConfigSchema(BaseModel):
    enabled: bool = True
    source: str = "external_10714"
    window_mode: str = "sequential_equal_blocks"
    n_windows: int = 107
    severities: dict[str, dict[str, Any]]
    scenarios: dict[str, Stress107ScenarioSchema]


class Stress107ManifestSchema(BaseModel):
    created_at_utc: str
    artifact_dir: str
    output_dir: str
    config_path: str
    dataset_50000: str
    full_dataset: str
    external_rows: int
    n_windows: int
    window_mode: str
    severities: list[str]
    scenarios: list[str]
    seeds: list[int]
    models: list[str]
    missing_sets: dict[str, list[str]]
    experiment_modes: list[str]
    source: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, help="Result directory containing saved robustness model artifacts.")
    parser.add_argument("--output-dir", required=True, help="New output directory for Stress107 CSV and report files.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_schema(model_cls, data: dict[str, Any]):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_output_dir(path: Path, overwrite: bool) -> Path:
    if path.exists():
        existing_files = [item for item in path.rglob("*") if item.is_file()]
        if existing_files and not overwrite:
            raise FileExistsError(f"Output directory already contains files: {path}. Use --overwrite or choose a new directory.")
        if overwrite:
            shutil.rmtree(path)
    for child in ["stress_tests", "reports"]:
        (path / child).mkdir(parents=True, exist_ok=True)
    return path


def model_path(artifact_dir: Path, seed: int, missing_set: str, mode: str, model_type: str) -> Path:
    if mode in {FULL_REFERENCE, INFERENCE_DROPOUT}:
        return artifact_dir / "models" / f"seed_{seed}" / FULL_REFERENCE / f"{model_type}.joblib"
    return artifact_dir / "models" / f"seed_{seed}" / mode / missing_set / f"{model_type}.joblib"


def sequential_windows(n_rows: int, n_windows: int) -> list[tuple[int, int]]:
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1.")
    if n_windows > n_rows:
        raise ValueError(f"n_windows={n_windows} cannot exceed n_rows={n_rows}.")
    return [(int(indices[0]), int(indices[-1]) + 1) for indices in np.array_split(np.arange(n_rows), n_windows)]


def scenario_multipliers(scenario: Stress107ScenarioSchema, severity: str) -> dict[str, float]:
    multipliers: dict[str, float] = {}
    if severity in scenario.DO_decrease_factors:
        multipliers["DO"] = float(scenario.DO_decrease_factors[severity])
    for indicator, severity_map in scenario.increase_indicators.items():
        if severity in severity_map:
            multipliers[indicator] = float(severity_map[severity])
    for indicator, severity_map in scenario.multipliers.items():
        if severity in severity_map:
            multipliers[indicator] = float(severity_map[severity])
    unknown = set(multipliers) - set(FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown stress indicator(s): {sorted(unknown)}")
    return multipliers


def perturbed_frame(external: pd.DataFrame, multipliers: dict[str, float]) -> pd.DataFrame:
    modified = external[FEATURE_COLUMNS].copy().astype(float)
    for column, multiplier in multipliers.items():
        modified[column] = modified[column] * float(multiplier)
    return clipped_features(modified)


def append_window_rows(
    rows: list[dict[str, Any]],
    *,
    missing_set: str,
    mode: str,
    seed: int,
    model_type: str,
    scenario_name: str,
    severity_name: str,
    severity_rank: int,
    perturbation_pct: float | None,
    multipliers: dict[str, float],
    windows: list[tuple[int, int]],
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    stress_pred: np.ndarray,
) -> None:
    baseline_cat = score_to_category(baseline_pred)
    stress_cat = score_to_category(stress_pred)
    actual_cat = score_to_category(y_true)
    multiplier_text = json.dumps(multipliers, sort_keys=True)
    for window_id, (start, end) in enumerate(windows):
        idx = np.arange(start, end)
        deltas = stress_pred[idx] - baseline_pred[idx]
        drops = baseline_pred[idx] - stress_pred[idx]
        worse = [category_rank(stress_cat[i]) < category_rank(baseline_cat[i]) for i in idx]
        rows.append(
            {
                "source": "external_10714",
                "missing_set": missing_set,
                "experiment_mode": mode,
                "experiment": experiment_name(mode, missing_set),
                "seed": seed,
                "model_type": model_type,
                "stress_scenario": scenario_name,
                "severity": severity_name,
                "severity_rank": severity_rank,
                "perturbation_pct": perturbation_pct,
                "multipliers": multiplier_text,
                "window_id": window_id,
                "start_index": start,
                "end_index_exclusive": end,
                "n": int(end - start),
                "actual_mean_score": float(np.mean(y_true[idx])),
                "baseline_mean_score": float(np.mean(baseline_pred[idx])),
                "stress_mean_score": float(np.mean(stress_pred[idx])),
                "mean_delta_stress_minus_baseline": float(np.mean(deltas)),
                "median_delta_stress_minus_baseline": float(np.median(deltas)),
                "mean_score_drop_baseline_minus_stress": float(np.mean(drops)),
                "median_score_drop_baseline_minus_stress": float(np.median(drops)),
                "pct_score_decreased": float(np.mean(stress_pred[idx] < baseline_pred[idx])),
                "pct_drop_ge_1": float(np.mean(drops >= 1.0)),
                "pct_drop_ge_5": float(np.mean(drops >= 5.0)),
                "pct_category_worse": float(np.mean(worse)),
                "actual_categories": ",".join(sorted(set(actual_cat[i] for i in idx))),
                "baseline_categories": ",".join(sorted(set(baseline_cat[i] for i in idx))),
                "stress_categories": ",".join(sorted(set(stress_cat[i] for i in idx))),
                "detected_mean_decrease": bool(np.mean(deltas) < 0.0),
                "detected_mean_drop_ge_1": bool(np.mean(drops) >= 1.0),
                "detected_any_category_worse": bool(np.mean(worse) > 0.0),
            }
        )


def detection_summary(window_frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "source",
        "missing_set",
        "experiment_mode",
        "experiment",
        "model_type",
        "stress_scenario",
        "severity",
        "severity_rank",
        "perturbation_pct",
    ]
    return (
        window_frame.groupby(group_columns, as_index=False)
        .agg(
            n_seed_window_cases=("window_id", "count"),
            n_rows_total=("n", "sum"),
            mean_delta_score=("mean_delta_stress_minus_baseline", "mean"),
            median_delta_score=("median_delta_stress_minus_baseline", "median"),
            mean_score_drop=("mean_score_drop_baseline_minus_stress", "mean"),
            median_score_drop=("median_score_drop_baseline_minus_stress", "median"),
            mean_pct_score_decreased=("pct_score_decreased", "mean"),
            mean_pct_drop_ge_1=("pct_drop_ge_1", "mean"),
            mean_pct_drop_ge_5=("pct_drop_ge_5", "mean"),
            mean_pct_category_worse=("pct_category_worse", "mean"),
            window_detection_rate_mean_decrease=("detected_mean_decrease", "mean"),
            window_detection_rate_drop_ge_1=("detected_mean_drop_ge_1", "mean"),
            window_detection_rate_any_category_worse=("detected_any_category_worse", "mean"),
        )
        .sort_values(group_columns)
    )


def severity_monotonicity(window_frame: pd.DataFrame, severity_order: list[str]) -> pd.DataFrame:
    group_columns = [
        "source",
        "missing_set",
        "experiment_mode",
        "experiment",
        "model_type",
        "stress_scenario",
        "seed",
        "window_id",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in window_frame.groupby(group_columns, sort=True):
        if not set(severity_order).issubset(set(group["severity"])):
            continue
        ordered = group.set_index("severity").loc[severity_order]
        deltas = ordered["mean_delta_stress_minus_baseline"].to_numpy(dtype=float)
        drops = ordered["mean_score_drop_baseline_minus_stress"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "monotonic_more_negative_delta": bool(np.all(np.diff(deltas) <= 1e-12)),
                "monotonic_larger_score_drop": bool(np.all(np.diff(drops) >= -1e-12)),
                **{f"mean_delta_{severity}": float(value) for severity, value in zip(severity_order, deltas)},
                **{f"mean_score_drop_{severity}": float(value) for severity, value in zip(severity_order, drops)},
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    summary_group_columns = [
        "source",
        "missing_set",
        "experiment_mode",
        "experiment",
        "model_type",
        "stress_scenario",
    ]
    aggregation = {
        "n_seed_window_cases": ("window_id", "count"),
        "severity_monotonicity_rate_delta": ("monotonic_more_negative_delta", "mean"),
        "severity_monotonicity_rate_drop": ("monotonic_larger_score_drop", "mean"),
    }
    for severity in severity_order:
        aggregation[f"mean_delta_{severity}"] = (f"mean_delta_{severity}", "mean")
        aggregation[f"mean_score_drop_{severity}"] = (f"mean_score_drop_{severity}", "mean")
    return detail.groupby(summary_group_columns, as_index=False).agg(**aggregation).sort_values(summary_group_columns)


def setting_rows(config: Stress107ConfigSchema, windows: list[tuple[int, int]]) -> list[dict[str, Any]]:
    rows = [
        {"item": "source", "value": config.source},
        {"item": "window_mode", "value": config.window_mode},
        {"item": "n_windows", "value": config.n_windows},
        {"item": "window_size_min", "value": min(end - start for start, end in windows)},
        {"item": "window_size_max", "value": max(end - start for start, end in windows)},
        {"item": "window_size_mean", "value": float(np.mean([end - start for start, end in windows]))},
        {"item": "interpretation", "value": "Controlled synthetic 107 sequential event-window stress test; not real event validation."},
        {"item": "detection_rule_mean_decrease", "value": "window mean stress prediction is lower than baseline prediction"},
        {"item": "detection_rule_drop_ge_1", "value": "window mean baseline-minus-stress score drop is at least 1 point"},
    ]
    for rank, (severity, spec) in enumerate(config.severities.items()):
        rows.append({"item": f"severity_{rank}", "value": f"{severity}: {json.dumps(spec, sort_keys=True)}"})
    for scenario, spec in config.scenarios.items():
        rows.append({"item": f"scenario_{scenario}", "value": spec.model_dump_json() if hasattr(spec, "model_dump_json") else spec.json()})
    return rows


def key_conclusions(
    detection: pd.DataFrame,
    monotonic: pd.DataFrame,
    *,
    n_windows: int,
    severities: list[str],
    scenarios: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "item": "scope",
            "value": f"{n_windows} consecutive event windows, {len(scenarios)} scenarios, {len(severities)} severity levels.",
        },
        {
            "item": "wording",
            "value": "Use '107 sequential event-window stress test', not '107-fold cross-validation'.",
        },
        {
            "item": "claim_boundary",
            "value": "This reduces dependence on a single selected window; it does not prove absence of all sampling bias.",
        },
    ]
    if not detection.empty:
        rows.extend(
            [
                {
                    "item": "min_detection_rate_mean_decrease",
                    "value": float(detection["window_detection_rate_mean_decrease"].min()),
                },
                {
                    "item": "mean_detection_rate_mean_decrease",
                    "value": float(detection["window_detection_rate_mean_decrease"].mean()),
                },
                {
                    "item": "mean_detection_rate_drop_ge_1",
                    "value": float(detection["window_detection_rate_drop_ge_1"].mean()),
                },
            ]
        )
    if not monotonic.empty:
        rows.extend(
            [
                {
                    "item": "mean_severity_monotonicity_rate_drop",
                    "value": float(monotonic["severity_monotonicity_rate_drop"].mean()),
                },
                {
                    "item": "min_severity_monotonicity_rate_drop",
                    "value": float(monotonic["severity_monotonicity_rate_drop"].min()),
                },
            ]
        )
    return rows


def main() -> None:
    args = parse_args()
    artifact_dir = resolve_path(args.artifact_dir).resolve()
    output_dir = resolve_output_dir(resolve_path(args.output_dir).resolve(), args.overwrite)
    source_manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    config_path = resolve_path(args.config or source_manifest["config_path"])
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    stress_config = validate_schema(Stress107ConfigSchema, config["stress107_event_windows"])
    if not stress_config.enabled:
        raise ValueError("stress107_event_windows.enabled is false.")
    if stress_config.window_mode != "sequential_equal_blocks":
        raise ValueError("Only sequential_equal_blocks is supported for Stress107.")

    subset = pd.read_csv(PROJECT_ROOT / config["dataset_50000"])
    full = pd.read_csv(PROJECT_ROOT / config["full_dataset"])
    external = extract_external_set(full, subset, config.get("expected_external_rows"), config.get("external_max_rows"))
    windows = sequential_windows(len(external), stress_config.n_windows)
    severity_order = list(stress_config.severities.keys())
    seeds = args.seeds or list(source_manifest["seeds"])
    models = args.models or list(source_manifest["models"])
    missing_sets = dict(source_manifest["missing_sets"])
    experiments: list[tuple[str, str]] = [(COMPLETE_SET, FULL_REFERENCE)]
    for missing_set in missing_sets:
        experiments.extend(
            [
                (missing_set, INFERENCE_DROPOUT),
                (missing_set, REDUCED_RETRAINING),
                (missing_set, INDICATOR_RECONSTRUCTION),
            ]
        )

    logger.info(
        "Stress107 artifact_dir={} output_dir={} rows={} windows={} scenarios={} severities={} seeds={} models={}",
        artifact_dir,
        output_dir,
        len(external),
        stress_config.n_windows,
        len(stress_config.scenarios),
        len(severity_order),
        seeds,
        models,
    )

    rows: list[dict[str, Any]] = []
    y_true = external["Score"].to_numpy()
    for seed in seeds:
        for model_type in models:
            for missing_set, mode in experiments:
                path = model_path(artifact_dir, int(seed), missing_set, mode, model_type)
                logger.info("stress107 seed={} model={} experiment={}", seed, model_type, experiment_name(mode, missing_set))
                bundle = joblib.load(path)
                started = time.perf_counter()
                baseline_pred = predict_bundle(mode, bundle, external)
                logger.debug("baseline prediction completed in {:.3f}s", time.perf_counter() - started)
                for scenario_name, scenario in stress_config.scenarios.items():
                    for severity_rank, severity in enumerate(severity_order):
                        multipliers = scenario_multipliers(scenario, severity)
                        scenario_frame = perturbed_frame(external, multipliers)
                        started = time.perf_counter()
                        stress_pred = predict_bundle(mode, bundle, scenario_frame)
                        logger.debug(
                            "stress prediction scenario={} severity={} completed in {:.3f}s",
                            scenario_name,
                            severity,
                            time.perf_counter() - started,
                        )
                        append_window_rows(
                            rows,
                            missing_set=missing_set,
                            mode=mode,
                            seed=int(seed),
                            model_type=model_type,
                            scenario_name=scenario_name,
                            severity_name=severity,
                            severity_rank=severity_rank,
                            perturbation_pct=stress_config.severities[severity].get("perturbation_pct"),
                            multipliers=multipliers,
                            windows=windows,
                            y_true=y_true,
                            baseline_pred=baseline_pred,
                            stress_pred=stress_pred,
                        )

    window_frame = pd.DataFrame(rows)
    detection = detection_summary(window_frame)
    monotonic = severity_monotonicity(window_frame, severity_order)
    write_csv(output_dir / "stress_tests" / "stress107_setting.csv", setting_rows(stress_config, windows))
    window_frame.to_csv(output_dir / "stress_tests" / "stress107_window_summary.csv", index=False)
    detection.to_csv(output_dir / "stress_tests" / "stress107_detection_summary.csv", index=False)
    monotonic.to_csv(output_dir / "stress_tests" / "stress107_severity_monotonicity.csv", index=False)
    write_csv(
        output_dir / "stress_tests" / "stress107_key_conclusions.csv",
        key_conclusions(
            detection,
            monotonic,
            n_windows=stress_config.n_windows,
            severities=severity_order,
            scenarios=list(stress_config.scenarios),
        ),
    )
    output_json(
        output_dir / "manifest.json",
        Stress107ManifestSchema(
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            artifact_dir=str(artifact_dir),
            output_dir=str(output_dir),
            config_path=str(config_path),
            dataset_50000=config["dataset_50000"],
            full_dataset=config["full_dataset"],
            external_rows=len(external),
            n_windows=stress_config.n_windows,
            window_mode=stress_config.window_mode,
            severities=severity_order,
            scenarios=list(stress_config.scenarios),
            seeds=[int(seed) for seed in seeds],
            models=list(models),
            missing_sets=missing_sets,
            experiment_modes=[FULL_REFERENCE, INFERENCE_DROPOUT, REDUCED_RETRAINING, INDICATOR_RECONSTRUCTION],
            source=stress_config.source,
            note="Controlled synthetic Stress107 sequential event-window stress test; not real event validation.",
        ),
    )
    logger.success("Stress107 outputs written to: {}", output_dir)


if __name__ == "__main__":
    main()
