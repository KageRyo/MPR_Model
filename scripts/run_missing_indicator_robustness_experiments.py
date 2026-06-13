from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
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
from pydantic import BaseModel
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reproduce_results import build_model, require_model_support, resolve_compute_device  # noqa: E402
from run_revision_missing_indicator_experiments import (  # noqa: E402
    CATEGORY_ORDER,
    METRICS_FOR_CI,
    clipped_features,
    collect_versions,
    command_output,
    display_path,
    extract_external_set,
    get_cpu_name,
    get_memory_total_kb,
    output_json,
    write_csv,
)
from src.settings import FEATURE_COLUMNS  # noqa: E402
from src.wqi import categorize_score  # noqa: E402


FULL_REFERENCE = "full_reference"
INFERENCE_DROPOUT = "inference_dropout"
REDUCED_RETRAINING = "reduced_retraining"
INDICATOR_RECONSTRUCTION = "indicator_reconstruction"
COMPLETE_SET = "complete"
SOURCES = ["internal_test", "external_10714"]


class ManifestSchema(BaseModel):
    created_at_utc: str
    config_path: str
    output_dir: str
    dataset_50000: str
    full_dataset: str
    train_rows_per_seed: int
    internal_test_rows_per_seed: int
    external_rows: int
    seeds: list[int]
    models: list[str]
    missing_sets: dict[str, list[str]]
    experiment_modes: list[str]
    feature_columns: list[str]
    compute_device: str
    gpu_id: int
    lightgbm_gpu_backend: str


class HardwareSchema(BaseModel):
    platform: str
    python: str
    cpu: str | None = None
    memory_total_kb: int | None = None
    nvidia_smi: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/missing_indicator_robustness_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--compute-device", choices=["cpu", "gpu", "auto"], default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--lightgbm-gpu-backend", choices=["gpu", "cuda"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_output_dir(path: Path, overwrite: bool) -> Path:
    if path.exists():
        existing_files = [item for item in path.rglob("*") if item.is_file()]
        if existing_files and not overwrite:
            raise FileExistsError(f"Output directory already contains files: {path}. Use --output-dir or --overwrite.")
        if overwrite:
            shutil.rmtree(path)
    for child in ["models", "predictions", "metrics", "stats", "stress_tests", "timing", "reports", "splits"]:
        (path / child).mkdir(parents=True, exist_ok=True)
    return path


def score_to_category(values: np.ndarray) -> list[str]:
    return [categorize_score(float(value))[0] for value in values]


def category_rank(label: str) -> int:
    return CATEGORY_ORDER.index(label) if label in CATEGORY_ORDER else -1


def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.max(y_true) - np.min(y_true)
    return float(mean_absolute_error(y_true, y_pred) / denom) if denom else 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    actual_cat = score_to_category(y_true)
    predicted_cat = score_to_category(y_pred)
    residual = y_true - y_pred
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "nmae": nmae(y_true, y_pred),
        "accuracy": float(accuracy_score(actual_cat, predicted_cat)),
        "macro_f1": float(f1_score(actual_cat, predicted_cat, average="macro")),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
    }


def train_model(
    model_type: str,
    features: list[str],
    frame: pd.DataFrame,
    target: np.ndarray,
    compute_device: str,
    gpu_id: int,
    lightgbm_gpu_backend: str,
):
    require_model_support(model_type)
    estimator = build_model(
        model_type,
        compute_device=compute_device,
        gpu_id=gpu_id,
        lightgbm_gpu_backend=lightgbm_gpu_backend,
    )
    estimator.fit(frame[features], target)
    return estimator


def available_features(missing_indicators: list[str]) -> list[str]:
    return [column for column in FEATURE_COLUMNS if column not in set(missing_indicators)]


def experiment_name(mode: str, missing_set: str) -> str:
    return FULL_REFERENCE if mode == FULL_REFERENCE else f"{mode}_{missing_set}"


def predict_bundle(mode: str, bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    if mode == FULL_REFERENCE:
        return bundle["wqi_model"].predict(frame[FEATURE_COLUMNS])
    if mode == INFERENCE_DROPOUT:
        dropped = frame[FEATURE_COLUMNS].copy()
        for indicator in bundle["missing_indicators"]:
            dropped[indicator] = np.nan
        return bundle["wqi_model"].predict(dropped)
    if mode == REDUCED_RETRAINING:
        return bundle["wqi_model"].predict(frame[bundle["available_features"]])
    if mode == INDICATOR_RECONSTRUCTION:
        reconstructed = frame[FEATURE_COLUMNS].copy()
        for indicator in bundle["missing_indicators"]:
            reconstructed[indicator] = bundle["indicator_models"][indicator].predict(frame[bundle["available_features"]])
        return bundle["wqi_model"].predict(reconstructed[FEATURE_COLUMNS])
    raise ValueError(f"Unsupported mode: {mode}")


def append_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    missing_set: str,
    mode: str,
    seed: int,
    model_type: str,
    row_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    actual_cat = score_to_category(y_true)
    predicted_cat = score_to_category(y_pred)
    for row_id, actual, predicted, actual_label, predicted_label in zip(row_ids, y_true, y_pred, actual_cat, predicted_cat):
        rows.append(
            {
                "source": source,
                "missing_set": missing_set,
                "experiment_mode": mode,
                "experiment": experiment_name(mode, missing_set),
                "seed": seed,
                "model_type": model_type,
                "row_id": int(row_id),
                "actual": round(float(actual), 6),
                "predicted": round(float(predicted), 6),
                "abs_error": round(float(abs(actual - predicted)), 6),
                "residual": round(float(actual - predicted), 6),
                "actual_category": actual_label,
                "predicted_category": predicted_label,
            }
        )


def append_metric_row(
    rows: list[dict[str, Any]],
    *,
    source: str,
    missing_set: str,
    mode: str,
    seed: int,
    model_type: str,
    model_path: Path,
    latency_s: float,
    training_s: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    rows.append(
        {
            "source": source,
            "missing_set": missing_set,
            "experiment_mode": mode,
            "experiment": experiment_name(mode, missing_set),
            "seed": seed,
            "model_type": model_type,
            "model_path": display_path(model_path),
            "n": len(y_true),
            "latency_s": latency_s,
            "training_s": training_s,
            **prediction_metrics(y_true, y_pred),
        }
    )


def append_reconstruction_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    missing_set: str,
    seed: int,
    model_type: str,
    frame: pd.DataFrame,
    bundle: dict[str, Any],
) -> None:
    for indicator in bundle["missing_indicators"]:
        actual = frame[indicator].to_numpy()
        predicted = bundle["indicator_models"][indicator].predict(frame[bundle["available_features"]])
        rows.append(
            {
                "source": source,
                "missing_set": missing_set,
                "seed": seed,
                "model_type": model_type,
                "target_indicator": indicator,
                "n": len(actual),
                "r2": float(r2_score(actual, predicted)),
                "mae": float(mean_absolute_error(actual, predicted)),
                "rmse": rmse(actual, predicted),
            }
        )


def aggregate_metric_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(metric_rows)
    rows: list[dict[str, Any]] = []
    group_columns = ["source", "missing_set", "experiment_mode", "experiment", "model_type"]
    for keys, group in frame.groupby(group_columns, sort=True):
        source, missing_set, mode, experiment, model_type = keys
        rows.append(
            {
                "source": source,
                "missing_set": missing_set,
                "experiment_mode": mode,
                "experiment": experiment,
                "model_type": model_type,
                "n_runs": len(group),
                "n_per_run": int(group["n"].iloc[0]),
                "r2_mean": group["r2"].mean(),
                "r2_std": group["r2"].std(ddof=0),
                "mae_mean": group["mae"].mean(),
                "mae_std": group["mae"].std(ddof=0),
                "rmse_mean": group["rmse"].mean(),
                "rmse_std": group["rmse"].std(ddof=0),
                "nmae_mean": group["nmae"].mean(),
                "nmae_std": group["nmae"].std(ddof=0),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=0),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=0),
                "latency_s_mean": group["latency_s"].mean(),
                "latency_s_std": group["latency_s"].std(ddof=0),
                "training_s_mean": group["training_s"].mean(),
                "training_s_std": group["training_s"].std(ddof=0),
            }
        )
    return rows


def best_by_experiment_source(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = pd.DataFrame(aggregate_metric_rows(metric_rows))
    rows: list[dict[str, Any]] = []
    group_columns = ["source", "missing_set", "experiment_mode", "experiment"]
    for _, group in summary.groupby(group_columns, sort=True):
        best = group.sort_values(["mae_mean", "rmse_mean", "model_type"]).iloc[0].to_dict()
        best["selection_metric"] = "lowest_mae_mean"
        rows.append(best)
    return rows


def bootstrap_ci_from_metric_rows(metric_rows: list[dict[str, Any]], n_boot: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(metric_rows)
    rows: list[dict[str, Any]] = []
    group_columns = ["source", "missing_set", "experiment_mode", "experiment", "model_type"]
    for keys, group in frame.groupby(group_columns, sort=True):
        source, missing_set, mode, experiment, model_type = keys
        group = group.reset_index(drop=True)
        for metric in METRICS_FOR_CI:
            values = group[metric].to_numpy(dtype=float)
            boot_values = [
                float(np.mean(values[rng.choice(np.arange(len(values)), size=len(values), replace=True)]))
                for _ in range(n_boot)
            ]
            rows.append(
                {
                    "source": source,
                    "missing_set": missing_set,
                    "experiment_mode": mode,
                    "experiment": experiment,
                    "model_type": model_type,
                    "metric": metric,
                    "point_estimate": float(np.mean(values)),
                    "bootstrap_mean": float(np.mean(boot_values)),
                    "ci95_low": float(np.quantile(boot_values, 0.025)),
                    "ci95_high": float(np.quantile(boot_values, 0.975)),
                    "n_bootstrap": n_boot,
                    "n_runs": len(values),
                    "n_per_run": int(group["n"].iloc[0]),
                }
            )
    return rows


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [np.nan] * len(p_values)
    valid = [(idx, value) for idx, value in enumerate(p_values) if pd.notna(value)]
    ordered = sorted(valid, key=lambda item: item[1])
    running = 0.0
    m = len(ordered)
    for rank, (idx, value) in enumerate(ordered, start=1):
        corrected = min(1.0, (m - rank + 1) * value)
        running = max(running, corrected)
        adjusted[idx] = running
    return adjusted


def paired_tests_from_metric_rows(metric_rows: list[dict[str, Any]], metric: str = "mae") -> list[dict[str, Any]]:
    frame = pd.DataFrame(metric_rows)
    frame[metric] = frame[metric].astype(float)
    pending: list[dict[str, Any]] = []
    p_values: list[float] = []
    group_columns = ["source", "missing_set", "experiment_mode", "experiment"]
    for keys, group in frame.groupby(group_columns, sort=True):
        source, missing_set, mode, experiment = keys
        wide = group.pivot_table(index="seed", columns="model_type", values=metric, aggfunc="first")
        for model_a, model_b in __import__("itertools").combinations(sorted(wide.columns), 2):
            paired = wide[[model_a, model_b]].dropna()
            diff = (paired[model_a] - paired[model_b]).to_numpy()
            if len(diff) == 0:
                p_value = np.nan
            else:
                try:
                    _, p_value = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                except ValueError:
                    p_value = np.nan
            pending.append(
                {
                    "source": source,
                    "missing_set": missing_set,
                    "experiment_mode": mode,
                    "experiment": experiment,
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_pairs": len(paired),
                    "mean_difference_a_minus_b": float(np.mean(diff)) if len(diff) else np.nan,
                    "wilcoxon_p_value": p_value,
                    "better_model_by_mean": model_a if len(diff) and np.mean(diff) < 0 else model_b,
                }
            )
            p_values.append(p_value)
    adjusted = holm_adjust(p_values)
    for row, p_adj in zip(pending, adjusted):
        row["holm_adjusted_p_value"] = p_adj
        row["significant_at_0_05"] = bool(p_adj < 0.05) if pd.notna(p_adj) else False
    return pending


def category_error_rows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_columns = ["source", "missing_set", "experiment_mode", "experiment", "model_type", "actual_category"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        source, missing_set, mode, experiment, model_type, actual_category = keys
        rows.append(
            {
                "source": source,
                "missing_set": missing_set,
                "experiment_mode": mode,
                "experiment": experiment,
                "model_type": model_type,
                "actual_category": actual_category,
                "n": len(group),
                "mae": float(mean_absolute_error(group["actual"], group["predicted"])),
                "rmse": rmse(group["actual"].to_numpy(), group["predicted"].to_numpy()),
                "accuracy_within_band": float((group["actual_category"] == group["predicted_category"]).mean()),
            }
        )
    return rows


def event_window_indices(n_rows: int, window_fraction: float, context_multiplier: int) -> dict[str, tuple[int, int]]:
    window_size = max(1, int(round(n_rows * window_fraction)))
    event_start = max(0, (n_rows - window_size) // 2)
    event_end = event_start + window_size
    context_size = window_size * max(1, context_multiplier)
    before_start = max(0, event_start - context_size)
    after_end = min(n_rows, event_end + context_size)
    return {
        "before": (before_start, event_start),
        "event": (event_start, event_end),
        "after": (event_end, after_end),
    }


def event_window_frame(external: pd.DataFrame, event_slice: tuple[int, int], multipliers: dict[str, float]) -> pd.DataFrame:
    modified = external[FEATURE_COLUMNS].copy()
    start, end = event_slice
    for column, multiplier in multipliers.items():
        modified.loc[start : end - 1, column] = modified.loc[start : end - 1, column] * float(multiplier)
    return clipped_features(modified)


def event_window_subset(external: pd.DataFrame, windows: dict[str, tuple[int, int]]) -> tuple[pd.DataFrame, dict[str, tuple[int, int]]]:
    ordered_segments = ["before", "event", "after"]
    indices: list[int] = []
    relative_windows: dict[str, tuple[int, int]] = {}
    cursor = 0
    for segment in ordered_segments:
        start, end = windows[segment]
        segment_indices = list(range(start, end))
        indices.extend(segment_indices)
        relative_windows[segment] = (cursor, cursor + len(segment_indices))
        cursor += len(segment_indices)
    return external.iloc[indices].reset_index(drop=True), relative_windows


def append_event_window_rows(
    rows: list[dict[str, Any]],
    *,
    missing_set: str,
    mode: str,
    seed: int,
    model_type: str,
    scenario_name: str,
    baseline_pred: np.ndarray,
    stress_pred: np.ndarray,
    calculation_windows: dict[str, tuple[int, int]],
    reported_windows: dict[str, tuple[int, int]],
) -> None:
    baseline_cat = score_to_category(baseline_pred)
    stress_cat = score_to_category(stress_pred)
    for segment, (start, end) in calculation_windows.items():
        if end <= start:
            continue
        reported_start, reported_end = reported_windows[segment]
        idx = np.arange(start, end)
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
                "segment": segment,
                "start_index": int(reported_start),
                "end_index_exclusive": int(reported_end),
                "n": int(end - start),
                "baseline_mean_score": float(np.mean(baseline_pred[idx])),
                "stress_mean_score": float(np.mean(stress_pred[idx])),
                "mean_delta_stress_minus_baseline": float(np.mean(stress_pred[idx] - baseline_pred[idx])),
                "median_delta_stress_minus_baseline": float(np.median(stress_pred[idx] - baseline_pred[idx])),
                "pct_score_decreased": float(np.mean(stress_pred[idx] < baseline_pred[idx])),
                "pct_category_worse": float(np.mean(worse)),
            }
        )


def build_bundles_for_missing_set(
    *,
    model_type: str,
    missing_set: str,
    missing_indicators: list[str],
    train: pd.DataFrame,
    y_train: np.ndarray,
    full_bundle: dict[str, Any],
    full_training_s: float,
    output_dir: Path,
    seed: int,
    compute_device: str,
    gpu_id: int,
    lightgbm_gpu_backend: str,
    save_models: bool,
) -> dict[str, tuple[dict[str, Any], Path, float]]:
    available = available_features(missing_indicators)
    bundles: dict[str, tuple[dict[str, Any], Path, float]] = {}
    full_path = output_dir / "models" / f"seed_{seed}" / FULL_REFERENCE / f"{model_type}.joblib"
    bundles[INFERENCE_DROPOUT] = (
        {**full_bundle, "missing_indicators": missing_indicators, "available_features": available},
        full_path,
        full_training_s,
    )

    started = time.perf_counter()
    reduced_model = train_model(model_type, available, train, y_train, compute_device, gpu_id, lightgbm_gpu_backend)
    reduced_training_s = time.perf_counter() - started
    reduced_bundle = {"wqi_model": reduced_model, "missing_indicators": missing_indicators, "available_features": available}
    reduced_path = output_dir / "models" / f"seed_{seed}" / REDUCED_RETRAINING / missing_set / f"{model_type}.joblib"
    if save_models:
        reduced_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(reduced_bundle, reduced_path)
    bundles[REDUCED_RETRAINING] = (reduced_bundle, reduced_path, reduced_training_s)

    started = time.perf_counter()
    indicator_models: dict[str, Any] = {}
    reconstructed_train = train[FEATURE_COLUMNS].copy()
    for indicator in missing_indicators:
        indicator_model = train_model(model_type, available, train, train[indicator].to_numpy(), compute_device, gpu_id, lightgbm_gpu_backend)
        indicator_models[indicator] = indicator_model
        reconstructed_train[indicator] = indicator_model.predict(train[available])
    stage2_model = train_model(model_type, FEATURE_COLUMNS, reconstructed_train, y_train, compute_device, gpu_id, lightgbm_gpu_backend)
    reconstruction_training_s = time.perf_counter() - started
    reconstruction_bundle = {
        "wqi_model": stage2_model,
        "indicator_models": indicator_models,
        "missing_indicators": missing_indicators,
        "available_features": available,
    }
    reconstruction_path = output_dir / "models" / f"seed_{seed}" / INDICATOR_RECONSTRUCTION / missing_set / f"{model_type}.joblib"
    if save_models:
        reconstruction_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(reconstruction_bundle, reconstruction_path)
    bundles[INDICATOR_RECONSTRUCTION] = (reconstruction_bundle, reconstruction_path, reconstruction_training_s)
    return bundles


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    compute_device = resolve_compute_device(args.compute_device or config.get("compute_device", "cpu"))
    gpu_id = args.gpu_id if args.gpu_id is not None else int(config.get("gpu_id", 0))
    lightgbm_gpu_backend = args.lightgbm_gpu_backend or config.get("lightgbm_gpu_backend", "gpu")
    output_dir = resolve_output_dir((PROJECT_ROOT / (args.output_dir or config["output_dir"])).resolve(), args.overwrite)

    subset = pd.read_csv(PROJECT_ROOT / config["dataset_50000"])
    full = pd.read_csv(PROJECT_ROOT / config["full_dataset"])
    external = extract_external_set(full, subset, config.get("expected_external_rows"), config.get("external_max_rows"))
    subset["wqi5_category"] = subset["Score"].apply(lambda value: categorize_score(float(value))[0])
    strata = subset["wqi5_category"].to_numpy()
    models = list(config["models"])
    seeds = list(config["seeds"])
    missing_sets = {name: list(spec["missing_indicators"]) for name, spec in config["missing_sets"].items()}
    n_bootstrap = int(config.get("n_bootstrap", 200))
    save_models = bool(config.get("save_models", True))

    logger.info(
        "compute_device={} gpu_id={} lightgbm_gpu_backend={} train_source_rows={} external_rows={} missing_sets={}",
        compute_device,
        gpu_id,
        lightgbm_gpu_backend,
        len(subset),
        len(external),
        list(missing_sets),
    )

    manifest = ManifestSchema(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        config_path=str(config_path),
        output_dir=str(output_dir),
        dataset_50000=config["dataset_50000"],
        full_dataset=config["full_dataset"],
        train_rows_per_seed=int(len(subset) * (1 - float(config["test_size"]))),
        internal_test_rows_per_seed=int(len(subset) * float(config["test_size"])),
        external_rows=len(external),
        seeds=seeds,
        models=models,
        missing_sets=missing_sets,
        experiment_modes=[FULL_REFERENCE, INFERENCE_DROPOUT, REDUCED_RETRAINING, INDICATOR_RECONSTRUCTION],
        feature_columns=FEATURE_COLUMNS,
        compute_device=compute_device,
        gpu_id=gpu_id,
        lightgbm_gpu_backend=lightgbm_gpu_backend,
    )
    output_json(output_dir / "manifest.json", manifest)
    output_json(
        output_dir / "hardware.json",
        HardwareSchema(
            platform=platform.platform(),
            python=sys.version.replace("\n", " "),
            cpu=get_cpu_name(),
            memory_total_kb=get_memory_total_kb(),
            nvidia_smi=command_output(["nvidia-smi"]),
        ),
    )
    output_json(output_dir / "versions.json", collect_versions())

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    event_config = config.get("event_window_stress", {})
    event_enabled = bool(event_config.get("enabled", True))
    windows = event_window_indices(
        len(external),
        float(event_config.get("window_fraction", 0.01)),
        int(event_config.get("context_multiplier", 1)),
    )
    event_external, event_relative_windows = event_window_subset(external, windows)
    event_scenarios = event_config.get("scenarios", {})

    for seed in seeds:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=config["test_size"], random_state=seed)
        train_idx, test_idx = next(splitter.split(subset[FEATURE_COLUMNS], strata))
        train = subset.iloc[train_idx].reset_index(drop=True)
        test = subset.iloc[test_idx].reset_index(drop=True)
        y_train = train["Score"].to_numpy()
        source_frames = {
            "internal_test": (test, test["Score"].to_numpy(), test_idx.astype(int)),
            "external_10714": (external, external["Score"].to_numpy(), np.arange(len(subset), len(subset) + len(external), dtype=int)),
        }
        for idx in train_idx:
            split_rows.append({"seed": seed, "split": "train", "row_id": int(idx)})
        for idx in test_idx:
            split_rows.append({"seed": seed, "split": "internal_test", "row_id": int(idx)})

        for model_type in models:
            logger.info(f"seed={seed}: training model={model_type}")
            started = time.perf_counter()
            full_model = train_model(model_type, FEATURE_COLUMNS, train, y_train, compute_device, gpu_id, lightgbm_gpu_backend)
            full_training_s = time.perf_counter() - started
            full_bundle = {"wqi_model": full_model, "missing_indicators": [], "available_features": FEATURE_COLUMNS}
            full_path = output_dir / "models" / f"seed_{seed}" / FULL_REFERENCE / f"{model_type}.joblib"
            if save_models:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(full_bundle, full_path)

            all_bundles: list[tuple[str, str, dict[str, Any], Path, float]] = [
                (COMPLETE_SET, FULL_REFERENCE, full_bundle, full_path, full_training_s)
            ]
            for missing_set, indicators in missing_sets.items():
                built = build_bundles_for_missing_set(
                    model_type=model_type,
                    missing_set=missing_set,
                    missing_indicators=indicators,
                    train=train,
                    y_train=y_train,
                    full_bundle=full_bundle,
                    full_training_s=full_training_s,
                    output_dir=output_dir,
                    seed=seed,
                    compute_device=compute_device,
                    gpu_id=gpu_id,
                    lightgbm_gpu_backend=lightgbm_gpu_backend,
                    save_models=save_models,
                )
                for mode, (bundle, path, training_s) in built.items():
                    all_bundles.append((missing_set, mode, bundle, path, training_s))

            for missing_set, mode, bundle, path, training_s in all_bundles:
                for source, (frame, y_true, row_ids) in source_frames.items():
                    started = time.perf_counter()
                    y_pred = predict_bundle(mode, bundle, frame)
                    latency_s = time.perf_counter() - started
                    append_prediction_rows(
                        prediction_rows,
                        source=source,
                        missing_set=missing_set,
                        mode=mode,
                        seed=seed,
                        model_type=model_type,
                        row_ids=row_ids,
                        y_true=y_true,
                        y_pred=y_pred,
                    )
                    append_metric_row(
                        metric_rows,
                        source=source,
                        missing_set=missing_set,
                        mode=mode,
                        seed=seed,
                        model_type=model_type,
                        model_path=path,
                        latency_s=latency_s,
                        training_s=training_s,
                        y_true=y_true,
                        y_pred=y_pred,
                    )
                    if mode == INDICATOR_RECONSTRUCTION:
                        append_reconstruction_rows(
                            reconstruction_rows,
                            source=source,
                            missing_set=missing_set,
                            seed=seed,
                            model_type=model_type,
                            frame=frame,
                            bundle=bundle,
                        )

                if event_enabled:
                    baseline_pred = predict_bundle(mode, bundle, event_external)
                    for scenario_name, multipliers in event_scenarios.items():
                        scenario_frame = event_window_frame(event_external, event_relative_windows["event"], multipliers)
                        stress_pred = predict_bundle(mode, bundle, scenario_frame)
                        append_event_window_rows(
                            event_rows,
                            missing_set=missing_set,
                            mode=mode,
                            seed=seed,
                            model_type=model_type,
                            scenario_name=scenario_name,
                            baseline_pred=baseline_pred,
                            stress_pred=stress_pred,
                            calculation_windows=event_relative_windows,
                            reported_windows=windows,
                        )

    predictions = pd.DataFrame(prediction_rows)
    metric_frame = pd.DataFrame(metric_rows)
    write_csv(output_dir / "splits" / "split_indices.csv", split_rows)
    predictions.to_csv(output_dir / "predictions" / "predictions_long.csv", index=False)
    metric_frame.to_csv(output_dir / "metrics" / "metrics_by_seed.csv", index=False)
    write_csv(output_dir / "metrics" / "metrics_summary.csv", aggregate_metric_rows(metric_rows))
    write_csv(output_dir / "metrics" / "best_by_experiment_source.csv", best_by_experiment_source(metric_rows))
    write_csv(output_dir / "metrics" / "indicator_reconstruction_metrics.csv", reconstruction_rows)
    write_csv(output_dir / "metrics" / "error_by_wqi_band.csv", category_error_rows(predictions))
    write_csv(output_dir / "stats" / "bootstrap_ci.csv", bootstrap_ci_from_metric_rows(metric_rows, n_bootstrap))
    write_csv(output_dir / "stats" / "paired_error_tests.csv", paired_tests_from_metric_rows(metric_rows))
    write_csv(output_dir / "stress_tests" / "event_window_stress_summary.csv", event_rows)
    logger.success(f"completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
