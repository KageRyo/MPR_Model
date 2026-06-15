from __future__ import annotations

import argparse
import csv
import itertools
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
from pydantic import BaseModel, Field
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
from src.settings import FEATURE_COLUMNS  # noqa: E402
from src.wqi import categorize_score  # noqa: E402


CORE_FEATURES = ["DO", "EC", "SS"]
MISSING_INDICATORS = ["BOD", "NH3N"]
FULL_REFERENCE = "full_reference"
FULL_DROPOUT = "full_inference_dropout"
REDUCED_RETRAINING = "reduced_retraining"
TWO_STAGE = "two_stage_reconstruction"
SOURCES = ["internal_test", "external_10714"]
METRICS_FOR_CI = ["r2", "mae", "rmse", "nmae", "accuracy", "macro_f1"]
CATEGORY_ORDER = ["Terrible", "Bad", "Poor", "Fair", "Good", "Excellent"]


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
    experiments: list[str]
    feature_columns: list[str]
    missing_indicators: list[str]
    compute_device: str
    gpu_id: int
    lightgbm_gpu_backend: str


class HardwareSchema(BaseModel):
    platform: str
    python: str
    cpu: str | None = None
    memory_total_kb: int | None = None
    nvidia_smi: str | None = None


class VersionsSchema(BaseModel):
    numpy: str
    pandas: str
    sklearn: str
    scipy: str
    joblib: str
    xgboost: str | None = None
    lightgbm: str | None = None
    pydantic: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/missing_indicator_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--compute-device", choices=["cpu", "gpu", "auto"], default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--lightgbm-gpu-backend", choices=["gpu", "cuda"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_output_dir(path: Path, overwrite: bool) -> Path:
    if path.exists():
        existing_files = [item for item in path.rglob("*") if item.is_file()]
        if existing_files and not overwrite:
            raise FileExistsError(f"Output directory already contains files: {path}. Use --output-dir or --overwrite.")
        if overwrite:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    for child in ["models", "predictions", "metrics", "stats", "stress_tests", "splits"]:
        (path / child).mkdir(parents=True, exist_ok=True)
    return path


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def get_cpu_name() -> str | None:
    try:
        with Path("/proc/cpuinfo").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def get_memory_total_kb() -> int | None:
    try:
        with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def collect_versions() -> VersionsSchema:
    import joblib as joblib_module
    import pydantic
    import scipy
    import sklearn

    try:
        import xgboost

        xgboost_version = xgboost.__version__
    except Exception:
        xgboost_version = None
    try:
        import lightgbm

        lightgbm_version = lightgbm.__version__
    except Exception:
        lightgbm_version = None

    return VersionsSchema(
        numpy=np.__version__,
        pandas=pd.__version__,
        sklearn=sklearn.__version__,
        scipy=scipy.__version__,
        joblib=joblib_module.__version__,
        xgboost=xgboost_version,
        lightgbm=lightgbm_version,
        pydantic=pydantic.__version__,
    )


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


def clipped_features(frame: pd.DataFrame) -> pd.DataFrame:
    clipped = frame.copy()
    lower_bounds = {"DO": 0.0, "BOD": 0.0, "NH3N": 0.0, "EC": 0.0, "SS": 0.0}
    upper_bounds = {"DO": 150.0, "BOD": 200.0, "NH3N": 50.0, "EC": 50000.0, "SS": 5000.0}
    for column in FEATURE_COLUMNS:
        clipped[column] = clipped[column].clip(lower=lower_bounds[column], upper=upper_bounds[column])
    return clipped


def extract_external_set(full: pd.DataFrame, subset: pd.DataFrame, expected_rows: int | None, max_rows: int | None) -> pd.DataFrame:
    if not full.iloc[: len(subset)].reset_index(drop=True).equals(subset.reset_index(drop=True)):
        raise ValueError("Configured 50000-row dataset is not an exact prefix of the full dataset.")
    external = full.iloc[len(subset) :].reset_index(drop=True)
    if expected_rows is not None and len(external) != expected_rows:
        raise ValueError(f"Expected {expected_rows} external rows, found {len(external)}.")
    if max_rows is not None:
        external = external.iloc[:max_rows].reset_index(drop=True)
    return external


def predict_bundle(experiment: str, model_bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    if experiment == FULL_REFERENCE:
        return model_bundle["wqi_model"].predict(frame[FEATURE_COLUMNS])
    if experiment == FULL_DROPOUT:
        dropped = frame[FEATURE_COLUMNS].copy()
        dropped[MISSING_INDICATORS] = np.nan
        return model_bundle["wqi_model"].predict(dropped)
    if experiment == REDUCED_RETRAINING:
        return model_bundle["wqi_model"].predict(frame[CORE_FEATURES])
    if experiment == TWO_STAGE:
        core = frame[CORE_FEATURES]
        reconstructed = frame[FEATURE_COLUMNS].copy()
        reconstructed["BOD"] = model_bundle["bod_model"].predict(core)
        reconstructed["NH3N"] = model_bundle["nh3n_model"].predict(core)
        return model_bundle["wqi_model"].predict(reconstructed[FEATURE_COLUMNS])
    raise ValueError(f"Unsupported experiment: {experiment}")


def train_wqi_model(model_type: str, features: list[str], frame: pd.DataFrame, y: np.ndarray, compute_device: str, gpu_id: int, lightgbm_gpu_backend: str):
    require_model_support(model_type)
    estimator = build_model(
        model_type,
        compute_device=compute_device,
        gpu_id=gpu_id,
        lightgbm_gpu_backend=lightgbm_gpu_backend,
    )
    estimator.fit(frame[features], y)
    return estimator


def append_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    experiment: str,
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
                "experiment": experiment,
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
    experiment: str,
    seed: int,
    model_type: str,
    model_path: Path,
    latency_s: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    metrics = prediction_metrics(y_true, y_pred)
    rows.append(
            {
                "source": source,
                "experiment": experiment,
                "seed": seed,
                "model_type": model_type,
                "model_path": display_path(model_path),
                "n": len(y_true),
                "latency_s": latency_s,
                **metrics,
        }
    )


def append_reconstruction_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    seed: int,
    model_type: str,
    row_ids: np.ndarray,
    actual_frame: pd.DataFrame,
    bod_pred: np.ndarray,
    nh3n_pred: np.ndarray,
) -> None:
    for target, predictions in [("BOD", bod_pred), ("NH3N", nh3n_pred)]:
        actual = actual_frame[target].to_numpy()
        metrics = {
            "source": source,
            "seed": seed,
            "model_type": model_type,
            "target_indicator": target,
            "n": len(actual),
            "r2": float(r2_score(actual, predictions)),
            "mae": float(mean_absolute_error(actual, predictions)),
            "rmse": rmse(actual, predictions),
        }
        rows.append(metrics)


def bootstrap_ci_from_metric_rows(metric_rows: list[dict[str, Any]], n_boot: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    frame = pd.DataFrame(metric_rows)
    for (source, experiment, model_type), group in frame.groupby(["source", "experiment", "model_type"], sort=True):
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
    m = len(ordered)
    running = 0.0
    for rank, (idx, value) in enumerate(ordered, start=1):
        corrected = min(1.0, (m - rank + 1) * value)
        running = max(running, corrected)
        adjusted[idx] = running
    return adjusted


def paired_tests_from_metric_rows(metric_rows: list[dict[str, Any]], metric: str = "mae") -> list[dict[str, Any]]:
    frame = pd.DataFrame(metric_rows)
    frame[metric] = frame[metric].astype(float)
    rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    p_values: list[float] = []
    for (source, experiment), group in frame.groupby(["source", "experiment"], sort=True):
        wide = group.pivot_table(index="seed", columns="model_type", values=metric, aggfunc="first")
        for model_a, model_b in itertools.combinations(sorted(wide.columns), 2):
            paired = wide[[model_a, model_b]].dropna()
            diff = (paired[model_a] - paired[model_b]).to_numpy()
            if len(diff) == 0:
                p_value = np.nan
            else:
                try:
                    _, p_value = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                except ValueError:
                    p_value = np.nan
            rng = np.random.default_rng(43)
            if len(diff) > 0:
                boot = [float(np.mean(diff[rng.choice(np.arange(len(diff)), size=len(diff), replace=True)])) for _ in range(200)]
                ci_low = float(np.quantile(boot, 0.025))
                ci_high = float(np.quantile(boot, 0.975))
                mean_diff = float(np.mean(diff))
            else:
                ci_low = ci_high = mean_diff = np.nan
            pending.append(
                {
                    "source": source,
                    "experiment": experiment,
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_pairs": len(paired),
                    "mean_difference_a_minus_b": mean_diff,
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "wilcoxon_p_value": p_value,
                    "better_model_by_mean": model_a if pd.notna(mean_diff) and mean_diff < 0 else model_b,
                }
            )
            p_values.append(p_value)
    adjusted = holm_adjust(p_values)
    for row, p_adj in zip(pending, adjusted):
        row["holm_adjusted_p_value"] = p_adj
        row["significant_at_0_05"] = bool(p_adj < 0.05) if pd.notna(p_adj) else False
        rows.append(row)
    return rows


def aggregate_metric_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(metric_rows)
    rows: list[dict[str, Any]] = []
    for (source, experiment, model_type), group in frame.groupby(["source", "experiment", "model_type"], sort=True):
        rows.append(
            {
                "source": source,
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
    for (source, experiment), group in summary.groupby(["source", "experiment"], sort=True):
        best = group.sort_values(["mae_mean", "rmse_mean", "model_type"]).iloc[0].to_dict()
        best["selection_metric"] = "lowest_mae_mean"
        rows.append(best)
    return rows


def category_error_rows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (source, experiment, model_type, actual_category), group in predictions.groupby(
        ["source", "experiment", "model_type", "actual_category"], sort=True
    ):
        rows.append(
            {
                "source": source,
                "experiment": experiment,
                "model_type": model_type,
                "actual_category": actual_category,
                "n": len(group),
                "mae": mean_absolute_error(group["actual"], group["predicted"]),
                "rmse": rmse(group["actual"].to_numpy(), group["predicted"].to_numpy()),
                "accuracy_within_band": float((group["actual_category"] == group["predicted_category"]).mean()),
            }
        )
    return rows


def stress_frames(external: pd.DataFrame, scenario_config: dict[str, dict[str, float]]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name, multipliers in scenario_config.items():
        modified = external[FEATURE_COLUMNS].copy()
        for column, multiplier in multipliers.items():
            modified[column] = modified[column] * float(multiplier)
        frames[name] = clipped_features(modified)
    return frames


def append_stress_rows(
    rows: list[dict[str, Any]],
    *,
    experiment: str,
    seed: int,
    model_type: str,
    baseline_frame: pd.DataFrame,
    scenario_name: str,
    scenario_frame: pd.DataFrame,
    model_bundle: dict[str, Any],
) -> None:
    baseline_pred = predict_bundle(experiment, model_bundle, baseline_frame)
    scenario_pred = predict_bundle(experiment, model_bundle, scenario_frame)
    baseline_cat = score_to_category(baseline_pred)
    scenario_cat = score_to_category(scenario_pred)
    worse = [category_rank(after) < category_rank(before) for before, after in zip(baseline_cat, scenario_cat)]
    rows.append(
        {
            "experiment": experiment,
            "seed": seed,
            "model_type": model_type,
            "stress_scenario": scenario_name,
            "n": len(scenario_pred),
            "baseline_mean_score": float(np.mean(baseline_pred)),
            "stress_mean_score": float(np.mean(scenario_pred)),
            "mean_delta_stress_minus_baseline": float(np.mean(scenario_pred - baseline_pred)),
            "median_delta_stress_minus_baseline": float(np.median(scenario_pred - baseline_pred)),
            "pct_score_decreased": float(np.mean(scenario_pred < baseline_pred)),
            "pct_category_worse": float(np.mean(worse)),
        }
    )


def append_stress_rows_for_experiment(
    rows: list[dict[str, Any]],
    *,
    experiment: str,
    seed: int,
    model_type: str,
    baseline_frame: pd.DataFrame,
    stress_inputs: dict[str, pd.DataFrame],
    model_bundle: dict[str, Any],
) -> None:
    baseline_pred = predict_bundle(experiment, model_bundle, baseline_frame)
    baseline_cat = score_to_category(baseline_pred)
    for scenario_name, scenario_frame in stress_inputs.items():
        scenario_pred = predict_bundle(experiment, model_bundle, scenario_frame)
        scenario_cat = score_to_category(scenario_pred)
        worse = [category_rank(after) < category_rank(before) for before, after in zip(baseline_cat, scenario_cat)]
        rows.append(
            {
                "experiment": experiment,
                "seed": seed,
                "model_type": model_type,
                "stress_scenario": scenario_name,
                "n": len(scenario_pred),
                "baseline_mean_score": float(np.mean(baseline_pred)),
                "stress_mean_score": float(np.mean(scenario_pred)),
                "mean_delta_stress_minus_baseline": float(np.mean(scenario_pred - baseline_pred)),
                "median_delta_stress_minus_baseline": float(np.median(scenario_pred - baseline_pred)),
                "pct_score_decreased": float(np.mean(scenario_pred < baseline_pred)),
                "pct_category_worse": float(np.mean(worse)),
            }
        )


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
    expected_external_rows = config.get("expected_external_rows")
    external_max_rows = config.get("external_max_rows")
    external = extract_external_set(full, subset, expected_external_rows, external_max_rows)
    subset["wqi5_category"] = subset["Score"].apply(lambda value: categorize_score(value)[0])
    y = subset["Score"].to_numpy()
    strata = subset["wqi5_category"].to_numpy()
    models = list(config["models"])
    seeds = list(config["seeds"])
    n_bootstrap = int(config.get("n_bootstrap", 200))
    save_models = bool(config.get("save_models", True))

    logger.info(
        "compute_device={} gpu_id={} lightgbm_gpu_backend={} train_source_rows={} external_rows={}",
        compute_device,
        gpu_id,
        lightgbm_gpu_backend,
        len(subset),
        len(external),
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
        experiments=[FULL_REFERENCE, FULL_DROPOUT, REDUCED_RETRAINING, TWO_STAGE, "stress_scenarios"],
        feature_columns=FEATURE_COLUMNS,
        missing_indicators=MISSING_INDICATORS,
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
    stress_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    stress_enabled = bool(config.get("stress_test", {}).get("enabled", True))
    stress_scenarios = config.get("stress_test", {}).get("scenarios", {})
    stress_inputs = stress_frames(external, stress_scenarios) if stress_enabled else {}

    for seed in seeds:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=config["test_size"], random_state=seed)
        train_idx, test_idx = next(splitter.split(subset[FEATURE_COLUMNS], strata))
        train = subset.iloc[train_idx].reset_index(drop=True)
        test = subset.iloc[test_idx].reset_index(drop=True)
        y_train = train["Score"].to_numpy()
        y_test = test["Score"].to_numpy()
        y_external = external["Score"].to_numpy()
        test_row_ids = test_idx.astype(int)
        external_row_ids = np.arange(len(subset), len(subset) + len(external), dtype=int)
        for idx in train_idx:
            split_rows.append({"seed": seed, "split": "train", "row_id": int(idx)})
        for idx in test_idx:
            split_rows.append({"seed": seed, "split": "internal_test", "row_id": int(idx)})

        for model_type in models:
            logger.info(f"seed={seed}: training model={model_type}")
            full_started = time.perf_counter()
            full_model = train_wqi_model(
                model_type,
                FEATURE_COLUMNS,
                train,
                y_train,
                compute_device,
                gpu_id,
                lightgbm_gpu_backend,
            )
            full_training_s = time.perf_counter() - full_started
            full_bundle = {"wqi_model": full_model}
            full_model_path = output_dir / "models" / f"seed_{seed}" / "full_model" / f"{model_type}.joblib"
            if save_models:
                full_model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(full_bundle, full_model_path)

            reduced_started = time.perf_counter()
            reduced_model = train_wqi_model(
                model_type,
                CORE_FEATURES,
                train,
                y_train,
                compute_device,
                gpu_id,
                lightgbm_gpu_backend,
            )
            reduced_training_s = time.perf_counter() - reduced_started
            reduced_bundle = {"wqi_model": reduced_model}
            reduced_model_path = output_dir / "models" / f"seed_{seed}" / REDUCED_RETRAINING / f"{model_type}.joblib"
            if save_models:
                reduced_model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(reduced_bundle, reduced_model_path)

            two_stage_started = time.perf_counter()
            bod_model = train_wqi_model(model_type, CORE_FEATURES, train, train["BOD"].to_numpy(), compute_device, gpu_id, lightgbm_gpu_backend)
            nh3n_model = train_wqi_model(
                model_type,
                CORE_FEATURES,
                train,
                train["NH3N"].to_numpy(),
                compute_device,
                gpu_id,
                lightgbm_gpu_backend,
            )
            reconstructed_train = train[FEATURE_COLUMNS].copy()
            reconstructed_train["BOD"] = bod_model.predict(train[CORE_FEATURES])
            reconstructed_train["NH3N"] = nh3n_model.predict(train[CORE_FEATURES])
            stage2_model = train_wqi_model(
                model_type,
                FEATURE_COLUMNS,
                reconstructed_train,
                y_train,
                compute_device,
                gpu_id,
                lightgbm_gpu_backend,
            )
            two_stage_training_s = time.perf_counter() - two_stage_started
            two_stage_bundle = {"bod_model": bod_model, "nh3n_model": nh3n_model, "wqi_model": stage2_model}
            two_stage_model_path = output_dir / "models" / f"seed_{seed}" / TWO_STAGE / f"{model_type}.joblib"
            if save_models:
                two_stage_model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(two_stage_bundle, two_stage_model_path)

            training_times = {
                FULL_REFERENCE: full_training_s,
                FULL_DROPOUT: full_training_s,
                REDUCED_RETRAINING: reduced_training_s,
                TWO_STAGE: two_stage_training_s,
            }
            bundles = {
                FULL_REFERENCE: (full_bundle, full_model_path),
                FULL_DROPOUT: (full_bundle, full_model_path),
                REDUCED_RETRAINING: (reduced_bundle, reduced_model_path),
                TWO_STAGE: (two_stage_bundle, two_stage_model_path),
            }

            for source, frame, y_true, row_ids in [
                ("internal_test", test, y_test, test_row_ids),
                ("external_10714", external, y_external, external_row_ids),
            ]:
                for experiment, (bundle, model_path) in bundles.items():
                    started = time.perf_counter()
                    y_pred = predict_bundle(experiment, bundle, frame)
                    latency_s = time.perf_counter() - started
                    append_prediction_rows(
                        prediction_rows,
                        source=source,
                        experiment=experiment,
                        seed=seed,
                        model_type=model_type,
                        row_ids=row_ids,
                        y_true=y_true,
                        y_pred=y_pred,
                    )
                    append_metric_row(
                        metric_rows,
                        source=source,
                        experiment=experiment,
                        seed=seed,
                        model_type=model_type,
                        model_path=model_path,
                        latency_s=latency_s,
                        y_true=y_true,
                        y_pred=y_pred,
                    )
                    metric_rows[-1]["training_s"] = training_times[experiment]

            for source, frame, row_ids in [
                ("internal_test", test, test_row_ids),
                ("external_10714", external, external_row_ids),
            ]:
                bod_pred = bod_model.predict(frame[CORE_FEATURES])
                nh3n_pred = nh3n_model.predict(frame[CORE_FEATURES])
                append_reconstruction_rows(
                    reconstruction_rows,
                    source=source,
                    seed=seed,
                    model_type=model_type,
                    row_ids=row_ids,
                    actual_frame=frame,
                    bod_pred=bod_pred,
                    nh3n_pred=nh3n_pred,
                )

            if stress_enabled:
                for experiment, (bundle, _) in bundles.items():
                    append_stress_rows_for_experiment(
                        stress_rows,
                        experiment=experiment,
                        seed=seed,
                        model_type=model_type,
                        baseline_frame=external,
                        stress_inputs=stress_inputs,
                        model_bundle=bundle,
                    )

    predictions = pd.DataFrame(prediction_rows)
    metric_frame = pd.DataFrame(metric_rows)
    write_csv(output_dir / "splits" / "split_indices.csv", split_rows)
    predictions.to_csv(output_dir / "predictions" / "predictions_long.csv", index=False)
    metric_frame.to_csv(output_dir / "metrics" / "metrics_by_seed.csv", index=False)
    write_csv(output_dir / "metrics" / "metrics_summary.csv", aggregate_metric_rows(metric_rows))
    write_csv(output_dir / "metrics" / "best_by_experiment_source.csv", best_by_experiment_source(metric_rows))
    write_csv(output_dir / "metrics" / "stage1_reconstruction_metrics.csv", reconstruction_rows)
    write_csv(output_dir / "metrics" / "error_by_wqi_band.csv", category_error_rows(predictions))
    write_csv(output_dir / "stats" / "bootstrap_ci.csv", bootstrap_ci_from_metric_rows(metric_rows, n_bootstrap))
    write_csv(output_dir / "stats" / "paired_error_tests.csv", paired_tests_from_metric_rows(metric_rows))
    write_csv(output_dir / "stress_tests" / "stress_summary.csv", stress_rows)
    logger.success(f"completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
