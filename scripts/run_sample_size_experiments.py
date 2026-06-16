from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reproduce_results import build_model, nmae, require_model_support, rmse
from src.settings import FEATURE_COLUMNS
from src.wqi import categorize_score

DEFAULT_DATASETS = [
    "data/dataV1_1000.csv",
    "data/dataV1_5000.csv",
    "data/dataV1_10000.csv",
    "data/dataV1_50000.csv",
]
DEFAULT_MODELS = ["lr", "mpr", "svm", "rf", "xgboost", "lightgbm"]
MODEL_DIR_NAMES = {
    "lr": "LR",
    "mpr": "MPR",
    "svm": "SVM",
    "rf": "RF",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_empty_dir(path: Path) -> Path:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to write into a non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def path_for_record(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def score_to_category(values: np.ndarray) -> list[str]:
    return [categorize_score(value)[0] for value in values]


def evaluate_split(model_type: str, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true_cat = score_to_category(y_true)
    y_pred_cat = score_to_category(y_pred)
    return {
        "model_type": model_type,
        "split": split,
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "nmae": nmae(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true_cat, y_pred_cat)),
        "macro_f1": float(f1_score(y_true_cat, y_pred_cat, average="macro")),
        "residual_mean": float(np.mean(y_true - y_pred)),
        "residual_std": float(np.std(y_true - y_pred)),
    }


def dataset_size_from_frame(path: Path, frame: pd.DataFrame) -> int:
    suffix = path.stem.rsplit("_", maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() else int(len(frame))


def gpu_is_visible() -> bool:
    try:
        subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return True


def resolve_compute_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "gpu" if gpu_is_visible() else "cpu"
    if requested_device == "gpu" and not gpu_is_visible():
        raise RuntimeError(
            "GPU execution was requested, but nvidia-smi is not available from this environment. "
            "Use --compute-device cpu or run from an environment with NVIDIA driver access."
        )
    return requested_device


def write_manifest(
    output_dir: Path,
    model_dir: Path,
    *,
    datasets: list[str],
    models: list[str],
    n_splits: int,
    shuffle_seed: int,
    compute_device: str,
    requested_compute_device: str,
    gpu_id: int,
    lightgbm_gpu_backend: str,
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": datasets,
        "models": models,
        "n_splits": n_splits,
        "split_protocol": "StratifiedKFold with 5 folds; each fold uses 80% training and 20% testing.",
        "shuffle_seed": shuffle_seed,
        "compute_device": compute_device,
        "requested_compute_device": requested_compute_device,
        "gpu_id": gpu_id,
        "lightgbm_gpu_backend": lightgbm_gpu_backend,
        "output_dir": path_for_record(output_dir),
        "model_dir": path_for_record(model_dir),
        "model_layout": "<model_dir>/<Model>/<dataset>/fold_<fold>/<model>.pkl",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_experiments(args: argparse.Namespace) -> tuple[Path, Path]:
    requested_compute_device = args.compute_device
    compute_device = resolve_compute_device(requested_compute_device)
    output_dir = prepare_empty_dir(
        resolve_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "results" / "sample_size_experiments"
    )
    model_dir = prepare_empty_dir(
        resolve_path(args.model_dir)
        if args.model_dir
        else PROJECT_ROOT / "models" / "sample_size_experiments"
    )

    logger.info(
        "compute_device={} requested={} gpu_id={} lightgbm_gpu_backend={}",
        compute_device,
        requested_compute_device,
        args.gpu_id,
        args.lightgbm_gpu_backend,
    )
    logger.info("output_dir={}", output_dir)
    logger.info("model_dir={}", model_dir)

    for model_type in args.models:
        require_model_support(model_type)

    metric_rows: list[dict] = []
    split_rows: list[dict] = []

    for dataset_arg in args.datasets:
        dataset_path = resolve_path(dataset_arg)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

        frame = pd.read_csv(dataset_path)
        required_columns = [*FEATURE_COLUMNS, "Score"]
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"{dataset_path} is missing required columns: {missing_columns}")

        dataset_size = dataset_size_from_frame(dataset_path, frame)
        dataset_name = dataset_path.stem
        frame["wqi5_category"] = frame["Score"].apply(lambda value: categorize_score(value)[0])
        x = frame[FEATURE_COLUMNS]
        y = frame["Score"].to_numpy()
        strata = frame["wqi5_category"].to_numpy()
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.shuffle_seed)

        logger.info("dataset={} rows={} folds={}", dataset_path, len(frame), args.n_splits)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x, strata), start=1):
            x_train = x.iloc[train_idx]
            x_test = x.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            for row_index in train_idx:
                split_rows.append(
                    {
                        "dataset": dataset_name,
                        "dataset_size": dataset_size,
                        "fold": fold,
                        "split": "train",
                        "row_index": int(row_index),
                    }
                )
            for row_index in test_idx:
                split_rows.append(
                    {
                        "dataset": dataset_name,
                        "dataset_size": dataset_size,
                        "fold": fold,
                        "split": "test",
                        "row_index": int(row_index),
                    }
                )

            for model_type in args.models:
                logger.info("dataset={} fold={} model={}", dataset_name, fold, model_type)
                estimator = build_model(
                    model_type,
                    compute_device=compute_device,
                    gpu_id=args.gpu_id,
                    lightgbm_gpu_backend=args.lightgbm_gpu_backend,
                )
                if estimator is None:
                    raise RuntimeError(f"Model support check passed but build_model returned None: {model_type}")
                if model_type == "xgboost":
                    estimator.set_params(model__verbosity=0)
                if model_type == "lightgbm":
                    estimator.set_params(model__verbose=-1)

                fit_started = time.perf_counter()
                estimator.fit(x_train, y_train)
                fit_time_s = time.perf_counter() - fit_started

                train_predict_started = time.perf_counter()
                y_train_pred = estimator.predict(x_train)
                train_predict_time_s = time.perf_counter() - train_predict_started

                test_predict_started = time.perf_counter()
                y_test_pred = estimator.predict(x_test)
                test_predict_time_s = time.perf_counter() - test_predict_started

                model_family_dir = MODEL_DIR_NAMES.get(model_type, model_type)
                artifact_path = model_dir / model_family_dir / dataset_name / f"fold_{fold}" / f"{model_type}.pkl"
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(estimator, artifact_path)

                common = {
                    "dataset": dataset_name,
                    "dataset_size": dataset_size,
                    "fold": fold,
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "fit_time_s": float(fit_time_s),
                    "artifact_path": path_for_record(artifact_path),
                }
                train_metrics = evaluate_split(model_type, "train", y_train, y_train_pred)
                train_metrics.update(
                    {
                        **common,
                        "predict_time_s": float(train_predict_time_s),
                    }
                )
                test_metrics = evaluate_split(model_type, "test", y_test, y_test_pred)
                test_metrics.update(
                    {
                        **common,
                        "predict_time_s": float(test_predict_time_s),
                    }
                )
                metric_rows.extend([train_metrics, test_metrics])

    metrics_frame = pd.DataFrame(metric_rows)
    summary_rows: list[dict] = []
    group_columns = ["dataset", "dataset_size", "model_type", "split"]
    for keys, group in metrics_frame.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        for metric in ["r2", "mae", "rmse", "nmae", "accuracy", "macro_f1", "fit_time_s", "predict_time_s"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        row["folds"] = int(group["fold"].nunique())
        row["train_rows"] = int(group["train_rows"].iloc[0])
        row["test_rows"] = int(group["test_rows"].iloc[0])
        summary_rows.append(row)

    write_csv(output_dir / "metrics" / "metrics_by_fold.csv", metric_rows)
    write_csv(output_dir / "metrics" / "metrics_summary.csv", summary_rows)
    write_csv(output_dir / "splits" / "split_indices.csv", split_rows)
    write_manifest(
        output_dir,
        model_dir,
        datasets=args.datasets,
        models=args.models,
        n_splits=args.n_splits,
        shuffle_seed=args.shuffle_seed,
        compute_device=compute_device,
        requested_compute_device=requested_compute_device,
        gpu_id=args.gpu_id,
        lightgbm_gpu_backend=args.lightgbm_gpu_backend,
    )
    logger.success("completed. Outputs written to {} and models to {}", output_dir, model_dir)
    return output_dir, model_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--compute-device", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--lightgbm-gpu-backend", choices=["gpu", "cuda"], default="gpu")
    args = parser.parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
