from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

from loguru import logger

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.settings import FEATURE_COLUMNS
from src.wqi import categorize_score, direct_wqi5_score

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.max(y_true) - np.min(y_true)
    return float(mean_absolute_error(y_true, y_pred) / denom) if denom else 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_model(model_type: str):
    if model_type == "lr":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
    if model_type == "mpr":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
    if model_type == "svm":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("model", SVR(C=10.0, epsilon=0.1, kernel="rbf")),
            ]
        )
    if model_type == "rf":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("model", RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)),
            ]
        )
    if model_type == "xgboost":
        if XGBRegressor is None:
            return None
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=0,
                    ),
                ),
            ]
        )
    if model_type == "lightgbm":
        if LGBMRegressor is None:
            return None
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("model", LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=0)),
            ]
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def require_model_support(model_type: str) -> None:
    if model_type == "xgboost" and XGBRegressor is None:
        raise ImportError("xgboost is not installed but is required by the current experiment config.")
    if model_type == "lightgbm" and LGBMRegressor is None:
        raise ImportError("lightgbm is not installed but is required by the current experiment config.")


def score_to_category(values: np.ndarray) -> list[str]:
    return [categorize_score(value)[0] for value in values]


def evaluate_predictions(model_type: str, y_true: np.ndarray, y_pred: np.ndarray, latency_s: float) -> dict:
    y_true_cat = score_to_category(y_true)
    y_pred_cat = score_to_category(y_pred)
    return {
        "model_type": model_type,
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "nmae": nmae(y_true, y_pred),
        "accuracy": accuracy_score(y_true_cat, y_pred_cat),
        "macro_f1": f1_score(y_true_cat, y_pred_cat, average="macro"),
        "latency_s": latency_s,
        "residual_mean": float(np.mean(y_true - y_pred)),
        "residual_std": float(np.std(y_true - y_pred)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_output_dir(base_output_dir: Path, overwrite: bool) -> Path:
    if base_output_dir.exists():
        existing_csvs = list(base_output_dir.glob("*.csv"))
        if existing_csvs and not overwrite:
            raise FileExistsError(
                f"Output directory already contains CSV results: {base_output_dir}. "
                "Use --output-dir to write elsewhere or pass --overwrite explicitly."
            )
        if overwrite and base_output_dir.exists():
            shutil.rmtree(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    return base_output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_path = (PROJECT_ROOT / config["dataset"]).resolve()
    configured_output_dir = args.output_dir or config["output_dir"]
    output_dir = resolve_output_dir((PROJECT_ROOT / configured_output_dir).resolve(), overwrite=args.overwrite)
    if not data_path.exists():
        raise FileNotFoundError(f"Configured dataset does not exist: {data_path}")
    frame = pd.read_csv(data_path)
    frame["wqi5_category"] = frame["Score"].apply(lambda value: categorize_score(value)[0])

    X = frame[FEATURE_COLUMNS]
    y = frame["Score"].to_numpy()
    strata = frame["wqi5_category"].to_numpy()

    repeated_rows: list[dict] = []
    category_rows: list[dict] = []

    for seed in config["seeds"]:
        logger.info(f"seed={seed}: preparing stratified split")
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config["test_size"],
            random_state=seed,
        )
        train_idx, test_idx = next(splitter.split(X, strata))
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        for model_type in config["models"]:
            logger.info(f"seed={seed}: running model={model_type}")
            if model_type == "direct_wqi5":
                started = time.perf_counter()
                y_pred = np.array(
                    [
                        direct_wqi5_score(
                            do=row.DO,
                            bod=row.BOD,
                            nh3n=row.NH3N,
                            ec=row.EC,
                            ss=row.SS,
                        )
                        for row in X_test.itertuples(index=False)
                    ]
                )
                latency_s = time.perf_counter() - started
            else:
                require_model_support(model_type)
                estimator = build_model(model_type)
                started = time.perf_counter()
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
                latency_s = time.perf_counter() - started

            metrics = evaluate_predictions(model_type, y_test, y_pred, latency_s)
            metrics["seed"] = seed
            repeated_rows.append(metrics)

            for category in sorted(set(score_to_category(y_test))):
                mask = np.array(score_to_category(y_test)) == category
                category_rows.append(
                    {
                        "seed": seed,
                        "model_type": model_type,
                        "category": category,
                        "count": int(mask.sum()),
                        "mae": mean_absolute_error(y_test[mask], y_pred[mask]),
                        "rmse": rmse(y_test[mask], y_pred[mask]),
                    }
                )

    summary_rows: list[dict] = []
    residual_rows: list[dict] = []
    summary_frame = pd.DataFrame(repeated_rows)
    for model_type, group in summary_frame.groupby("model_type"):
        summary_rows.append(
            {
                "model_type": model_type,
                "r2_mean": group["r2"].mean(),
                "r2_std": group["r2"].std(ddof=0),
                "mae_mean": group["mae"].mean(),
                "mae_std": group["mae"].std(ddof=0),
                "rmse_mean": group["rmse"].mean(),
                "rmse_std": group["rmse"].std(ddof=0),
                "nmae_mean": group["nmae"].mean(),
                "nmae_std": group["nmae"].std(ddof=0),
                "latency_s_mean": group["latency_s"].mean(),
            }
        )
        residual_rows.append(
            {
                "model_type": model_type,
                "residual_mean_mean": group["residual_mean"].mean(),
                "residual_std_mean": group["residual_std"].mean(),
            }
        )

    write_csv(output_dir / "repeated_split_results.csv", repeated_rows)
    write_csv(output_dir / "metrics_summary.csv", summary_rows)
    write_csv(output_dir / "residual_statistics.csv", residual_rows)
    write_csv(output_dir / "category_metrics.csv", category_rows)
    logger.success(f"completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
