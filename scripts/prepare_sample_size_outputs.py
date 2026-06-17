from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTED_DATASET_SIZES = (1000, 10000, 50000)
MODEL_ORDER = {
    "lr": 0,
    "mpr": 1,
    "svm": 2,
    "rf": 3,
    "xgboost": 4,
    "lightgbm": 5,
}
T_CRITICAL_95_DF4 = 2.7764451051977987


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare sample-size sensitivity outputs."
    )
    parser.add_argument(
        "--metrics-dir",
        default="results/sample_size_experiments/metrics",
        help="Directory containing metrics_summary.csv and metrics_by_fold.csv.",
    )
    parser.add_argument("--output-dir", default="statistics/outputs")
    return parser.parse_args()


def ci95_from_fold_summary(mean: float, std: float, n_folds: int) -> tuple[float, float]:
    half_width = T_CRITICAL_95_DF4 * std / math.sqrt(n_folds)
    return mean - half_width, mean + half_width


def make_sample_size_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary[summary["dataset_size"].isin(SELECTED_DATASET_SIZES)].copy()
    rows: list[dict[str, float | int | str]] = []

    for (dataset_size, model_type), group in summary.groupby(["dataset_size", "model_type"], sort=False):
        train = group[group["split"] == "train"].iloc[0]
        test = group[group["split"] == "test"].iloc[0]
        n_folds = int(test["folds"])
        r2_low, r2_high = ci95_from_fold_summary(float(test["r2_mean"]), float(test["r2_std"]), n_folds)
        mae_low, mae_high = ci95_from_fold_summary(float(test["mae_mean"]), float(test["mae_std"]), n_folds)
        rmse_low, rmse_high = ci95_from_fold_summary(float(test["rmse_mean"]), float(test["rmse_std"]), n_folds)

        rows.append(
            {
                "dataset_size": int(dataset_size),
                "model_type": str(model_type),
                "n_folds": n_folds,
                "train_n_mean": float(train["train_rows"]),
                "test_n_mean": float(test["test_rows"]),
                "train_r2_mean": float(train["r2_mean"]),
                "train_mae_mean": float(train["mae_mean"]),
                "train_rmse_mean": float(train["rmse_mean"]),
                "test_r2_mean": float(test["r2_mean"]),
                "test_mae_mean": float(test["mae_mean"]),
                "test_rmse_mean": float(test["rmse_mean"]),
                "test_r2_std": float(test["r2_std"]),
                "test_mae_std": float(test["mae_std"]),
                "test_rmse_std": float(test["rmse_std"]),
                "training_s_mean": float(test["fit_time_s_mean"]),
                "test_latency_s_mean": float(test["predict_time_s_mean"]),
                "test_r2_ci95_low": r2_low,
                "test_r2_ci95_high": r2_high,
                "test_mae_ci95_low": mae_low,
                "test_mae_ci95_high": mae_high,
                "test_rmse_ci95_low": rmse_low,
                "test_rmse_ci95_high": rmse_high,
            }
        )

    out = pd.DataFrame(rows)
    out["_model_order"] = out["model_type"].map(MODEL_ORDER)
    return out.sort_values(["dataset_size", "_model_order"]).drop(columns=["_model_order"])


def make_sample_size_metrics_by_fold(by_fold: pd.DataFrame) -> pd.DataFrame:
    by_fold = by_fold[by_fold["dataset_size"].isin(SELECTED_DATASET_SIZES)].copy()
    by_fold["n_samples"] = by_fold.apply(
        lambda row: int(row["train_rows"]) if row["split"] == "train" else int(row["test_rows"]),
        axis=1,
    )
    by_fold["training_s"] = by_fold["fit_time_s"]
    by_fold["prediction_s"] = by_fold["predict_time_s"]

    columns = [
        "dataset_size",
        "dataset",
        "fold",
        "model_type",
        "split",
        "n_samples",
        "r2",
        "mae",
        "rmse",
        "nmae",
        "accuracy",
        "macro_f1",
        "training_s",
        "prediction_s",
    ]
    out = by_fold[columns].copy()
    out["_model_order"] = out["model_type"].map(MODEL_ORDER)
    out["_split_order"] = out["split"].map({"train": 0, "test": 1})
    return out.sort_values(["dataset_size", "fold", "_model_order", "_split_order"]).drop(
        columns=["_model_order", "_split_order"]
    )


def main() -> None:
    args = parse_args()
    metrics_dir = PROJECT_ROOT / args.metrics_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(metrics_dir / "metrics_summary.csv")
    by_fold = pd.read_csv(metrics_dir / "metrics_by_fold.csv")

    make_sample_size_sensitivity(summary).to_csv(output_dir / "sample_size_sensitivity.csv", index=False)
    make_sample_size_metrics_by_fold(by_fold).to_csv(output_dir / "sample_size_metrics_by_fold.csv", index=False)


if __name__ == "__main__":
    main()
