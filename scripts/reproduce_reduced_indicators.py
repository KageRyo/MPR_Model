from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reproduce_results import (  # noqa: E402
    build_model,
    evaluate_predictions,
    require_model_support,
    resolve_compute_device,
    resolve_output_dir,
    rmse,
    score_to_category,
    write_csv,
)
from src.settings import FEATURE_COLUMNS  # noqa: E402
from src.wqi import categorize_score, direct_wqi5_score  # noqa: E402


FULL_FEATURES = tuple(FEATURE_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reduced_indicator_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--compute-device", choices=["cpu", "gpu", "auto"], default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--lightgbm-gpu-backend", choices=["gpu", "cuda"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scenario_features(config: dict) -> dict[str, list[str]]:
    scenarios = config.get("scenarios", {})
    if not scenarios:
        raise ValueError("Reduced-indicator config must define at least one scenario.")

    parsed: dict[str, list[str]] = {}
    allowed_features = set(FEATURE_COLUMNS)
    for scenario_name, scenario_config in scenarios.items():
        features = list(scenario_config.get("features", []))
        if not features:
            raise ValueError(f"Scenario {scenario_name!r} must define features.")
        if len(set(features)) != len(features):
            raise ValueError(f"Scenario {scenario_name!r} has duplicate features: {features}")
        unknown = sorted(set(features) - allowed_features)
        if unknown:
            raise ValueError(f"Scenario {scenario_name!r} has unsupported features: {unknown}")
        parsed[scenario_name] = features
    return parsed


def direct_wqi5_predictions(frame: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            direct_wqi5_score(
                do=row.DO,
                bod=row.BOD,
                nh3n=row.NH3N,
                ec=row.EC,
                ss=row.SS,
            )
            for row in frame[list(FULL_FEATURES)].itertuples(index=False)
        ]
    )


def append_category_rows(
    rows: list[dict],
    *,
    seed: int,
    scenario: str,
    purpose: str,
    features: list[str],
    model_type: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    y_test_categories = np.array(score_to_category(y_test))
    for category in sorted(set(y_test_categories)):
        mask = y_test_categories == category
        rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "purpose": purpose,
                "features": "|".join(features),
                "feature_count": len(features),
                "model_type": model_type,
                "category": category,
                "count": int(mask.sum()),
                "mae": mean_absolute_error(y_test[mask], y_pred[mask]),
                "rmse": rmse(y_test[mask], y_pred[mask]),
            }
        )


def summarize_results(repeated_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    summary_frame = pd.DataFrame(repeated_rows)
    summary_rows: list[dict] = []
    for (scenario, model_type), group in summary_frame.groupby(["scenario", "model_type"]):
        first_row = group.iloc[0]
        summary_rows.append(
            {
                "scenario": scenario,
                "purpose": first_row["purpose"],
                "features": first_row["features"],
                "feature_count": int(first_row["feature_count"]),
                "model_type": model_type,
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
                "residual_mean_mean": group["residual_mean"].mean(),
                "residual_std_mean": group["residual_std"].mean(),
            }
        )

    best_rows: list[dict] = []
    summary_for_best = pd.DataFrame(summary_rows)
    surrogate_summary = summary_for_best[summary_for_best["model_type"] != "direct_wqi5"]
    for scenario, group in surrogate_summary.groupby("scenario"):
        best = group.sort_values(["mae_mean", "rmse_mean", "model_type"]).iloc[0].to_dict()
        best["selection_metric"] = "lowest_mae_mean"
        best_rows.append(best)
    return summary_rows, best_rows


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_path = (PROJECT_ROOT / config["dataset"]).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Configured dataset does not exist: {data_path}")

    configured_output_dir = args.output_dir or config["output_dir"]
    output_dir = resolve_output_dir((PROJECT_ROOT / configured_output_dir).resolve(), overwrite=args.overwrite)

    requested_compute_device = args.compute_device or config.get("compute_device", "cpu")
    compute_device = resolve_compute_device(requested_compute_device)
    gpu_id = args.gpu_id if args.gpu_id is not None else int(config.get("gpu_id", 0))
    lightgbm_gpu_backend = args.lightgbm_gpu_backend or config.get("lightgbm_gpu_backend", "gpu")
    scenarios = scenario_features(config)
    scenario_purposes = {
        scenario_name: str(scenario_config.get("purpose", ""))
        for scenario_name, scenario_config in config.get("scenarios", {}).items()
    }
    models = list(config["models"])
    include_direct_baseline = bool(config.get("include_direct_wqi5_full_baseline", True))

    logger.info(
        "compute_device={} requested={} gpu_id={} lightgbm_gpu_backend={}",
        compute_device,
        requested_compute_device,
        gpu_id,
        lightgbm_gpu_backend,
    )

    frame = pd.read_csv(data_path)
    frame["wqi5_category"] = frame["Score"].apply(lambda value: categorize_score(value)[0])
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
        train_idx, test_idx = next(splitter.split(frame[FEATURE_COLUMNS], strata))
        y_train = y[train_idx]
        y_test = y[test_idx]

        for scenario, features in scenarios.items():
            logger.info(f"seed={seed}: running scenario={scenario} features={features}")
            purpose = scenario_purposes.get(scenario, "")
            X_train = frame.iloc[train_idx][features]
            X_test = frame.iloc[test_idx][features]

            if include_direct_baseline and len(features) == len(FULL_FEATURES) and set(features) == set(FULL_FEATURES):
                started = time.perf_counter()
                y_pred = direct_wqi5_predictions(frame.iloc[test_idx])
                latency_s = time.perf_counter() - started
                metrics = evaluate_predictions("direct_wqi5", y_test, y_pred, latency_s)
                metrics.update(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "purpose": purpose,
                        "features": "|".join(features),
                        "feature_count": len(features),
                    }
                )
                repeated_rows.append(metrics)
                append_category_rows(
                    category_rows,
                    seed=seed,
                    scenario=scenario,
                    purpose=purpose,
                    features=features,
                    model_type="direct_wqi5",
                    y_test=y_test,
                    y_pred=y_pred,
                )

            for model_type in models:
                if model_type == "direct_wqi5":
                    logger.warning("Skipping direct_wqi5 from reduced-indicator model list; it is handled as full-input baseline.")
                    continue
                logger.info(f"seed={seed}: scenario={scenario}: running model={model_type}")
                require_model_support(model_type)
                estimator = build_model(
                    model_type,
                    compute_device=compute_device,
                    gpu_id=gpu_id,
                    lightgbm_gpu_backend=lightgbm_gpu_backend,
                )
                started = time.perf_counter()
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
                latency_s = time.perf_counter() - started

                metrics = evaluate_predictions(model_type, y_test, y_pred, latency_s)
                metrics.update(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "purpose": purpose,
                        "features": "|".join(features),
                        "feature_count": len(features),
                    }
                )
                repeated_rows.append(metrics)
                append_category_rows(
                    category_rows,
                    seed=seed,
                    scenario=scenario,
                    purpose=purpose,
                    features=features,
                    model_type=model_type,
                    y_test=y_test,
                    y_pred=y_pred,
                )

    summary_rows, best_rows = summarize_results(repeated_rows)
    write_csv(output_dir / "reduced_indicator_results.csv", repeated_rows)
    write_csv(output_dir / "reduced_indicator_summary.csv", summary_rows)
    write_csv(output_dir / "reduced_indicator_category_metrics.csv", category_rows)
    write_csv(output_dir / "best_surrogate_by_scenario.csv", best_rows)
    logger.success(f"completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
