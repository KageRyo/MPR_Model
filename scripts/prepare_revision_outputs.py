from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METRICS_FOR_TABLES = ["r2", "mae", "rmse", "accuracy", "macro_f1"]
TABLE6_METRICS = ["r2", "mae", "rmse", "macro_f1"]
FEATURE_COLUMNS = ["DO", "BOD", "NH3N", "EC", "SS"]
MODEL_ARTIFACT_PREFIX = {
    "lightgbm": "modelLGBM",
    "lr": "modelLR",
    "mpr": "modelMPR",
    "rf": "modelRF",
    "svm": "modelSVM",
    "xgboost": "modelXGB",
}
MODEL_DIR = {
    "lightgbm": "LightGBM",
    "lr": "LR",
    "mpr": "MPR",
    "rf": "RF",
    "svm": "SVM",
    "xgboost": "XGBoost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the 2026-06-14 stress/missing-indicator revision outputs."
    )
    parser.add_argument("--bundle-dir", default="results_20260614_stress")
    parser.add_argument("--output-dir", default="statistics/outputs")
    parser.add_argument("--update-production-model", action="store_true")
    parser.add_argument(
        "--archive-legacy-50000-artifacts",
        "--archive-legacy-xgboost",
        dest="archive_legacy_50000_artifacts",
        action="store_true",
    )
    return parser.parse_args()


def read_csv(csv_dir: Path, name: str) -> pd.DataFrame:
    path = csv_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing revision result CSV: {path}")
    return pd.read_csv(path)


def ci_lookup(ci: pd.DataFrame) -> dict[tuple[str, str, str, str, str, str], tuple[float, float]]:
    lookup: dict[tuple[str, str, str, str, str, str], tuple[float, float]] = {}
    for _, row in ci.iterrows():
        key = (
            str(row["source"]),
            str(row["missing_set"]),
            str(row["experiment_mode"]),
            str(row["experiment"]),
            str(row["model_type"]),
            str(row["metric"]),
        )
        lookup[key] = (float(row["ci95_low"]), float(row["ci95_high"]))
    return lookup


def add_ci_columns(frame: pd.DataFrame, ci: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    lookup = ci_lookup(ci)
    out = frame.copy()
    for metric in metrics:
        lows: list[float | None] = []
        highs: list[float | None] = []
        for _, row in out.iterrows():
            key = (
                str(row["source"]),
                str(row["missing_set"]),
                str(row["experiment_mode"]),
                str(row["experiment"]),
                str(row["model_type"]),
                metric,
            )
            low, high = lookup.get(key, (None, None))
            lows.append(low)
            highs.append(high)
        out[f"{metric}_ci95_low"] = lows
        out[f"{metric}_ci95_high"] = highs
    return out


def make_table6(metrics_summary: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame:
    table = metrics_summary[
        (metrics_summary["source"] == "external_10714")
        & (metrics_summary["missing_set"] == "complete")
        & (metrics_summary["experiment_mode"] == "full_reference")
    ].copy()
    table = add_ci_columns(table, ci, TABLE6_METRICS)
    columns = [
        "model_type",
        "n_runs",
        "n_per_run",
        "r2_mean",
        "r2_ci95_low",
        "r2_ci95_high",
        "mae_mean",
        "mae_ci95_low",
        "mae_ci95_high",
        "rmse_mean",
        "rmse_ci95_low",
        "rmse_ci95_high",
        "macro_f1_mean",
        "macro_f1_ci95_low",
        "macro_f1_ci95_high",
        "accuracy_mean",
        "latency_s_mean",
        "training_s_mean",
    ]
    return table[columns].sort_values(["mae_mean", "rmse_mean", "model_type"])


def interpretation_for(row: pd.Series) -> str:
    missing_set = row["missing_set"]
    mode = row["experiment_mode"]
    r2 = float(row["r2_mean"])
    if missing_set == "complete":
        return "complete five-indicator reference surrogate; direct WQI5 remains the reference method"
    if missing_set == "missing_nh3n" and mode == "reduced_retraining":
        return "useful auxiliary estimate when NH3N is unavailable; not a formula replacement"
    if missing_set == "missing_nh3n" and mode == "indicator_reconstruction":
        return "two-stage reconstruction remains feasible for NH3N but adds pipeline complexity"
    if missing_set == "missing_bod" and mode == "inference_dropout":
        return "full model shows partial robustness to inference-time BOD loss"
    if missing_set == "missing_bod" and r2 < 0.2:
        return "external generalization is weak; BOD is a critical indicator"
    if missing_set == "missing_bod_nh3n" and mode == "inference_dropout":
        return "partial stress setting; insufficient as a complete-input replacement"
    if missing_set == "missing_bod_nh3n" and r2 < 0.0:
        return "not reliable on external hold-out; only a coarse screening boundary case"
    return "deployment-constrained auxiliary result; interpret conservatively"


def make_table7(best: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame:
    table = best[best["source"] == "external_10714"].copy()
    table = add_ci_columns(table, ci, ["r2", "mae", "rmse", "macro_f1"])
    table["available_indicators"] = table["missing_set"].map(
        {
            "complete": "DO / BOD / NH3N / EC / SS",
            "missing_bod": "DO / NH3N / EC / SS",
            "missing_nh3n": "DO / BOD / EC / SS",
            "missing_bod_nh3n": "DO / EC / SS",
        }
    )
    table["interpretation"] = table.apply(interpretation_for, axis=1)
    order = {
        "full_reference": 0,
        "inference_dropout": 1,
        "reduced_retraining": 2,
        "indicator_reconstruction": 3,
    }
    table["mode_order"] = table["experiment_mode"].map(order).fillna(99)
    columns = [
        "missing_set",
        "experiment_mode",
        "available_indicators",
        "model_type",
        "r2_mean",
        "r2_ci95_low",
        "r2_ci95_high",
        "mae_mean",
        "mae_ci95_low",
        "mae_ci95_high",
        "rmse_mean",
        "rmse_ci95_low",
        "rmse_ci95_high",
        "macro_f1_mean",
        "macro_f1_ci95_low",
        "macro_f1_ci95_high",
        "accuracy_mean",
        "interpretation",
    ]
    return table.sort_values(["missing_set", "mode_order", "mae_mean"])[columns]


def make_table8(cpu_timing: pd.DataFrame) -> pd.DataFrame:
    out = cpu_timing.copy()
    if "latency_s_mean" in out.columns:
        out["throughput_rows_per_s_mean"] = out["n_rows"] / out["latency_s_mean"]
    if "latency_s_std" in out.columns:
        out["latency_ms_mean"] = out["latency_s_mean"] * 1000.0
        out["latency_ms_std"] = out["latency_s_std"] * 1000.0
    preferred_columns = [
        "source",
        "missing_set",
        "experiment_mode",
        "experiment",
        "model_type",
        "n_repeats",
        "n_rows",
        "latency_s_mean",
        "latency_s_std",
        "latency_ms_mean",
        "latency_ms_std",
        "latency_per_row_ms_mean",
        "latency_per_row_ms_std",
        "throughput_rows_per_s_mean",
    ]
    existing = [column for column in preferred_columns if column in out.columns]
    if existing:
        return out[existing].copy()
    return out


def make_stress107_summary(detection: pd.DataFrame, monotonicity: pd.DataFrame) -> pd.DataFrame:
    detection_summary = (
        detection.groupby(["stress_scenario", "severity"], as_index=False)
        .agg(
            mean_window_detection_rate_mean_decrease=("window_detection_rate_mean_decrease", "mean"),
            mean_window_detection_rate_drop_ge_1=("window_detection_rate_drop_ge_1", "mean"),
            mean_pct_category_worse=("mean_pct_category_worse", "mean"),
            mean_score_drop=("mean_score_drop", "mean"),
        )
    )
    mono_summary = (
        monotonicity.groupby("stress_scenario", as_index=False)
        .agg(
            mean_severity_monotonicity_rate_drop=("severity_monotonicity_rate_drop", "mean"),
            min_severity_monotonicity_rate_drop=("severity_monotonicity_rate_drop", "min"),
        )
    )
    return detection_summary.merge(mono_summary, on="stress_scenario", how="left")


def make_feature_score_correlations() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    datasets = {
        "processed_60714": PROJECT_ROOT / "data" / "dataV1.csv",
        "revision_subset_50000": PROJECT_ROOT / "data" / "dataV1_50000.csv",
    }
    for dataset_name, path in datasets.items():
        frame = pd.read_csv(path)
        for feature in FEATURE_COLUMNS:
            rows.append(
                {
                    "dataset": dataset_name,
                    "feature": feature,
                    "n": int(frame[[feature, "Score"]].dropna().shape[0]),
                    "pearson_r": float(frame[feature].corr(frame["Score"], method="pearson")),
                    "spearman_r": float(frame[feature].corr(frame["Score"], method="spearman")),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    table6: pd.DataFrame,
    table7: pd.DataFrame,
    table8: pd.DataFrame,
    stress_summary: pd.DataFrame,
) -> None:
    best_full = table6.iloc[0]
    missing_nh3n = table7[
        (table7["missing_set"] == "missing_nh3n")
        & (table7["experiment_mode"] == "reduced_retraining")
    ].head(1)
    missing_bod_nh3n = table7[
        (table7["missing_set"] == "missing_bod_nh3n")
        & (table7["experiment_mode"] == "reduced_retraining")
    ].head(1)
    lines = [
        "# Revision Statistical Summary",
        "",
        "## Scope",
        "",
        "This report freezes the 2026-06-14 missing-indicator robustness, Stress107, and CPU-only timing outputs.",
        "It replaces the earlier percentage-agreement tables with R2, MAE, RMSE, Macro-F1, bootstrap confidence intervals, and paired model tests.",
        "",
        "The task remains WQI5 surrogate regression, not future water-quality forecasting. Direct WQI5 computation remains the reference method when all five indicators are available.",
        "",
        "## Main Findings",
        "",
        f"- Complete-input external hold-out best model: `{best_full['model_type']}` with R2={best_full['r2_mean']:.4f}, MAE={best_full['mae_mean']:.4f}, RMSE={best_full['rmse_mean']:.4f}.",
    ]
    if not missing_nh3n.empty:
        row = missing_nh3n.iloc[0]
        lines.append(
            f"- Missing NH3N reduced retraining remains useful as an auxiliary setting: `{row['model_type']}` with R2={row['r2_mean']:.4f}, MAE={row['mae_mean']:.4f}."
        )
    if not missing_bod_nh3n.empty:
        row = missing_bod_nh3n.iloc[0]
        lines.append(
            f"- DO/EC/SS-only reduced retraining is not reliable on the external hold-out: `{row['model_type']}` with R2={row['r2_mean']:.4f}, MAE={row['mae_mean']:.4f}."
        )
    lines.extend(
        [
            "- Stress107 uses 107 sequential event windows, not 107-fold cross-validation.",
            "- CPU-only timing is the deployment-oriented inference-time reference; GPU/multicore acceleration is acceptable for experiment reproduction.",
            "",
            "## Output Tables",
            "",
            "- `revision_table6_complete_input_performance.csv`",
            "- `revision_table7_missing_indicator_robustness.csv`",
            "- `revision_table8_cpu_only_timing.csv`",
            "- `revision_table9_stress107_summary.csv`",
            "- `revision_feature_score_correlations.csv`",
            "- `revision_bootstrap_ci.csv`",
            "- `revision_paired_error_tests.csv`",
            "",
            "## Reporting Boundary",
            "",
            "Stress107 reduces dependence on a single selected middle window, but it does not prove absence of all sampling bias and is not a real pollution-event validation.",
        ]
    )
    (output_dir / "revision_statistical_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_production_models(output_dir: Path) -> dict[str, Any]:
    seed_metrics = pd.read_csv(output_dir / "revision_metrics_by_seed.csv")
    full_reference = seed_metrics[
        (seed_metrics["source"] == "external_10714")
        & (seed_metrics["experiment"] == "full_reference")
    ].copy()

    artifacts: list[dict[str, Any]] = []
    for model_type in sorted(MODEL_DIR):
        candidates = full_reference[full_reference["model_type"] == model_type].sort_values(["mae", "rmse", "seed"])
        if candidates.empty:
            raise RuntimeError(f"No full_reference seed artifact found for {model_type}")
        selected = candidates.iloc[0]
        bundle_path = PROJECT_ROOT / str(selected["model_path"])
        bundle = joblib.load(bundle_path)
        model = bundle["wqi_model"] if isinstance(bundle, dict) and "wqi_model" in bundle else bundle
        if not hasattr(model, "predict"):
            raise TypeError(f"Selected {model_type} artifact does not expose predict(): {bundle_path}")

        target_dir = PROJECT_ROOT / "models" / MODEL_DIR[model_type]
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{MODEL_ARTIFACT_PREFIX[model_type]}Ver.2.0-revision-50000-seed{int(selected['seed'])}.pkl"
        joblib.dump(model, target_path)

        artifacts.append(
            {
                "model_type": model_type,
                "production_artifact": str(target_path.relative_to(PROJECT_ROOT)),
                "source_artifact": str(bundle_path.relative_to(PROJECT_ROOT)),
                "seed": int(selected["seed"]),
                "source": str(selected["source"]),
                "experiment": str(selected["experiment"]),
                "metrics": {
                    "r2": float(selected["r2"]),
                    "mae": float(selected["mae"]),
                    "rmse": float(selected["rmse"]),
                    "accuracy": float(selected["accuracy"]),
                    "macro_f1": float(selected["macro_f1"]),
                },
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "lowest external_10714 MAE per model among complete-input full_reference seed artifacts",
        "artifacts": artifacts,
        "api_contract": "complete-input WQI5 surrogate; required features are DO, BOD, NH3N, EC, SS",
        "not_for": "missing-indicator replacement or future water-quality forecasting",
    }
    manifest_path = PROJECT_ROOT / "models" / "production_model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def archive_legacy_50000_artifacts() -> None:
    for model_type, directory_name in MODEL_DIR.items():
        source_dir = PROJECT_ROOT / "models" / directory_name
        if not source_dir.exists():
            continue
        archive_dir = PROJECT_ROOT / "models" / "archive" / "legacy_v1" / directory_name
        for source in sorted(source_dir.glob("*50000.pkl")):
            if "revision" in source.name:
                continue
            target = archive_dir / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                source.unlink()
            else:
                shutil.move(str(source), str(target))


def main() -> None:
    args = parse_args()
    bundle_dir = (PROJECT_ROOT / args.bundle_dir).resolve()
    csv_dir = bundle_dir / "organized" / "csv"
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    best = read_csv(csv_dir, "v2_best_by_experiment_source.csv")
    metrics_summary = read_csv(csv_dir, "v2_metrics_summary.csv")
    metrics_by_seed = read_csv(csv_dir, "v2_metrics_by_seed.csv")
    bootstrap_ci = read_csv(csv_dir, "v2_bootstrap_ci.csv")
    paired_tests = read_csv(csv_dir, "v2_paired_error_tests.csv")
    cpu_timing = read_csv(csv_dir, "v2_cpu_only_inference_timing_summary.csv")
    stress_detection = read_csv(csv_dir, "stress107_detection_summary.csv")
    stress_mono = read_csv(csv_dir, "stress107_severity_monotonicity.csv")

    table6 = make_table6(metrics_summary, bootstrap_ci)
    table7 = make_table7(best, bootstrap_ci)
    table8 = make_table8(cpu_timing)
    table9 = make_stress107_summary(stress_detection, stress_mono)
    correlations = make_feature_score_correlations()

    table6.to_csv(output_dir / "revision_table6_complete_input_performance.csv", index=False)
    table7.to_csv(output_dir / "revision_table7_missing_indicator_robustness.csv", index=False)
    table8.to_csv(output_dir / "revision_table8_cpu_only_timing.csv", index=False)
    table9.to_csv(output_dir / "revision_table9_stress107_summary.csv", index=False)
    bootstrap_ci.to_csv(output_dir / "revision_bootstrap_ci.csv", index=False)
    paired_tests.to_csv(output_dir / "revision_paired_error_tests.csv", index=False)
    metrics_by_seed.to_csv(output_dir / "revision_metrics_by_seed.csv", index=False)
    stress_detection.to_csv(output_dir / "revision_stress107_detection_summary.csv", index=False)
    stress_mono.to_csv(output_dir / "revision_stress107_severity_monotonicity.csv", index=False)
    correlations.to_csv(output_dir / "revision_feature_score_correlations.csv", index=False)

    write_report(output_dir, table6, table7, table8, table9)

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(bundle_dir.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "tables": [
            "revision_table6_complete_input_performance.csv",
            "revision_table7_missing_indicator_robustness.csv",
            "revision_table8_cpu_only_timing.csv",
            "revision_table9_stress107_summary.csv",
            "revision_feature_score_correlations.csv",
        ],
        "large_artifacts_policy": "Large raw models/predictions remain under ignored results_20260614_stress/raw and are not committed.",
    }

    if args.archive_legacy_50000_artifacts:
        archive_legacy_50000_artifacts()
    if args.update_production_model:
        manifest["production_models"] = write_production_models(output_dir)

    (output_dir / "revision_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.success(f"Revision outputs written to {output_dir}")


if __name__ == "__main__":
    main()
