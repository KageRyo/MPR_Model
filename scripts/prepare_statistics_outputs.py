from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import t, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METRICS_FOR_OUTPUTS = ["r2", "mae", "rmse", "accuracy", "macro_f1"]
COMPLETE_INPUT_METRICS = ["r2", "mae", "rmse", "macro_f1"]
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
        description="Prepare publication statistics outputs from the missing-indicator result bundle."
    )
    parser.add_argument("--bundle-dir", default="results/package")
    parser.add_argument("--output-dir", default="statistics/outputs")
    parser.add_argument(
        "--predictions-csv",
        default="results/missing_indicator_robustness/predictions/predictions_long.csv",
        help="Prediction rows used for hold-out paired absolute-error tests.",
    )
    parser.add_argument("--update-production-model", action="store_true")
    parser.add_argument(
        "--archive-legacy-50000-artifacts",
        dest="archive_legacy_50000_artifacts",
        action="store_true",
    )
    return parser.parse_args()


def read_csv(csv_dir: Path, name: str) -> pd.DataFrame:
    path = csv_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing result CSV: {path}")
    return pd.read_csv(path)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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


def format_p_value(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    if value == 0.0:
        return "<1e-300"
    return format(float(value), ".17g")


def mean_diff_ci(diff: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    if len(diff) < 2:
        return (np.nan, np.nan)
    mean = float(np.mean(diff))
    sem = float(np.std(diff, ddof=1) / math.sqrt(len(diff)))
    critical = float(t.ppf((1.0 + confidence) / 2.0, df=len(diff) - 1))
    return (mean - critical * sem, mean + critical * sem)


def paired_comparison_result(model_a: str, model_b: str, mean_diff: float, significant: bool) -> str:
    lower_model = model_a if mean_diff < 0 else model_b
    if significant:
        return f"{lower_model} has significantly lower mean absolute error"
    return f"not significant; lower mean absolute error: {lower_model}"


def paired_tests_from_prediction_rows(predictions_csv: Path, source: str = "external_10714") -> pd.DataFrame:
    usecols = [
        "source",
        "missing_set",
        "experiment_mode",
        "experiment",
        "seed",
        "model_type",
        "row_id",
        "abs_error",
    ]
    predictions = pd.read_csv(predictions_csv, usecols=usecols)
    predictions = predictions[predictions["source"] == source].copy()
    predictions["abs_error"] = predictions["abs_error"].astype(float)

    pending: list[dict[str, Any]] = []
    p_values: list[float] = []
    group_columns = ["source", "missing_set", "experiment_mode", "experiment"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        group_source, missing_set, mode, experiment = keys
        wide = group.pivot_table(
            index=["seed", "row_id"],
            columns="model_type",
            values="abs_error",
            aggfunc="first",
        )
        for model_a, model_b in itertools.combinations(sorted(wide.columns), 2):
            paired = wide[[model_a, model_b]].dropna()
            diff = (paired[model_a] - paired[model_b]).to_numpy(dtype=float)
            if len(diff) == 0:
                p_value = np.nan
                mean_diff = np.nan
                ci_low = np.nan
                ci_high = np.nan
                mean_a = np.nan
                mean_b = np.nan
            else:
                mean_diff = float(np.mean(diff))
                ci_low, ci_high = mean_diff_ci(diff)
                mean_a = float(paired[model_a].mean())
                mean_b = float(paired[model_b].mean())
                try:
                    _, p_value = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="asymptotic")
                except ValueError:
                    p_value = np.nan
            pending.append(
                {
                    "source": group_source,
                    "missing_set": missing_set,
                    "experiment_mode": mode,
                    "experiment": experiment,
                    "metric": "absolute_error",
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_pairs": int(len(diff)),
                    "mean_absolute_error_a": mean_a,
                    "mean_absolute_error_b": mean_b,
                    "mean_difference_a_minus_b": mean_diff,
                    "mean_difference_ci95_low": ci_low,
                    "mean_difference_ci95_high": ci_high,
                    "wilcoxon_p_value": p_value,
                    "lower_mean_error_model": model_a if len(diff) and mean_diff < 0 else model_b,
                }
            )
            p_values.append(p_value)

    adjusted = holm_adjust(p_values)
    for row, p_adj in zip(pending, adjusted):
        significant = bool(p_adj < 0.05) if pd.notna(p_adj) else False
        row["holm_adjusted_p_value"] = p_adj
        row["wilcoxon_p_value_formatted"] = format_p_value(row["wilcoxon_p_value"])
        row["holm_adjusted_p_value_formatted"] = format_p_value(p_adj)
        row["model_comparison_result"] = paired_comparison_result(
            str(row["model_a"]),
            str(row["model_b"]),
            float(row["mean_difference_a_minus_b"]),
            significant,
        )
        row["significant_at_0_05"] = significant

    output = pd.DataFrame(pending)
    columns = [
        "source",
        "missing_set",
        "experiment_mode",
        "experiment",
        "metric",
        "model_a",
        "model_b",
        "n_pairs",
        "mean_absolute_error_a",
        "mean_absolute_error_b",
        "mean_difference_a_minus_b",
        "mean_difference_ci95_low",
        "mean_difference_ci95_high",
        "wilcoxon_p_value",
        "wilcoxon_p_value_formatted",
        "holm_adjusted_p_value",
        "holm_adjusted_p_value_formatted",
        "lower_mean_error_model",
        "model_comparison_result",
        "significant_at_0_05",
    ]
    return output[columns].sort_values(["missing_set", "experiment_mode", "experiment", "model_a", "model_b"])


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


def make_complete_input_performance(metrics_summary: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame:
    output = metrics_summary[
        (metrics_summary["source"] == "external_10714")
        & (metrics_summary["missing_set"] == "complete")
        & (metrics_summary["experiment_mode"] == "full_reference")
    ].copy()
    output = add_ci_columns(output, ci, COMPLETE_INPUT_METRICS)
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
    return output[columns].sort_values(["mae_mean", "rmse_mean", "model_type"])


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


def make_missing_indicator_robustness(best: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame:
    output = best[best["source"] == "external_10714"].copy()
    output = add_ci_columns(output, ci, ["r2", "mae", "rmse", "macro_f1"])
    output["available_indicators"] = output["missing_set"].map(
        {
            "complete": "DO / BOD / NH3N / EC / SS",
            "missing_bod": "DO / NH3N / EC / SS",
            "missing_nh3n": "DO / BOD / EC / SS",
            "missing_bod_nh3n": "DO / EC / SS",
        }
    )
    output["interpretation"] = output.apply(interpretation_for, axis=1)
    order = {
        "full_reference": 0,
        "inference_dropout": 1,
        "reduced_retraining": 2,
        "indicator_reconstruction": 3,
    }
    output["mode_order"] = output["experiment_mode"].map(order).fillna(99)
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
    return output.sort_values(["missing_set", "mode_order", "mae_mean"])[columns]


def make_cpu_only_timing(cpu_timing: pd.DataFrame) -> pd.DataFrame:
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
    detection = detection[
        (detection["missing_set"] == "complete")
        & (detection["experiment_mode"] == "full_reference")
    ].copy()
    monotonicity = monotonicity[
        (monotonicity["missing_set"] == "complete")
        & (monotonicity["experiment_mode"] == "full_reference")
    ].copy()
    detection_summary = (
        detection.groupby(
            ["missing_set", "experiment_mode", "model_type", "stress_scenario", "severity"],
            as_index=False,
        )
        .agg(
            mean_window_detection_rate_mean_decrease=("window_detection_rate_mean_decrease", "mean"),
            mean_window_detection_rate_drop_ge_1=("window_detection_rate_drop_ge_1", "mean"),
            mean_window_detection_rate_any_category_worse=("window_detection_rate_any_category_worse", "mean"),
            mean_pct_category_worse=("mean_pct_category_worse", "mean"),
            mean_pct_score_decreased=("mean_pct_score_decreased", "mean"),
            mean_pct_drop_ge_1=("mean_pct_drop_ge_1", "mean"),
            mean_pct_drop_ge_5=("mean_pct_drop_ge_5", "mean"),
            mean_score_drop=("mean_score_drop", "mean"),
            mean_delta_score=("mean_delta_score", "mean"),
        )
    )
    mono_summary = (
        monotonicity.groupby(
            ["missing_set", "experiment_mode", "model_type", "stress_scenario"],
            as_index=False,
        )
        .agg(
            mean_severity_monotonicity_rate_drop=("severity_monotonicity_rate_drop", "mean"),
            min_severity_monotonicity_rate_drop=("severity_monotonicity_rate_drop", "min"),
        )
    )
    return detection_summary.merge(
        mono_summary,
        on=["missing_set", "experiment_mode", "model_type", "stress_scenario"],
        how="left",
    ).sort_values(["model_type", "stress_scenario", "severity"])


def make_feature_score_correlations() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    datasets = {
        "processed_60714": PROJECT_ROOT / "data" / "dataV1.csv",
        "subset_50000": PROJECT_ROOT / "data" / "dataV1_50000.csv",
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


def paired_summary_markdown(paired_tests: pd.DataFrame) -> list[str]:
    subset = paired_tests[
        (paired_tests["source"] == "external_10714")
        & (paired_tests["missing_set"] == "complete")
        & (paired_tests["experiment_mode"] == "full_reference")
        & (paired_tests["experiment"] == "full_reference")
    ].copy()
    if subset.empty:
        return []
    lines = [
        "## Pairwise Error Tests",
        "",
        "Complete-input full-reference comparisons use external hold-out row absolute-error differences. Negative A-B values favor model A; positive values favor model B.",
        "",
        "| Comparison | Mean absolute-error difference (A - B) | 95% CI | Holm p | Result | Significant |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in subset.iterrows():
        comparison = f"{row['model_a']} vs {row['model_b']}"
        mean_diff = format(float(row["mean_difference_a_minus_b"]), ".6g")
        ci = (
            f"[{format(float(row['mean_difference_ci95_low']), '.6g')}, "
            f"{format(float(row['mean_difference_ci95_high']), '.6g')}]"
        )
        significant = "yes" if bool(row["significant_at_0_05"]) else "no"
        lines.append(
            f"| {comparison} | {mean_diff} | {ci} | {row['holm_adjusted_p_value_formatted']} | "
            f"{row['model_comparison_result']} | {significant} |"
        )
    lines.append("")
    return lines


def write_report(
    output_dir: Path,
    complete_input_performance: pd.DataFrame,
    missing_indicator_robustness: pd.DataFrame,
    cpu_only_timing: pd.DataFrame,
    stress_summary: pd.DataFrame,
    paired_tests: pd.DataFrame,
) -> None:
    best_full = complete_input_performance.iloc[0]
    missing_nh3n = missing_indicator_robustness[
        (missing_indicator_robustness["missing_set"] == "missing_nh3n")
        & (missing_indicator_robustness["experiment_mode"] == "reduced_retraining")
    ].head(1)
    missing_bod_nh3n = missing_indicator_robustness[
        (missing_indicator_robustness["missing_set"] == "missing_bod_nh3n")
        & (missing_indicator_robustness["experiment_mode"] == "reduced_retraining")
    ].head(1)
    lines = [
        "# Statistical Summary",
        "",
        "## Scope",
        "",
        "This report summarizes the missing-indicator robustness, Stress107, and CPU-only timing outputs.",
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
            "- Pairwise model comparisons use external hold-out row absolute-error differences with Wilcoxon signed-rank tests and Holm correction.",
            "",
            "## Output Files",
            "",
            "- `complete_input_performance.csv`",
            "- `missing_indicator_robustness.csv`",
            "- `cpu_only_timing.csv`",
            "- `stress107_summary.csv`",
            "- `feature_score_correlations.csv`",
            "- `bootstrap_ci.csv`",
            "- `paired_error_tests.csv`",
            "- `sample_size_sensitivity.csv`",
            "- `sample_size_metrics_by_fold.csv`",
            "",
            "## Sample-Size Sensitivity",
            "",
            "The sample-size experiment evaluates six surrogate models using 1,000, 10,000, and 50,000 rows under stratified 80/20 splits. The summary output is `sample_size_sensitivity.csv`, and fold-level results are provided in `sample_size_metrics_by_fold.csv`.",
            "",
        ]
    )
    lines.extend(paired_summary_markdown(paired_tests))
    lines.extend(
        [
            "## Reporting Boundary",
            "",
            "Stress107 reduces dependence on a single selected middle window, but it does not prove absence of all sampling bias and is not a real pollution-event validation.",
        ]
    )
    (output_dir / "statistical_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_production_models(output_dir: Path) -> dict[str, Any]:
    seed_metrics = pd.read_csv(output_dir / "metrics_by_seed.csv")
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
        target_path = target_dir / f"{MODEL_ARTIFACT_PREFIX[model_type]}Ver.2.0-50000-seed{int(selected['seed'])}.pkl"
        joblib.dump(model, target_path)

        artifacts.append(
            {
                "model_type": model_type,
                "production_artifact": str(target_path.relative_to(PROJECT_ROOT)),
                "source_artifact": display_path(bundle_path),
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
        for source in sorted(source_dir.glob("*50000*.pkl")):
            if "-seed" in source.name:
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
    predictions_csv = (PROJECT_ROOT / args.predictions_csv).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    best = read_csv(csv_dir, "missing_indicator_best_by_experiment_source.csv")
    metrics_summary = read_csv(csv_dir, "missing_indicator_metrics_summary.csv")
    metrics_by_seed = read_csv(csv_dir, "missing_indicator_metrics_by_seed.csv")
    bootstrap_ci = read_csv(csv_dir, "missing_indicator_bootstrap_ci.csv")
    paired_tests = paired_tests_from_prediction_rows(predictions_csv)
    cpu_timing = read_csv(csv_dir, "cpu_only_inference_timing_summary.csv")
    stress_detection = read_csv(csv_dir, "stress107_detection_summary.csv")
    stress_mono = read_csv(csv_dir, "stress107_severity_monotonicity.csv")

    complete_input_performance = make_complete_input_performance(metrics_summary, bootstrap_ci)
    missing_indicator_robustness = make_missing_indicator_robustness(best, bootstrap_ci)
    cpu_only_timing = make_cpu_only_timing(cpu_timing)
    stress107_summary = make_stress107_summary(stress_detection, stress_mono)
    correlations = make_feature_score_correlations()

    complete_input_performance.to_csv(output_dir / "complete_input_performance.csv", index=False)
    missing_indicator_robustness.to_csv(output_dir / "missing_indicator_robustness.csv", index=False)
    cpu_only_timing.to_csv(output_dir / "cpu_only_timing.csv", index=False)
    stress107_summary.to_csv(output_dir / "stress107_summary.csv", index=False)
    bootstrap_ci.to_csv(output_dir / "bootstrap_ci.csv", index=False)
    paired_tests.to_csv(output_dir / "paired_error_tests.csv", index=False)
    metrics_by_seed.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    stress_detection.to_csv(output_dir / "stress107_detection_summary.csv", index=False)
    stress_mono.to_csv(output_dir / "stress107_severity_monotonicity.csv", index=False)
    correlations.to_csv(output_dir / "feature_score_correlations.csv", index=False)

    write_report(
        output_dir,
        complete_input_performance,
        missing_indicator_robustness,
        cpu_only_timing,
        stress107_summary,
        paired_tests,
    )

    manifest: dict[str, Any] = {
        "bundle_dir": display_path(bundle_dir),
        "output_dir": display_path(output_dir),
        "outputs": [
            "complete_input_performance.csv",
            "missing_indicator_robustness.csv",
            "cpu_only_timing.csv",
            "stress107_summary.csv",
            "feature_score_correlations.csv",
            "paired_error_tests.csv",
            "sample_size_sensitivity.csv",
            "sample_size_metrics_by_fold.csv",
        ],
        "sample_size_metrics_source": "results/sample_size_experiments/metrics",
        "sample_size_outputs_scope": "requested 1,000, 10,000, and 50,000 row settings; local 5,000 row results are excluded from the publication-facing sample-size CSVs.",
        "paired_error_tests_source": display_path(predictions_csv),
        "paired_error_tests_method": "Wilcoxon signed-rank tests over paired external hold-out row absolute-error differences with Holm correction; CI is a 95% t interval for the mean paired difference.",
        "large_artifacts_policy": "Large raw models/predictions remain under ignored results/missing_indicator_robustness and results/stress107 directories and are not committed.",
    }

    if args.archive_legacy_50000_artifacts:
        archive_legacy_50000_artifacts()
    if args.update_production_model:
        manifest["production_models"] = write_production_models(output_dir)

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.success(f"Statistics outputs written to {output_dir}")


if __name__ == "__main__":
    main()
