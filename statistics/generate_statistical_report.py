from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_PATH = OUTPUT_DIR / "statistical_analysis_report.md"

MODEL_ORDER = ["LightGBM", "XGBoost", "RF", "SVM", "MPR", "LR"]
WQI_BAND_ORDER = ["Excellent", "Good", "Fair", "Poor", "Bad", "Terrible"]
CORE_METRICS = ["r2", "mae", "rmse", "mean_predictive_accuracy_pct"]
KEY_COMPARISONS = {
    ("LightGBM", "XGBoost"),
    ("LightGBM", "RF"),
    ("XGBoost", "RF"),
    ("LightGBM", "SVM"),
    ("LightGBM", "LR"),
    ("LightGBM", "MPR"),
}


def read_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing generated table: {path}")
    return pd.read_csv(path)


def fmt(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, str):
        return value
    value = float(value)
    if abs(value) >= 1000 or (0 < abs(value) < 0.0001):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str] | None = None) -> str:
    headers = headers or columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def prediction_metrics_table(metrics: pd.DataFrame, ci: pd.DataFrame) -> str:
    ci_wide = ci.pivot_table(index="model", columns="metric", values=["ci95_low", "ci95_high"], aggfunc="first")
    rows = []
    for model in MODEL_ORDER:
        group = metrics[metrics["model"] == model]
        if group.empty:
            continue
        row = group.iloc[0]
        rows.append(
            {
                "Model": model,
                "R2": fmt(row["r2"]),
                "MAE": fmt(row["mae"]),
                "RMSE": fmt(row["rmse"]),
                "MPA (%)": fmt(row["mean_predictive_accuracy_pct"]),
                "R2 95% CI": f"[{fmt(ci_wide.loc[model, ('ci95_low', 'r2')])}, {fmt(ci_wide.loc[model, ('ci95_high', 'r2')])}]",
                "MAE 95% CI": f"[{fmt(ci_wide.loc[model, ('ci95_low', 'mae')])}, {fmt(ci_wide.loc[model, ('ci95_high', 'mae')])}]",
                "RMSE 95% CI": f"[{fmt(ci_wide.loc[model, ('ci95_low', 'rmse')])}, {fmt(ci_wide.loc[model, ('ci95_high', 'rmse')])}]",
                "MPA 95% CI": (
                    f"[{fmt(ci_wide.loc[model, ('ci95_low', 'mean_predictive_accuracy_pct')])}, "
                    f"{fmt(ci_wide.loc[model, ('ci95_high', 'mean_predictive_accuracy_pct')])}]"
                ),
            }
        )
    table = pd.DataFrame(rows)
    return markdown_table(table, list(table.columns))


def best_rmse_by_sample_table(metric_ci: pd.DataFrame) -> str:
    subset = metric_ci[metric_ci["metric"] == "valid_rmse"].copy()
    idx = subset.groupby("sample_size")["mean"].idxmin()
    best = subset.loc[idx].sort_values("sample_size")
    rows = []
    for _, row in best.iterrows():
        ci = "NA" if pd.isna(row["ci95_low"]) else f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]"
        rows.append(
            {
                "Sample size": int(row["sample_size"]),
                "Best model": row["model"],
                "Mean RMSE": fmt(row["mean"]),
                "95% CI": ci,
                "Runs": int(row["n_runs"]),
            }
        )
    table = pd.DataFrame(rows)
    return markdown_table(table, list(table.columns))


def key_p_value_table(tests: pd.DataFrame) -> str:
    rows = []
    for _, row in tests.iterrows():
        pair = (row["model_a"], row["model_b"])
        reverse = (row["model_b"], row["model_a"])
        if pair not in KEY_COMPARISONS and reverse not in KEY_COMPARISONS:
            continue
        rows.append(
            {
                "Comparison": f"{row['model_a']} vs {row['model_b']}",
                "Mean absolute-error difference (A - B)": fmt(row["mean_difference_a_minus_b"]),
                "Bootstrap 95% CI for mean difference": f"[{fmt(row['bootstrap_ci95_low'])}, {fmt(row['bootstrap_ci95_high'])}]",
                "Wilcoxon p": row["wilcoxon_p_value"],
                "Holm p": row["holm_adjusted_p_value"],
                "Lower mean error": row["better_model_by_mean"],
            }
        )
    table = pd.DataFrame(rows)
    return markdown_table(table, list(table.columns))


def residual_table(residuals: pd.DataFrame) -> str:
    rows = []
    for model in MODEL_ORDER:
        group = residuals[residuals["model"] == model]
        if group.empty:
            continue
        row = group.iloc[0]
        rows.append(
            {
                "Model": model,
                "Residual mean": fmt(row["residual_mean"]),
                "Residual std": fmt(row["residual_std"]),
                "Skewness": fmt(row["residual_skewness"]),
                "Kurtosis": fmt(row["residual_kurtosis"]),
                "KS p": fmt(row["normality_ks_p_value"]),
            }
        )
    table = pd.DataFrame(rows)
    return markdown_table(table, list(table.columns))


def band_error_table(bands: pd.DataFrame) -> str:
    idx = bands.groupby("actual_wqi_band")["mae"].idxmin()
    best = bands.loc[idx].copy()
    best["band_order"] = best["actual_wqi_band"].map({label: i for i, label in enumerate(WQI_BAND_ORDER)})
    best = best.sort_values("band_order")
    rows = []
    for _, row in best.iterrows():
        rows.append(
            {
                "Actual WQI band": row["actual_wqi_band"],
                "Lowest-MAE model": row["model"],
                "n": int(row["n"]),
                "MAE": fmt(row["mae"]),
                "RMSE": fmt(row["rmse"]),
                "Bias": fmt(row["bias"]),
                "MPA (%)": fmt(row["mean_predictive_accuracy_pct"]),
            }
        )
    table = pd.DataFrame(rows)
    return markdown_table(table, list(table.columns))


def write_report() -> None:
    metric_ci = read_csv("metric_ci_by_runs")
    prediction_metrics = read_csv("test_prediction_metrics")
    prediction_ci = read_csv("test_bootstrap_ci")
    paired_tests = read_csv("test_paired_error_tests")
    residuals = read_csv("residual_diagnostics")
    band_errors = read_csv("error_by_wqi_band")
    present_bands = set(band_errors["actual_wqi_band"].unique())
    missing_bands = [band for band in WQI_BAND_ORDER if band not in present_bands]
    missing_band_note = (
        f" The 10,714-record hold-out set contains no {', '.join(missing_bands)} rows."
        if missing_bands
        else ""
    )

    lines = [
        "# Statistical Analysis Summary",
        "",
        "## Scope",
        "",
        "This summary reports statistical checks for WQI5 surrogate regression. The source records are the archived experiment workbook and the committed CSV datasets under `data/`. The analysis is a post-processing workflow: it does not retrain model artifacts, and derived metrics are recomputed from the recorded actual and predicted values.",
        "",
        "The primary task is continuous WQI5 score estimation. Reported hold-out metrics include R2, MAE, RMSE, residual diagnostics, and Mean Predictive Accuracy (MPA):",
        "",
        "```text",
        "MPA (%) = mean_i [(1 - |y_i - yhat_i| / y_i) * 100]",
        "```",
        "",
        "For positive reference scores, MPA is equivalent to `100% - MAPE(%)`.",
        "",
        "## Confidence Intervals and Tests",
        "",
        "- Run-level 95% intervals summarize repeated subset-benchmark metric logs.",
        "- Row-level bootstrap 95% intervals summarize the 10,714-record hold-out prediction set for R2, MAE, RMSE, and MPA.",
        "- Model comparisons use paired Wilcoxon signed-rank tests on absolute errors, followed by Holm correction.",
        "- Reported p-values smaller than floating-point reporting precision are shown as `<1e-300`.",
        "",
        "### Interval Definitions",
        "",
        "`metric_ci_by_runs.csv` reports run-level intervals computed as `mean +/- t_(0.975, n-1) * sample_std / sqrt(n)` from repeated benchmark metric logs. When only one run is available, the point estimate is reported and the interval bounds are left empty.",
        "",
        "`test_bootstrap_ci.csv` reports row-level bootstrap intervals. For each model, the 10,714 hold-out rows are resampled with replacement, the metric is recomputed on each bootstrap sample, and the 2.5th and 97.5th percentiles are reported.",
        "",
        "`test_paired_error_tests.csv` reports paired absolute-error differences on the inference evaluation set. For each row, `diff_i = |y_i - yhat_A_i| - |y_i - yhat_B_i|`; the interval is the 2.5th to 97.5th percentile range of bootstrapped mean differences. A negative mean difference means model A has lower average absolute error; a positive mean difference means model B has lower average absolute error. Intervals that include zero indicate that the average difference is small relative to its bootstrap uncertainty.",
        "",
        "## Best Validation RMSE by Sample Size",
        "",
        best_rmse_by_sample_table(metric_ci),
        "",
        "## Hold-out Prediction Metrics",
        "",
        prediction_metrics_table(prediction_metrics, prediction_ci),
        "",
        "## Pairwise Error Tests on the 10,714-Sample Inference Evaluation Set",
        "",
        "Each comparison uses paired absolute errors from the same evaluation rows. The reported difference is `model A absolute error - model B absolute error`.",
        "",
        key_p_value_table(paired_tests),
        "",
        "The full pairwise table is available in `statistics/outputs/test_paired_error_tests.csv`.",
        "",
        "## Residual Diagnostics",
        "",
        residual_table(residuals),
        "",
        "The KS p-values indicate departures from normal residual distributions. The residual statistics are used as diagnostics for bias, dispersion, asymmetry, and tail behavior.",
        "",
        "## Residual Figures",
        "",
        "### Overview",
        "",
        "![Residual overview](figures/residual_overview.png)",
        "",
        "![Residual Q-Q overview](figures/residual_qq_overview.png)",
        "",
        "### Model Diagnostics",
        "",
        "![LightGBM residual diagnostics](figures/residual_diagnostics_lightgbm.png)",
        "",
        "![XGBoost residual diagnostics](figures/residual_diagnostics_xgboost.png)",
        "",
        "![RF residual diagnostics](figures/residual_diagnostics_rf.png)",
        "",
        "![SVM residual diagnostics](figures/residual_diagnostics_svm.png)",
        "",
        "![MPR residual diagnostics](figures/residual_diagnostics_mpr.png)",
        "",
        "![LR residual diagnostics](figures/residual_diagnostics_lr.png)",
        "",
        "### Residual Histograms",
        "",
        "![LightGBM residual histogram](figures/residual_lightgbm.png)",
        "",
        "![XGBoost residual histogram](figures/residual_xgboost.png)",
        "",
        "![RF residual histogram](figures/residual_rf.png)",
        "",
        "![SVM residual histogram](figures/residual_svm.png)",
        "",
        "![MPR residual histogram](figures/residual_mpr.png)",
        "",
        "![LR residual histogram](figures/residual_lr.png)",
        "",
        "## Error by WQI Band",
        "",
        "WQI bands follow the backend category configuration used by the WaterMirror API: Excellent, Good, Fair, Poor, Bad, and Terrible. These rows describe regression error within each actual WQI band."
        + missing_band_note,
        "",
        band_error_table(band_errors),
        "",
        "## Generated Files",
        "",
        "- `metric_ci_by_runs.csv`",
        "- `paired_tests_by_runs.csv`",
        "- `test_prediction_metrics.csv`",
        "- `test_bootstrap_ci.csv`",
        "- `test_paired_error_tests.csv`",
        "- `residual_diagnostics.csv`",
        "- `error_by_wqi_band.csv`",
        "- `dataset_distribution_robustness.csv`",
        "- `sample_size_stability.csv`",
        "- `figures/residual_overview.png`",
        "- `figures/residual_qq_overview.png`",
        "- `figures/residual_<model>.png`",
        "- `figures/residual_diagnostics_<model>.png`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    write_report()
