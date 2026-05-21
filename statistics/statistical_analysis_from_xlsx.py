from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, kstest, norm, rankdata, skew, kurtosis, spearmanr, t, wilcoxon
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
INPUT_XLSX = BASE_DIR / "整理.xlsx"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_XLSX = OUTPUT_DIR / "statistical_analysis_outputs.xlsx"

MODEL_BLOCKS = {
    "XGBoost": 0,
    "LightGBM": 9,
    "RF": 18,
    "SVM": 27,
    "MPR": 36,
    "LR": 45,
}

PRED_BLOCKS = {
    "XGBoost": 0,
    "LightGBM": 5,
    "RF": 10,
    "SVM": 15,
    "MPR": 20,
    "LR": 25,
}

METRIC_COLUMNS = [
    "group",
    "train_r2",
    "valid_r2",
    "train_rmse",
    "valid_rmse",
    "train_mae",
    "valid_mae",
    "train_time_sec",
]

SAMPLE_SHEETS = ["100筆", "1000筆", "5000筆", "10000筆", "20000筆", "50000筆"]
DATASET_MAP = {
    100: DATA_DIR / "dataV1_100.csv",
    1000: DATA_DIR / "dataV1_1000.csv",
    5000: DATA_DIR / "dataV1_5000.csv",
    10000: DATA_DIR / "dataV1_10000.csv",
    20000: DATA_DIR / "dataV1_20000.csv",
    50000: DATA_DIR / "dataV1_50000.csv",
}
FULL_DATASET_PATH = DATA_DIR / "dataV1.csv"
FEATURE_COLUMNS = ["DO", "BOD", "NH3N", "EC", "SS", "Score"]
RUN_METRICS = ["valid_r2", "valid_mae", "valid_rmse", "train_time_sec"]
TEST_METRICS = ["r2", "mae", "rmse", "category_accuracy", "macro_f1"]


def ci_mean(series: pd.Series, confidence: float = 0.95) -> tuple[float, float, float, int]:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0

    mean = float(np.mean(x))
    if n < 2:
        return mean, np.nan, np.nan, n

    se = np.std(x, ddof=1) / np.sqrt(n)
    margin = float(t.ppf((1 + confidence) / 2, df=n - 1) * se)
    return mean, mean - margin, mean + margin, n


def bootstrap_ci_mean(
    values: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    batch_size: int = 250,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boot_means = np.empty(n_boot, dtype=float)
    cursor = 0
    while cursor < n_boot:
        current = min(batch_size, n_boot - cursor)
        idx = rng.integers(0, len(values), size=(current, len(values)))
        boot_means[cursor : cursor + current] = values[idx].mean(axis=1)
        cursor += current
    return float(np.mean(boot_means)), float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))


def holm_adjust(p_values: list[float]) -> list[float]:
    arr = np.asarray(p_values, dtype=float)
    adjusted = np.full(arr.shape, np.nan)
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return adjusted.tolist()

    valid = arr[valid_mask]
    order = np.argsort(valid)
    ordered = valid[order]
    m = len(ordered)
    holm = np.empty(m, dtype=float)
    running = 0.0
    for i, p in enumerate(ordered):
        value = min(1.0, (m - i) * p)
        running = max(running, value)
        holm[i] = running

    restored = np.empty(m, dtype=float)
    restored[order] = holm
    adjusted[valid_mask] = restored
    return adjusted.tolist()


def rank_biserial_from_diff(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return np.nan

    abs_diff = np.abs(diff)
    ranks = rankdata(abs_diff, method="average")
    pos_sum = np.sum(ranks[diff > 0])
    neg_sum = np.sum(ranks[diff < 0])
    total = len(diff) * (len(diff) + 1) / 2
    return float((pos_sum - neg_sum) / total)


def wqi_category(x: float) -> str:
    if x <= 30:
        return "Terrible"
    if x <= 50:
        return "Poor"
    if x <= 70:
        return "Medium"
    if x <= 85:
        return "Good"
    return "Excellent"


def load_metric_logs(path: Path) -> pd.DataFrame:
    all_rows: list[pd.DataFrame] = []
    for sheet in SAMPLE_SHEETS:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        sample_size = int(sheet.replace("筆", ""))
        for model, start_col in MODEL_BLOCKS.items():
            block = raw.iloc[1:, start_col : start_col + 8].copy()
            block.columns = METRIC_COLUMNS
            block = block.dropna(subset=["group"])
            block = block[~block["group"].astype(str).str.contains("平均值", na=False)]
            for col in METRIC_COLUMNS[1:]:
                block[col] = pd.to_numeric(block[col], errors="coerce")
            block["model"] = model
            block["sample_size"] = sample_size
            all_rows.append(block)
    return pd.concat(all_rows, ignore_index=True)


def summarize_metric_ci(metric_logs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sample_size, model), group in metric_logs.groupby(["sample_size", "model"], sort=True):
        for metric in RUN_METRICS:
            mean, low, high, n = ci_mean(group[metric])
            rows.append(
                {
                    "sample_size": sample_size,
                    "model": model,
                    "metric": metric,
                    "n_runs": n,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "std": float(group[metric].std(ddof=1)) if n >= 2 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def paired_metric_tests(metric_logs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_size in sorted(metric_logs["sample_size"].unique()):
        subset = metric_logs[metric_logs["sample_size"] == sample_size]
        for metric in ["valid_r2", "valid_mae", "valid_rmse"]:
            pivot = subset.pivot_table(index="group", columns="model", values=metric, aggfunc="first")
            p_values = []
            pending_rows = []
            for model_a, model_b in combinations(MODEL_BLOCKS, 2):
                if model_a not in pivot.columns or model_b not in pivot.columns:
                    continue
                paired = pivot[[model_a, model_b]].dropna()
                if len(paired) < 2:
                    pending_rows.append(
                        {
                            "sample_size": sample_size,
                            "metric": metric,
                            "model_a": model_a,
                            "model_b": model_b,
                            "n_pairs": len(paired),
                            "mean_difference_a_minus_b": np.nan,
                            "bootstrap_ci95_low": np.nan,
                            "bootstrap_ci95_high": np.nan,
                            "wilcoxon_p_value": np.nan,
                            "rank_biserial": np.nan,
                            "better_model_by_mean": np.nan,
                        }
                    )
                    p_values.append(np.nan)
                    continue

                diff = (paired[model_a] - paired[model_b]).to_numpy()
                try:
                    _, p_value = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                except ValueError:
                    p_value = np.nan

                _, ci_low, ci_high = bootstrap_ci_mean(diff)
                if metric == "valid_r2":
                    better = model_a if np.mean(diff) > 0 else model_b
                else:
                    better = model_a if np.mean(diff) < 0 else model_b

                pending_rows.append(
                    {
                        "sample_size": sample_size,
                        "metric": metric,
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_pairs": len(paired),
                        "mean_difference_a_minus_b": float(np.mean(diff)),
                        "bootstrap_ci95_low": ci_low,
                        "bootstrap_ci95_high": ci_high,
                        "wilcoxon_p_value": p_value,
                        "rank_biserial": rank_biserial_from_diff(diff),
                        "better_model_by_mean": better,
                    }
                )
                p_values.append(p_value)

            adjusted = holm_adjust(p_values)
            for row, p_adj in zip(pending_rows, adjusted):
                row["holm_adjusted_p_value"] = p_adj
                row["significant_at_0_05"] = bool(p_adj < 0.05) if pd.notna(p_adj) else False
                rows.append(row)

    return pd.DataFrame(rows)


def load_test_predictions(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="10714筆測試", header=None)
    rows = []
    for model, start_col in PRED_BLOCKS.items():
        block = raw.iloc[2:, start_col : start_col + 4].copy()
        block.columns = ["actual", "predicted", "recorded_error", "recorded_accuracy"]
        block["actual"] = pd.to_numeric(block["actual"], errors="coerce")
        block["predicted"] = pd.to_numeric(block["predicted"], errors="coerce")
        block = block.dropna(subset=["actual", "predicted"]).reset_index(drop=True)
        block["row_id"] = np.arange(len(block))
        block["model"] = model
        block["residual"] = block["predicted"] - block["actual"]
        block["abs_error"] = np.abs(block["residual"])
        block["squared_error"] = block["residual"] ** 2
        block["actual_category"] = block["actual"].map(wqi_category)
        block["pred_category"] = block["predicted"].map(wqi_category)
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def compute_prediction_metrics(df: pd.DataFrame) -> pd.Series:
    actual = df["actual"].to_numpy()
    predicted = df["predicted"].to_numpy()
    residual = predicted - actual
    return pd.Series(
        {
            "r2": r2_score(actual, predicted),
            "mae": mean_absolute_error(actual, predicted),
            "rmse": np.sqrt(mean_squared_error(actual, predicted)),
            "residual_mean": float(np.mean(residual)),
            "residual_std": float(np.std(residual, ddof=1)),
            "residual_skewness": float(skew(residual)),
            "residual_kurtosis": float(kurtosis(residual, fisher=True)),
            "category_accuracy": float(np.mean(df["actual_category"] == df["pred_category"])),
            "macro_f1": f1_score(df["actual_category"], df["pred_category"], average="macro"),
            "n_test": len(df),
        }
    )


def bootstrap_prediction_ci(predictions: pd.DataFrame, n_boot: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        group = group.reset_index(drop=True)
        n = len(group)
        boot_metrics = []
        for _ in range(n_boot):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            sampled = group.iloc[idx]
            boot_metrics.append(compute_prediction_metrics(sampled))
        boot_df = pd.DataFrame(boot_metrics)
        for metric in TEST_METRICS:
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": float(boot_df[metric].mean()),
                    "ci95_low": float(boot_df[metric].quantile(0.025)),
                    "ci95_high": float(boot_df[metric].quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


def paired_error_tests(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wide_abs = predictions.pivot_table(index="row_id", columns="model", values="abs_error", aggfunc="first")
    p_values = []
    pending_rows = []
    for model_a, model_b in combinations(MODEL_BLOCKS, 2):
        paired = wide_abs[[model_a, model_b]].dropna()
        diff = (paired[model_a] - paired[model_b]).to_numpy()
        try:
            _, p_value = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            p_value = np.nan
        _, ci_low, ci_high = bootstrap_ci_mean(diff)
        pending_rows.append(
            {
                "metric": "absolute_error",
                "model_a": model_a,
                "model_b": model_b,
                "n_pairs": len(paired),
                "mean_difference_a_minus_b": float(np.mean(diff)),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "wilcoxon_p_value": p_value,
                "rank_biserial": rank_biserial_from_diff(diff),
                "better_model_by_mean": model_a if np.mean(diff) < 0 else model_b,
            }
        )
        p_values.append(p_value)

    adjusted = holm_adjust(p_values)
    for row, p_adj in zip(pending_rows, adjusted):
        row["holm_adjusted_p_value"] = p_adj
        row["significant_at_0_05"] = bool(p_adj < 0.05) if pd.notna(p_adj) else False
        rows.append(row)
    return pd.DataFrame(rows)


def category_level_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, actual_category), group in predictions.groupby(["model", "actual_category"], sort=True):
        rows.append(
            {
                "model": model,
                "actual_category": actual_category,
                "n": len(group),
                "mae": mean_absolute_error(group["actual"], group["predicted"]),
                "rmse": np.sqrt(mean_squared_error(group["actual"], group["predicted"])),
                "bias": float(np.mean(group["residual"])),
                "within_category_accuracy": float(np.mean(group["actual_category"] == group["pred_category"])),
            }
        )
    return pd.DataFrame(rows)


def residual_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        residual = group["residual"].to_numpy()
        mean = float(np.mean(residual))
        std = float(np.std(residual, ddof=1))
        if std == 0:
            ks_stat, ks_p = np.nan, np.nan
        else:
            ks_stat, ks_p = kstest(residual, lambda x: norm.cdf(x, loc=mean, scale=std))
        rows.append(
            {
                "model": model,
                "n": len(group),
                "residual_mean": mean,
                "residual_std": std,
                "residual_skewness": float(skew(residual)),
                "residual_kurtosis": float(kurtosis(residual, fisher=True)),
                "normality_ks_statistic": float(ks_stat),
                "normality_ks_p_value": float(ks_p),
            }
        )
    return pd.DataFrame(rows)


def dataset_distribution_robustness() -> pd.DataFrame:
    full_df = pd.read_csv(FULL_DATASET_PATH)
    rows = []
    for sample_size, dataset_path in DATASET_MAP.items():
        subset_df = pd.read_csv(dataset_path)
        for column in FEATURE_COLUMNS:
            subset_values = subset_df[column].to_numpy()
            full_values = full_df[column].to_numpy()
            ks_stat, ks_p = ks_2samp(subset_values, full_values)
            pooled_sd = np.sqrt((np.var(subset_values, ddof=1) + np.var(full_values, ddof=1)) / 2)
            smd = 0.0 if pooled_sd == 0 else (np.mean(subset_values) - np.mean(full_values)) / pooled_sd
            rows.append(
                {
                    "sample_size": sample_size,
                    "variable": column,
                    "subset_mean": float(np.mean(subset_values)),
                    "full_mean": float(np.mean(full_values)),
                    "subset_std": float(np.std(subset_values, ddof=1)),
                    "full_std": float(np.std(full_values, ddof=1)),
                    "mean_shift": float(np.mean(subset_values) - np.mean(full_values)),
                    "standardized_mean_difference": float(smd),
                    "ks_statistic": float(ks_stat),
                    "ks_p_value": float(ks_p),
                }
            )
    return pd.DataFrame(rows)


def sample_size_stability(metric_ci: pd.DataFrame) -> pd.DataFrame:
    rows = []
    summary = metric_ci.pivot_table(index=["model", "sample_size"], columns="metric", values="mean", aggfunc="first").reset_index()
    for model, group in summary.groupby("model", sort=True):
        group = group.sort_values("sample_size")
        for metric in ["valid_r2", "valid_mae", "valid_rmse"]:
            coef, p_value = spearmanr(group["sample_size"], group[metric])
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_sample_sizes": len(group),
                    "spearman_rho": float(coef),
                    "spearman_p_value": float(p_value),
                    "first_mean": float(group.iloc[0][metric]),
                    "last_mean": float(group.iloc[-1][metric]),
                    "relative_change_pct": float((group.iloc[-1][metric] - group.iloc[0][metric]) / group.iloc[0][metric] * 100),
                }
            )
    return pd.DataFrame(rows)


def save_csvs(frames: dict[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metric_logs = load_metric_logs(INPUT_XLSX)
    metric_ci = summarize_metric_ci(metric_logs)
    metric_tests = paired_metric_tests(metric_logs)

    predictions = load_test_predictions(INPUT_XLSX)
    prediction_metrics = pd.DataFrame(
        [
            {"model": model, **compute_prediction_metrics(group).to_dict()}
            for model, group in predictions.groupby("model", sort=True)
        ]
    )
    prediction_ci = bootstrap_prediction_ci(predictions)
    prediction_tests = paired_error_tests(predictions)
    category_metrics = category_level_metrics(predictions)
    residual_stats = residual_diagnostics(predictions)

    dist_robustness = dataset_distribution_robustness()
    stability = sample_size_stability(metric_ci)

    frames = {
        "long_metric_logs": metric_logs,
        "metric_ci_by_runs": metric_ci,
        "paired_tests_by_runs": metric_tests,
        "test_predictions_long": predictions,
        "test_prediction_metrics": prediction_metrics,
        "test_bootstrap_ci": prediction_ci,
        "test_paired_error_tests": prediction_tests,
        "category_level_metrics": category_metrics,
        "residual_diagnostics": residual_stats,
        "dataset_distribution_robustness": dist_robustness,
        "sample_size_stability": stability,
    }

    save_csvs(frames)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f"Saved Excel: {OUTPUT_XLSX}")
    for name in frames:
        print(f"Saved CSV: {OUTPUT_DIR / f'{name}.csv'}")


if __name__ == "__main__":
    main()
