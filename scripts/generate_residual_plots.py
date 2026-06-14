from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "statistics" / "outputs" / "test_predictions_long.csv"
OUTPUT_DIR = PROJECT_ROOT / "statistics" / "outputs" / "figures"

MODEL_ORDER = ["LightGBM", "XGBoost", "RF", "SVM", "MPR", "LR"]
MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "rf": "RF",
    "svm": "SVM",
    "mpr": "MPR",
    "lr": "LR",
}
MODEL_COLORS = {
    "LightGBM": "#1b5e20",
    "XGBoost": "#1565c0",
    "RF": "#ef6c00",
    "SVM": "#6a1b9a",
    "MPR": "#c62828",
    "LR": "#37474f",
}
HIST_FILL = "#87CEEB"
HIST_EDGE = "#5A5A5A"


def apply_classic_axes_style(ax) -> None:
    ax.tick_params(labelsize=8, width=0.8, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")


def save_model_histogram(group: pd.DataFrame, model: str) -> None:
    residual = group["residual"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.hist(residual, bins=20, color=HIST_FILL, edgecolor=HIST_EDGE, linewidth=0.6, alpha=0.95)
    ax.set_title(f"{model} Model Residual Distribution", fontsize=9, pad=8)
    ax.set_xlabel("Residuals", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    apply_classic_axes_style(ax)
    fig.savefig(OUTPUT_DIR / f"residual_{model.lower()}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_model_diagnostics(group: pd.DataFrame, model: str) -> None:
    color = MODEL_COLORS.get(model, "#455a64")
    actual = group["actual"].to_numpy()
    predicted = group["predicted"].to_numpy()
    residual = group["residual"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()

    axes[0].scatter(actual, residual, s=9, alpha=0.35, color=color, edgecolors="none")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"{model}: Residual vs Actual")
    axes[0].set_xlabel("Actual WQI5")
    axes[0].set_ylabel("Residual (actual - predicted)")
    apply_classic_axes_style(axes[0])

    axes[1].scatter(predicted, residual, s=9, alpha=0.35, color=color, edgecolors="none")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title(f"{model}: Residual vs Predicted")
    axes[1].set_xlabel("Predicted WQI5")
    axes[1].set_ylabel("Residual (actual - predicted)")
    apply_classic_axes_style(axes[1])

    axes[2].hist(residual, bins=30, color=HIST_FILL, edgecolor=HIST_EDGE, linewidth=0.6, alpha=0.95)
    axes[2].axvline(np.mean(residual), color="black", linestyle="--", linewidth=1, label=f"mean={np.mean(residual):.3f}")
    axes[2].set_title(f"{model} Model Residual Distribution", fontsize=10)
    axes[2].set_xlabel("Residuals")
    axes[2].set_ylabel("Frequency")
    axes[2].legend(frameon=False)
    apply_classic_axes_style(axes[2])

    n = len(residual)
    sorted_resid = np.sort(residual)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = norm.ppf(probs, loc=np.mean(residual), scale=np.std(residual, ddof=1))
    axes[3].scatter(theo, sorted_resid, s=9, alpha=0.5, color=color, edgecolors="none")
    lo = min(theo.min(), sorted_resid.min())
    hi = max(theo.max(), sorted_resid.max())
    axes[3].plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    axes[3].set_title(f"{model}: Q-Q Plot")
    axes[3].set_xlabel("Theoretical Quantiles")
    axes[3].set_ylabel("Observed Residual Quantiles")
    apply_classic_axes_style(axes[3])

    fig.suptitle(f"{model} Residual Diagnostics", fontsize=13)
    fig.savefig(OUTPUT_DIR / f"residual_diagnostics_{model.lower()}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_overview_figure(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=False, constrained_layout=True)
    axes = axes.ravel()

    for ax, model in zip(axes, MODEL_ORDER):
        group = frame[frame["model"] == model]
        color = MODEL_COLORS.get(model, "#455a64")
        ax.scatter(group["actual"], group["residual"], s=7, alpha=0.28, color=color, edgecolors="none")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.set_title(model)
        ax.set_xlabel("Actual WQI5")
        ax.set_ylabel("Residual")
        apply_classic_axes_style(ax)

    fig.suptitle("Residual vs Actual Across Models", fontsize=14)
    fig.savefig(OUTPUT_DIR / "residual_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_qq_overview(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    axes = axes.ravel()

    for ax, model in zip(axes, MODEL_ORDER):
        group = frame[frame["model"] == model]
        residual = np.sort(group["residual"].to_numpy())
        n = len(residual)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theo = norm.ppf(probs, loc=np.mean(residual), scale=np.std(residual, ddof=1))
        color = MODEL_COLORS.get(model, "#455a64")
        ax.scatter(theo, residual, s=7, alpha=0.35, color=color, edgecolors="none")
        lo = min(theo.min(), residual.min())
        hi = max(theo.max(), residual.max())
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=0.9)
        ax.set_title(model)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Observed Residual Quantiles")
        apply_classic_axes_style(ax)

    fig.suptitle("Q-Q Diagnostics Across Models", fontsize=14)
    fig.savefig(OUTPUT_DIR / "residual_qq_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=str(INPUT_CSV))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--source", default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--missing-set", default=None)
    return parser.parse_args()


def load_prediction_frame(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if args.source and "source" in frame.columns:
        frame = frame[frame["source"] == args.source]
    if args.experiment and "experiment" in frame.columns:
        frame = frame[frame["experiment"] == args.experiment]
    if args.missing_set and "missing_set" in frame.columns:
        frame = frame[frame["missing_set"] == args.missing_set]

    if "model" not in frame.columns and "model_type" in frame.columns:
        frame["model"] = frame["model_type"].map(MODEL_LABELS).fillna(frame["model_type"])
    if "residual" not in frame.columns:
        frame["residual"] = frame["actual"] - frame["predicted"]
    return frame


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    if not input_csv.is_absolute():
        input_csv = PROJECT_ROOT / input_csv
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}.")

    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_prediction_frame(input_csv, args)
    if frame.empty:
        raise ValueError("No prediction rows remain after applying filters.")

    for model in MODEL_ORDER:
        group = frame[frame["model"] == model].copy()
        if group.empty:
            continue
        save_model_histogram(group, model)
        save_model_diagnostics(group, model)

    save_overview_figure(frame)
    save_qq_overview(frame)

    print(f"Saved figures in: {OUTPUT_DIR}")
    for model in MODEL_ORDER:
        path = OUTPUT_DIR / f"residual_{model.lower()}.png"
        if path.exists():
            print(f"Saved: {path}")
        diag_path = OUTPUT_DIR / f"residual_diagnostics_{model.lower()}.png"
        if diag_path.exists():
            print(f"Saved: {diag_path}")
    print(f"Saved: {OUTPUT_DIR / 'residual_overview.png'}")
    print(f"Saved: {OUTPUT_DIR / 'residual_qq_overview.png'}")


if __name__ == "__main__":
    main()
