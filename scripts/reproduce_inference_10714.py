from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
XLSX_PATH = PROJECT_ROOT / "statistics" / "整理.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "statistics" / "outputs" / "inference_reproduction"

FEATURE_COLUMNS = ["DO", "BOD", "NH3N", "EC", "SS"]
MODEL_PATHS = {
    "XGBoost": MODELS_DIR / "XGBoost" / "modelXGBVer.1.0-50000.pkl",
    "LightGBM": MODELS_DIR / "LightGBM" / "modelLGBMVer.1.0-50000.pkl",
    "RF": None,
    "SVM": MODELS_DIR / "SVM" / "modelSVMVer.1.0-50000.pkl",
    "MPR": MODELS_DIR / "MPR" / "modelMPRVer.1.0-50000.pkl",
    "LR": MODELS_DIR / "LR" / "modelLRVer.1.0-50000.pkl",
}
PRED_BLOCKS = {
    "XGBoost": 0,
    "LightGBM": 5,
    "RF": 10,
    "SVM": 15,
    "MPR": 20,
    "LR": 25,
}


def extract_inference_set() -> pd.DataFrame:
    full = pd.read_csv(DATA_DIR / "dataV1.csv")
    subset_50000 = pd.read_csv(DATA_DIR / "dataV1_50000.csv")
    if not full.iloc[: len(subset_50000)].reset_index(drop=True).equals(subset_50000.reset_index(drop=True)):
        raise ValueError("dataV1_50000.csv is not an exact prefix of dataV1.csv; inference-set extraction assumption failed.")
    inference_set = full.iloc[len(subset_50000) :].reset_index(drop=True)
    if len(inference_set) != 10714:
        raise ValueError(f"Expected 10714 inference evaluation rows, found {len(inference_set)}.")
    return inference_set


def load_xlsx_predictions() -> pd.DataFrame:
    raw = pd.read_excel(XLSX_PATH, sheet_name="10714筆測試", header=None)
    rows = []
    for model, start_col in PRED_BLOCKS.items():
        block = raw.iloc[2:, start_col : start_col + 4].copy()
        block.columns = ["actual", "predicted_xlsx", "recorded_error", "recorded_accuracy"]
        block["actual"] = pd.to_numeric(block["actual"], errors="coerce")
        block["predicted_xlsx"] = pd.to_numeric(block["predicted_xlsx"], errors="coerce")
        block = block.dropna(subset=["actual", "predicted_xlsx"]).reset_index(drop=True)
        block["row_id"] = np.arange(len(block))
        block["model"] = model
        rows.append(block[["row_id", "model", "actual", "predicted_xlsx"]])
    return pd.concat(rows, ignore_index=True)


def reproduce_predictions(inference_set: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = inference_set[FEATURE_COLUMNS]
    for model, path in MODEL_PATHS.items():
        if path is None or not path.exists():
            continue
        estimator = joblib.load(path)
        predicted = estimator.predict(features)
        rows.append(
            pd.DataFrame(
                {
                    "row_id": np.arange(len(inference_set)),
                    "model": model,
                    "predicted_reproduced": predicted,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["row_id", "model", "predicted_reproduced"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inference_set = extract_inference_set()
    xlsx_predictions = load_xlsx_predictions()

    if not xlsx_predictions[xlsx_predictions["model"] == "XGBoost"]["actual"].reset_index(drop=True).round(2).equals(
        inference_set["Score"].round(2).reset_index(drop=True)
    ):
        raise ValueError("Inference scores do not match the actual values recorded in the Excel 10714-sheet.")

    reproduced = reproduce_predictions(inference_set)
    comparison = xlsx_predictions.merge(reproduced, on=["row_id", "model"], how="left")
    comparison["reproduction_abs_diff"] = np.abs(comparison["predicted_xlsx"] - comparison["predicted_reproduced"])

    summary_rows = []
    for model, group in comparison.groupby("model", sort=True):
        has_reproduced = group["predicted_reproduced"].notna().any()
        summary_rows.append(
            {
                "model": model,
                "n_rows": len(group),
                "has_reproduced_model_artifact": has_reproduced,
                "max_abs_diff_vs_xlsx": float(group["reproduction_abs_diff"].max()) if has_reproduced else np.nan,
                "mean_abs_diff_vs_xlsx": float(group["reproduction_abs_diff"].mean()) if has_reproduced else np.nan,
            }
        )

    inference_set.to_csv(OUTPUT_DIR / "inference_10714.csv", index=False)
    xlsx_predictions.to_csv(OUTPUT_DIR / "inference_predictions_from_xlsx.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "inference_prediction_reproduction_comparison.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "inference_reproduction_summary.csv", index=False)

    print(f"Saved: {OUTPUT_DIR / 'inference_10714.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'inference_predictions_from_xlsx.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'inference_prediction_reproduction_comparison.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'inference_reproduction_summary.csv'}")


if __name__ == "__main__":
    main()
