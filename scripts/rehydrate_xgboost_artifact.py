from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wqsurrogatemodels.settings import FEATURE_COLUMNS
from wqsurrogatemodels.xgboost_artifacts import (
    XGBoostArtifactRehydrationError,
    rehydrate_xgboost_pickle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a prediction-validated XGBoost pickle that no longer emits the legacy Booster warning."
    )
    parser.add_argument("--source", required=True, type=Path, help="Existing XGBoost pipeline pickle.")
    parser.add_argument("--target", required=True, type=Path, help="New candidate path; must not already exist.")
    parser.add_argument(
        "--verification-csv",
        type=Path,
        help="Optional CSV of representative complete-input records used for parity validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verification_frame = None
    if args.verification_csv is not None:
        verification_frame = pd.read_csv(args.verification_csv)
        missing_columns = [column for column in FEATURE_COLUMNS if column not in verification_frame.columns]
        if missing_columns:
            raise SystemExit(
                f"Verification CSV is missing required columns: {', '.join(missing_columns)}"
            )

    try:
        result = rehydrate_xgboost_pickle(
            args.source,
            args.target,
            verification_frame=verification_frame,
        )
    except XGBoostArtifactRehydrationError as exc:
        raise SystemExit(str(exc)) from exc

    warning_status = "detected" if result.source_legacy_warning_detected else "not detected"
    print(f"Created candidate: {result.target}")
    print(f"Legacy source warning: {warning_status}")
    print(f"Prediction parity: verified on {result.verification_rows} record(s)")
    print("Review and update the production manifest separately before promoting this candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
