from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from wqsurrogatemodels.xgboost_artifacts import (
    XGBoostArtifactRehydrationError,
    rehydrate_xgboost_pickle,
)


class PredictableModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["DO"].to_numpy(dtype=float) + frame["BOD"].to_numpy(dtype=float)


def test_rehydrate_xgboost_pickle_creates_prediction_verified_candidate(tmp_path: Path):
    source = tmp_path / "source.pkl"
    target = tmp_path / "candidate.pkl"
    joblib.dump(PredictableModel(), source)
    records = pd.DataFrame(
        [{"DO": 7.2, "BOD": 2.1, "NH3N": 0.3, "EC": 450.0, "SS": 12.0}]
    )

    result = rehydrate_xgboost_pickle(source, target, verification_frame=records)

    assert result.target == target.resolve()
    assert result.verification_rows == 1
    assert target.exists()
    assert joblib.load(target).predict(records)[0] == pytest.approx(9.3)


def test_rehydrate_xgboost_pickle_refuses_to_overwrite_target(tmp_path: Path):
    source = tmp_path / "source.pkl"
    target = tmp_path / "candidate.pkl"
    joblib.dump(PredictableModel(), source)
    target.write_text("existing candidate", encoding="utf-8")

    with pytest.raises(XGBoostArtifactRehydrationError, match="Refusing to overwrite"):
        rehydrate_xgboost_pickle(source, target)
