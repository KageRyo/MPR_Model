from __future__ import annotations

from enum import Enum


class ModelTypeEnum(str, Enum):
    """Supported model types for WQI5 assessment."""

    DIRECT_WQI5 = "direct_wqi5"
    LR = "lr"
    MPR = "mpr"
    SVM = "svm"
    RF = "rf"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

    @property
    def display_name(self) -> str:
        """Human-friendly name for UI / documentation."""
        names = {
            self.DIRECT_WQI5: "Direct WQI5 (formula baseline)",
            self.LR: "Linear Regression",
            self.MPR: "Multiple Polynomial Regression",
            self.SVM: "Support Vector Machine",
            self.RF: "Random Forest",
            self.XGBOOST: "XGBoost",
            self.LIGHTGBM: "LightGBM",
        }
        return names[self]

    @property
    def is_ml_model(self) -> bool:
        """Whether this model type requires a trained ML artifact."""
        return self != self.DIRECT_WQI5
