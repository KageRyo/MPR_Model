# Model Hyperparameters

This document describes the current reproducibility workflow.

## Reproducibility Workflow

The current reproducibility workflow uses the following main settings:

| Model | Library | Preprocessing | Key Hyperparameters |
| --- | --- | --- | --- |
| `direct_wqi5` | formula baseline | none | direct WQI5 equation |
| `lr` | scikit-learn | mean imputation + standard scaling | default `LinearRegression()` |
| `mpr` | scikit-learn | mean imputation + polynomial features + standard scaling | `degree=2`, `include_bias=False` |
| `svm` | scikit-learn | mean imputation + standard scaling | `kernel=rbf`, `C=10.0`, `epsilon=0.1` |
| `rf` | scikit-learn | mean imputation | `n_estimators=300`, `random_state=0`, `n_jobs=-1` |
| `xgboost` | xgboost | mean imputation | `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9`, `random_state=0` |
| `lightgbm` | lightgbm | mean imputation | `n_estimators=300`, `learning_rate=0.05`, `random_state=0` |

GPU execution is optional and disabled by default. When
`--compute-device gpu` is used, only `xgboost` and `lightgbm` receive GPU
parameters. `xgboost` uses `tree_method="hist"` and `device="cuda:<gpu_id>"`.
LightGBM uses `device_type="gpu"` for OpenCL-capable builds; its CUDA tree
learner requires a CUDA-enabled LightGBM build. The scikit-learn models remain
CPU-based in this reproducibility workflow.
