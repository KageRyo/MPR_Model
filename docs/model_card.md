# Model Card

## Task Definition

`WQSurrogateModels` performs WQI5-based current-state water quality assessment from five input indicators.

This repository does not perform temporal forecasting.

`Score` denotes a dimensionless WQI5 index on a `0-100` scale.

## Inputs

- `DO`
- `BOD`
- `NH3N`
- `EC`
- `SS`

## Outputs

- `score`
- `category`
- `rating_range`
- `assessment`
- `warnings`

## Supported Models

- `direct_wqi5`
- `lr`
- `mpr`
- `svm`
- `rf`
- `xgboost`
- `lightgbm`

## Local Inference Artifacts

Model binaries are local artifacts and are not committed to Git. The
`models/production_model_manifest.json` file documents the paths expected by
the API for each supported surrogate model.

The local inference artifacts are complete-input WQI5 surrogates:

```text
DO, BOD, NH3N, EC, SS -> WQI5 score
```

They are not missing-indicator models. Experiment artifacts remain in ignored
local output directories.

## Intended Use

- backend assessment for `WaterMirror`
- API-based batch or single-record WQI5 assessment
- reproducibility and comparison of direct and surrogate approaches

## Not Intended Use

- temporal forecasting
- causal inference
- unsupported water quality indices beyond the documented WQI5 framing
- replacing deterministic WQI5 when all five indicators are available
- treating reduced-indicator models as reliable substitutes for complete-input WQI5
- describing the 107-window stress test as real pollution-event validation
- decision-making without reviewing domain-specific limitations

## Limitations

- Dataset files are private inputs and are not distributed.
- The documented reference data do not contain timestamps.
- Optional model families such as `xgboost` and `lightgbm` require their corresponding runtime dependencies.
- External hold-out results show that `BOD` is a critical indicator; reduced-input settings without `BOD` should be interpreted conservatively.
- CPU-only timing is a rough inference-time reference, not direct proof of performance on a low-end edge device.
