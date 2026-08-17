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

Research datasets and trained model binaries are not distributed through this repository or its releases. See [Data Availability and Redistribution](data-availability.md) for the repository's distribution policy. The `models/production_model_manifest.json` file contains metadata and expected local paths only; it does not contain serialized model parameters or executable model binaries.

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

- This repository does not include the complete dataset, processed experiment
  table, or experiment subsets. See [data_preparation.md](data_preparation.md)
  for the official download source and preparation contract.
- The prepared experiment input does not contain timestamps.
- Optional model families such as `xgboost` and `lightgbm` require their corresponding runtime dependencies.
- External hold-out results show that `BOD` is a critical indicator; reduced-input settings without `BOD` should be interpreted conservatively.
- CPU-only timing is a rough inference-time reference, not direct proof of performance on a low-end edge device.
