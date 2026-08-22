# Limitations

- This repository does not include the complete dataset, processed experiment
  table, or experiment subsets. Monitoring data can be downloaded from the
  official source described in [data_preparation.md](data_preparation.md).
- The prepared experiment input does not contain timestamps, so this repository
  should not be described as a temporal forecasting system.
- The repository documents only the aggregate row counts, six-column schema,
  and preprocessing operations needed to prepare a compatible local input.
- The supported task is cross-sectional WQI5 assessment rather than
  time-dependent prediction.
- Optional surrogate model families such as `xgboost` and `lightgbm` require their exact recorded runtime dependencies when loading production artifacts; see [Production Artifact and ARM64 Runtime](edge-runtime.md).
- Direct WQI5 computation remains the reference method under complete-input conditions; surrogate models should not be described as mathematically superior to the deterministic formula.
- Reduced-indicator experiments estimate the reference WQI5 score under incomplete-input scenarios. They support incomplete-input analysis, not claims of future water-quality forecasting.
- The external hold-out results show that missing `NH3N` can still support useful auxiliary estimation, but missing `BOD` and especially `BOD + NH3N` have weak external generalization as reduced-input substitutes.
- The 107-window stress test is a controlled synthetic event-window stress test, not real typhoon, rainfall, or pollution-event validation.
- The 107-window stress test reduces dependence on one selected event window; it does not prove the absence of all sampling bias.
- CPU-only inference timing is a rough timing reference measured from saved artifacts. It is not direct validation on a low-end edge device unless that hardware is explicitly tested.
- GPU and multicore CPU acceleration may be used for repeated model-comparison experiments, but GPU timing should not be used as edge-device deployment evidence.
