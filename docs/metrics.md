# Metrics

This repository models `Score` as a continuous WQI5 regression target.

## Regression Metrics

- `R²`: coefficient of determination
- `MAE`: mean absolute error
- `RMSE`: root mean squared error
- `NMAE`: normalized mean absolute error
- `Residual Mean`: mean of `y_i - ŷ_i`
- `Residual Std.`: standard deviation of `y_i - ŷ_i`

Residuals are defined as:

```text
residual_i = y_i - ŷ_i
```

where `y_i` is the reference WQI5 score and `ŷ_i` is the model-estimated WQI5 score.

## Archived Percentage-Agreement Metric

Legacy comparison tables included a percentage-agreement metric derived from
absolute percentage error. It is not used as a primary reporting metric because
the task is continuous WQI5 regression. Primary reporting should use `R²`,
`MAE`, `RMSE`, `NMAE`, residual statistics, WQI-band accuracy / macro-F1,
confidence intervals, and paired tests.

## WQI Bands

WQI5 scores are mapped to six backend categories for API responses and WaterMirror display:

- `Excellent`: `85 < WQI5 <= 100`
- `Good`: `70 < WQI5 <= 85`
- `Fair`: `50 < WQI5 <= 70`
- `Poor`: `30 < WQI5 <= 50`
- `Bad`: `15 < WQI5 <= 30`
- `Terrible`: `0 <= WQI5 <= 15`

The statistical outputs include regression error summaries stratified by actual WQI band.
