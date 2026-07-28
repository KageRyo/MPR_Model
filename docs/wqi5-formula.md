# WQI5 Formula

`Score` denotes the calculated WQI5 score used throughout this repository.

## Definition

- `Score`: calculated WQI5 score
- Unit: dimensionless index score
- Range: `0-100`
- Inputs: `DO`, `BOD`, `NH3N`, `EC`, `SS`

## Current Direct Baseline

The direct baseline is implemented in [`src/wqi.py`](../src/wqi.py) as `direct_wqi5_score()`.

Each raw indicator is transformed into a sub-index value:

- `QDO`
- `QBOD`
- `QNH3N`
- `QEC`
- `QSS`

Each sub-index is clipped to the range `[0, 100]` using `clamp_score()`. The WQI5 score is then computed as the arithmetic mean of the five clipped sub-indices:

```text
WQI5 = (clamp(QDO) + clamp(QBOD) + clamp(QNH3N) + clamp(QEC) + clamp(QSS)) / 5
```

In the current implementation, the five sub-indices are equally weighted.

## Indicator-to-Subindex Transformations

The repository currently uses these direct formulas:

```text
QDO   = -0.08841347 + 0.8996848*DO - 4.907377e-2*DO^2 + 1.5696e-3*DO^3 - 1.5216e-5*DO^4 + 4.545e-8*DO^5
QBOD  = 1123.6 / (1 + 9.99 * exp(0.2 * BOD))
QNH3N = 9.79 + 56.76 / (NH3N + 0.6236888)
QSS   = 100.1 - 2.433*SS + 2.282e-2*SS^2 - 7.90e-5*SS^3
QEC   = 101.7 / (1 + 0.0062 * exp(8.32e-3 * EC))
```

The computed output is rounded to three decimal places after averaging.

## Interpretation Notes

- The resulting `Score` is a constructed WQI5 index, not a directly observed physical unit.
- Correlation analyses between the five raw indicators and `Score` should be interpreted as relationships between features and a constructed index derived from those same features.
- The direct baseline is intended as a transparent reference implementation for current-state assessment, not temporal forecasting.
