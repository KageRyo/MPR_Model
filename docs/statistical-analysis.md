# Statistical Analysis

This document describes the statistics used for the missing-indicator,
107-window stress-test, and CPU-only timing results.

## Interpretation Boundaries

The task is WQI5 surrogate regression. The target `Score` is computed from the
same five indicators used under complete-input conditions, so complete-input
results are WQI5 approximation results, not future water-quality forecasting.

Direct WQI5 computation remains the reference method when `DO`, `BOD`, `NH3N`,
`EC`, and `SS` are all available. Reduced-input, inference-dropout, and
indicator-reconstruction experiments evaluate auxiliary behavior under
incomplete-input constraints.

## Primary Metrics

Primary reporting uses:

- `R²`
- `MAE`
- `RMSE`
- `NMAE`
- WQI-band `Accuracy`
- WQI-band `Macro-F1`
- residual mean and residual standard deviation where useful

Percentage-agreement metrics are not used as primary reporting metrics.

## Confidence Intervals

The missing-indicator robustness workflow runs five stratified seeds over a
locally prepared `50,000`-row subset:

- `40,000` training rows per seed
- `10,000` internal-test rows per seed
- fixed `10,714` held-out rows from the compatible full input

Seed-level bootstrap `95%` confidence intervals are reported for:

- `R²`
- `MAE`
- `RMSE`
- `NMAE`
- `Accuracy`
- `Macro-F1`

The confidence-interval output is:

```text
statistics/outputs/bootstrap_ci.csv
```

The complete-input performance output is generated from
`results/complete_input_gpu/metrics_summary.csv` and reports repeated-split
means and standard deviations for trained surrogate models:

```text
statistics/outputs/complete_input_performance.csv
```

The missing-indicator robustness output uses the missing-indicator bootstrap
confidence intervals:

```text
statistics/outputs/missing_indicator_robustness.csv
```

## Significance Testing

The pairwise model-comparison table uses the complete-input GPU repeated-split
results:

```text
results/complete_input_gpu/repeated_split_results.csv
```

The comparison excludes `direct_wqi5` because it is a formula baseline, not a
trained surrogate model. For each trained model pair and seed, the paired
difference is:

```text
diff_seed = MAE_ModelA_seed - MAE_ModelB_seed
```

Negative values favor `ModelA`; positive values favor `ModelB`. The reported
`A-B` value is the mean paired MAE difference across the five seeds. `A-B 95%
CI` is the paired t confidence interval for that mean difference.

The `p-value` column is the Holm-adjusted paired t-test p-value across the 15
trained-model pair comparisons. The test operates on the available seed-level
MAE values.

The paired-test output is:

```text
statistics/outputs/paired_error_tests.csv
```

## 107-Window Stress Analysis

The 107-window stress analysis is a controlled synthetic event-window stress
test. Held-out rows are divided into `107` consecutive
non-overlapping windows. Each window is perturbed under 30%, 100%, and 300%
severity levels for five pollution-like scenarios.

It should not be called `107-fold cross-validation`. The windows are event
locations, not model-training folds. The `stress107` output filename prefix is a
repository-specific label, not a new validation methodology.

The 107-window stress-test summary is:

```text
statistics/outputs/stress107_summary.csv
```

This analysis reduces concern that the stress-test conclusion depends on a
single selected middle window. It does not prove absence of all sampling bias
and does not replace validation on real timestamped pollution-event data.

## Feature-Score Correlation

Feature-score correlations are descriptive because WQI5 `Score` is constructed
from the same five indicators. They summarize relationships used by the
surrogate-regression experiments without publishing row-level data.

The correlation output is:

```text
statistics/outputs/feature_score_correlations.csv
```

## CPU-Only Timing

GPU and multicore CPU acceleration may be used to reproduce model-comparison
experiments efficiently. CPU-only timing is reported separately using inference
from saved model artifacts.

The CPU-only timing output is:

```text
statistics/outputs/cpu_only_timing.csv
```

This output is a conservative inference-time reference. It should not be
described as direct validation on a low-end edge device unless such hardware is
actually tested.

## Report

The generated statistical reports are:

```text
statistics/outputs/statistical_summary.md
statistics/outputs/statistical_analysis_report.md
```
