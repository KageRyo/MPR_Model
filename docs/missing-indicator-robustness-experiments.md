# Missing-Indicator Robustness Experiments

This workflow evaluates WQI5 surrogate behavior under single-indicator and
multi-indicator missing conditions. It is an incomplete-input sensitivity
experiment, not temporal forecasting and not a substitute for complete-input
deterministic WQI5 computation.

## Data Split

The workflow uses a private local `50,000`-row subset for stratified `80/20`
training and internal testing. Remaining rows from the compatible
`60,714`-row input form a fixed held-out inference set.

For each seed:

- train: `40,000` rows
- internal test: `10,000` rows
- external hold-out: `10,714` rows

The external hold-out is not folded into cross-validation. It remains a fixed
external inference set.

## Missing Settings

| Missing set | Missing indicator(s) | Available indicators |
| --- | --- | --- |
| `missing_bod` | `BOD` | `DO`, `NH3N`, `EC`, `SS` |
| `missing_nh3n` | `NH3N` | `DO`, `BOD`, `EC`, `SS` |
| `missing_bod_nh3n` | `BOD`, `NH3N` | `DO`, `EC`, `SS` |

## Experiment Modes

| Mode | Purpose |
| --- | --- |
| `full_reference` | Complete five-indicator surrogate evaluated with complete input. |
| `inference_dropout` | Complete five-indicator surrogate evaluated with selected indicator(s) set to missing at inference time. |
| `reduced_retraining` | Surrogate trained and evaluated only with the available indicators. |
| `indicator_reconstruction` | Missing indicator(s) are reconstructed from available indicators, then WQI5 is estimated from the reconstructed full-input vector. |

`indicator_reconstruction` trains the first-stage reconstruction models only on
the training split. The second-stage WQI surrogate is trained on reconstructed
training rows so training and evaluation use the same reconstructed-input
structure.

## Event-Window Stress Test

The **107-window sequential event-window stress test** divides the held-out
rows into `107` consecutive,
non-overlapping event windows. Each window contains approximately `1%` of the
external samples. Each window is evaluated under low-, medium-, and
high-severity synthetic perturbations:

| Severity | Pollution indicators | DO |
| --- | ---: | ---: |
| `low_30pct` | `x1.30` | `x0.70` |
| `medium_100pct` | `x2.00` | `x0.50` |
| `high_300pct` | `x4.00` | `x0.30` |

The default scenarios are:

- `organic_pollution`: `BOD` increases and `DO` decreases.
- `ammonia_pollution`: `NH3N` increases.
- `suspended_solids_event`: `SS` increases.
- `conductivity_event`: `EC` increases.
- `combined_pollution`: `BOD`, `NH3N`, `SS`, and `EC` increase while `DO`
  decreases.

This should be described as a `107-window sequential event-window stress test`,
not as `107-fold cross-validation`, because the windows are perturbation
locations, not training-validation folds. The `stress107` filename prefix is a
repository-specific label, not a new validation methodology.

This is a controlled synthetic event-window stress test. It should not be
described as real typhoon, rainfall, or pollution-event validation unless a
timestamped real event dataset is added.

The valid claim is that this 107-window test reduces the concern that the
stress-test conclusion depends on a single selected window. It does not prove
that all sampling bias is impossible.

When a scenario perturbs an indicator that is absent from a reduced-input
setting, reduced sensitivity is expected. For example, `missing_nh3n` settings
should not be expected to fully detect `ammonia_pollution` from `NH3N` because
`NH3N` is not available to that model.

## CPU-Only Timing

GPU acceleration may be used for experiment reproduction. CPU-only inference
timing is measured separately from saved model artifacts without retraining.
Use constrained timing variables such as:

```bash
CUDA_VISIBLE_DEVICES=""
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

CPU-only timing is a rough inference-time reference for constrained CPU
environments. It should not be mixed with GPU training or GPU timing claims.

## Run

```bash
python scripts/run_missing_indicator_robustness_experiments.py \
  --config configs/missing_indicator_robustness_config.yaml \
  --output-dir results/missing_indicator_robustness_run
```

Measure CPU-only inference timing from saved artifacts:

```bash
python scripts/measure_missing_indicator_cpu_timing.py \
  --output-dir results/missing_indicator_robustness_run
```

Export a readable workbook:

```bash
python scripts/export_missing_indicator_robustness_excel.py \
  --output-dir results/missing_indicator_robustness_run
```

## Outputs

The output directory is organized as:

- `manifest.json`: run configuration and dataset summary.
- `hardware.json`: local hardware snapshot.
- `versions.json`: library versions.
- `models/`: saved joblib model artifacts.
- `predictions/predictions_long.csv`: row-level internal and external predictions.
- `metrics/metrics_by_seed.csv`: per-seed metrics.
- `metrics/metrics_summary.csv`: mean and standard deviation across seeds.
- `metrics/best_by_experiment_source.csv`: lowest-mean-MAE model per setting.
- `metrics/indicator_reconstruction_metrics.csv`: first-stage reconstruction metrics.
- `metrics/error_by_wqi_band.csv`: WQI-band error summaries.
- `stats/bootstrap_ci.csv`: seed-level bootstrap confidence intervals.
- `stats/paired_error_tests.csv`: paired Wilcoxon comparisons over per-seed MAE.
- `stress_tests/event_window_stress_summary.csv`: localized event-window stress responses.
- `stress_tests/stress107_window_summary.csv`: window-level responses for the
  107-window stress test.
- `stress_tests/stress107_detection_summary.csv`: detection-rate summaries by
  scenario, severity, model, and experiment mode.
- `stress_tests/stress107_severity_monotonicity.csv`: checks whether the 30%,
  100%, and 300% perturbations produce progressively larger score drops.
- `stress_tests/stress107_key_conclusions.csv`: compact interpretation notes for
  result discussion.
- `timing/cpu_only_inference_timing.csv`: raw CPU-only timing repeats.
- `timing/cpu_only_inference_timing_summary.csv`: CPU-only timing summary.
- `reports/missing_indicator_robustness_summary.xlsx`: formatted workbook.

The dataset, split indices, row-level predictions, workbooks, and model
artifacts are private local outputs and must not be committed.

## Interpretation Boundary

Complete-input deterministic WQI5 remains the reference method when all five
indicators are available. Reduced, dropout, or reconstructed-indicator settings
evaluate auxiliary behavior under incomplete-input constraints. Negative
external hold-out R2 should be interpreted as a generalization limitation, not
hidden or reframed as successful substitution for complete-input WQI5.
