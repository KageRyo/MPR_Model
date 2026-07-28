# Experiment Protocol

This document describes the reproducibility workflow. Dataset paths refer to
private local inputs that are not included in the repository.

## Task Definition

This repository evaluates `current-state WQI5 surrogate regression` from five water indicators:

- `DO`
- `BOD`
- `NH3N`
- `EC`
- `SS`

The task is not forecasting. The reference dataset has no timestamp field.

## Split Strategy

- Validation type: `cross-sectional`
- Split method: `StratifiedShuffleSplit`
- Stratification target: WQI5 category derived from `Score`
- Seeds: `0, 1, 2, 3, 4`
- Test size: `20%`
- Default split counts for a compatible `60,714`-row input:
  - train: `48,571`
  - test: `12,143`
- No timestamp, station-sequence, or lag feature is used in the current protocol.

## Models

- `direct_wqi5`
- `lr`
- `mpr`
- `svm`
- `rf`
- `xgboost`
- `lightgbm`

`direct_wqi5` is the deterministic reference when all five indicators are
available. Comparisons of learned surrogates should state clearly that the
models approximate the reference WQI5 score rather than replace the
deterministic formula.

If `xgboost` or `lightgbm` is missing from the runtime environment, treat that as an environment setup failure rather than silently dropping the model from the configured experiment.

By default, the reproducibility script runs on CPU to preserve portability. GPU
execution can be enabled for supported gradient boosting models only:

```bash
python scripts/reproduce_results.py \
  --config configs/experiment_config.yaml \
  --output-dir results/complete_input_gpu_run \
  --compute-device gpu \
  --gpu-id 0 \
  --overwrite
```

`xgboost` uses CUDA through `device="cuda:<gpu_id>"` and `tree_method="hist"`.
The installed LightGBM package should be verified before use; this environment
supports `device_type="gpu"` through the OpenCL backend, while
`device_type="cuda"` requires a LightGBM build compiled with CUDA support.
The scikit-learn models in this workflow remain CPU-based.

## Reduced-Indicator Experiment

The reduced-indicator experiment evaluates surrogate regressors when one or
more indicators are removed to simulate incomplete or delayed sensing.

The primary missing-indicator workflow includes
single-indicator missing settings, two-stage indicator reconstruction, the
107-window stress test, and CPU-only inference timing.

Run:

```bash
python scripts/reproduce_reduced_indicators.py \
  --config configs/reduced_indicator_config.yaml \
  --output-dir results/reduced_indicator_run
```

See [reduced-indicator-analysis.md](reduced-indicator-analysis.md). Direct WQI5
is treated as the complete-input reference and is not reported for reduced-input
scenarios.

## Sample-Size Experiment

The sample-size experiment compares the six learned surrogate models across
fixed data volumes of `1,000`, `5,000`, `10,000`, and `50,000` rows. Each CSV
must be prepared locally with the schema in
[data_preparation.md](data_preparation.md).

This workflow uses `5` stratified folds. Each fold trains on `80%` of the
selected dataset and tests on the remaining `20%`, reporting train and test
`R²`, `MAE`, and `RMSE` along with the other standard regression and WQI-band
metrics.

Run:

```bash
python scripts/run_sample_size_experiments.py \
  --datasets \
    data/subset_1000.csv \
    data/subset_5000.csv \
    data/subset_10000.csv \
    data/subset_50000.csv \
  --compute-device gpu \
  --gpu-id 0
```

See [sample-size-experiments.md](sample-size-experiments.md). The workflow
writes to `results/sample_size_experiments` and
`models/sample_size_experiments` by default, and refuses to write into non-empty
directories.

## Missing-Indicator Core Suite

The missing-indicator core suite uses a local `50,000`-row subset for
stratified `80/20` training and internal testing, then evaluates the trained
models on the remaining held-out rows of the compatible full input.

Run:

```bash
python scripts/run_missing_indicator_experiments.py \
  --config configs/missing_indicator_config.yaml \
  --output-dir results/missing_indicator_core_run \
  --compute-device gpu \
  --gpu-id 0
```

See [missing-indicator-core-experiments.md](missing-indicator-core-experiments.md).

## Missing-Indicator Suite

The missing-indicator suite supports single-indicator missing settings, the
107-window stress test, and CPU-only inference timing.
It evaluates:

- `missing_bod`: BOD unavailable.
- `missing_nh3n`: NH3N unavailable.
- `missing_bod_nh3n`: BOD and NH3N unavailable.

For each missing setting, the workflow reports:

- `inference_dropout`: complete-input model evaluated with selected indicators
  set to missing at inference time.
- `reduced_retraining`: model trained and evaluated only with available
  indicators.
- `indicator_reconstruction`: missing indicator(s) reconstructed from available
  indicators before WQI5 surrogate inference.

The workflow includes a 107-window stress test, which divides held-out rows into `107`
consecutive non-overlapping event windows and applies 30%, 100%, and 300%
synthetic perturbations. The `stress107` filename prefix is repository-specific,
not a new validation method. The suite also includes a separate CPU-only
inference timing workflow from saved artifacts.

Run:

```bash
python scripts/run_missing_indicator_robustness_experiments.py \
  --config configs/missing_indicator_robustness_config.yaml \
  --output-dir results/missing_indicator_robustness_run
```

Measure CPU-only inference timing:

```bash
python scripts/measure_missing_indicator_cpu_timing.py \
  --output-dir results/missing_indicator_robustness_run
```

Export the workbook:

```bash
python scripts/export_missing_indicator_robustness_excel.py \
  --output-dir results/missing_indicator_robustness_run
```

See [missing-indicator-robustness-experiments.md](missing-indicator-robustness-experiments.md).

## Metrics

Regression metrics:

- `R²`
- `MAE`
- `RMSE`
- `NMAE`
- WQI-band `Accuracy`
- WQI-band `Macro-F1`

Operational metrics:

- training/inference runtime
- residual mean
- residual standard deviation

Percentage-agreement metrics are not used as primary reporting metrics.
See [metrics.md](metrics.md) for metric definitions and for
guidance on separating regression metrics from WQI-band summaries.

## Outputs

Running `scripts/reproduce_results.py` writes:

- `results/metrics_summary.csv`
- `results/repeated_split_results.csv`
- `results/residual_statistics.csv`
- `results/category_metrics.csv`

These files are intended to support reproducible regeneration of result tables after the experiment hyperparameters are locked.

For verification runs, use a separate output directory so existing experiment
outputs are not overwritten:

```bash
python scripts/reproduce_results.py --config configs/experiment_config.yaml --output-dir results/verification_run
```

Use `--overwrite` only when you intentionally want to replace an existing result directory.
