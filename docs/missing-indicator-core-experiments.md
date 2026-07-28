# Missing-Indicator Core Experiments

This workflow evaluates WQI5 surrogate behavior under missing-indicator and
incomplete-input settings. It is designed for method comparison, not temporal
water-quality forecasting. A broader set of missing conditions is documented
in
[`missing-indicator-robustness-experiments.md`](missing-indicator-robustness-experiments.md).

## Data Split

The experiment uses a private local `50,000`-row subset for training and
internal testing. Remaining rows from the compatible `60,714`-row input form a
fixed held-out inference set. The script verifies that the subset is an exact
prefix of the full input.

For each seed, the subset is split with stratified `80/20` sampling by
WQI5 category:

- train: `40,000` rows
- internal test: `10,000` rows
- external inference: `10,714` rows

## Experiments

| Experiment | Purpose |
| --- | --- |
| `full_reference` | Complete five-indicator model evaluated with complete five-indicator input. |
| `full_inference_dropout` | Complete five-indicator model evaluated with `BOD` and `NH3N` set to missing at inference time. |
| `reduced_retraining` | Model trained and evaluated using only `DO`, `EC`, and `SS`. |
| `two_stage_reconstruction` | Reconstructs `BOD` and `NH3N` from `DO`, `EC`, and `SS`, then evaluates WQI5 with a full-input surrogate. |
| `stress_scenarios` | Scenario-based perturbation tests for pollution-like parameter shifts. |

`full_inference_dropout` and `full_reference` use the same trained full-input
model. The dropout condition simulates unavailable BOD and NH3N measurements at
inference time.

The two-stage reconstruction workflow trains the indicator-reconstruction models
only on the training split. The second-stage WQI surrogate is trained on the
training rows after `BOD` and `NH3N` have been reconstructed, so the
second-stage model sees the same reconstructed-feature structure during
training and evaluation.

```text
DO / EC / SS -> BOD
DO / EC / SS -> NH3N
DO / reconstructed BOD / reconstructed NH3N / EC / SS -> WQI5 surrogate
```

## Run

```bash
python scripts/run_missing_indicator_experiments.py \
  --config configs/missing_indicator_config.yaml \
  --output-dir results/missing_indicator_core_run \
  --compute-device gpu \
  --gpu-id 0
```

The script refuses to overwrite an output directory that already contains files
unless `--overwrite` is passed explicitly.

If training and primary prediction outputs have completed but derived
statistics need to be regenerated, use the statistics-output workflow
documented in the repository README.

## Outputs

The output directory contains:

- `manifest.json`: run configuration and dataset summary.
- `hardware.json`: local hardware snapshot.
- `versions.json`: library versions.
- `models/`: saved joblib model artifacts by seed, experiment, and model.
- `splits/split_indices.csv`: train/test split row indices for each seed.
- `predictions/predictions_long.csv`: row-level internal-test and external
  inference predictions.
- `metrics/metrics_by_seed.csv`: per-seed metrics.
- `metrics/metrics_summary.csv`: mean and standard deviation across seeds.
- `metrics/best_by_experiment_source.csv`: lowest-mean-MAE model per experiment
  and source.
- `metrics/stage1_reconstruction_metrics.csv`: BOD and NH3N reconstruction
  metrics for the two-stage workflow.
- `metrics/error_by_wqi_band.csv`: WQI-band error summaries.
- `stats/bootstrap_ci.csv`: seed-level bootstrap confidence intervals over
  repeated splits.
- `stats/paired_error_tests.csv`: paired Wilcoxon comparisons over per-seed
  MAE with Holm correction.
- `stress_tests/stress_summary.csv`: scenario-based stress-test response
  summaries.

The dataset, split indices, predictions, and model artifacts are private local
outputs and must not be committed.

## Interpretation Boundary

Complete-input WQI5 computation remains the reference method when all five
indicators are available. These experiments evaluate whether surrogate models
can provide useful auxiliary estimates under missing indicators, delayed
measurements, indicator reconstruction, or controlled stress-test scenarios.
