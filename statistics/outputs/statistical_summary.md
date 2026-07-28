# Statistical Summary

## Scope

This report summarizes complete-input GPU performance, missing-indicator results, 107-window stress-test results, and CPU-only timing outputs.
It reports R2, MAE, RMSE, Macro-F1, bootstrap confidence intervals, and paired model tests.

The task remains WQI5 surrogate regression, not future water-quality forecasting. Direct WQI5 computation remains the reference method when all five indicators are available.

## Main Findings

- Complete-input GPU repeated-split best model: `xgboost` with R2=0.9996, MAE=0.2638, RMSE=0.4241.
- Missing NH3N reduced retraining remains useful as an auxiliary setting: `lightgbm` with R2=0.9494, MAE=2.3694.
- DO/EC/SS-only reduced retraining is not reliable on the external hold-out: `rf` with R2=-0.1401, MAE=14.7246.
- The 107-window stress test uses sequential event windows, not 107-fold cross-validation.
- CPU-only timing is a rough inference-time reference; GPU/multicore acceleration is used only for experiment reproduction.
- Pairwise model comparisons use complete-input GPU repeated-split MAE with Holm-adjusted paired t-test p-values across the 15 model pairs.

## Output Files

- `complete_input_performance.csv`
- `missing_indicator_robustness.csv`
- `cpu_only_timing.csv`
- `stress107_summary.csv`
- `feature_score_correlations.csv`
- `bootstrap_ci.csv`
- `paired_error_tests.csv`
- `sample_size_sensitivity.csv`
- `sample_size_metrics_by_fold.csv`

## Sample-Size Sensitivity

The sample-size experiment evaluates six surrogate models using 1,000, 10,000, and 50,000 rows under stratified 80/20 splits. The summary output is `sample_size_sensitivity.csv`, and fold-level results are provided in `sample_size_metrics_by_fold.csv`.

## Pairwise Error Tests

Complete-input GPU comparisons use repeated-split MAE values paired by seed. `A-B` is `ModelA MAE - ModelB MAE`, so negative values favor ModelA. The p-value column is Holm-adjusted across the 15 model-pair comparisons.

| ModelA | ModelB | A-B | A-B 95% CI | p-value | Significant |
| --- | --- | ---: | --- | ---: | --- |
| lightgbm | lr | -7.2673097338681032 | [-7.2903447555694596, -7.2442747121667468] | 1.529e-10 | yes |
| lightgbm | mpr | -4.8720865610707973 | [-4.8930542623153244, -4.8511188598262702] | 3.810e-10 | yes |
| lightgbm | rf | -0.049528002755215586 | [-0.063680933021625516, -0.035375072488805649] | 9.300e-04 | yes |
| lightgbm | svm | -1.6278333380378112 | [-1.6532062884584264, -1.6024603876171959] | 2.383e-08 | yes |
| lightgbm | xgboost | 0.024438810628445939 | [0.017977096504734304, 0.030900524752157574] | 9.300e-04 | yes |
| lr | mpr | 2.3952231727973059 | [2.3651589946174059, 2.4252873509772059] | 1.503e-08 | yes |
| lr | rf | 7.217781731112888 | [7.1900131095167659, 7.2455503527090102] | 2.876e-10 | yes |
| lr | svm | 5.6394763958302923 | [5.6130180551146207, 5.6659347365459638] | 4.892e-10 | yes |
| lr | xgboost | 7.2917485444965493 | [7.2654937846460816, 7.318003304347017] | 2.376e-10 | yes |
| mpr | rf | 4.8225585583155812 | [4.7980446556258594, 4.8470724610053031] | 6.067e-10 | yes |
| mpr | svm | 3.2442532230329868 | [3.2197203410170565, 3.2687861050489171] | 2.641e-09 | yes |
| mpr | xgboost | 4.8965253716992434 | [4.8762651345572534, 4.9167856088412334] | 3.551e-10 | yes |
| rf | svm | -1.5783053352825955 | [-1.5946078207007606, -1.5620028498644305] | 8.045e-09 | yes |
| rf | xgboost | 0.073966813383661525 | [0.063806057868705426, 0.084127568898617625] | 1.061e-04 | yes |
| svm | xgboost | 1.6522721486662573 | [1.6300503522804322, 1.6744939450520824] | 1.652e-08 | yes |

## Reporting Boundary

The p-values test paired MAE differences from `results/complete_input_gpu/repeated_split_results.csv`, matched by `seed`. The complete-input GPU archive contains split-level metrics rather than per-row predictions, so the paired tests use five seed-level paired values per model comparison.
All 15 model pairs are significant under this paired t-test because every pair has same-direction MAE differences across the five seeds. These tests use seed-level MAE values rather than row-level absolute errors.

The 107-window stress test reduces dependence on a single selected middle window, but it does not prove absence of all sampling bias and is not a real pollution-event validation.
