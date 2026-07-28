# Data Preparation

The dataset used by the experiment workflows is not distributed with this
repository. This page documents only its size, schema, and preprocessing
contract so that users can prepare a compatible local input.

## Dataset Summary

- Original records: approximately `87,005`
- Records after preprocessing: `60,714`
- File format: CSV with a header row
- Data layout: cross-sectional rows with no timestamp field
- Feature columns: `DO`, `BOD`, `NH3N`, `EC`, `SS`
- Target column: `Score`, a dimensionless WQI5 score
- Expected column order: `DO,BOD,NH3N,EC,SS,Score`

All six columns are numeric. Experiment scripts derive WQI5 categories from
`Score` when stratified splits are requested.

## Preprocessing

The processed table was produced with the following operations:

1. Align source fields to the six-column schema.
2. Remove invalid or unusable records.
3. Apply dataset alignment and filtering.
4. Trim extreme values from both tails (lower `1%` and upper `1%`).
5. Export the retained rows in CSV format.

The reduction from approximately `87,005` to `60,714` rows is the combined
result of these operations; it is not a single-variable keep rate. Training
pipelines apply mean imputation when a compatible local input contains missing
values.

## Local Use

Files under `data/` are private local inputs and are ignored by Git. Configure
the dataset paths in `.env` and `configs/*.yaml` before running data-dependent
endpoints or experiments. Do not commit local datasets, subsets, or derived
row-level exports.
