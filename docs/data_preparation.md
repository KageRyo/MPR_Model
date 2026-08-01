# Data Preparation

This repository does not distribute the complete source dataset, the processed
experiment table, or experiment subsets. It provides a preprocessing contract
for preparing schema-compatible local inputs; the study dataset itself is not
distributed and cannot be reconstructed from this repository alone. Download
water-quality monitoring data from the [Ministry of Environment's National Environmental Water Quality Monitoring Information Network](https://wq.moenv.gov.tw/EWQP/zh/ConService/DownLoad/HistoryData.aspx),
then use the schema and preprocessing contract below to prepare a local input.
These instructions do not guarantee the same rows or record counts as the study
dataset. Data obtained from the source remain subject to the applicable source
terms.

## Reported Study Dataset Summary

The following values describe the dataset used in the reported study.

- Original records: approximately `87,005`
- Records after preprocessing: `60,714`
- File format: CSV with a header row
- Prepared data layout: cross-sectional rows with no timestamp field
- Feature columns: `DO`, `BOD`, `NH3N`, `EC`, `SS`
- Target column: `Score`, a dimensionless WQI5 score
- Expected column order: `DO,BOD,NH3N,EC,SS,Score`

All six columns are numeric. Experiment scripts derive WQI5 categories from
`Score` when stratified splits are requested.

## Training-Data-Volume Experiments

To assess sensitivity to training-data volume, the sample-size workflow uses
locally prepared subsets of `1,000`, `5,000`, `10,000`, and `50,000` rows.
Each subset must follow the same six-column schema and preprocessing contract
described on this page. See [sample-size-experiments.md](sample-size-experiments.md)
for the split protocol, run commands, and reported metrics.

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

Files under `data/` are locally prepared inputs and are ignored by Git.
Configure the dataset paths in `.env` and `configs/*.yaml` before running
data-dependent endpoints or experiments. Do not commit downloaded data, local
subsets, or derived row-level exports.
