# Data Availability and Redistribution

This document defines the data and model artifacts that are included in, excluded from, or generated from WQSurrogateModels.

## Included in the Repository

The repository includes source code, API and methodology documentation, configuration examples, reproducibility scripts, aggregate experiment outputs, statistical summaries, figures, and the metadata-only production model manifest.

Aggregate outputs may be published or committed when they do not contain restricted row-level records, private source data, credentials, or serialized model parameters whose redistribution rights have not been established.

## Not Included in the Repository or Releases

The complete study-specific row-level dataset, processed experiment table, experiment subsets, downloaded source data, and derived row-level exports are not distributed through this repository or its releases. The `data/` directory is ignored by Git and is intended for locally prepared inputs only.

Trained surrogate model binaries, including serialized `.pkl` or `.joblib` artifacts, are not distributed through this repository or its releases unless their redistribution rights are separately established. `models/production_model_manifest.json` records expected local artifact paths and metadata; it does not contain model parameters.

## Preparing Compatible Local Data

Users who have the required data access can prepare schema-compatible local inputs by following [Data Preparation](data_preparation.md). The documented source is the [Ministry of Environment's National Environmental Water Quality Monitoring Information Network](https://wq.moenv.gov.tw/EWQP/zh/ConService/DownLoad/HistoryData.aspx), and the preparation instructions do not guarantee the same rows or record counts as the reported study dataset.

The experiment schema is:

```text
DO,BOD,NH3N,EC,SS,Score
```

The API input schema uses the five feature columns without `Score`:

```text
DO,BOD,NH3N,EC,SS
```

## Source Terms and Redistribution

Data obtained from external monitoring sources remain subject to the source provider's access terms, licenses, and redistribution conditions. The Apache-2.0 license for this repository does not grant rights to third-party datasets, derived restricted records, or trained artifacts created from materials with separate terms.

Before sharing a dataset, row-level export, trained model binary, or other derived artifact, verify that its source terms permit the intended redistribution and preserve the required attribution or notices. When those rights are unclear, share the preparation procedure or aggregate summary instead of the restricted artifact.

## Related Documentation

- [Data Preparation](data_preparation.md)
- [Model Card](model_card.md)
- [Experiment Protocol](experiment_protocol.md)
- [Project README](../README.md)
