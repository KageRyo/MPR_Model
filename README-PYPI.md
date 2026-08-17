# WQSurrogateModels

`wqsurrogatemodels` is the installable Python backend for WQI5-based current-state
water quality assessment.

## Install

```bash
pip install wqsurrogatemodels
```

The package requires Python 3.10 or newer. Optional surrogate-model libraries
are available with:

```bash
pip install "wqsurrogatemodels[models]"
```

## Use the direct WQI5 baseline

```python
from wqsurrogatemodels import direct_wqi5_score

score = direct_wqi5_score(do=7.2, bod=2.1, nh3n=0.3, ec=450, ss=12)
```

## Run the API

```bash
uvicorn wqsurrogatemodels.api:app --host 0.0.0.0 --port 8001
```

The primary HTTP contract is under `/api/v2/*`. Datasets and trained surrogate
artifacts are supplied separately and are not included in the PyPI distribution.
Set `PROJECT_ROOT` to the directory containing local `data/` and `models/`
directories when running the API outside a source checkout.

## Package boundary

The distribution contains the runtime package, dependency metadata, this
description, and the Apache 2.0 license. It excludes research datasets,
serialized model binaries, generated results, experiment configurations,
reproducibility scripts, repository-only tests, and research documentation.

Repository and API documentation:
<https://github.com/KageRyo/WQSurrogateModels>
