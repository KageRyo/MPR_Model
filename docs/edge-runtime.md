# Production Artifact and ARM64 Runtime

Production surrogate artifacts are Python/joblib serializations, so the runtime
that loads them is part of the deployment contract. This document records the
supported artifact runtime for x86 and ARM64 deployments; it does not replace
the Jetson Nano smoke test and benchmark work.

## Compatibility contract

`models/production_model_manifest.json` is the source of truth for each
production artifact. The service checks its declared package versions before it
calls `joblib.load`. Readiness reports describe a missing or incompatible
dependency; assessment requests retain the stable `model_unavailable` HTTP 503
contract and record the detailed reason in the service log.

| Model family | Required packages |
| --- | --- |
| `direct_wqi5` | No persisted artifact or optional model dependency |
| `lr`, `mpr`, `svm`, `rf` | `joblib==1.5.3`, `scikit-learn==1.5.2` |
| `xgboost` | Base requirements plus `xgboost==2.1.4` |
| `lightgbm` | Base requirements plus `lightgbm==4.6.0` |

The production XGBoost artifact records XGBoost 2.1.4 in its serialized model
configuration. The LightGBM artifact format only records its major model format
version, so 4.6.0 is the pinned and load-validated deployment baseline. If an
artifact is re-exported, update its manifest runtime entry and re-run the
compatibility tests before promoting it.

XGBoost emits its upstream warning about loading a serialized Booster when this
legacy pickle is loaded, even under the pinned 2.1.4 runtime. The artifact has
been load- and prediction-validated with that runtime. The API intercepts only
that known verbose warning and logs a concise migration notice; unrelated
warnings keep their usual handling.

To create a candidate without the legacy warning, rehydrate the artifact using
the pinned runtime. The command never overwrites the source or an existing
target, and checks prediction parity before writing the candidate:

```bash
uv run --locked --extra xgboost python scripts/rehydrate_xgboost_artifact.py \
  --source models/XGBoost/modelXGBVer.2.0-50000-seed2.pkl \
  --target /path/to/modelXGBVer.2.0-50000-seed2-rehydrated.pkl \
  --verification-csv /path/to/representative_inputs.csv
```

`--verification-csv` is optional; without it the script uses three complete
input probes. Treat the output as a candidate: update the manifest, checksum,
and deployment evidence only in a separate promotion change.

## Installation strategy

Exact model-runtime pins live in `pyproject.toml`; `uv.lock` resolves their
transitive dependencies for each supported platform. The production manifest
ties those pins to individual artifacts. This avoids a separate constraints
file drifting away from the package metadata or the deployed artifact.

Install only the runtime a deployment needs:

```bash
# direct_wqi5 and scikit-learn artifacts, including Random Forest
uv sync --locked

# a deployment that loads only one optional family
uv sync --locked --extra xgboost
uv sync --locked --extra lightgbm

# a deployment that must load either optional family
uv sync --locked --extra models
```

Equivalent `pip` commands are `pip install .`, `pip install ".[xgboost]"`,
`pip install ".[lightgbm]"`, and `pip install ".[models]"`. Use the matching
optional extra rather than installing XGBoost and LightGBM on a `direct_wqi5`
or Random Forest-only deployment.

The Docker image follows the same rule. Its default build contains the base
runtime only; enable an optional family deliberately:

```bash
docker build -t wqsurrogatemodels:base .
docker build --build-arg MODEL_EXTRAS=xgboost -t wqsurrogatemodels:xgboost .
docker build --build-arg MODEL_EXTRAS=lightgbm -t wqsurrogatemodels:lightgbm .
docker build --build-arg MODEL_EXTRAS=models -t wqsurrogatemodels:models .
```

`INSTALL_MODEL_EXTRAS=true` remains the docker-compose-compatible shortcut for
installing both optional families.

The image includes `libgomp1`, the OpenMP runtime required to import the
published LightGBM wheel on slim Debian images.

## ARM64 availability

The pinned versions have Linux aarch64 binary paths compatible with the
`manylinux2014` / glibc 2.17 baseline:

- `scikit-learn==1.5.2` supplies CPython 3.10–3.12 aarch64 wheels.
- `xgboost==2.1.4` supplies a `manylinux2014_aarch64` wheel.
- `lightgbm==4.6.0` supplies a `manylinux2014_aarch64` wheel.

CI builds the optional-model image for `linux/arm64` to catch dependency
resolution regressions. That confirms an ARM64 container build path, not
device behaviour: Jetson Nano / JetPack image compatibility, mounted-artifact
loading, and latency/power measurements remain acceptance work for issues #55
and #57.

## Training versus inference

The pinned libraries are the inference contract for the committed production
artifacts. Training may use separate, explicitly recorded environments, but a
newly promoted artifact must carry its own runtime entry in the production
manifest. Do not assume that a training environment's newer package versions
can safely unpickle an older production artifact.
