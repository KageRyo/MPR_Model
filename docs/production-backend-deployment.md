# Production Backend Deployment

This guide deploys the WQSurrogateModels API as the backend for WaterMirror.
Use the direct WQI5 baseline unless an approved local surrogate artifact is
available. The repository and published package deliberately do not distribute
research datasets or serialized model binaries.

## Choose a Runtime

Use `uv` when the host manages Python directly. Use Docker Compose when the
service should run as a long-lived container with read-only model and data
mounts. Both paths use the same `.env` configuration and v2 API.

| Runtime | Best for | Start command |
| --- | --- | --- |
| `uv` | local development or a host-managed service | `uv run python main.py` |
| Docker Compose | a repeatable container deployment | `docker compose up -d --build` |

## 1. Prepare Configuration and Local Resources

Start from the checked-in template. Do not commit the resulting `.env` file if
it contains deployment-specific values.

```bash
cp .env.example .env
mkdir -p models data
```

The default configuration starts the direct WQI5 baseline, which does not need
a model artifact or local dataset for readiness. For a Docker deployment, keep
the container `API_PORT` at `8001`; choose a different external port with
`HOST_PORT`.

```dotenv
# Docker Compose bind-mount sources on the host.
LOCAL_MODEL_DIR=/srv/wq/models
LOCAL_DATA_DIR=/srv/wq/data
HOST_PORT=8001

# Paths inside the application/container.
MODEL_DIR=models
PRODUCTION_MODEL_MANIFEST=production_model_manifest.json
DATASET_FILE=dataV1.csv

DEFAULT_MODEL=direct_wqi5
API_HOST=0.0.0.0
API_PORT=8001
AUTO_PORT=false
REQUIRE_DATASET_FOR_READINESS=false
REQUEST_TIMEOUT_MS=10000

# Replace this with the exact WaterMirror web origins in production.
CORS_ALLOW_ORIGINS=https://water.example.com
```

`PROJECT_ROOT` is optional. Set it only when running the installed package with
`data/` and `models/` outside the working directory. A malformed boolean,
integer, model type, path, or CORS list fails startup with an explicit
configuration message; it is not silently converted to a different value.

### Approved surrogate artifacts

For a surrogate default model (`lr`, `mpr`, `svm`, `rf`, `xgboost`, or
`lightgbm`), place the locally approved artifact and its
`production_model_manifest.json` under `LOCAL_MODEL_DIR`. The manifest records
the expected path, SHA-256, version, feature schema, evaluation metadata, and
runtime compatibility. The API validates the checksum before loading an
artifact.

Keep artifacts and datasets under the applicable access terms. Do not add them
to Git, container images, releases, or the PyPI package. When the selected
artifact uses XGBoost or LightGBM, set `INSTALL_MODEL_EXTRAS=true` before the
Docker build so their runtime libraries are installed.

## 2. Run with Docker Compose

Docker Compose mounts `LOCAL_MODEL_DIR` at `/app/models` and `LOCAL_DATA_DIR`
at `/app/data`, both read-only. Build and start the service:

```bash
docker compose up -d --build
docker compose ps
```

Inspect the container logs during initial deployment:

```bash
docker compose logs --follow wq-surrogate-models
```

To stop the deployment without deleting the mounted host data:

```bash
docker compose down
```

For a reverse proxy, terminate TLS at the proxy and forward requests to the
configured `HOST_PORT`. Keep the proxy health check on `/api/v2/health`; do not
substitute readiness for liveness.

## 3. Run Directly with `uv`

Install exactly the locked runtime and start the API:

```bash
uv sync --locked
uv run python main.py
```

For an XGBoost or LightGBM deployment, include the optional model dependencies:

```bash
uv sync --locked --extra models
uv run python main.py
```

Set `PROJECT_ROOT` if the local `models/` and `data/` directories are not below
the process working directory. Leave `AUTO_PORT=false` for a stable deployment
URL; enabling it can make a proxy or frontend point at the wrong port.

## 4. Verify Health, Readiness, and Assessment

Health answers whether the API process is reachable. It returns `200` and the
canonical backend `version`; it does not claim that every configured assessment
dependency is usable.

```bash
curl --fail http://localhost:8001/api/v2/health
```

Readiness answers whether the configured runtime can serve assessments. It
returns `200` with `status: "ready"` or `503` with `status: "not_ready"`, along
with safe dataset and model availability summaries. It never includes local
filesystem paths.

```bash
curl --fail http://localhost:8001/api/v2/ready
```

Run a direct-baseline assessment as an end-to-end smoke check:

```bash
curl --fail --request POST http://localhost:8001/api/v2/assessment \
  --header 'Content-Type: application/json' \
  --data '{"DO":96.2,"BOD":1.5,"NH3N":0.22,"EC":171,"SS":2.6,"model_type":"direct_wqi5"}'
```

The `/api/v2/models` response exposes only a model type, availability, and
version metadata. Use it to decide whether a WaterMirror model choice can be
offered; do not infer local paths or artifact contents from the API.

## 5. Connect WaterMirror

Configure the WaterMirror deployment with the public backend URL and matching
default model:

```dotenv
EXPO_PUBLIC_API_BASE_URL=https://api.example.com
EXPO_PUBLIC_DEFAULT_MODEL=direct_wqi5
EXPO_PUBLIC_REQUEST_TIMEOUT_MS=10000
```

Set the backend `CORS_ALLOW_ORIGINS` to the exact public WaterMirror origin,
for example `https://water.example.com`. Use `*` only for local development.
WaterMirror should call `/api/v2/*`; legacy root endpoints remain only for
backward compatibility.

## Troubleshooting

| Symptom | Check | Resolution |
| --- | --- | --- |
| Container exits at startup | `docker compose logs wq-surrogate-models` | Correct the named invalid environment value; configuration parsing fails fast. |
| Host port cannot bind | `docker compose ps` | Change `HOST_PORT`, not the container `API_PORT`; update the proxy or WaterMirror URL. |
| Health is `200` but readiness is `503` | `curl /api/v2/ready` and logs | Confirm the configured default model, manifest, and read-only mounts. If `REQUIRE_DATASET_FOR_READINESS=true`, also provide a CSV with a `Score` column. |
| Surrogate assessment reports `model_unavailable` | `/api/v2/models` and manifest checksum | Supply the approved artifact at its manifest path, use the compatible runtime, and rebuild with `INSTALL_MODEL_EXTRAS=true` for XGBoost or LightGBM. |
| Percentile or category request reports `dataset_unavailable` | Mounted `LOCAL_DATA_DIR` | Provide the configured dataset file with the expected `Score` column. |
| Browser request is blocked | Browser network panel and backend CORS setting | Add the exact WaterMirror origin to `CORS_ALLOW_ORIGINS`; do not use a wildcard in production. |
| A client receives an API error | Response `X-Request-ID` header | Search the structured backend logs for that request ID. Do not send raw CSV contents or measurement payloads in support reports. |

For endpoint payload details, see the [API reference](api-reference.md). For
the local development pairing, see [Full-Stack Local Run](fullstack-local-run.md).
