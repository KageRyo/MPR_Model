# API Reference

This document summarizes the primary HTTP API exposed by `WQSurrogateModels`.

## Base Path

New clients should use `/api/v2/*`.

## Health and Discovery

### `GET /api/v2/health`

Checks service availability.

Example:

```bash
curl http://localhost:8001/api/v2/health
```

### `GET /api/v2/ready`

Checks whether the service is ready to accept assessments. Clients should use
this endpoint for readiness checks and reserve `/health` for process reachability.
It returns `200` with `status: "ready"` when configured required dependencies
are available, or `503` with `status: "not_ready"` when they are not. The
response reports whether the optional local dataset is available and a safe
availability summary for every supported model type; it never includes local
filesystem paths.

### `GET /api/v2/models`

Returns the supported `model_type` values.

### `GET /api/v2/categories`

Returns WQI5 category distribution metadata.

### `GET /api/v2/percentile?score=82.5`

Returns percentile information for a supplied score.

## Assessment

### `POST /api/v2/assessment`

Submits a single record for assessment.

Example request:

```json
{
  "DO": 70,
  "BOD": 2,
  "NH3N": 0.3,
  "EC": 400,
  "SS": 10,
  "model_type": "direct_wqi5"
}
```

Example `curl`:

```bash
curl -X POST http://localhost:8001/api/v2/assessment \
  -H "Content-Type: application/json" \
  -d '{"DO":70,"BOD":2,"NH3N":0.3,"EC":400,"SS":10,"model_type":"direct_wqi5"}'
```

Example response:

```json
{
  "score": 75.685,
  "category": "Good",
  "rating_range": "70 < WQI5 ≤ 85",
  "model_type": "direct_wqi5",
  "latency_ms": 0.123,
  "assessment": {
    "DO": "Fair",
    "BOD": "Good",
    "NH3N": "Fair",
    "EC": "Good",
    "SS": "Good"
  },
  "warnings": []
}
```

### `POST /api/v2/assessment/csv/summary`

Accepts a CSV upload and returns a summary assessment.

Expected CSV header:

```text
DO,BOD,NH3N,EC,SS
```

The optional multipart form field `model_type` uses one of the documented model
type values. If omitted, the configured backend default is used.

### `POST /api/v2/assessment/csv/rows`

Accepts a CSV upload and returns per-row results.

The response contains `scores` (one value per CSV row), `model_type`, and
`latency_ms`.

## Model Types

Supported `model_type` values:

- `direct_wqi5`
- `lr`
- `mpr`
- `svm`
- `rf`
- `xgboost`
- `lightgbm`

## Legacy Endpoints

Legacy root-level endpoints remain available for backward compatibility:

- `GET /status`
- `GET /models`
- `GET /categories`
- `GET /percentile`
- `POST /predict`
- `POST /score/total/`
- `POST /score/all/`

New code should use `/api/v2/*`.
