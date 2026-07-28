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
  "DO": 7.2,
  "BOD": 2.1,
  "NH3N": 0.3,
  "EC": 450,
  "SS": 12,
  "model_type": "direct_wqi5"
}
```

Example `curl`:

```bash
curl -X POST http://localhost:8001/api/v2/assessment \
  -H "Content-Type: application/json" \
  -d '{"DO":7.2,"BOD":2.1,"NH3N":0.3,"EC":450,"SS":12,"model_type":"direct_wqi5"}'
```

Example response:

```json
{
  "score": 82.5,
  "category": "Good",
  "rating_range": "70 < WQI5 ≤ 85",
  "model_type": "direct_wqi5",
  "latency_ms": 12.4,
  "assessment": {
    "DO": "Good",
    "BOD": "Fair"
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

### `POST /api/v2/assessment/csv/rows`

Accepts a CSV upload and returns per-row results.

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
