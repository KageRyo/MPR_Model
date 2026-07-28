# WaterMirror Integration Guide

This document describes the API contract between **WaterMirror** (frontend) and **WQSurrogateModels** (backend).

> **Primary contract (v2):** All new development should use endpoints under `/api/v2/*`.
> Legacy root-level endpoints are retained **only for backward compatibility** with existing WaterMirror deployments and are marked `deprecated`.

## Overview

WaterMirror communicates with this backend using the v2 contract (preferred) with fallback to legacy endpoints.

WaterMirror is the frontend application.
WQSurrogateModels is the backend and model repository.

---

## 1. Primary v2 Endpoints (Recommended)

### Health & Discovery

- `GET /api/v2/health`
- `GET /api/v2/models`
- `GET /api/v2/categories`
- `GET /api/v2/percentile?score=82.5`

### Assessment

- `POST /api/v2/assessment` — single record assessment (replaces `/predict`)
- `POST /api/v2/assessment/csv/summary` — CSV batch, returns mean score (replaces `/score/total/`)
- `POST /api/v2/assessment/csv/rows` — CSV batch, returns per-row scores (replaces `/score/all/`)

**Example `POST /api/v2/assessment` Request:**

```json
{
  "DO": 7.2,
  "BOD": 2.1,
  "NH3N": 0.3,
  "EC": 450,
  "SS": 12,
  "model_type": "lightgbm"
}
```

**Response uses `AssessmentResponseSchema` (Pydantic model).**

---

## 2. Deprecated Legacy Endpoints (Compatibility Only)

These are kept so existing WaterMirror and old demos do not break.
**Do not use in new code.**

| Deprecated          | New Primary (`/api/v2`)                  | Notes |
|---------------------|------------------------------------------|-------|
| `GET /status`       | `GET /api/v2/health`                     | Health check |
| `GET /models`       | `GET /api/v2/models`                     | Model list |
| `GET /categories`   | `GET /api/v2/categories`                 | Category distribution |
| `GET /percentile`   | `GET /api/v2/percentile`                 | Score percentile lookup |
| `POST /predict`     | `POST /api/v2/assessment`                | Single record |
| `POST /score/total/`| `POST /api/v2/assessment/csv/summary`    | CSV mean |
| `POST /score/all/`  | `POST /api/v2/assessment/csv/rows`       | CSV per-row |

All legacy endpoints are annotated with `deprecated=True` in the OpenAPI schema and will eventually be removed in a future major version.

---

## 3. Shared Response Types

### AssessmentResponseSchema (primary)

```json
{
  "score": 82.5,
  "category": "Good",
  "rating_range": "70 < WQI5 ≤ 85",
  "model_type": "lightgbm",
  "latency_ms": 12.4,
  "assessment": {
    "DO": "Good",
    "BOD": "Fair",
    ...
  },
  "warnings": []
}
```

Legacy responses keep the exact same shape for compatibility.

### Other Utility Endpoints Used by WaterMirror

All now have v2 equivalents (see table above).

---

## 4. Model Types

Supported values for `model_type` (unchanged):

| Value        | Description                     | Requires Pre-trained Model |
|--------------|----------------------------------|------------------------------|
| `direct_wqi5` | Formula-based baseline          | No                           |
| `lr`          | Linear Regression               | Yes                          |
| `mpr`         | Multiple Polynomial Regression  | Yes                          |
| `svm`         | Support Vector Machine          | Yes                          |
| `rf`          | Random Forest                   | Yes                          |
| `xgboost`     | XGBoost                         | Yes                          |
| `lightgbm`    | LightGBM                        | Yes                          |

---

## 5. CORS Configuration

When running WaterMirror (especially in development or web builds), the backend must allow requests from the frontend origin.

**Environment Variable:**

```env
CORS_ALLOW_ORIGINS=http://localhost:8081,https://your-watermirror-domain.com
```

- Use `*` only for local development.
- Separate multiple origins with commas.

---

## 6. Backward Compatibility Notes

- The backend guarantees that `model_type` is always returned as a **plain string** in all responses.
- Legacy root endpoints continue to function exactly as before (now internally delegate to the v2 service methods).
- WaterMirror has been updated to call the new `/api/v2/*` paths.
- When `model_type` is not provided, the backend uses the value from `DEFAULT_MODEL` environment variable (default: `direct_wqi5`).

---

## 7. Recommended Environment Variables for WaterMirror Integration

### Local backend on the same machine

```env
DEFAULT_MODEL=direct_wqi5
EXPO_PUBLIC_API_BASE_URL=http://localhost:8001
EXPO_PUBLIC_DEFAULT_MODEL=direct_wqi5
EXPO_PUBLIC_REQUEST_TIMEOUT_MS=10000
```

### Physical phone testing on the same LAN

```env
EXPO_PUBLIC_API_BASE_URL=http://<your-lan-ip>:8001
EXPO_PUBLIC_DEFAULT_MODEL=direct_wqi5
EXPO_PUBLIC_REQUEST_TIMEOUT_MS=10000
```

### Remote backend

```env
EXPO_PUBLIC_API_BASE_URL=https://api.example.com
EXPO_PUBLIC_DEFAULT_MODEL=direct_wqi5
EXPO_PUBLIC_REQUEST_TIMEOUT_MS=10000
```

---

**Primary contract moved to /api/v2/* while preserving backward compatibility.**

Last updated: 2026 (v2 refactor)
