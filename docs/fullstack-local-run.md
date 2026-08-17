# Full-Stack Local Run

This guide runs `WQSurrogateModels` and `WaterMirror` together on one machine.

## Prerequisites

- Python 3.10 or newer
- `uv` for the Python backend
- Node.js 20 or newer and npm for the Expo frontend

## 1. Start the backend

```bash
cd WQSurrogateModels
# Run this only when .env does not exist yet.
test -f .env || cp .env.example .env
uv sync --extra dev --extra models
uv run python main.py
```

For a local frontend/backend pair, keep these backend settings aligned with
the frontend URL:

```env
API_HOST=0.0.0.0
API_PORT=8001
AUTO_PORT=false
CORS_ALLOW_ORIGINS=*
```

Default local backend address:

```text
http://localhost:8001
```

## 2. Check the health endpoint

```bash
curl http://localhost:8001/api/v2/health
```

## 3. Configure WaterMirror

In `WaterMirror/.env`:

```dotenv
EXPO_PUBLIC_API_BASE_URL=http://localhost:8001
EXPO_PUBLIC_DEFAULT_MODEL=direct_wqi5
EXPO_PUBLIC_REQUEST_TIMEOUT_MS=10000
```

## 4. Start the frontend

```bash
cd WaterMirror
npm ci
npx expo start --web
```

`uv` is used for the Python backend only. WaterMirror is an Expo/React Native
application and uses Node.js/npm.

## 5. Test the flow

Use one of these paths:

1. Enter `DO`, `BOD`, `NH3N`, `EC`, and `SS` manually.
2. Upload a CSV with header `DO,BOD,NH3N,EC,SS`.
3. Confirm the app renders backend-returned `score`, `category`, `rating_range`, and `warnings`.

## Physical phone testing

If testing on a real phone, replace `localhost` in `WaterMirror/.env` with your machine's LAN IP:

```dotenv
EXPO_PUBLIC_API_BASE_URL=http://<your-lan-ip>:8001
```

The backend must listen on `0.0.0.0`, and the phone and development machine
must be on the same LAN. Keep `API_PORT` and the port in
`EXPO_PUBLIC_API_BASE_URL` the same; do not use `AUTO_PORT=true` for a paired
frontend/backend run unless you also update the frontend URL to the selected
port.

If Node.js is not installed locally, the existing WaterMirror Dockerfile can
run the Expo web frontend with Node 20:

```bash
cd WaterMirror
docker build -t watermirror-dev:local .
docker run --rm --name watermirror-web \
  --publish 8081:8081 \
  --env EXPO_PUBLIC_API_BASE_URL=http://localhost:8001 \
  --env EXPO_PUBLIC_DEFAULT_MODEL=direct_wqi5 \
  --env EXPO_PUBLIC_REQUEST_TIMEOUT_MS=10000 \
  --env REACT_NATIVE_PACKAGER_HOSTNAME=0.0.0.0 \
  watermirror-dev:local npx expo start --web --host lan --port 8081
```
