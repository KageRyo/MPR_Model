# Full-Stack Local Run

This guide runs `WQSurrogateModels` and `WaterMirror` together on one machine.

## 1. Start the backend

```bash
cd WQSurrogateModels
cp .env.example .env
python main.py
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
npm install
npx expo start
```

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
