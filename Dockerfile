FROM ghcr.io/astral-sh/uv:0.12.9 AS uv

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=uv /uv /uvx /bin/

# LightGBM's published wheels need the OpenMP runtime on slim Debian images.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The runtime image includes source code and the public manifest only. Local
# datasets and serialized model artifacts are supplied as read-only mounts.
COPY pyproject.toml uv.lock README-PYPI.md LICENSE ./
COPY src ./src
COPY main.py ./
COPY models/production_model_manifest.json ./models/production_model_manifest.json

# A direct-WQI5 deployment needs only the base package. Enable optional model
# libraries explicitly when the mounted production manifest selects them.
ARG INSTALL_MODEL_EXTRAS=false
ARG MODEL_EXTRAS=""
RUN if [ -n "$MODEL_EXTRAS" ]; then \
      uv sync --locked --no-dev --extra "$MODEL_EXTRAS"; \
    elif [ "$INSTALL_MODEL_EXTRAS" = "true" ]; then \
      uv sync --locked --no-dev --extra models; \
    else \
      uv sync --locked --no-dev; \
    fi

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8001/api/v2/health', timeout=3)" || exit 1

CMD ["python", "main.py"]
