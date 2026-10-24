# Enterprise AI Knowledge Assistant - one image for both the API and the UI.
#
#   docker compose up --build          # API :8000 + UI :8501 (see docker-compose.yml)
#
# Build args:
#   PRELOAD_MODELS=1 (default) bakes the embedding + reranker models (~180 MB) into
#   the image, so containers start fast and work without internet access.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Dependencies first, so code changes do not invalidate this (slow) layer.
# CPU-only torch: the default Linux wheel would add several GB of CUDA libraries.
COPY requirements-lock.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt

COPY . .

ARG PRELOAD_MODELS=1
RUN if [ "$PRELOAD_MODELS" = "1" ]; then python scripts/download_models.py; fi
# With models baked in, never call the Hugging Face Hub at runtime.
ENV HF_HUB_OFFLINE=${PRELOAD_MODELS}

# Run as an unprivileged user; only the data directory (a volume) is writable.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/documents /app/data/processed /opt/hf-cache \
    && chown -R appuser /app/data /opt/hf-cache
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
