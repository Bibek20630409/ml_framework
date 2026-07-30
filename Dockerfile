# ── Base ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (smaller, no CUDA) so it's cached across rebuilds.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

# ── Serving image (default) ──────────────────────────────────────────
FROM base AS serve
RUN pip install ".[serve,mlops,security,monitoring]"
EXPOSE 8000
# Health/metrics: GET /health, GET /metrics
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
    CMD curl -fs http://localhost:8000/health || exit 1
# Production multi-worker server. The env-based factory (serving/asgi.py) reads
# MLF_* vars — e.g. MLF_REGISTRY_MODEL, MLFLOW_TRACKING_URI, MLF_API_KEY (secret),
# MLF_RATE_LIMIT — so it loads from the MLflow registry and enforces auth.
CMD ["gunicorn", "ml_framework.serving.asgi:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "2", "-b", "0.0.0.0:8000", \
     "--timeout", "120"]

# ── Training / pipeline image ────────────────────────────────────────
FROM base AS train
RUN pip install ".[image,hpo,diagnostics,logging,mlops]"
ENTRYPOINT ["mlf"]
CMD ["train", "--config", "configs/example_tabular.yaml"]
