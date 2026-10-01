# ── Source layer (no ML runtime) ─────────────────────────────────────
# Deliberately torch-free. The serve-gbdt target below builds from *this* stage,
# which is the only way an image can avoid torch entirely — a shared base that
# installs it would put ~2 GB into every downstream image whether or not the model
# is a neural net.
FROM python:3.11-slim AS src
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

# ── Torch base (deep-learning targets) ───────────────────────────────
FROM src AS base
# CPU-only torch (smaller, no CUDA), installed in its own layer so it caches
# across rebuilds of everything above it. Installed explicitly from the CPU index
# rather than resolved from the `lightning` extra, which would pull the default
# CUDA build and several GB of nvidia wheels with it.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# ── Serving image, deep-learning models (default) ────────────────────
FROM base AS serve
RUN pip install ".[lightning,serve,mlops,security,monitoring]"
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

# ── Serving image, gradient-boosted trees ────────────────────────────
# Builds from `src`, not `base`: **no torch, no Lightning.** This is the payoff
# from making core/inference.py manifest-driven — the loader dispatches on
# manifest.model.backend, so serving an XGBoost bundle never imports a deep
# learning stack. Expect roughly an order of magnitude less image, and a cold
# start to match.
#
#   docker build --target serve-gbdt -t ml-framework-serve-gbdt .
#
# `parquet` is included because a serving container is exactly where the pyarrow
# coupling used to break: it arrived only via mlflow, which this image omits.
FROM src AS serve-gbdt
RUN pip install ".[gbdt,serve,security,monitoring,parquet]"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD curl -fs http://localhost:8000/health || exit 1
CMD ["gunicorn", "ml_framework.serving.asgi:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "2", "-b", "0.0.0.0:8000", \
     "--timeout", "120"]

# ── Training / pipeline image ────────────────────────────────────────
FROM base AS train
RUN pip install ".[lightning,image,gbdt,hpo,diagnostics,logging,mlops]"
ENTRYPOINT ["mlf"]
CMD ["train", "--config", "configs/example_tabular.yaml"]
