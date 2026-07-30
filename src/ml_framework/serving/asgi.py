"""
serving/asgi.py
───────────────
Module-level ASGI app built from environment variables — the entry point for
multi-worker servers (gunicorn / uvicorn ``--workers``), which need an import
string rather than a constructed object.

    gunicorn ml_framework.serving.asgi:app -k uvicorn.workers.UvicornWorker -w 4

Config (all optional) via env:
  MLF_ARTIFACTS · MLF_REGISTRY_MODEL · MLF_REGISTRY_STAGE · MLFLOW_TRACKING_URI
  MLF_API_KEY (secret) · MLF_RATE_LIMIT · MLF_MAX_INSTANCES
"""

from __future__ import annotations

import os

from .api import create_app

app = create_app(
    os.environ.get("MLF_ARTIFACTS", "outputs"),
    registry_model=os.environ.get("MLF_REGISTRY_MODEL") or None,
    registry_stage=os.environ.get("MLF_REGISTRY_STAGE", "production"),
    tracking_uri=os.environ.get("MLFLOW_TRACKING_URI") or None,
    api_key=os.environ.get("MLF_API_KEY") or None,
    rate_limit=os.environ.get("MLF_RATE_LIMIT", "60/minute") or None,
    max_instances=int(os.environ.get("MLF_MAX_INSTANCES", "10000")),
)
