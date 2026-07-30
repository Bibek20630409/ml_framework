"""
serving/api.py
──────────────
FastAPI inference service around the artifact bundle / MLflow registry. The model
is loaded once at startup and reused across requests.

Endpoints:
  GET  /health          → liveness/readiness (unauthenticated)
  GET  /metrics         → Prometheus metrics (unauthenticated)
  POST /predict         → {"predictions": [...]}         (auth + rate-limited)
  POST /predict_proba   → {"probabilities": [[...]]}      (auth + rate-limited)

Hardening (production): API-key auth (``X-API-Key``), per-client rate limiting,
and request-size caps. Secrets come from the environment, never config files.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ..core import Inferencer

log = logging.getLogger(__name__)

# Prometheus is optional (part of the [serve]/[mlops] extras).
_PRED_COUNTER: Any = None
_DRIFT_GAUGE: Any = None
try:
    from prometheus_client import Counter, Gauge
    from prometheus_fastapi_instrumentator import Instrumentator

    _PROM = True
    _PRED_COUNTER = Counter(
        "mlf_predictions_total",
        "Model predictions, labelled by predicted class",
        ["predicted_class"],
    )
    _DRIFT_GAUGE = Gauge(
        "mlf_feature_psi",
        "Input feature drift (PSI) vs the training reference distribution",
        ["feature"],
    )
except Exception:  # pragma: no cover - prometheus not installed
    _PROM = False

# slowapi is optional (part of the [security] extra).
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    _SLOWAPI = True
except Exception:  # pragma: no cover - slowapi not installed
    _SLOWAPI = False


class PredictRequest(BaseModel):
    instances: list[list[float]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    predictions: list[float]


class ProbaResponse(BaseModel):
    probabilities: list[list[float]]


def create_app(
    artifact_dir: str | Path = "outputs",
    *,
    registry_model: str | None = None,
    registry_stage: str = "production",
    tracking_uri: str | None = None,
    api_key: str | None = None,
    rate_limit: str | None = "60/minute",
    max_instances: int = 10_000,
) -> FastAPI:
    """Build the inference app.

    Model source: MLflow registry when ``registry_model`` is set, else the local
    bundle in ``artifact_dir``. Hardening: ``api_key`` (None → auth disabled, for
    local/dev), ``rate_limit`` (None → disabled), and ``max_instances`` per request.
    """
    state: dict[str, Any] = {"inferencer": None, "tracker": None}

    def _load() -> Inferencer:
        if registry_model:
            return Inferencer.from_registry(registry_model, registry_stage, tracking_uri)
        return Inferencer.from_artifacts(artifact_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["inferencer"] = _load()
        log.info("model loaded (registry=%s dir=%s)", registry_model, artifact_dir)
        yield

    app = FastAPI(title="ML Framework Inference API", version="1.0.0", lifespan=lifespan)

    # ── Rate limiting (per client IP) ─────────────────────
    if _SLOWAPI and rate_limit:
        limiter = Limiter(key_func=get_remote_address, default_limits=[rate_limit])
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
        app.add_middleware(SlowAPIMiddleware)

    # ── Auth (API key) ────────────────────────────────────
    def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        # Disabled when no key is configured (local/dev). Constant-time compare.
        if api_key:
            import hmac

            if not x_api_key or not hmac.compare_digest(x_api_key, api_key):
                raise HTTPException(status_code=401, detail="invalid or missing API key")

    auth = [Depends(require_api_key)]

    def _get_inferencer() -> Inferencer:
        inf = state["inferencer"]
        if inf is None:  # e.g. tests that skip startup
            inf = _load()
            state["inferencer"] = inf
        return inf

    def _get_tracker():
        if state["tracker"] is None:
            from ..monitoring import DriftTracker

            inf = _get_inferencer()
            state["tracker"] = DriftTracker(
                inf.reference_stats, inf.feature_cols, gauge=_DRIFT_GAUGE
            )
        return state["tracker"]

    def _to_array(req: PredictRequest) -> np.ndarray:
        if len(req.instances) > max_instances:
            raise HTTPException(
                status_code=413,
                detail=f"too many instances: {len(req.instances)} > {max_instances}",
            )
        arr = np.asarray(req.instances, dtype="float32")
        if arr.ndim != 2:
            raise HTTPException(status_code=422, detail="instances must be a 2D array")
        inf = _get_inferencer()
        expected = getattr(inf.model, "input_dim", None)
        if expected and arr.shape[1] != expected:
            raise HTTPException(
                status_code=422,
                detail=f"expected {expected} features, got {arr.shape[1]}",
            )
        return arr

    @app.get("/health")
    def health() -> dict:
        inf = _get_inferencer()
        return {"status": "ok", "task": inf.task, "model": inf.config.model.name}

    @app.post("/predict", response_model=PredictResponse, dependencies=auth)
    def predict(req: PredictRequest) -> PredictResponse:
        inf = _get_inferencer()
        arr = _to_array(req)
        preds = inf.predict(arr)
        _get_tracker().observe(arr)  # feed the rolling drift window
        if _PROM and _PRED_COUNTER is not None and inf.task != "regression":
            for p in preds:
                _PRED_COUNTER.labels(predicted_class=str(int(p))).inc()
        return PredictResponse(predictions=[float(p) for p in preds])

    @app.get("/drift", dependencies=auth)
    def drift() -> dict:
        """Current input drift (PSI per feature) over the rolling window."""
        return {"psi": _get_tracker().compute()}

    @app.post("/predict_proba", response_model=ProbaResponse, dependencies=auth)
    def predict_proba(req: PredictRequest) -> ProbaResponse:
        inf = _get_inferencer()
        if inf.task == "regression":
            raise HTTPException(status_code=400, detail="proba unavailable for regression")
        probs = inf.predict_proba(_to_array(req))
        return ProbaResponse(probabilities=[[float(v) for v in row] for row in probs])

    # Expose latency/throughput/error metrics at /metrics for Prometheus.
    if _PROM:
        Instrumentator().instrument(app).expose(app, include_in_schema=False)

    return app
