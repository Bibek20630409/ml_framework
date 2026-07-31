"""
serving/api.py
──────────────
FastAPI inference service over an artifact bundle (or an MLflow registry model).
The model is loaded once at startup and reused across requests.

Endpoints:
  GET  /health                  → liveness/readiness + what is loaded (unauthenticated)
  GET  /metrics                 → Prometheus metrics (unauthenticated)
  POST /predict                 → {"predictions": [...], "labels": [...]}
  POST /predict_proba           → {"probabilities": [[...]], "classes": [...]}
  POST /predict_with_confidence → {"predictions": [...], "confidence": [...]}
  GET  /drift                   → PSI per feature (tabular bundles only)

**Nothing here imports torch.** The app talks to an
:class:`~ml_framework.core.inference.Inferencer`, which dispatches on the bundle
manifest, so the same code serves a Lightning checkpoint and an XGBoost booster —
and a GBDT image never installs a deep-learning stack to do it.

Three things are manifest-derived rather than hardcoded, each replacing a v1 wart:

* **The request/response schemas**, chosen by ``data.kind`` from
  ``serving/schemas.py``, so ``/docs`` describes *this* model.
* **The ``/predict_proba`` refusal**, which reads ``signature.output.kind``
  instead of testing ``task == "regression"`` — a check that was wrong for every
  task not yet invented.
* **The input contract**, from ``signature.input``, replacing
  ``getattr(inf.model, "input_dim", None)`` — code that reached into a torch
  module to learn its own API and returned ``None`` for anything else.

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

from ..core.inference import Inferencer
from ..core.types import UnsupportedCapability
from .metrics import get_collectors, instrument
from .schemas import (
    ConfidenceResponse,
    ForecastRequest,
    ForecastResponse,
    PayloadError,
    ProbaResponse,
    request_model,
    response_model,
    supports_drift,
)

log = logging.getLogger(__name__)

# slowapi is optional (part of the [security] extra).
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    _SLOWAPI = True
except ImportError:  # pragma: no cover - slowapi not installed
    _SLOWAPI = False


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
    collectors = get_collectors()

    def _load() -> Inferencer:
        if registry_model:
            return Inferencer.from_registry(registry_model, registry_stage, tracking_uri)
        return Inferencer.from_artifacts(artifact_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["inferencer"] = _load()
        log.info("model loaded (registry=%s dir=%s)", registry_model, artifact_dir)
        yield

    from .. import __version__

    app = FastAPI(title="ML Framework Inference API", version=__version__, lifespan=lifespan)

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
                inf.reference_stats, inf.feature_cols, gauge=collectors.drift
            )
        return state["tracker"]

    # The bundle is loaded lazily, so the request schema cannot be a static
    # annotation without forcing a load at import time. The body is validated
    # explicitly instead, against the model chosen from the manifest.
    def _parse(body: dict[str, Any]) -> tuple[Inferencer, Any, np.ndarray]:
        inf = _get_inferencer()
        try:
            req = request_model(inf.data_kind).model_validate(body)
        except PayloadError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        n_rows = getattr(req, "n_rows", 0)
        if n_rows > max_instances:
            raise HTTPException(
                status_code=413, detail=f"too many instances: {n_rows} > {max_instances}"
            )
        # Checked rather than caught: a kind whose request schema cannot yet build
        # model input is a 501, but an AttributeError raised *inside* to_array is a
        # bug and must not be reported as "not implemented".
        to_array = getattr(req, "to_array", None)
        if to_array is None:
            raise HTTPException(
                status_code=501,
                detail=f"serving data kind '{inf.data_kind}' is not implemented yet",
            )
        try:
            arr = to_array(inf.feature_cols)
        except PayloadError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        expected = inf.n_features
        if expected and arr.shape[1] != expected:
            raise HTTPException(
                status_code=422, detail=f"expected {expected} features, got {arr.shape[1]}"
            )
        return inf, req, arr

    def _labels(inf: Inferencer, preds: np.ndarray) -> list[str] | None:
        """Human-readable class names, when the bundle records them."""
        names = inf.class_names
        if not names or not inf.produces_proba:
            return None
        return [names[int(p)] if 0 <= int(p) < len(names) else str(int(p)) for p in preds]

    # ── Routes ────────────────────────────────────────────
    @app.get("/health")
    def health() -> dict:
        """Keeps v1's ``status``/``task`` keys and adds what is actually loaded."""
        inf = _get_inferencer()
        return {
            "status": "ok",
            "task": inf.task,
            "model": inf.model_name,
            "backend": inf.backend_name,
            "data_kind": inf.data_kind,
            "bundle_version": inf.manifest.bundle_version,
            "framework_version": inf.manifest.framework_version,
            "n_features": inf.n_features,
        }

    def _forecast(body: dict[str, Any]) -> Any:
        """The timeseries branch of ``/predict``.

        A forecaster is not scored on rows: it continues from where it was fitted
        and is asked for a horizon. Routing it through ``_parse`` would demand a
        feature matrix that does not exist — which is why the payload varies by
        data kind rather than pretending every model takes one.
        """
        inf = _get_inferencer()
        # Named directly rather than looked up: this branch runs only for
        # timeseries, and going through `request_model` would hand back a
        # `type[BaseModel]` that has no `horizon` as far as a type checker is
        # concerned — silenced only by ignores that would also hide a real mistake.
        assert request_model(inf.data_kind) is ForecastRequest
        try:
            req = ForecastRequest.model_validate(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if req.horizon > max_instances:
            raise HTTPException(status_code=413, detail=f"horizon {req.horizon} > {max_instances}")

        values = inf.predict(req.horizon)
        lower, upper = inf.forecast_interval(req.horizon)
        return ForecastResponse(
            forecast=[float(v) for v in values],
            index=None,
            lower=None if lower is None else [float(v) for v in lower],
            upper=None if upper is None else [float(v) for v in upper],
        )

    @app.post("/predict", dependencies=auth)
    def predict(body: dict[str, Any]) -> Any:
        if _get_inferencer().data_kind == "timeseries":
            return _forecast(body)
        inf, _, arr = _parse(body)
        preds = inf.predict(arr)
        if supports_drift(inf.data_kind):
            _get_tracker().observe(arr)  # feed the rolling drift window
        if collectors.predictions is not None and inf.produces_proba:
            for p in preds:
                collectors.predictions.labels(predicted_class=str(int(p))).inc()
        model = response_model(inf.data_kind)
        return model(predictions=[float(p) for p in preds], labels=_labels(inf, preds))

    def _require_proba() -> Inferencer:
        """Refuse before parsing.

        Whether a model produces probabilities is a property of the *manifest*, not
        of the request body — so a forecasting bundle must answer 400 ("this
        produces values over a horizon") rather than 501 ("that payload is not
        implemented"). Checking after parsing gave the second, less accurate reason.
        """
        inf = _get_inferencer()
        if not inf.produces_proba:
            raise HTTPException(
                status_code=400,
                detail=(f"this model produces {inf.signature.output.kind}, not probabilities"),
            )
        return inf

    @app.post("/predict_proba", response_model=ProbaResponse, dependencies=auth)
    def predict_proba(body: dict[str, Any]) -> ProbaResponse:
        _require_proba()
        inf, _, arr = _parse(body)
        try:
            probs = inf.predict_proba(arr)
        except UnsupportedCapability as exc:
            # From the manifest, not from a hardcoded task check.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ProbaResponse(
            probabilities=[[float(v) for v in row] for row in probs],
            classes=inf.class_names,
        )

    @app.post("/predict_with_confidence", response_model=ConfidenceResponse, dependencies=auth)
    def predict_with_confidence(body: dict[str, Any]) -> ConfidenceResponse:
        """``Inferencer.predict_with_confidence`` exposed over HTTP."""
        _require_proba()
        inf, _, arr = _parse(body)
        try:
            preds, conf = inf.predict_with_confidence(arr)
        except UnsupportedCapability as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ConfidenceResponse(
            predictions=[int(p) for p in preds],
            confidence=[float(c) for c in conf],
            labels=_labels(inf, preds),
        )

    @app.get("/drift", dependencies=auth)
    def drift() -> dict:
        """Current input drift (PSI per feature) over the rolling window.

        Tabular only. PSI over token ids or pixel bytes is a number without a
        meaning, so other kinds get an explicit 501 rather than a plausible-looking
        answer.
        """
        inf = _get_inferencer()
        if not supports_drift(inf.data_kind):
            raise HTTPException(
                status_code=501,
                detail=(
                    f"drift is computed over named numeric features; data kind "
                    f"'{inf.data_kind}' has none"
                ),
            )
        return {"psi": _get_tracker().compute()}

    # Expose latency/throughput/error metrics at /metrics for Prometheus.
    instrument(app)

    return app


__all__ = ["create_app"]
