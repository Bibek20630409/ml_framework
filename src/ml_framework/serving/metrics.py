"""
serving/metrics.py
──────────────────
Prometheus collectors, created **once per process** rather than once per import.

v1 built its ``Counter`` and ``Gauge`` at module scope in ``api.py``. That works
exactly as long as ``create_app`` is called once per process — and stops working
the moment it is not, because the default Prometheus registry raises
``Duplicated timeseries in CollectorRegistry`` on the second registration. Two
things in this framework now call it repeatedly: the test suite, and any process
that serves more than one model.

The fix is a lookup-then-create against the registry, which is idempotent. It also
means the collectors can be *fetched* by code that does not know whether
prometheus_client is installed at all, since :func:`get_collectors` returns a
struct of ``None``s in that case and every call site already tolerates it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

PREDICTIONS_TOTAL = "mlf_predictions_total"
FEATURE_PSI = "mlf_feature_psi"


@dataclass(frozen=True, slots=True)
class Collectors:
    """The metric handles, or ``None`` when prometheus_client is absent.

    A struct rather than a tuple so a third collector can be added without
    updating every unpacking site.
    """

    predictions: Any = None
    drift: Any = None

    @property
    def enabled(self) -> bool:
        return self.predictions is not None or self.drift is not None


_CACHE: Collectors | None = None


def _existing(registry: Any, name: str) -> Any:
    """The already-registered collector for ``name``, if there is one.

    ``_names_to_collectors`` is private API, but it is the only way to ask the
    default registry a question it otherwise answers by raising. The alternative —
    catching ``ValueError`` from a duplicate registration — cannot distinguish "we
    already made this" from "somebody else registered this name", and would hand
    back a collector we never validated.
    """
    mapping = getattr(registry, "_names_to_collectors", {})
    return mapping.get(name)


def get_collectors() -> Collectors:
    """The process-wide collectors, creating them on first call.

    Safe to call from every ``create_app``: the second call returns the same
    handles instead of raising a duplicate-timeseries error.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        from prometheus_client import REGISTRY, Counter, Gauge
    except ImportError:  # prometheus_client is part of the [serve]/[mlops] extras
        _CACHE = Collectors()
        return _CACHE

    predictions = _existing(REGISTRY, PREDICTIONS_TOTAL) or Counter(
        PREDICTIONS_TOTAL,
        "Model predictions, labelled by predicted class",
        ["predicted_class"],
    )
    drift = _existing(REGISTRY, FEATURE_PSI) or Gauge(
        FEATURE_PSI,
        "Input feature drift (PSI) vs the training reference distribution",
        ["feature"],
    )
    _CACHE = Collectors(predictions=predictions, drift=drift)
    return _CACHE


def instrument(app: Any) -> bool:
    """Attach the ASGI latency/throughput instrumentation and expose ``/metrics``.

    Returns whether it happened, so the caller can log the reason rather than
    silently serving an app with no ``/metrics`` route.
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:
        log.info("prometheus_fastapi_instrumentator not installed; /metrics disabled")
        return False
    Instrumentator().instrument(app).expose(app, include_in_schema=False)
    return True


def reset_for_tests() -> None:
    """Drop the cache so a test can re-exercise the creation path."""
    global _CACHE
    _CACHE = None


__all__ = [
    "FEATURE_PSI",
    "PREDICTIONS_TOTAL",
    "Collectors",
    "get_collectors",
    "instrument",
    "reset_for_tests",
]
