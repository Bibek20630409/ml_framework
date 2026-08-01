"""
backends/forecast.py
────────────────────
The third fit-loop shape: **fit-per-series, no X/y, predict-by-horizon.**

This is the backend that most justifies the split. Prophet and ARIMA do not
consume a feature matrix and do not take a batch of rows — they are handed a
series and asked what comes next. ``predict(X)`` is a lying signature for that,
which is why ``Capabilities.accepts`` distinguishes a ``series`` payload and why
the estimator's input is a horizon rather than a matrix.

What "fit-per-series" buys, even with one series: the estimator holds a mapping
of ``{series_id: model}``, so multi-series support becomes a *source* change
(emit more than one key) rather than a backend rewrite. Today the source emits a
single series, so the mapping has one entry.

Heavy imports live inside the plugins' ``build()``, so importing this module —
or listing the backend with ``mlf backends`` — never imports prophet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel as PydanticModel
from pydantic import Field

from ..core.export import ExportResult, unsupported
from ..core.protocols import (
    ArtifactRef,
    BuildContext,
    FitResult,
    Int,
    Predictions,
    RunContext,
    TrialHooks,
)
from ..core.task import get_task_spec
from ..core.types import Capabilities, FrameworkError, UnsupportedCapability
from .base import BaseBackend, clean_metrics

log = logging.getLogger(__name__)

MODEL_FILE = "forecast.pkl"
ARTIFACT_FORMAT = "forecast-pickle"
# The key a single-series bundle is stored under.
DEFAULT_SERIES = "__series__"


class ForecastBackendError(FrameworkError):
    """A forecaster could not be fitted, saved or restored."""


class ForecastFitParams(PydanticModel):
    """Schema for ``fit.params`` on this backend.

    Short, and that is the point: there is no learning rate, no batch size and no
    epoch count, because there is no iterative loop. What the loop *does* have is a
    horizon and a seasonality, and both are properties of the data rather than of
    the fit — so they come from the bundle, not from here.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    # Retained across the whole search space so a trial can vary it; the source's
    # value is the default.
    seasonality: int | None = Field(default=None, ge=1)


# ── Estimator ─────────────────────────────────────────────
class ForecastEstimator:
    """Predict-only wrapper around one fitted model per series.

    ``predict`` takes a **horizon**, not a feature matrix. Accepting an int, or
    anything sized, keeps it usable from both the evaluation path (which knows how
    many points it wants) and the serving path (which is handed
    ``ForecastRequest.horizon``).
    """

    def __init__(self, models: dict[str, Any], task: str = "forecasting") -> None:
        self.models = models
        self.task = task
        self.task_spec = get_task_spec(task)

    @staticmethod
    def _horizon(inputs: Any) -> int:
        """How many steps were asked for, from whatever the caller passed."""
        if isinstance(inputs, (int, np.integer)):
            return int(inputs)
        if hasattr(inputs, "horizon"):
            return int(inputs.horizon)
        try:
            return int(len(inputs))
        except TypeError as exc:
            raise ForecastBackendError(
                f"a forecast needs a horizon; got {type(inputs).__name__}"
            ) from exc

    def predict(self, inputs: Any, *, series: str = DEFAULT_SERIES) -> np.ndarray:
        model = self.models.get(series)
        if model is None:
            raise ForecastBackendError(
                f"no fitted model for series '{series}'. Fitted: {sorted(self.models)}"
            )
        return np.asarray(model.forecast(self._horizon(inputs)), dtype="float64").reshape(-1)

    def predict_proba(self, inputs: Any) -> np.ndarray:
        raise UnsupportedCapability(
            "a forecast produces values over a horizon, not class probabilities"
        )

    def interval(self, inputs: Any, *, series: str = DEFAULT_SERIES) -> tuple[Any, Any]:
        """``(lower, upper)`` prediction bounds, or ``(None, None)``.

        Optional by design: the seasonal-naive baseline has no uncertainty model,
        and inventing one would be worse than admitting it has none.
        """
        model = self.models.get(series)
        bounds = getattr(model, "interval", None)
        if bounds is None:
            return None, None
        try:
            return bounds(self._horizon(inputs))
        except Exception as exc:  # noqa: BLE001 - intervals are informational
            log.debug("interval unavailable: %s", exc)
            return None, None


# ── Backend ───────────────────────────────────────────────
class ForecastBackend(BaseBackend):
    """Fit one model per series; predict by horizon."""

    name: ClassVar[str] = "forecast"
    capabilities: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset({"series"}),
        # A forecaster consumes the series on its own scale; standardizing it would
        # have to be undone before anything could be read.
        needs_scaling=False,
        native_missing=False,
        supports_sample_weight=False,
        # Values over a horizon, never class probabilities.
        produces_proba=False,
        # No intermediate epochs to prune on: a Prophet fit is one call.
        supports_pruning=False,
        supports_gpu=False,
        supports_mixed_precision=False,
        supports_lr_range_test=False,
        supports_resume=False,
    )

    # ── fit ──
    def fit(self, spec: Any, bundle: Any, cfg: Any, *, run: RunContext) -> FitResult:
        params = ForecastFitParams.model_validate(dict(cfg.fit.params))
        seasonality = params.seasonality or int(bundle.meta.get("seasonality", 1))

        models: dict[str, Any] = {}
        for series_id, values, index in self._series(bundle):
            model = spec.build(
                BuildContext(
                    task=bundle.task,
                    input_dim=bundle.input_dim,
                    output_dim=1,
                    feature_schema=bundle.schema,
                    params=cfg.model.params,
                    optim={"seasonality": seasonality},
                    seed=run.seed,
                )
            )
            log.info("fitting %s on series '%s' (%d points)", spec.name, series_id, len(values))
            model.fit(values, index=index)
            models[series_id] = model

        estimator = ForecastEstimator(models, bundle.task)
        val_metrics = self._val_metrics(estimator, bundle)
        log.info("val metrics: %s", val_metrics)
        return FitResult(estimator=estimator, val_metrics=val_metrics)

    @staticmethod
    def _series(bundle: Any):
        """``(series_id, values, index)`` per series in the training split.

        One entry today. Written as an iteration so that a source emitting several
        series needs no backend change — which is what "fit-per-series" means as a
        loop *shape* rather than as a count.
        """
        train = bundle.train
        if train.y is None:
            raise ForecastBackendError("the training split carries no series values")
        yield DEFAULT_SERIES, np.asarray(train.y, dtype="float64"), train.index

    def _val_metrics(self, estimator: ForecastEstimator, bundle: Any) -> dict[str, float]:
        """The task's metrics over the validation window.

        Computed rather than logged from a loop, exactly as the GBDT backend does —
        which is what keeps ``FitResult.val_metrics`` a plain dict the tuning driver
        can read without knowing which backend produced it.
        """
        val = bundle.val
        if val is None or val.y is None:
            return {}
        actual = np.asarray(val.y, dtype="float64")
        predicted = estimator.predict(len(actual))
        spec = get_task_spec(bundle.task)
        return clean_metrics(spec.compute(actual, predicted, None, prefix="val_"))

    # ── persistence ──
    def save(self, est: Any, dest: str | Path) -> ArtifactRef:
        """Pickle. Deliberately, and with the reason recorded.

        Prophet and statsmodels expose no portable serialization of a *fitted*
        model — Prophet's own documented path is pickle. Converting to ONNX would
        be lossy where it worked at all, and the manifest's ``format`` field means
        a future native format is a new value rather than a breaking change.
        """
        import joblib

        target_dir = Path(dest)
        target_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(getattr(est, "models", est), target_dir / MODEL_FILE)
        log.info("saved forecaster → %s", target_dir / MODEL_FILE)
        return ArtifactRef(path=f"{target_dir.name}/{MODEL_FILE}", format=ARTIFACT_FORMAT)

    def load(self, bundle_dir: str | Path, manifest: Any) -> ForecastEstimator:
        import joblib

        path = Path(bundle_dir) / manifest.model.artifact
        if not path.exists():
            raise ForecastBackendError(f"missing model artifact {path}")
        return ForecastEstimator(joblib.load(path), manifest.task)

    # ── prediction ──
    def predict_split(self, est: Any, bundle: Any, split: str) -> Predictions:
        """Forecast as many steps as the split holds.

        A forecaster does not score arbitrary rows: it continues from where it was
        fitted. So the "prediction" for a split is the next ``len(split)`` steps,
        aligned to that split's timestamps — which is why ``Predictions.index``
        exists and why a forecasting report is meaningless without it.
        """
        target = bundle.split(split)
        if target.y is None:
            raise ForecastBackendError(f"split '{split}' carries no values to score against")
        actual = np.asarray(target.y, dtype="float64")
        return Predictions(
            y_true=actual,
            y_pred=est.predict(len(actual)),
            y_prob=None,
            index=target.index,
        )

    # ── HPO ──
    def search_space(self) -> dict[str, Any]:
        """Seasonality is the one loop-level knob worth searching.

        Everything else that matters — trend flexibility, ARIMA order — is model
        shape and belongs to the plugin, exactly as tree depth does for GBDT.
        """
        return {"fit.params.seasonality": Int(1, 12)}

    def trial_hooks(self, trial: Any) -> TrialHooks:
        """No pruning: a Prophet fit is one call with no intermediate value.

        Returning empty hooks is the honest answer. The tuning driver falls back to
        a non-pruning search rather than believing it installed something.
        """
        return TrialHooks.empty()

    def params_model(self) -> type[PydanticModel]:
        return ForecastFitParams

    # Pickle only. The reason is on `save()` above and has not changed: Prophet
    # and statsmodels expose no portable serialization of a *fitted* model, so
    # there is nothing to convert to. Listing `onnx` here and failing at runtime
    # would be worse than refusing at the table.
    export_formats: ClassVar[tuple[str, ...]] = ("pickle",)

    def export(self, est: Any, dest: Path, fmt: str, *, manifest: Any = None) -> ExportResult:
        if fmt not in self.export_formats:
            unsupported(
                self.name,
                fmt,
                self.export_formats,
                "a fitted Prophet/statsmodels model has no portable graph form",
            )
        import joblib

        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(est, dest)
        log.info("exported pickle -> %s", dest)
        return ExportResult(
            path=dest,
            format="pickle",
            notes="joblib pickle; loads only with the same library versions",
        )

    def model_size(self, est: Any) -> dict[str, Any]:
        """Series count — the analogue of a parameter or tree count here."""
        models = getattr(est, "models", {})
        return {"series": len(models)} if models else {}


def build_backend() -> ForecastBackend:
    """Factory referenced by the registry's ``BackendSpec``."""
    return ForecastBackend()


__all__ = [
    "DEFAULT_SERIES",
    "ForecastBackend",
    "ForecastBackendError",
    "ForecastEstimator",
    "ForecastFitParams",
    "build_backend",
]
