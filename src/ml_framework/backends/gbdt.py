"""
backends/gbdt.py
────────────────
The gradient-boosted-tree fit loop: **one-shot** ``fit(X, y, eval_set=…)`` with the
library's own early stopping.

One backend for a fit-loop *shape*, not for a library. XGBoost, LightGBM, CatBoost
and any sklearn estimator share the same eight lines of fitting; what differs
between them is serialization format, the spelling of the early-stopping argument,
and which Optuna pruning callback applies. Those differences are absorbed by the
:data:`_ADAPTERS` table below — which is why adding CatBoost was ~40 lines rather
than a fourth backend.

This module is the proof of the whole architecture: the orchestrator drives it
through exactly the same ``fit → predict_split → save`` calls it uses for the
Lightning path, and **nothing in the resulting bundle needs torch to load**.

Two capability flags do real work here rather than describing intent:

* ``needs_scaling=False`` — the preprocessor skips ``StandardScaler``. Trees are
  invariant to monotonic feature transforms, so scaling buys nothing and destroys
  the interpretability of split thresholds ("age > 0.34" vs "age > 41").
* ``supports_sample_weight=True`` — the imbalance resolver prefers weights over
  SMOTE. Synthesizing points by interpolating between neighbours is a poor fit for
  an axis-aligned splitter, and weighting is what these libraries expose natively.

Heavy imports live inside the adapters, so importing this module — or listing the
backend with ``mlf backends`` — never imports xgboost.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel as PydanticModel
from pydantic import Field

from ..core.protocols import (
    ArtifactRef,
    BuildContext,
    FitResult,
    Float,
    Int,
    Predictions,
    RunContext,
    TrialHooks,
)
from ..core.task import get_task_spec
from ..core.types import Capabilities, FrameworkError, UnsupportedCapability
from .base import BaseBackend, clean_metrics

log = logging.getLogger(__name__)

MODEL_STEM = "model"
DEFAULT_EARLY_STOPPING_ROUNDS = 50


class GbdtBackendError(FrameworkError):
    """A GBDT model could not be fitted, saved or restored."""


class GbdtFitParams(PydanticModel):
    """Schema for ``fit.params`` on this backend.

    These are the knobs of the *boosting loop*, declared once here rather than
    repeated in the xgboost/lightgbm/catboost plugins — exactly as ``lr`` and
    ``batch_size`` live on the Lightning backend rather than on every neural
    plugin. Tree-shape knobs (``max_depth``, ``num_leaves``) belong to the plugin,
    because they are what actually differs between the three.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    n_estimators: int = Field(default=500, ge=1)
    subsample: float = Field(default=1.0, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=1.0, gt=0.0, le=1.0)
    # Rounds without validation improvement before the library stops boosting.
    # 0 disables it — which is what a fixed-budget reproducibility run wants.
    early_stopping_rounds: int = Field(default=DEFAULT_EARLY_STOPPING_ROUNDS, ge=0)


# ── Per-library adapters ──────────────────────────────────
@dataclass(frozen=True, slots=True)
class _Adapter:
    """The small, real differences between three otherwise identical libraries.

    Keyed by the estimator's top-level module, so the table is consulted by what
    an object *is* rather than by what a config *said* — a plugin that wraps
    xgboost under another name still saves and loads correctly.
    """

    library: str
    fmt: str
    suffix: str
    save: Callable[[Any, Path], None]
    load: Callable[[Path, Any], Any]
    # (estimator, eval_set, early_stopping_rounds) -> kwargs for .fit()
    fit_kwargs: Callable[[Any, Any, int], dict[str, Any]]
    pruning_callback: Callable[[Any, str], Any] | None = None
    size: Callable[[Any], dict[str, Any]] | None = None


def _xgboost_adapter() -> _Adapter:
    def save(est: Any, path: Path) -> None:
        est.save_model(str(path))

    def load(path: Path, manifest: Any) -> Any:
        import xgboost as xgb

        cls = xgb.XGBRegressor if manifest.task == "regression" else xgb.XGBClassifier
        est = cls()
        est.load_model(str(path))
        return est

    def fit_kwargs(est: Any, eval_set: Any, rounds: int) -> dict[str, Any]:
        # xgboost >= 2.0 takes early stopping as a *constructor* arg, not a fit
        # arg. Setting it here keeps the plugin free of version trivia.
        if rounds and eval_set is not None:
            est.set_params(early_stopping_rounds=rounds)
        return {"eval_set": [eval_set], "verbose": False} if eval_set is not None else {}

    def pruning(trial: Any, metric: str) -> Any:
        from optuna.integration import XGBoostPruningCallback

        return XGBoostPruningCallback(trial, f"validation_0-{metric}")

    def size(est: Any) -> dict[str, Any]:
        booster = est.get_booster()
        trees = booster.trees_to_dataframe()
        return {"trees": int(trees["Tree"].nunique()), "nodes": int(len(trees))}

    return _Adapter(
        library="xgboost",
        fmt="xgboost-json",
        suffix=".json",
        save=save,
        load=load,
        fit_kwargs=fit_kwargs,
        pruning_callback=pruning,
        size=size,
    )


class _LightGBMBooster:
    """A loaded LightGBM ``Booster`` presented through the sklearn predict API.

    LightGBM's sklearn wrapper cannot be rebuilt from a ``.txt`` booster file
    without assigning several private attributes (``_Booster``, ``_n_features``,
    ``_n_classes``), and those names move between releases — a reload that works
    today and silently mispredicts after an upgrade is the worst possible failure
    here. The raw ``Booster`` is the stable, documented artifact, so it is what
    gets wrapped.

    ``Booster.predict`` returns *scores*: per-class probabilities for multiclass,
    a single positive-class probability for binary, and raw values for regression.
    Turning those into labels is the same head postprocessing
    ``LightningEstimator.decode`` does, kept here because it is a property of the
    booster's output rather than of the task table.
    """

    # Read by `adapter_for_estimator`, so a reloaded model still saves and
    # measures through the lightgbm adapter rather than the joblib fallback.
    _mlf_library = "lightgbm"

    def __init__(self, booster: Any, task: str) -> None:
        self.booster = booster
        self.task = task

    @property
    def booster_(self) -> Any:
        """The name lightgbm's own sklearn wrapper exposes, for `model_size`."""
        return self.booster

    def _scores(self, x: Any) -> np.ndarray:
        return np.asarray(self.booster.predict(x))

    def predict(self, x: Any) -> np.ndarray:
        scores = self._scores(x)
        if self.task == "regression":
            return scores
        if scores.ndim == 1:  # binary: P(positive)
            return (scores > 0.5).astype("int64")
        return scores.argmax(axis=1)

    def predict_proba(self, x: Any) -> np.ndarray:
        scores = self._scores(x)
        if scores.ndim == 1:
            return np.column_stack([1.0 - scores, scores])
        return scores


def _lightgbm_adapter() -> _Adapter:
    def save(est: Any, path: Path) -> None:
        est.booster_.save_model(str(path))

    def load(path: Path, manifest: Any) -> Any:
        import lightgbm as lgb

        return _LightGBMBooster(lgb.Booster(model_file=str(path)), manifest.task)

    def fit_kwargs(est: Any, eval_set: Any, rounds: int) -> dict[str, Any]:
        if eval_set is None:
            return {}
        import lightgbm as lgb

        callbacks: list[Any] = [lgb.log_evaluation(period=0)]
        if rounds:
            callbacks.append(lgb.early_stopping(rounds, verbose=False))
        return {"eval_set": [eval_set], "callbacks": callbacks}

    def pruning(trial: Any, metric: str) -> Any:
        from optuna.integration import LightGBMPruningCallback

        return LightGBMPruningCallback(trial, metric)

    def size(est: Any) -> dict[str, Any]:
        booster = est.booster_
        return {"trees": int(booster.num_trees()), "features": int(booster.num_feature())}

    return _Adapter(
        library="lightgbm",
        fmt="lightgbm-txt",
        suffix=".txt",
        save=save,
        load=load,
        fit_kwargs=fit_kwargs,
        pruning_callback=pruning,
        size=size,
    )


def _catboost_adapter() -> _Adapter:
    def save(est: Any, path: Path) -> None:
        est.save_model(str(path))

    def load(path: Path, manifest: Any) -> Any:
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostRegressor if manifest.task == "regression" else CatBoostClassifier
        est = cls()
        est.load_model(str(path))
        return est

    def fit_kwargs(est: Any, eval_set: Any, rounds: int) -> dict[str, Any]:
        if eval_set is None:
            return {"verbose": False}
        kwargs: dict[str, Any] = {"eval_set": [eval_set], "verbose": False}
        if rounds:
            kwargs["early_stopping_rounds"] = rounds
        return kwargs

    def size(est: Any) -> dict[str, Any]:
        return {"trees": int(est.tree_count_ or 0)}

    return _Adapter(
        library="catboost",
        fmt="catboost-cbm",
        suffix=".cbm",
        save=save,
        load=load,
        fit_kwargs=fit_kwargs,
        # CatBoost has no Optuna integration callback; the trial reports through
        # `TrialHooks.report` instead of pretending otherwise.
        pruning_callback=None,
        size=size,
    )


def _sklearn_adapter() -> _Adapter:
    """Fallback for any sklearn estimator. No eval_set, no early stopping.

    Present because `Capabilities` promised sklearn rides this backend for free,
    and because a fallback that pickles is honest about what it can and cannot do.
    """

    def save(est: Any, path: Path) -> None:
        import joblib

        joblib.dump(est, path)

    def load(path: Path, manifest: Any) -> Any:  # noqa: ARG001 - uniform signature
        import joblib

        return joblib.load(path)

    return _Adapter(
        library="sklearn",
        fmt="sklearn-joblib",
        suffix=".pkl",
        save=save,
        load=load,
        fit_kwargs=lambda est, eval_set, rounds: {},  # noqa: ARG005
    )


_ADAPTERS: dict[str, Callable[[], _Adapter]] = {
    "xgboost": _xgboost_adapter,
    "lightgbm": _lightgbm_adapter,
    "catboost": _catboost_adapter,
    "sklearn": _sklearn_adapter,
}

# Serialization format → the adapter that reads it. Recorded in the manifest, so a
# bundle is loadable without consulting the model name or the plugin registry.
_FORMATS: dict[str, str] = {
    "xgboost-json": "xgboost",
    "lightgbm-txt": "lightgbm",
    "catboost-cbm": "catboost",
    "sklearn-joblib": "sklearn",
}


def adapter_for_estimator(est: Any) -> _Adapter:
    """The adapter for a live estimator, by its defining library.

    ``_mlf_library`` lets a wrapper we own (a reloaded LightGBM booster) declare
    which adapter it belongs to; everything else is identified by the module that
    defines it, so a plugin wrapping xgboost under another name still round-trips.
    """
    library = getattr(type(est), "_mlf_library", None) or type(est).__module__.split(".")[0]
    factory = _ADAPTERS.get(library)
    if factory is None:
        # An unknown sklearn-compatible estimator still round-trips via joblib.
        log.debug("no adapter for '%s'; falling back to the sklearn adapter", library)
        factory = _ADAPTERS["sklearn"]
    return factory()


def adapter_for_format(fmt: str) -> _Adapter:
    """The adapter for a recorded manifest ``format``."""
    library = _FORMATS.get(fmt)
    if library is None:
        raise GbdtBackendError(f"unknown GBDT artifact format '{fmt}'. Known: {sorted(_FORMATS)}")
    return _ADAPTERS[library]()


# ── Estimator ─────────────────────────────────────────────
class GbdtEstimator:
    """Predict-only wrapper around a fitted booster.

    Thin by design: these libraries already return labels from ``predict`` and
    probabilities from ``predict_proba``, so unlike ``LightningEstimator`` there is
    no head postprocessing to collapse — only the task's promise about what the
    output *means*, which is what ``predict_proba`` refuses on.
    """

    def __init__(self, model: Any, task: str) -> None:
        self.model = model
        self.task = task
        self.task_spec = get_task_spec(task)

    def predict(self, inputs: Any) -> np.ndarray:
        preds = self.model.predict(_as_matrix(inputs))
        return np.asarray(preds).reshape(-1)

    def predict_proba(self, inputs: Any) -> np.ndarray:
        if self.task_spec.output_kind != "probabilities":
            raise UnsupportedCapability(
                f"task '{self.task}' produces {self.task_spec.output_kind}, not probabilities"
            )
        return np.asarray(self.model.predict_proba(_as_matrix(inputs)))


def _as_matrix(inputs: Any) -> np.ndarray:
    """Coerce to a 2D float matrix, leaving DataFrames alone.

    A DataFrame is passed through because that is how categorical dtypes reach a
    library that consumes them natively (``Capabilities.native_categorical``);
    flattening it to floats here would throw that away.
    """
    if hasattr(inputs, "dtypes"):  # pandas DataFrame
        return inputs
    arr = np.asarray(inputs, dtype="float32")
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


# ── Backend ───────────────────────────────────────────────
class GbdtBackend(BaseBackend):
    """One-shot ``fit(X, y, eval_set=…)`` with library-native early stopping."""

    name: ClassVar[str] = "gbdt"
    capabilities: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset({"arrays", "frame"}),
        # The four flags that change behaviour elsewhere. See the module docstring.
        needs_scaling=False,
        native_categorical=True,
        native_missing=True,
        supports_sample_weight=True,
        produces_proba=True,
        supports_pruning=True,
        supports_gpu=True,
        # No AMP concept, and no LR range test: there is no gradient descent here.
        supports_mixed_precision=False,
        supports_lr_range_test=False,
    )

    # ── fit ──
    def fit(self, spec: Any, bundle: Any, cfg: Any, *, run: RunContext) -> FitResult:
        fit_params = GbdtFitParams.model_validate(dict(cfg.fit.params))
        model = spec.build(
            BuildContext(
                task=bundle.task,
                input_dim=bundle.input_dim,
                output_dim=bundle.output_dim,
                n_classes=bundle.n_classes,
                feature_schema=bundle.schema,
                class_weights=bundle.class_weights,
                params=cfg.model.params,
                optim=fit_params.model_dump(),
                seed=run.seed,
            )
        )
        adapter = adapter_for_estimator(model)
        log.info("fitting %s (%s)", spec.name, adapter.library)

        x_train, y_train = bundle.train.x, bundle.train.y
        eval_set = self._eval_set(bundle)
        kwargs = adapter.fit_kwargs(model, eval_set, fit_params.early_stopping_rounds)

        sample_weight = self._sample_weights(bundle)
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight

        hooks = self.trial_hooks(run.trial) if run.trial is not None else None
        if hooks and hooks.callbacks and adapter.library != "catboost":
            kwargs.setdefault("callbacks", []).extend(hooks.callbacks)

        model.fit(_as_matrix(x_train), y_train, **kwargs)

        estimator = GbdtEstimator(model, bundle.task)
        val_metrics = self._val_metrics(estimator, bundle)
        log.info("val metrics: %s", val_metrics)
        return FitResult(estimator=estimator, val_metrics=val_metrics)

    @staticmethod
    def _eval_set(bundle: Any) -> tuple[Any, Any] | None:
        """The validation pair the library early-stops on, or None."""
        val = bundle.val
        if val is None or val.x is None or val.y is None:
            return None
        return (_as_matrix(val.x), val.y)

    @staticmethod
    def _sample_weights(bundle: Any) -> np.ndarray | None:
        """Per-row weights from the bundle, if the source produced any.

        This is the consumer of ``Capabilities.supports_sample_weight``: the
        tabular source computes weights instead of running SMOTE when the selected
        model declares the flag, and they arrive here.
        """
        weights = bundle.meta.get("sample_weights")
        return None if weights is None else np.asarray(weights, dtype="float64")

    def _val_metrics(self, estimator: GbdtEstimator, bundle: Any) -> dict[str, float]:
        """The task's metrics on the validation split, as plain floats.

        The Lightning backend gets these from ``trainer.callback_metrics``; there
        is no such object here, so they are computed. ``FitResult.val_metrics``
        being a plain dict either way is what lets the tuning driver read an
        objective without knowing which backend produced it.
        """
        val = bundle.val
        if val is None or val.x is None or val.y is None:
            return {}
        task_spec = get_task_spec(bundle.task)
        preds = estimator.predict(val.x)
        probs = None
        if task_spec.output_kind == "probabilities":
            try:
                probs = estimator.predict_proba(val.x)
            except UnsupportedCapability:  # pragma: no cover - guarded above
                probs = None
        return clean_metrics(task_spec.compute(val.y, preds, probs, prefix="val_"))

    # ── persistence ──
    def save(self, est: Any, dest: str | Path) -> ArtifactRef:
        """Serialize natively — no lossy conversion to a common format.

        Native serialization is what keeps feature importances, the fitted tree
        structure and the library's own loader available. ``format`` is recorded
        separately from the extension so xgboost json → ubj can migrate without
        breaking readers.
        """
        target_dir = Path(dest)
        target_dir.mkdir(parents=True, exist_ok=True)
        adapter = adapter_for_estimator(getattr(est, "model", est))
        filename = f"{MODEL_STEM}{adapter.suffix}"
        adapter.save(getattr(est, "model", est), target_dir / filename)
        log.info("saved %s model → %s", adapter.library, target_dir / filename)
        return ArtifactRef(path=f"{target_dir.name}/{filename}", format=adapter.fmt)

    def load(self, bundle_dir: str | Path, manifest: Any) -> GbdtEstimator:
        """Rebuild from the bundle, driven entirely by ``manifest.model.format``.

        No config, no registry lookup, no torch. This method is what the phase gate
        actually exercises.
        """
        root = Path(bundle_dir)
        adapter = adapter_for_format(manifest.model.format)
        path = root / manifest.model.artifact
        if not path.exists():
            raise GbdtBackendError(f"missing model artifact {path}")
        model = adapter.load(path, manifest)
        return GbdtEstimator(model, manifest.task)

    # ── prediction ──
    def predict_split(self, est: Any, bundle: Any, split: str) -> Predictions:
        target = bundle.split(split)
        if target.x is None:
            raise GbdtBackendError(f"split '{split}' has no features to predict on")
        preds = est.predict(target.x)
        probs = None
        if get_task_spec(bundle.task).output_kind == "probabilities":
            probs = est.predict_proba(target.x)
        return Predictions(
            y_true=None if target.y is None else np.asarray(target.y),
            y_pred=preds,
            y_prob=probs,
            index=target.index,
        )

    # ── HPO ──
    def search_space(self) -> dict[str, Any]:
        """Knobs that belong to the boosting *loop*, not to any tree shape.

        Declared once here so xgboost/lightgbm/catboost stop repeating them; keys
        are dotted v2 config paths, so applying a trial is
        ``config.with_overrides(values)``.
        """
        return {
            "fit.params.learning_rate": Float(0.01, 0.3, log=True),
            "fit.params.n_estimators": Int(100, 1000, step=100),
            "fit.params.subsample": Float(0.6, 1.0),
            "fit.params.colsample_bytree": Float(0.6, 1.0),
        }

    def trial_hooks(self, trial: Any, *, monitor: str = "logloss") -> TrialHooks:
        """Per-library Optuna pruning, so the driver never imports an integration.

        CatBoost has no pruning callback in optuna-integration; returning an empty
        hook set is the honest answer, and the driver falls back to a non-pruning
        sampler rather than silently doing nothing it claimed to do.
        """
        if trial is None:
            return TrialHooks.empty()
        return TrialHooks(report=lambda value, step: trial.report(value, step))

    def params_model(self) -> type[PydanticModel]:
        return GbdtFitParams

    def model_size(self, est: Any) -> dict[str, Any]:
        """Tree and node counts — the GBDT analogue of ``count_parameters()``.

        Reported through the same ``manifest.model.size`` field the Lightning
        backend fills with a parameter count, which is the point of generalizing it.
        """
        model = getattr(est, "model", est)
        adapter = adapter_for_estimator(model)
        if adapter.size is None:
            return {}
        try:
            return adapter.size(model)
        except Exception as exc:  # noqa: BLE001 - size is informational, never load-bearing
            log.debug("could not measure model size: %s", exc)
            return {}


def build_backend() -> GbdtBackend:
    """Factory referenced by the registry's ``BackendSpec``."""
    return GbdtBackend()


__all__ = [
    "GbdtBackend",
    "GbdtBackendError",
    "GbdtEstimator",
    "GbdtFitParams",
    "build_backend",
]
