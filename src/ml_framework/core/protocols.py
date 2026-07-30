"""
core/protocols.py
─────────────────
**The crux of the architecture.** The contracts that let one orchestrator drive a
Lightning loop, an ``xgboost.fit`` call and a per-series Prophet fit without
knowing which it has.

Two rules decide the shape of everything here:

1. **The backend owns the fit loop; the model stays dumb.** ~15 models map onto 3
   fit-loop shapes (iterative/mini-batch, one-shot ``fit(X, y)``, fit-per-series).
   Putting ``fit()`` on the model means 15 duplicates, or a base class that owns
   the loop — which is a backend expressed through inheritance, where you cannot
   swap it, cannot test it in isolation, and an HF transformer cannot reuse the
   Lightning loop without subclassing ``BaseModel``. A model knows its
   architecture and its forward pass; it does not know about output dirs,
   checkpoint policy or trackers.

2. **:class:`Estimator` is predict-only**, because it is the only thing that
   crosses into the serving process. ``save``/``load`` live on the backend:
   ``load`` needs registry access to rebuild an architecture before loading
   weights, and making every estimator a registry client would drag the training
   dependencies into the serving image.

Nothing here imports torch, Lightning or any optional dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from .types import Capabilities, DataKind, Payload, Task

if TYPE_CHECKING:  # deferred: keeps `protocols` importable with no tracking deps
    from pydantic import BaseModel as PydanticModel

    from ..tracking.run_logger import RunLogger


# ── Data carried between the layers ───────────────────────
@dataclass(frozen=True, slots=True)
class Predictions:
    """A backend's output for one split, as arrays.

    ``index`` is not decoration: forecasting reports are meaningless without the
    timestamps the values belong to, and tabular ``predictions.csv`` benefits from
    keeping the original row order recoverable.
    """

    y_true: np.ndarray | None
    y_pred: np.ndarray
    y_prob: np.ndarray | None = None
    index: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(np.asarray(self.y_pred).shape[0])


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Where a model lives inside a bundle, and in what format.

    ``format`` is tracked separately from the file extension so serialization can
    migrate (xgboost json → ubj) without breaking existing readers. ``path`` is
    always relative to the bundle root and may name a *directory* (an HF model
    dir) — nothing outside the owning backend interprets it.
    """

    path: str
    format: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "format": self.format}


@dataclass(frozen=True, slots=True)
class FitResult:
    """What a backend hands back to the orchestrator.

    ``val_metrics`` is a plain ``dict[str, float]`` — *not* a Lightning
    ``callback_metrics`` object. That is what lets the HPO driver read an
    objective value from any backend instead of hardcoding ``val/loss``.
    """

    estimator: Estimator
    val_metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)
    # Extra files the backend wants in the bundle: {relative_path: source_path}.
    extra_files: Mapping[str, Path] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Budget:
    """The resolved training budget. ``None`` means "no limit from this axis"."""

    max_epochs: int | None = None
    max_seconds: float | None = None
    patience: int | None = None


@dataclass(frozen=True, slots=True)
class RunContext:
    """Everything a backend needs about *this run* that is not data or params.

    Passing this instead of the whole ``ExperimentConfig`` is what keeps backends
    from reading unrelated config blocks — the coupling the v1 mutable-global
    config was rewritten to remove.
    """

    output_dir: Path
    seed: int = 42
    budget: Budget = field(default_factory=Budget)
    run_logger: RunLogger | None = None
    accelerator: str = "auto"
    devices: str | int = "auto"
    precision: str = "32"
    deterministic: bool = True
    # Set during HPO so a backend can shorten its own loop / install pruning.
    trial: Any | None = None


@dataclass(frozen=True, slots=True)
class BuildContext:
    """The single argument to ``ModelSpec.build``.

    One frozen dataclass rather than a keyword explosion, so adding a field later
    does not touch every plugin.
    """

    task: Task
    input_dim: int = 0
    output_dim: int = 0
    n_classes: int | None = None
    # `data.types.FeatureSchema` once the data layer lands in P1; typed loosely
    # here only because this module must not import the data package.
    feature_schema: Any | None = None
    # numpy, never torch — the agnostic layer must not import torch; the Lightning
    # backend converts.
    class_weights: np.ndarray | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    optim: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 42
    device: str = "cpu"


# ── Search-space vocabulary ───────────────────────────────
# Declarative because it must be serializable (into hpo.json), narrowable from
# YAML (`tune.overrides`) and printable (`mlf models --show`). Keys in a search
# space are DOTTED CONFIG PATHS, so applying a trial is exactly
# `config.with_overrides(values)` — a mechanism that already exists and is tested.
@dataclass(frozen=True, slots=True)
class Float:
    low: float
    high: float
    log: bool = False
    step: float | None = None

    def suggest(self, trial: Any, name: str) -> float:
        return trial.suggest_float(name, self.low, self.high, log=self.log, step=self.step)


@dataclass(frozen=True, slots=True)
class Int:
    low: int
    high: int
    log: bool = False
    step: int = 1

    def suggest(self, trial: Any, name: str) -> int:
        return trial.suggest_int(name, self.low, self.high, log=self.log, step=self.step)


@dataclass(frozen=True, slots=True)
class Categorical:
    choices: tuple[Any, ...]

    def suggest(self, trial: Any, name: str) -> Any:
        return trial.suggest_categorical(name, list(self.choices))


@dataclass(frozen=True, slots=True)
class Const:
    value: Any

    def suggest(self, trial: Any, name: str) -> Any:  # noqa: ARG002 - protocol shape
        return self.value


ParamSpec = Float | Int | Categorical | Const
SearchSpace = Mapping[str, ParamSpec]


@dataclass(frozen=True, slots=True)
class TrialHooks:
    """Per-backend HPO plumbing, so the driver never imports ``optuna_integration``.

    ``callbacks`` are handed to whatever loop the backend runs (Lightning
    callbacks, xgboost callbacks, …). ``report`` lets a backend push an
    intermediate value itself when its library has no callback hook.
    """

    callbacks: tuple[Any, ...] = ()
    report: Callable[[float, int], None] | None = None

    @classmethod
    def empty(cls) -> TrialHooks:
        return cls()

    def __bool__(self) -> bool:
        return bool(self.callbacks) or self.report is not None


# ── Protocols ─────────────────────────────────────────────
@runtime_checkable
class Estimator(Protocol):
    """Predict-only. The only object that crosses into the serving process.

    Implementations apply the task's canonical head postprocessing exactly once
    (see ``TaskSpec.postprocess``), which is what collapses the sigmoid/softmax/
    identity branching currently duplicated across the step function, the
    evaluator and the inferencer.
    """

    def predict(self, inputs: Any) -> np.ndarray:
        """Hard predictions: labels for classification, values for regression.

        ``inputs`` is parameterized by ``data.kind`` — an (n, d) array/frame for
        tabular, a tensor/PIL batch for image, ``list[str]`` for text, a
        ``ForecastRequest`` for timeseries. ``predict(X)`` would be a lying
        signature for forecasting, so the payload varies by kind rather than
        pretending otherwise.
        """
        ...

    def predict_proba(self, inputs: Any) -> np.ndarray:
        """Class probabilities.

        Raises :class:`~ml_framework.core.types.UnsupportedCapability` when the
        model does not produce them (``Capabilities.produces_proba is False``).
        """
        ...


@runtime_checkable
class TrainingBackend(Protocol):
    """One per fit-loop **shape**, not per library.

    ``lightning`` (MLP/CNN/LSTM/TFT/HF transformer), ``gbdt``
    (XGBoost/LightGBM/CatBoost/sklearn) and ``forecast``
    (Prophet/statsmodels/naive) cover ~15 models. A backend absorbs the small
    per-library differences internally, which is why adding CatBoost is ~40 lines
    rather than a new backend.
    """

    name: ClassVar[str]
    capabilities: ClassVar[Capabilities]

    def fit(self, spec: Any, bundle: Any, cfg: Any, *, run: RunContext) -> FitResult: ...

    def save(self, est: Estimator, dest: Path) -> ArtifactRef:
        """Serialize ``est`` under ``dest`` and return where/what it wrote."""
        ...

    def load(self, bundle_dir: Path, manifest: Any) -> Estimator:
        """Rebuild an estimator from a bundle. The counterpart of :meth:`save`."""
        ...

    def predict_split(self, est: Estimator, bundle: Any, split: str) -> Predictions: ...

    def search_space(self) -> SearchSpace:
        """Backend-level knobs (lr, batch_size) declared once here rather than
        repeated in every plugin that rides this loop."""
        ...

    def trial_hooks(self, trial: Any) -> TrialHooks: ...

    def params_model(self) -> type[PydanticModel]:
        """Pydantic schema validating ``fit.params`` for this backend."""
        ...


@runtime_checkable
class Preprocessor(Protocol):
    """Owns **all** fitted transform state.

    Everything lands in ``bundle/preprocessor/`` with a ``preprocessor.json``
    naming the dotted class path and its files. **Nothing outside the
    preprocessor reads that directory** — that invariant is what removes the
    hardcoded ``scaler.pkl`` special cases from the inferencer and the MLflow
    logger.
    """

    def fit(self, split: Any, schema: Any) -> None:
        """Fit on the **training split only**. Fitting on val/test is leakage."""
        ...

    def transform(self, x: Any) -> Any: ...

    def save(self, dest: Path) -> Mapping[str, Any]:
        """Write state under ``dest``; return the manifest fragment describing it."""
        ...

    @classmethod
    def load(cls, src: Path, spec: Mapping[str, Any]) -> Preprocessor: ...

    @property
    def collate_fn(self) -> Callable[[Sequence[Any]], Any] | None:
        """Optional batch collation (text padding). ``None`` for the default."""
        ...


@runtime_checkable
class Splitter(Protocol):
    """Produces train/val/test index sets.

    Splitting is a protocol rather than a function because the *correct* split is
    a property of the data, not of the model: using a shuffled split on a
    time-series silently leaks the future into training, which is the most
    damaging silent failure in this domain.
    """

    name: ClassVar[str]

    def split(self, n: int, *, y: np.ndarray | None = None, **kwargs: Any) -> Any: ...


# ── Payload compatibility ─────────────────────────────────
# The canonical payload for each data kind, used to check a model/backend's
# `Capabilities.accepts` at build time.
DEFAULT_PAYLOAD: Mapping[DataKind, Payload] = {
    "tabular": "arrays",
    "image": "dataset",
    "text": "dataset",
    "timeseries": "series",
}
