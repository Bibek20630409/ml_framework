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

    # The two backends spell validation metrics differently, and neither is wrong:
    # Lightning's logger convention is `val/acc` (the slash groups them in
    # TensorBoard), while metrics computed from arrays come back as `val_acc`.
    # Normalizing at the source would rename keys the trackers already publish, so
    # the *lookup* absorbs the difference instead — in one place, here, rather than
    # in every consumer.
    _METRIC_PREFIXES: ClassVar[tuple[str, ...]] = ("", "val_", "val/")

    def metric(self, name: str) -> float | None:
        """This result's value for ``name``, whichever convention produced it.

        ``metric("acc")`` finds ``acc``, ``val_acc`` or ``val/acc``. Returns
        ``None`` when the backend did not report it, so a caller can say which
        metric is missing rather than raising a bare ``KeyError``.
        """
        for prefix in self._METRIC_PREFIXES:
            key = f"{prefix}{name}"
            if key in self.val_metrics:
                return float(self.val_metrics[key])
        return None


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
    strategy: str = "auto"
    deterministic: bool = True
    # Set during HPO so a backend can shorten its own loop / install pruning.
    trial: Any | None = None
    # A checkpoint to continue from. Resolved and existence-checked by the
    # orchestrator, so a backend that supports resuming can use it directly and one
    # that does not is never handed a path it would have to explain away.
    resume_from: Path | None = None


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

    def export(self, est: Estimator, dest: Path, fmt: str, *, manifest: Any = None) -> Any:
        """Convert ``est`` to ``fmt`` at ``dest``, or refuse.

        On the backend because only the backend knows what its estimator
        physically is. A central exporter would need a branch per backend, which
        is the coupling the plugin design exists to remove. Refusing is a normal
        outcome — a Prophet model has no ONNX graph — and it must raise rather
        than write something that is not the requested format.
        """
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


# An opaque handle to a table owned by a `DataBackend`: a `pd.DataFrame` under
# `local`, a `pyspark.sql.DataFrame` under `spark`. `Any` rather than a union, for
# the same reason `Split.x` is `Any` — naming the alternatives would make this
# module import pandas and pyspark.
Table = Any


@runtime_checkable
class DataBackend(Protocol):
    """The engine that reads and reduces a table, chosen per run by ``data.backend``.

    A data backend decides *how the bytes become a matrix*, not how the model is
    trained: ``local`` reads with pandas in-process, ``spark`` reads and reduces
    across a cluster and then collects. Training is single-node either way —
    :class:`~ml_framework.data.types.DataBundle` holds numpy arrays and the
    splitters index into them.

    Every method below replaces one pandas idiom that exists in the data layer
    today; there are none here without a call site, per the rule on
    :class:`~ml_framework.core.types.Capabilities`.

    **Exactly two methods collect: :meth:`column` and :meth:`to_pandas`.** They are
    named so they are greppable, and the split between them is a correctness rule,
    not a convenience:

    *Arrays that must line up row-for-row have to come out of a single collect.*

    Under a distributed engine each collect re-executes the query plan, and two
    executions need not agree on row order — pyspark documents
    ``monotonically_increasing_id`` (which :meth:`sort_by` relies on) as
    non-deterministic for exactly this reason. So a caller that needs features
    *and* labels, or a label-end *and* its observation time, calls
    :meth:`to_pandas` **once** and slices the result. :meth:`column` is for the
    genuinely standalone array, where there is no second array to fall out of step
    with.

    :meth:`select` exists to make that affordable: it narrows the table *without*
    collecting, so the one materialization carries only the columns needed.
    """

    name: ClassVar[str]
    engine: ClassVar[str]
    """The library doing the work. Printed by ``mlf data-backends``."""

    # ── read ──
    def read_table(self, path: str) -> Table:
        """A CSV file, a Parquet file, or a directory of Parquet part-files."""
        ...

    # ── inspect: nothing leaves the cluster ──
    def columns(self, table: Table) -> tuple[str, ...]: ...

    def n_rows(self, table: Table) -> int: ...

    def dtypes(self, table: Table, columns: Sequence[str]) -> Mapping[str, str]:
        """Column dtypes as **numpy-style** names, identically on every backend.

        Normalized rather than passed through, because these reach
        ``FeatureSchema.dtypes`` and from there the bundle manifest and the
        serving signature: Spark's native ``bigint``/``double`` would make the
        artifact depend on which engine happened to build it.
        """
        ...

    # ── reduce: still lazy ──
    def sort_by(self, table: Table, column: str) -> Table:
        """**Stable** ascending sort, with ties broken deterministically.

        Not a detail. ``builders._label_columns`` sorts so the positions a
        splitter computes line up with the rows it splits; an engine whose tie
        order varies between runs would silently change the folds.
        """
        ...

    def select(self, table: Table, columns: Sequence[str]) -> Table:
        """Narrow the table to ``columns``. **Lazy — this does not collect.**

        Under ``spark`` the projection pushes down into Parquet, so the single
        :meth:`to_pandas` that follows carries only what the caller needs.
        """
        ...

    # ── collect: the only two methods that move data to the driver ──
    def column(self, table: Table, name: str, *, dtype: str | None = None) -> np.ndarray:
        """One standalone column as a numpy array.

        Only for arrays with no row-alignment partner — the CV label vector, whose
        row count is checked separately. Two calls to this method are **two
        collects** and may disagree on row order; when two arrays must line up, go
        through :meth:`to_pandas` once instead.
        """
        ...

    def to_pandas(self, table: Table) -> Any:
        """The table on the driver, as pandas, in **one** materialization.

        The atomic collect: everything sliced out of the returned frame is
        guaranteed to be row-aligned. Also the escape hatch for code that is
        genuinely pandas-shaped (pandera contracts, the timeseries exogenous
        frame), and the one call that can turn a distributed run into a driver
        OOM. Identity under ``local``.
        """
        ...

    # ── clean + write ──
    # These five exist for one consumer, `pipeline.spark_preprocess.preprocess`.
    # They were deliberately withheld until that consumer was expressed in terms
    # of this protocol, rather than added speculatively alongside the read path.
    def filter_notnull(self, table: Table, column: str) -> Table:
        """Drop rows whose ``column`` is null. Lazy."""
        ...

    def drop_all_null_rows(self, table: Table, columns: Sequence[str]) -> Table:
        """Drop rows where **every** one of ``columns`` is null. Lazy.

        Deliberately "all" and not "any": a single missing feature is ordinary,
        while a row that is null across every feature carries no signal at all.
        """
        ...

    def drop_duplicates(self, table: Table) -> Table:
        """Drop exact duplicate rows. Lazy."""
        ...

    def cast(self, table: Table, column: str, dtype: str) -> Table:
        """Cast ``column``, naming ``dtype`` in the **numpy** spelling.

        Numpy-style for the same reason :meth:`dtypes` reports that way: the
        caller should not have to know whether it is talking to ``float64`` or
        ``double``.
        """
        ...

    def write_parquet(self, table: Table, path: str) -> None:
        """Write to ``path`` as a **directory** of Parquet part-files, overwriting.

        A directory on every backend, including ``local``, because
        :meth:`read_table` distinguishes a Parquet directory from a CSV by
        inspecting the path — a bare file with no suffix would come back as a CSV
        read. Writing the same shape everywhere is what lets a stage run under one
        engine and be consumed by a run under the other.
        """
        ...


# ── Payload compatibility ─────────────────────────────────
# The canonical payload for each data kind, used to check a model/backend's
# `Capabilities.accepts` at build time.
DEFAULT_PAYLOAD: Mapping[DataKind, Payload] = {
    "tabular": "arrays",
    "image": "dataset",
    "text": "dataset",
    "timeseries": "series",
}

# Every payload a kind can be materialized as. Usually one — but a data *kind* is
# a statement about the data, not about the shape a model wants it in, and time
# series make the difference visible: Prophet is handed the ordered values
# (``series``), an LSTM the same data cut into sliding windows (``arrays``). The
# source produces whichever the selected model declares it accepts.
#
# `validate_combination` checks against this set rather than the single default,
# so a model is refused only when it can consume *none* of what its kind offers.
KIND_PAYLOADS: Mapping[DataKind, frozenset[Payload]] = {
    "tabular": frozenset({"arrays", "frame"}),
    "image": frozenset({"dataset"}),
    "text": frozenset({"dataset"}),
    "timeseries": frozenset({"series", "arrays"}),
}
