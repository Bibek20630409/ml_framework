"""
data/types.py
─────────────
:class:`DataBundle` — the framework-agnostic handoff between the data layer and
whatever trains on it.

The v1 design put ingestion, splitting, scaling, imbalance handling *and*
DataLoader construction inside a ``LightningDataModule``, so a fit loop that is
not Lightning's could not consume the result at any price. A bundle is the
smallest thing that breaks that: plain arrays (or a lazy ``Dataset``) plus the
schema describing them.

Three rules this module holds to:

1. **No torch.** ``class_weights`` is a numpy array, not a tensor — the agnostic
   layer must not import torch, and the Lightning backend converts. (The v1
   ``compute_class_weights`` still returns a tensor at its original import path;
   see ``data.preprocess.tabular``.)
2. **The payload tag keeps the container honest.** An image split holds a lazy
   ``Dataset``, not an (n, d) matrix, and ``Split.payload`` says so. A backend
   declares what it can consume via ``Capabilities.accepts``, which is what turns
   "xgboost cannot consume an image folder" into a build-time error.
3. **Fitted transform state lives on the preprocessor**, never loose on the
   bundle. Nothing outside the preprocessor reads its files — that invariant is
   what removes the hardcoded ``scaler.pkl`` special cases downstream.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.types import DataKind, Payload, Task


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """What the columns/labels *are*, independent of how they are stored.

    ``feature_names`` is load-bearing rather than documentation: it goes into the
    bundle manifest's signature so the serving layer can reject silently
    reordered columns — the most common serving defect — and so drift monitoring
    reads feature names instead of guessing them.

    ``time_col``/``freq`` are unused until the time-series work; they live here
    because a schema without them would have to grow a parallel type later.
    """

    feature_names: tuple[str, ...] = ()
    dtypes: Mapping[str, str] = field(default_factory=dict)
    # Positional indices of categorical features, for backends that consume them
    # natively (`Capabilities.native_categorical`) instead of one-hot.
    categorical_idx: tuple[int, ...] = ()
    target_name: str | None = None
    class_names: tuple[str, ...] | None = None
    time_col: str | None = None
    freq: str | None = None

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True, slots=True)
class Split:
    """One of train/val/test.

    ``x`` holds whatever ``payload`` says it holds: an ``np.ndarray`` for
    ``arrays``, a ``DataFrame`` for ``frame``, a torch ``Dataset`` for
    ``dataset``. Typing it ``Any`` is deliberate — narrowing it would force this
    module to import torch to name the alternative.

    ``index`` carries the original row identity (row order for tabular,
    timestamps for a series). Forecasting reports are meaningless without it, and
    for tabular it makes ``predictions.csv`` traceable back to input rows.
    """

    payload: Payload = "arrays"
    x: Any = None
    y: np.ndarray | None = None
    index: np.ndarray | None = None

    @property
    def n(self) -> int:
        """Row count, from whichever member can answer.

        ``y`` first: for a ``dataset`` payload the labels are the cheap thing to
        measure, and ``len()`` on a lazily-loading dataset may touch the disk.
        """
        for candidate in (self.y, self.x):
            if candidate is None:
                continue
            try:
                return int(len(candidate))
            except TypeError:  # not sized (a generator-backed source)
                continue
        return 0

    def __bool__(self) -> bool:
        return self.x is not None or self.y is not None


@dataclass(frozen=True, slots=True)
class DataBundle:
    """Everything downstream of ingestion, in one object with no torch in it.

    ``input_dim``/``output_dim`` are the derived values the v1 code wrote onto the
    datamodule instance after ``setup()``. They are computed once, here, by the
    source that knows how — rather than re-derived by whoever needs them.
    ``output_dim`` follows the v1 head convention exactly: 1 for binary (single
    logit) and regression, ``n_classes`` for multiclass.
    """

    train: Split
    val: Split
    test: Split
    schema: FeatureSchema
    task: Task
    data_kind: DataKind
    input_dim: int = 0
    output_dim: int = 0
    # numpy, never torch. Binary → 1 element (BCE pos_weight); multiclass →
    # one balanced weight per class; None when no imbalance correction applies.
    class_weights: np.ndarray | None = None
    # Fitted on the TRAIN split only. Owns every piece of fitted transform state.
    preprocessor: Any | None = None
    # Drift baseline: the raw (pre-transform) training feature distribution,
    # because serving computes drift on the raw features clients send.
    reference_stats: dict | None = None
    # Source-specific extras the adapter may need but the contract should not
    # name: `sample_weights` for a WeightedRandomSampler, for instance.
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def payload(self) -> Payload:
        return self.train.payload

    @property
    def n_classes(self) -> int | None:
        """Number of classes, or ``None`` for regression.

        Recovered from ``output_dim`` and the task rather than stored twice: a
        binary head has ``output_dim == 1`` but two classes, and keeping a
        separate field would let the two disagree.
        """
        if self.task == "binary":
            return 2
        if self.task in ("multiclass", "token_classification"):
            # For a tagger this is the size of the *tag* vocabulary. It reaches the
            # manifest, which is what lets `predictions.csv` and the serving
            # response name tags instead of printing integers.
            return self.output_dim
        return None

    def split(self, name: str) -> Split:
        """``bundle.split("test")`` — so callers can be parameterized by name."""
        try:
            value = getattr(self, name)
        except AttributeError:
            value = None
        if not isinstance(value, Split):
            raise KeyError(f"Unknown split '{name}' (expected train|val|test)")
        return value

    def sizes(self) -> dict[str, int]:
        return {name: self.split(name).n for name in ("train", "val", "test")}
