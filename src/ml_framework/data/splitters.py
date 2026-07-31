"""
data/splitters.py
─────────────────
Train/val/test partitioning, as **index sets**.

Splitting is its own object rather than a function because the *correct* split is
a property of the data, not of the model. Using a shuffled split on a time series
silently leaks the future into training — the most damaging silent failure in this
domain, and one that shows up as a suspiciously good validation score rather than
as an error.

Returning indices rather than sliced arrays is what makes the splitters reusable:
the same index sets partition an array, a DataFrame or a lazily-loaded
``Dataset``, and they can be recorded for reproducibility.

:class:`RandomSplitter` reproduces the v1 ``split_dataset`` **exactly** — the same
``train_test_split`` calls with the same ``random_state`` and the same
stratification, in the same order — so a run at a fixed seed produces the
partition it produced before this refactor. The v1 ``split_dataset`` helper still
exists (it slices with these indices) and its tests are unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..core.types import FrameworkError

log = logging.getLogger(__name__)


class SplitError(FrameworkError):
    """A split could not be produced (empty side, missing time/group column)."""


@dataclass(frozen=True, slots=True)
class SplitIndices:
    """Positional indices into the source rows, one array per split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        for name in ("train", "val", "test"):
            if len(getattr(self, name)) == 0:
                raise SplitError(
                    f"the '{name}' split is empty — the dataset is too small for the "
                    f"requested val_size/test_size"
                )

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def log(self) -> None:
        log.info("split → train:%d val:%d test:%d", len(self.train), len(self.val), len(self.test))


def _check_sizes(val_size: float, test_size: float) -> None:
    if not 0.0 < val_size < 1.0 or not 0.0 < test_size < 1.0:
        raise SplitError("val_size and test_size must each be in (0, 1)")
    if val_size + test_size >= 1.0:
        raise SplitError("val_size + test_size must be < 1.0")


# ── Random / stratified ───────────────────────────────────
@dataclass(frozen=True, slots=True)
class RandomSplitter:
    """Shuffled holdout, stratified for classification. The v1 default.

    Auto-selects between a plain holdout and a fold-derived split by dataset size:
    below ``holdout_threshold`` rows, a single 5-fold cut yields a larger, more
    stable training set than carving 30% off the top.

    The regression branch uses plain ``KFold`` rather than ``StratifiedKFold`` —
    stratifying a continuous target is what crashed the original framework, and
    the regression-on-a-small-dataset regression test guards it.
    """

    name: ClassVar[str] = "random"

    seed: int = 42
    task: str = "multiclass"
    val_size: float = 0.15
    test_size: float = 0.15
    holdout_threshold: int = 5000

    def split(self, n: int, *, y: np.ndarray | None = None, **_: Any) -> SplitIndices:
        from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

        _check_sizes(self.val_size, self.test_size)
        idx = np.arange(n)
        is_reg = self.task == "regression"
        if not is_reg and y is None:
            raise SplitError(f"task '{self.task}' requires labels to stratify the split")
        stratify = None if is_reg else y

        if n >= self.holdout_threshold:
            log.info("n=%d >= %d → stratified holdout", n, self.holdout_threshold)
            tmp, test = train_test_split(
                idx, test_size=self.test_size, random_state=self.seed, stratify=stratify
            )
            strat2 = None if is_reg else y[tmp]  # type: ignore[index]
            rel_val = self.val_size / (1.0 - self.test_size)
            train, val = train_test_split(
                tmp, test_size=rel_val, random_state=self.seed, stratify=strat2
            )
        else:
            log.info("n=%d < %d → fold-derived split", n, self.holdout_threshold)
            splitter = (
                KFold(n_splits=5, shuffle=True, random_state=self.seed)
                if is_reg
                else StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
            )
            train, hold = next(iter(splitter.split(idx, y)))
            strat3 = None if is_reg else y[hold]  # type: ignore[index]
            val, test = train_test_split(
                hold, test_size=0.5, random_state=self.seed, stratify=strat3
            )

        out = SplitIndices(np.asarray(train), np.asarray(val), np.asarray(test))
        out.log()
        return out


# ── Temporal ──────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TemporalSplitter:
    """Contiguous ``[train | gap | val | gap | test]`` in time order.

    Never shuffled, never stratified: every training row must precede every
    validation row, which precedes every test row.

    ``gap`` drops rows between the segments. It is not cosmetic — with lag or
    rolling-window features, the last training rows and the first validation rows
    share source observations, so a zero gap leaks even though the split itself is
    chronological.
    """

    name: ClassVar[str] = "temporal"

    val_size: float = 0.15
    test_size: float = 0.15
    gap: int = 0

    def split(
        self, n: int, *, y: np.ndarray | None = None, time: np.ndarray | None = None, **_: Any
    ) -> SplitIndices:
        _check_sizes(self.val_size, self.test_size)
        if self.gap < 0:
            raise SplitError("gap must be >= 0")

        # `time` is optional: a source that has already sorted its rows passes
        # nothing, and row order *is* the chronology. A stable sort keeps ties in
        # their original order rather than permuting them arbitrarily.
        order = np.arange(n) if time is None else np.argsort(np.asarray(time), kind="stable")

        n_test = max(1, int(round(self.test_size * n)))
        n_val = max(1, int(round(self.val_size * n)))
        n_train = n - n_test - n_val - 2 * self.gap
        if n_train < 1:
            raise SplitError(
                f"n={n} leaves no training rows after val={n_val}, test={n_test} "
                f"and 2 x gap={self.gap}"
            )

        train = order[:n_train]
        val = order[n_train + self.gap : n_train + self.gap + n_val]
        test = order[n_train + 2 * self.gap + n_val :]
        out = SplitIndices(train, val, test)
        out.log()
        return out


# ── Grouped ───────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class GroupSplitter:
    """Holdout that keeps every row of a group on one side of the split.

    For repeated entities (multiple visits per patient, multiple sessions per
    user) a row-level split puts the same entity in train and test, which inflates
    every metric. Nothing in v1 prevented that.
    """

    name: ClassVar[str] = "group"

    seed: int = 42
    val_size: float = 0.15
    test_size: float = 0.15

    def split(
        self, n: int, *, y: np.ndarray | None = None, groups: np.ndarray | None = None, **_: Any
    ) -> SplitIndices:
        from sklearn.model_selection import GroupShuffleSplit

        _check_sizes(self.val_size, self.test_size)
        if groups is None:
            raise SplitError("GroupSplitter requires a `groups` array")
        g = np.asarray(groups)
        if len(g) != n:
            raise SplitError(f"groups has {len(g)} entries for {n} rows")
        if len(np.unique(g)) < 3:
            raise SplitError("GroupSplitter needs at least 3 distinct groups")

        idx = np.arange(n)
        first = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.seed)
        tmp, test = next(iter(first.split(idx, y, groups=g)))
        rel_val = self.val_size / (1.0 - self.test_size)
        second = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=self.seed)
        rel_train, rel_val_idx = next(iter(second.split(tmp, None, groups=g[tmp])))

        out = SplitIndices(tmp[rel_train], tmp[rel_val_idx], test)
        out.log()
        return out


# ── Cross-validation ──────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CrossValidationSplitter:
    """k folds, each a complete ``SplitIndices``.

    Yields *whole* train/val/test partitions rather than the usual (train, test)
    pairs, because everything downstream — early stopping, the preprocessor fitted
    on train only, the per-fold metrics — needs a validation set. Fold *i* is the
    test set; a ``val_size`` slice of what remains becomes validation; the rest is
    training.

    Deliberately not a ``Splitter``: the protocol returns one partition and this
    returns k. Making it fit would have meant either a lying signature or a
    protocol that returns a list nobody else wants.

    Stratified for classification, plain KFold for regression — stratifying a
    continuous target is what crashed the original framework, and the same guard
    applies here.
    """

    name: ClassVar[str] = "cv"

    folds: int = 5
    seed: int = 42
    task: str = "multiclass"
    val_size: float = 0.15

    def split(self, n: int, *, y: np.ndarray | None = None, **_: Any) -> list[SplitIndices]:
        from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

        if self.folds < 2:
            raise SplitError(f"cross-validation needs at least 2 folds, got {self.folds}")
        if n < self.folds:
            raise SplitError(f"cannot make {self.folds} folds from {n} rows")

        is_reg = self.task == "regression"
        if not is_reg and y is None:
            raise SplitError(f"task '{self.task}' requires labels to stratify the folds")

        idx = np.arange(n)
        splitter = (
            KFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
            if is_reg
            else StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        )

        out: list[SplitIndices] = []
        for rest, test in splitter.split(idx, y):
            # `val_size` is a fraction of the *whole* dataset, so the validation
            # set stays the size the config asked for rather than shrinking with k.
            rel_val = min(0.5, max(1.0 / len(rest), self.val_size * n / len(rest)))
            stratify = None if is_reg else y[rest]  # type: ignore[index]
            train, val = train_test_split(
                rest, test_size=rel_val, random_state=self.seed, stratify=stratify
            )
            out.append(SplitIndices(np.asarray(train), np.asarray(val), np.asarray(test)))

        log.info("cross-validation: %d folds over %d rows", self.folds, n)
        return out


@dataclass(frozen=True, slots=True)
class RollingOriginSplitter:
    """Time-series cross-validation: k origins, each forecasting the next horizon.

    The temporal counterpart of :class:`CrossValidationSplitter`, and **not**
    interchangeable with it. Shuffled k-fold puts future rows in training and past
    rows in test; the resulting score is not an estimate of anything you can
    deploy. Here every fold's training data precedes its test window, so each fold
    is a rehearsal of the thing you will actually do — stand at a point in time and
    forecast forward.

    ``expanding=True`` (the default) grows the training set with each origin, which
    is what a production retrain does. ``expanding=False`` slides a fixed window,
    which is what you want when old data is actively misleading (a regime change,
    a changed measurement process).

    ``gap`` drops observations between train and test. With lag features the last
    training rows and the first test rows share source observations, so a zero gap
    leaks even though the split is chronological.
    """

    name: ClassVar[str] = "rolling_origin"

    folds: int = 3
    horizon: int = 1
    gap: int = 0
    expanding: bool = True
    # Rows before the first origin. None → derived so the folds fit.
    min_train: int | None = None

    def split(self, n: int, *, y: np.ndarray | None = None, **_: Any) -> list[SplitIndices]:
        if self.folds < 1:
            raise SplitError(f"rolling-origin needs at least 1 fold, got {self.folds}")
        if self.horizon < 1:
            raise SplitError(f"horizon must be >= 1, got {self.horizon}")

        # Each fold consumes `horizon` rows for test and `horizon` for validation,
        # plus two gaps. Everything before the first origin is the initial train.
        per_fold = self.horizon
        needed = self.folds * per_fold + per_fold + 2 * self.gap
        min_train = self.min_train if self.min_train is not None else max(per_fold, n - needed)
        if min_train < 1:
            raise SplitError(
                f"n={n} is too short for {self.folds} folds of horizon {self.horizon} "
                f"(needs at least {needed + 1} rows)"
            )

        order = np.arange(n)
        out: list[SplitIndices] = []
        for fold in range(self.folds):
            train_end = min_train + fold * per_fold
            val_start = train_end + self.gap
            val_end = val_start + per_fold
            test_start = val_end + self.gap
            test_end = test_start + per_fold
            if test_end > n:
                log.warning(
                    "rolling-origin: stopping after %d folds — the series ends at %d", fold, n
                )
                break
            train_start = 0 if self.expanding else max(0, train_end - min_train)
            out.append(
                SplitIndices(
                    order[train_start:train_end],
                    order[val_start:val_end],
                    order[test_start:test_end],
                )
            )

        if not out:
            raise SplitError(
                f"n={n} produced no rolling-origin folds at horizon {self.horizon}; "
                f"shorten the horizon or the fold count"
            )
        log.info(
            "rolling-origin: %d folds, horizon %d, %s window",
            len(out),
            self.horizon,
            "expanding" if self.expanding else "sliding",
        )
        return out


# ── Registry ──────────────────────────────────────────────
# A plain name → class map rather than a PluginRegistry: splitters take no
# optional dependencies and need no capability metadata, so a spec-carrying
# registry would be ceremony.
#
# Only the single-partition splitters are here. `CrossValidationSplitter` and
# `RollingOriginSplitter` return a *list* of partitions, so they do not satisfy
# the same protocol and are selected by `split.folds` rather than by name.
SPLITTERS: dict[str, type] = {
    RandomSplitter.name: RandomSplitter,
    TemporalSplitter.name: TemporalSplitter,
    GroupSplitter.name: GroupSplitter,
}


def get_splitter_class(name: str) -> type:
    try:
        return SPLITTERS[name]
    except KeyError:
        raise SplitError(
            f"Unknown split strategy '{name}'. Available: {sorted(SPLITTERS)}"
        ) from None


# ── v1 compatibility ──────────────────────────────────────
def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    task: str,
    val_size: float,
    test_size: float,
    holdout_threshold: int,
):
    """Train/val/test split returning **sliced arrays**, as v1 did.

    Kept at its original signature (and re-exported from ``core``) because it is
    public API with its own tests. It is now a two-line wrapper over
    :class:`RandomSplitter`, so there is one implementation of the logic rather
    than two that can drift.
    """
    parts = RandomSplitter(
        seed=seed,
        task=task,
        val_size=val_size,
        test_size=test_size,
        holdout_threshold=holdout_threshold,
    ).split(len(x), y=y)
    return (
        x[parts.train],
        x[parts.val],
        x[parts.test],
        y[parts.train],
        y[parts.val],
        y[parts.test],
    )
