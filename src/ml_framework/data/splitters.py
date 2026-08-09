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

Two families live here, and the distinction is load-bearing:

* **Single-partition** splitters (:class:`RandomSplitter`,
  :class:`TemporalSplitter`, :class:`GroupSplitter`) return one
  :class:`SplitIndices` and are selected by ``split.strategy``.
* **Cross-validation** splitters (:class:`CrossValidationSplitter`,
  :class:`RollingOriginSplitter`, :class:`PurgedKFoldSplitter`,
  :class:`CombinatorialPurgedSplitter`) return a *list* of them and are selected
  by ``split.cv_strategy`` once ``split.folds >= 2``.

The cross-validation family exists because "estimate this model's performance
honestly" has a different correct answer per data domain, and using the wrong one
does not raise — it reports a better number. Shuffled k-fold on a time series
trains on the future; unpurged k-fold on overlapping labels trains on the test
set's own observations. Both look like success.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..core.types import FrameworkError

log = logging.getLogger(__name__)


# Tasks with exactly one class label per row -- the only thing stratification can
# balance. Everything else falls back to an unstratified fold: a regression target
# is continuous, a forecast is a series, a token-tagged sentence carries one label
# *per token*, and a seq2seq target is a string. Previously spelled
# `task == "regression"` at four call sites, which quietly asserted that every
# non-regression task has a row label -- true until token tagging arrived, and
# then wrong in a way that surfaces as an sklearn error deep in a split.
STRATIFIED_TASKS: frozenset[str] = frozenset({"binary", "multiclass"})


def stratifies(task: str) -> bool:
    """Whether ``task`` has a per-row class label to balance folds against."""
    return task in STRATIFIED_TASKS


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
        unstratified = not stratifies(self.task)
        if not unstratified and y is None:
            raise SplitError(f"task '{self.task}' requires labels to stratify the split")
        stratify = None if unstratified else y

        if n >= self.holdout_threshold:
            log.info("n=%d >= %d → stratified holdout", n, self.holdout_threshold)
            tmp, test = train_test_split(
                idx, test_size=self.test_size, random_state=self.seed, stratify=stratify
            )
            strat2 = None if unstratified else y[tmp]  # type: ignore[index]
            rel_val = self.val_size / (1.0 - self.test_size)
            train, val = train_test_split(
                tmp, test_size=rel_val, random_state=self.seed, stratify=strat2
            )
        else:
            log.info("n=%d < %d → fold-derived split", n, self.holdout_threshold)
            splitter = (
                KFold(n_splits=5, shuffle=True, random_state=self.seed)
                if unstratified
                else StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
            )
            train, hold = next(iter(splitter.split(idx, y)))
            strat3 = None if unstratified else y[hold]  # type: ignore[index]
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

        unstratified = not stratifies(self.task)
        if not unstratified and y is None:
            raise SplitError(f"task '{self.task}' requires labels to stratify the folds")

        idx = np.arange(n)
        splitter = (
            KFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
            if unstratified
            else StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        )

        out: list[SplitIndices] = []
        for rest, test in splitter.split(idx, y):
            # `val_size` is a fraction of the *whole* dataset, so the validation
            # set stays the size the config asked for rather than shrinking with k.
            rel_val = min(0.5, max(1.0 / len(rest), self.val_size * n / len(rest)))
            stratify = None if unstratified else y[rest]  # type: ignore[index]
            train, val = train_test_split(
                rest, test_size=rel_val, random_state=self.seed, stratify=stratify
            )
            out.append(SplitIndices(np.asarray(train), np.asarray(val), np.asarray(test)))

        log.info("cross-validation: %d folds over %d rows", self.folds, n)
        return out


def label_spans(
    n: int,
    *,
    horizon: int = 0,
    label_end: np.ndarray | None = None,
    times: np.ndarray | None = None,
) -> np.ndarray:
    """The last row each row's label depends on, as positional indices.

    A row's *label span* is the window of observations its target is computed
    from. A 10-step-ahead return at row ``i`` is not known until row ``i + 10``,
    so rows ``i`` and ``i + 10`` share information even though they are distinct
    rows. Purging needs to know that, and this is the one place the two ways of
    saying it are turned into the same array of positions.

    ``horizon`` is the approximation: every label spans the same fixed number of
    rows. It covers the common case (a fixed-horizon target) without asking a
    source to carry an extra column.

    ``label_end`` is the exact statement — one end-of-label *time* per row
    (López de Prado's ``t1``). Converting a time to a row position requires the
    rows' own observation times, so ``times`` is required alongside it; there is
    no way to infer "which row is this timestamp" from the timestamps of a
    different quantity. Rows are assumed sorted by ``times``, which is what every
    temporal source in this framework guarantees.

    With neither, every span is the row itself and purging becomes a no-op —
    which is exactly right for a target that is known at the row it sits on.
    """
    idx = np.arange(n)
    if label_end is not None:
        ends = np.asarray(label_end)
        if len(ends) != n:
            raise SplitError(f"label_end has {len(ends)} entries for {n} rows")
        if times is None:
            raise SplitError(
                "label_end needs the rows' own observation times to be mapped onto "
                "row positions — set split.time_col alongside split.label_end_col, "
                "or use split.label_horizon to state the span in rows instead"
            )
        observed = np.asarray(times)
        if len(observed) != n:
            raise SplitError(f"times has {len(observed)} entries for {n} rows")
        # `side="right" - 1` gives the last observation at or before the label's
        # end, so a label ending between two rows does not purge the later one.
        positions = np.searchsorted(observed, ends, side="right") - 1
        # A span never points behind its own row: a label that ends before the
        # observation it belongs to is a data error, and letting it produce a
        # backwards span would make that row's purge window empty — silently
        # turning the bad data into no protection at all.
        return np.maximum(idx, np.clip(positions, 0, n - 1))
    if horizon < 0:
        raise SplitError(f"label_horizon must be >= 0, got {horizon}")
    return np.minimum(idx + horizon, n - 1)


def purge_and_embargo(
    train: np.ndarray,
    test: np.ndarray,
    *,
    spans: np.ndarray,
    embargo: int = 0,
) -> np.ndarray:
    """``train`` with every row that shares information with ``test`` removed.

    Two distinct mechanisms, both required, and neither sufficient alone:

    **Purge** drops training rows whose label span reaches into the test window.
    Those rows were labelled using observations the test set is scoring on, so
    keeping them lets the model fit a target it is about to be graded against.
    Note it is the *span* that is checked, not the row index — a training row
    hundreds of positions before the test set still leaks if its label horizon
    reaches inside.

    **Embargo** drops training rows in a window immediately *after* the test set.
    Purging handles leakage forward in time; the embargo handles it backward,
    through serial correlation. A feature computed at ``test_end + 1`` is
    correlated with observations inside the test window even though no label
    spans it, and with strongly autocorrelated data that correlation is enough to
    inflate the score.

    Returns the surviving training indices in their original order.
    """
    if len(test) == 0:
        return train
    if embargo < 0:
        raise SplitError(f"embargo must be >= 0, got {embargo}")

    test_start = int(np.min(test))
    test_end = int(np.max(test))
    # Purge: the training row starts before the test window ends, and its label
    # reaches at or past the test window's start. Straddling in either direction
    # is contamination.
    overlaps = (train <= test_end) & (spans[train] >= test_start)
    # Embargo: `embargo` rows after the test window, dropped regardless of span.
    embargoed = (train > test_end) & (train <= test_end + embargo)
    return train[~(overlaps | embargoed)]


def embargo_rows(n: int, embargo: float | int) -> int:
    """``embargo`` as a row count, accepting either a fraction or a count.

    A fraction below 1.0 is read as a share of the dataset (``0.01`` → 1% of
    rows), which is how the finance literature states it and what keeps a config
    portable across dataset sizes. A value of 1.0 or greater is read as a literal
    row count, because "embargo 1 row" is a thing people mean and reading it as
    100% of the data would be a spectacular silent failure.
    """
    if embargo <= 0:
        return 0
    if embargo < 1.0:
        return int(np.ceil(embargo * n))
    return int(embargo)


@dataclass(frozen=True, slots=True)
class PurgedKFoldSplitter:
    """k **contiguous** folds with purging and an embargo. No shuffling.

    The splitter for data where rows are not independent — overlapping labels,
    serially correlated features, anything where knowing row *j* tells you
    something about row *i*. Plain :class:`CrossValidationSplitter` assumes
    independence and quietly reports a score that assumes it too.

    Two departures from :class:`CrossValidationSplitter`, both deliberate:

    * **Folds are contiguous slices, not shuffled.** A shuffled fold interleaves
      test rows through the training set, which leaves every training row
      adjacent to a test row and makes purging remove most of the data.
    * **Validation is carved from the end of the training block**, and purged
      against the test window as well. A validation set that leaks is a model
      selected on a leaked number, which is the same defect one level up.

    Unlike :class:`RollingOriginSplitter` this does *not* enforce that training
    precedes testing — every fold is used as a test set in turn, including the
    earliest. That is correct when the concern is overlapping labels rather than
    forecasting a future you cannot see; use rolling-origin when the deployment
    story is "stand at a point in time and predict forward".
    """

    name: ClassVar[str] = "purged"

    folds: int = 5
    embargo: float = 0.0
    label_horizon: int = 0
    val_size: float = 0.15

    def split(
        self,
        n: int,
        *,
        y: np.ndarray | None = None,
        label_end: np.ndarray | None = None,
        time: np.ndarray | None = None,
        **_: Any,
    ) -> list[SplitIndices]:
        if self.folds < 2:
            raise SplitError(f"purged cross-validation needs at least 2 folds, got {self.folds}")
        if n < self.folds:
            raise SplitError(f"cannot make {self.folds} folds from {n} rows")

        spans = label_spans(n, horizon=self.label_horizon, label_end=label_end, times=time)
        gap = embargo_rows(n, self.embargo)
        idx = np.arange(n)
        bounds = np.linspace(0, n, self.folds + 1).astype(int)

        out: list[SplitIndices] = []
        for i in range(self.folds):
            test = idx[bounds[i] : bounds[i + 1]]
            if len(test) == 0:
                continue
            rest = idx[~np.isin(idx, test)]
            kept = purge_and_embargo(rest, test, spans=spans, embargo=gap)
            if len(kept) < 2:
                raise SplitError(
                    f"fold {i} has {len(kept)} rows left after purging — the label horizon "
                    f"({self.label_horizon}) or embargo ({self.embargo}) is too large for "
                    f"{n} rows in {self.folds} folds"
                )
            # Validation comes off the *end* of what survived, then is purged
            # against the test window in turn. Taking it at random would scatter
            # validation rows through the purged region we just cleared.
            n_val = max(1, int(round(self.val_size * len(kept))))
            train, val = kept[:-n_val], kept[-n_val:]
            if len(train) == 0:
                raise SplitError(
                    f"fold {i} has no training rows after reserving {n_val} for validation"
                )
            out.append(SplitIndices(train, val, test))

        if not out:  # pragma: no cover - guarded by the fold-count checks above
            raise SplitError(f"purged cross-validation produced no folds from {n} rows")
        log.info(
            "purged k-fold: %d folds over %d rows, horizon %d, embargo %d rows",
            len(out),
            n,
            self.label_horizon,
            gap,
        )
        return out


@dataclass(frozen=True, slots=True)
class CombinatorialPurgedSplitter:
    """CPCV: every way of choosing ``test_groups`` of ``groups`` as the test set.

    Purged k-fold gives one backtest path — each row is tested exactly once, in
    one particular arrangement. That single path is itself a sample, and judging a
    model on it is judging it on one draw. CPCV splits the data into ``groups``
    contiguous blocks and tests every combination of ``test_groups`` of them,
    producing ``C(groups, test_groups)`` folds and enough paths to see the
    *distribution* of the score rather than one number from it.

    The cost is combinatorial and stated plainly rather than discovered: 6 groups
    of 2 is 15 fits, 10 of 2 is 45. :attr:`max_folds` caps it, keeping the first
    ``max_folds`` combinations in lexicographic order — a deterministic prefix
    rather than a random sample, so a capped run is reproducible.

    Test groups are generally **not contiguous**, which is the point: a fold that
    tests blocks 0 and 4 trains on the blocks between them, and the purge is
    applied around each test block separately.
    """

    name: ClassVar[str] = "cpcv"

    groups: int = 6
    test_groups: int = 2
    embargo: float = 0.0
    label_horizon: int = 0
    val_size: float = 0.15
    max_folds: int = 20

    def split(
        self,
        n: int,
        *,
        y: np.ndarray | None = None,
        label_end: np.ndarray | None = None,
        time: np.ndarray | None = None,
        **_: Any,
    ) -> list[SplitIndices]:
        from itertools import combinations

        if self.groups < 2:
            raise SplitError(f"CPCV needs at least 2 groups, got {self.groups}")
        if not 1 <= self.test_groups < self.groups:
            raise SplitError(
                f"test_groups must be in [1, {self.groups - 1}], got {self.test_groups}"
            )
        if n < self.groups:
            raise SplitError(f"cannot make {self.groups} groups from {n} rows")
        if self.max_folds < 1:
            raise SplitError(f"max_folds must be >= 1, got {self.max_folds}")

        spans = label_spans(n, horizon=self.label_horizon, label_end=label_end, times=time)
        gap = embargo_rows(n, self.embargo)
        idx = np.arange(n)
        bounds = np.linspace(0, n, self.groups + 1).astype(int)
        blocks = [idx[bounds[i] : bounds[i + 1]] for i in range(self.groups)]

        out: list[SplitIndices] = []
        for combo in combinations(range(self.groups), self.test_groups):
            if len(out) >= self.max_folds:
                log.warning(
                    "CPCV: stopping at max_folds=%d; C(%d,%d) would be %d folds",
                    self.max_folds,
                    self.groups,
                    self.test_groups,
                    _n_combinations(self.groups, self.test_groups),
                )
                break
            test = np.concatenate([blocks[g] for g in combo])
            rest = idx[~np.isin(idx, test)]
            # Each test block is purged separately: one purge against the union
            # would use the span between the first and last block and delete the
            # training data sitting between two distant test blocks.
            kept = rest
            for g in combo:
                kept = purge_and_embargo(kept, blocks[g], spans=spans, embargo=gap)
            if len(kept) < 2:
                raise SplitError(
                    f"CPCV fold {combo} has {len(kept)} rows left after purging — reduce "
                    f"label_horizon ({self.label_horizon}), embargo ({self.embargo}) "
                    f"or test_groups ({self.test_groups})"
                )
            n_val = max(1, int(round(self.val_size * len(kept))))
            train, val = kept[:-n_val], kept[-n_val:]
            if len(train) == 0:
                raise SplitError(
                    f"CPCV fold {combo} has no training rows after reserving "
                    f"{n_val} for validation"
                )
            out.append(SplitIndices(train, val, test))

        if not out:  # pragma: no cover - guarded by the group checks above
            raise SplitError(f"CPCV produced no folds from {n} rows")
        log.info(
            "CPCV: %d folds (C(%d,%d)=%d), %d rows, horizon %d, embargo %d rows",
            len(out),
            self.groups,
            self.test_groups,
            _n_combinations(self.groups, self.test_groups),
            n,
            self.label_horizon,
            gap,
        )
        return out


def _n_combinations(n: int, k: int) -> int:
    from math import comb

    return comb(n, k)


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
# Two maps, because there are genuinely two protocols. `SPLITTERS` holds the
# single-partition splitters selected by `split.strategy`. `CV_SPLITTERS` holds
# the ones that return a *list* of partitions, selected by `split.cv_strategy`
# once `split.folds >= 2`. Collapsing them into one map would mean a caller
# cannot tell from the name whether it is getting one partition or k.
SPLITTERS: dict[str, type] = {
    RandomSplitter.name: RandomSplitter,
    TemporalSplitter.name: TemporalSplitter,
    GroupSplitter.name: GroupSplitter,
}

CV_SPLITTERS: dict[str, type] = {
    CrossValidationSplitter.name: CrossValidationSplitter,
    RollingOriginSplitter.name: RollingOriginSplitter,
    PurgedKFoldSplitter.name: PurgedKFoldSplitter,
    CombinatorialPurgedSplitter.name: CombinatorialPurgedSplitter,
}


def get_splitter_class(name: str) -> type:
    try:
        return SPLITTERS[name]
    except KeyError:
        raise SplitError(
            f"Unknown split strategy '{name}'. Available: {sorted(SPLITTERS)}"
        ) from None


def get_cv_splitter_class(name: str) -> type:
    """The list-returning splitter for ``name``. Companion to
    :func:`get_splitter_class` for the cross-validation family."""
    try:
        return CV_SPLITTERS[name]
    except KeyError:
        raise SplitError(
            f"Unknown cv_strategy '{name}'. Available: {sorted(CV_SPLITTERS)}"
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
