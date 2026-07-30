"""Splitters: the random one must not change, the others must not leak."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from ml_framework.data.splitters import (
    GroupSplitter,
    RandomSplitter,
    SplitError,
    SplitIndices,
    TemporalSplitter,
    get_splitter_class,
    split_dataset,
)


def _legacy_split(x, y, *, seed, task, val_size, test_size, holdout_threshold):
    """The v1 ``split_dataset`` body, inlined.

    This is the regression oracle for :class:`RandomSplitter`: a partition change
    would silently move rows between train and test and invalidate every metric
    comparison across the refactor, while breaking no other assertion.
    """
    n = len(x)
    is_reg = task == "regression"
    stratify = None if is_reg else y
    if n >= holdout_threshold:
        x_tmp, x_test, y_tmp, y_test = train_test_split(
            x, y, test_size=test_size, random_state=seed, stratify=stratify
        )
        strat2 = None if is_reg else y_tmp
        rel_val = val_size / (1.0 - test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_tmp, y_tmp, test_size=rel_val, random_state=seed, stratify=strat2
        )
    else:
        splitter = (
            KFold(n_splits=5, shuffle=True, random_state=seed)
            if is_reg
            else StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        )
        tr_idx, hold_idx = next(iter(splitter.split(x, y)))
        x_train, y_train = x[tr_idx], y[tr_idx]
        x_hold, y_hold = x[hold_idx], y[hold_idx]
        strat3 = None if is_reg else y_hold
        x_val, x_test, y_val, y_test = train_test_split(
            x_hold, y_hold, test_size=0.5, random_state=seed, stratify=strat3
        )
    return x_train, x_val, x_test, y_train, y_val, y_test


def _data(n: int, n_classes: int | None, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype("float32")
    if n_classes is None:
        y = rng.normal(size=n).astype("float32")
    else:
        y = np.array([i % n_classes for i in range(n)])
    return x, y


# ── RandomSplitter: the partition must not move ───────────
@pytest.mark.unit
@pytest.mark.parametrize(
    "task,n,n_classes",
    [
        ("binary", 100, 2),  # fold-derived branch
        ("multiclass", 300, 3),
        ("regression", 100, None),
        ("binary", 6000, 2),  # holdout branch
        ("multiclass", 8000, 4),
        ("regression", 7000, None),
    ],
)
@pytest.mark.parametrize("seed", [0, 42])
def test_split_dataset_partition_is_identical_to_the_v1_implementation(task, n, n_classes, seed):
    x, y = _data(n, n_classes)
    kwargs = {
        "seed": seed,
        "task": task,
        "val_size": 0.15,
        "test_size": 0.15,
        "holdout_threshold": 5000,
    }
    expected = _legacy_split(x, y, **kwargs)
    actual = split_dataset(x, y, **kwargs)
    for exp, act in zip(expected, actual, strict=True):
        assert np.array_equal(exp, act)


@pytest.mark.unit
def test_random_splitter_returns_a_disjoint_covering_partition():
    x, y = _data(300, 3)
    parts = RandomSplitter(seed=0, task="multiclass").split(len(x), y=y)
    joined = np.concatenate([parts.train, parts.val, parts.test])
    assert len(np.unique(joined)) == len(joined) == 300


@pytest.mark.unit
def test_random_splitter_refuses_classification_without_labels():
    with pytest.raises(SplitError, match="requires labels"):
        RandomSplitter(task="multiclass").split(100)


@pytest.mark.unit
def test_splitters_reject_sizes_that_leave_no_training_rows():
    with pytest.raises(SplitError, match="must be < 1.0"):
        RandomSplitter(task="regression", val_size=0.5, test_size=0.6).split(100)


@pytest.mark.unit
def test_split_indices_refuses_an_empty_side():
    """An empty split is a configuration error, not something to discover in the
    fit loop as a zero-length dataloader."""
    with pytest.raises(SplitError, match="'val' split is empty"):
        SplitIndices(np.arange(5), np.array([], dtype=int), np.arange(5, 8))


# ── TemporalSplitter: chronology is the whole point ───────
@pytest.mark.unit
def test_temporal_split_is_contiguous_and_ordered():
    parts = TemporalSplitter(val_size=0.2, test_size=0.2).split(100)
    assert parts.train.max() < parts.val.min() < parts.test.min()
    assert np.array_equal(parts.train, np.arange(len(parts.train)))
    assert len(parts.train) + len(parts.val) + len(parts.test) == 100


@pytest.mark.unit
def test_temporal_split_sorts_by_the_time_column_not_row_order():
    # Rows arrive shuffled; the split must still be chronological.
    times = np.array([5, 1, 4, 2, 3, 9, 7, 6, 8, 10])
    parts = TemporalSplitter(val_size=0.2, test_size=0.2).split(10, time=times)
    ordered = np.concatenate([parts.train, parts.val, parts.test])
    assert np.array_equal(times[ordered], np.sort(times))


@pytest.mark.unit
def test_temporal_gap_drops_rows_between_segments():
    """The gap is what stops lag features from leaking across the boundary, so the
    dropped rows must be genuinely absent — not merely reassigned."""
    parts = TemporalSplitter(val_size=0.2, test_size=0.2, gap=3).split(100)
    kept = len(parts.train) + len(parts.val) + len(parts.test)
    assert kept == 100 - 2 * 3
    assert parts.val.min() - parts.train.max() == 4  # 3 dropped rows + 1


@pytest.mark.unit
def test_temporal_split_refuses_when_the_gap_consumes_training_data():
    with pytest.raises(SplitError, match="no training rows"):
        TemporalSplitter(val_size=0.3, test_size=0.3, gap=20).split(20)


# ── GroupSplitter: no entity on two sides ─────────────────
@pytest.mark.unit
def test_group_split_keeps_every_group_on_one_side():
    groups = np.repeat(np.arange(20), 5)  # 20 entities x 5 rows
    parts = GroupSplitter(seed=0).split(len(groups), groups=groups)
    train_g, val_g, test_g = (set(groups[p]) for p in (parts.train, parts.val, parts.test))
    assert not (train_g & val_g) and not (train_g & test_g) and not (val_g & test_g)


@pytest.mark.unit
def test_group_split_requires_groups():
    with pytest.raises(SplitError, match="requires a `groups`"):
        GroupSplitter().split(100)


@pytest.mark.unit
def test_group_split_refuses_too_few_groups_to_fill_three_sides():
    with pytest.raises(SplitError, match="at least 3 distinct groups"):
        GroupSplitter().split(10, groups=np.repeat([0, 1], 5))


# ── Lookup ────────────────────────────────────────────────
@pytest.mark.unit
def test_get_splitter_class_by_name():
    assert get_splitter_class("random") is RandomSplitter
    assert get_splitter_class("temporal") is TemporalSplitter


@pytest.mark.unit
def test_unknown_split_strategy_names_the_available_ones():
    with pytest.raises(SplitError, match="Available:"):
        get_splitter_class("nope")
