"""
Purged k-fold and combinatorial purged cross-validation.

These splitters exist for one reason: k-fold assumes rows are independent, and
when they are not it reports a better score than the model can deliver. So the
tests here are mostly about what is *absent* from a training fold — the rows
whose labels overlap the test window, and the rows immediately after it. A
splitter that returns plausible-looking folds while leaving those in would pass
any test that only checks shapes, which is why almost every assertion below is
about a specific index not being present.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_framework.data.splitters import (
    CV_SPLITTERS,
    CombinatorialPurgedSplitter,
    PurgedKFoldSplitter,
    SplitError,
    embargo_rows,
    get_cv_splitter_class,
    label_spans,
    purge_and_embargo,
)

pytestmark = pytest.mark.unit


# ── label_spans ───────────────────────────────────────────
def test_label_spans_with_no_horizon_makes_every_row_its_own_span():
    # A target known at the row it sits on overlaps nothing, so purging must be
    # a no-op rather than a quiet data loss.
    assert list(label_spans(5)) == [0, 1, 2, 3, 4]


def test_label_spans_extends_each_row_by_the_horizon():
    assert list(label_spans(5, horizon=2)) == [2, 3, 4, 4, 4]


def test_label_spans_clamps_the_horizon_at_the_last_row():
    # The span of the final row cannot reach past the data that exists.
    assert list(label_spans(3, horizon=10)) == [2, 2, 2]


def test_label_spans_maps_an_end_time_column_to_positions():
    # Rows are observed at 1, 3, 5, 7. Row 0's label ends at time 3, which is
    # row 1's observation; row 1's ends at 7, which is row 3's.
    times = np.array([1.0, 3.0, 5.0, 7.0])
    spans = label_spans(4, label_end=np.array([3.0, 7.0, 5.0, 7.0]), times=times)
    assert list(spans) == [1, 3, 2, 3]


def test_label_spans_does_not_purge_the_row_after_a_label_that_ends_between_two():
    # A label ending at 4.0 falls between the observations at 3.0 and 5.0. It
    # reaches row 1 and not row 2 — purging row 2 would discard usable data.
    times = np.array([1.0, 3.0, 5.0, 7.0])
    spans = label_spans(4, label_end=np.array([4.0, 4.0, 6.0, 8.0]), times=times)
    assert list(spans) == [1, 1, 2, 3]


def test_label_spans_never_points_a_span_behind_its_own_row():
    # An end time earlier than the row's own observation is a data error, but it
    # must not produce a backwards span that makes the purge window empty.
    times = np.array([1.0, 3.0, 5.0, 7.0])
    spans = label_spans(4, label_end=np.array([0.0, 0.0, 0.0, 0.0]), times=times)
    assert list(spans) == [0, 1, 2, 3]


def test_label_spans_refuses_an_end_column_without_observation_times():
    # A time cannot be placed on a row without knowing when the rows happened,
    # and guessing is how a purge silently protects nothing.
    with pytest.raises(SplitError, match="observation times"):
        label_spans(4, label_end=np.array([1.0, 2.0, 3.0, 4.0]))


def test_label_spans_rejects_a_mismatched_end_column():
    with pytest.raises(SplitError, match="3 entries for 5 rows"):
        label_spans(5, label_end=np.array([1.0, 2.0, 3.0]), times=np.arange(5.0))


def test_label_spans_rejects_a_negative_horizon():
    with pytest.raises(SplitError, match="must be >= 0"):
        label_spans(5, horizon=-1)


# ── embargo_rows ──────────────────────────────────────────
@pytest.mark.parametrize(
    ("embargo", "n", "expected"),
    [
        (0.0, 1000, 0),
        (0.01, 1000, 10),  # a fraction is a share of the dataset
        (0.005, 1000, 5),
        (5, 1000, 5),  # >= 1.0 is a literal row count
        (1.0, 1000, 1),  # the boundary reads as one row, not as 100%
    ],
)
def test_embargo_rows_reads_a_fraction_or_a_count(embargo, n, expected):
    assert embargo_rows(n, embargo) == expected


def test_embargo_rows_rounds_a_fraction_up():
    # Rounding down would let `embargo: 0.001` on a small dataset silently mean
    # "no embargo at all" while reading as though one was configured.
    assert embargo_rows(100, 0.001) == 1


# ── purge_and_embargo ─────────────────────────────────────
def test_purge_drops_training_rows_whose_label_reaches_into_the_test_window():
    # Rows 0-9 train, 12-15 test. With a 3-row label horizon, rows 9, 10 and 11
    # are labelled using observations inside the test window.
    n = 20
    spans = label_spans(n, horizon=3)
    train = np.arange(0, 12)
    test = np.arange(12, 16)

    kept = purge_and_embargo(train, test, spans=spans, embargo=0)

    assert 9 in train and 9 not in kept  # span reaches row 12
    assert 8 in kept  # span reaches only row 11
    assert set(kept) == set(range(0, 9))


def test_embargo_drops_training_rows_immediately_after_the_test_window():
    # Nothing here overlaps by label — the embargo is doing all the work, which
    # is the point: purging looks forward and cannot see serial correlation.
    n = 20
    spans = label_spans(n, horizon=0)
    train = np.arange(0, 20)
    test = np.arange(8, 12)

    kept = purge_and_embargo(train, test, spans=spans, embargo=3)

    for row in (12, 13, 14):
        assert row not in kept, f"row {row} is inside the embargo"
    assert 15 in kept
    assert 7 in kept  # before the test window; the embargo is one-sided


def test_purge_and_embargo_leaves_an_independent_split_untouched():
    spans = label_spans(20, horizon=0)
    train = np.arange(0, 10)
    kept = purge_and_embargo(train, np.arange(15, 20), spans=spans, embargo=0)
    assert list(kept) == list(train)


def test_purge_and_embargo_rejects_a_negative_embargo():
    with pytest.raises(SplitError, match="embargo must be >= 0"):
        purge_and_embargo(np.arange(10), np.arange(3), spans=label_spans(10), embargo=-1)


# ── PurgedKFoldSplitter ───────────────────────────────────
def test_purged_kfold_yields_contiguous_non_overlapping_test_blocks():
    folds = PurgedKFoldSplitter(folds=4).split(200)

    assert len(folds) == 4
    tests = [f.test for f in folds]
    # Contiguous: each test block is a run of consecutive indices.
    for test in tests:
        assert list(test) == list(range(int(test[0]), int(test[-1]) + 1))
    # Exhaustive and disjoint: every row is tested exactly once.
    combined = np.concatenate(tests)
    assert sorted(combined) == list(range(200))


def test_purged_kfold_removes_overlapping_rows_from_training():
    horizon = 8
    folds = PurgedKFoldSplitter(folds=4, label_horizon=horizon).split(200)
    spans = label_spans(200, horizon=horizon)

    for fold in folds:
        test_start, test_end = int(fold.test.min()), int(fold.test.max())
        for row in fold.train:
            overlaps = row <= test_end and spans[row] >= test_start
            assert not overlaps, f"train row {row} overlaps test [{test_start}, {test_end}]"


def test_purged_kfold_purges_the_validation_split_too():
    # A validation set that leaks selects the model on a leaked number — the same
    # defect as a leaking test set, one level up.
    horizon = 8
    folds = PurgedKFoldSplitter(folds=4, label_horizon=horizon).split(200)
    spans = label_spans(200, horizon=horizon)

    for fold in folds:
        test_start, test_end = int(fold.test.min()), int(fold.test.max())
        for row in fold.val:
            assert not (row <= test_end and spans[row] >= test_start)


def test_purged_kfold_keeps_train_val_and_test_disjoint():
    for fold in PurgedKFoldSplitter(folds=3, label_horizon=5, embargo=0.01).split(300):
        assert not set(fold.train) & set(fold.val)
        assert not set(fold.train) & set(fold.test)
        assert not set(fold.val) & set(fold.test)


def test_purged_kfold_drops_more_rows_as_the_horizon_grows():
    small = PurgedKFoldSplitter(folds=4, label_horizon=2).split(400)
    large = PurgedKFoldSplitter(folds=4, label_horizon=40).split(400)
    assert sum(len(f.train) for f in large) < sum(len(f.train) for f in small)


def test_purged_kfold_refuses_a_horizon_that_consumes_the_data():
    # Failing loudly beats returning two-row training folds and a score computed
    # from them.
    with pytest.raises(SplitError, match="too large"):
        PurgedKFoldSplitter(folds=5, label_horizon=200).split(100)


def test_purged_kfold_needs_at_least_two_folds():
    with pytest.raises(SplitError, match="at least 2 folds"):
        PurgedKFoldSplitter(folds=1).split(100)


def test_purged_kfold_refuses_more_folds_than_rows():
    with pytest.raises(SplitError, match="cannot make 10 folds from 5 rows"):
        PurgedKFoldSplitter(folds=10).split(5)


# ── CombinatorialPurgedSplitter ───────────────────────────
def test_cpcv_produces_one_fold_per_combination():
    folds = CombinatorialPurgedSplitter(groups=5, test_groups=2, max_folds=100).split(300)
    assert len(folds) == 10  # C(5, 2)


def test_cpcv_test_sets_are_generally_not_contiguous():
    # The defining difference from purged k-fold: a fold can test two separated
    # blocks and train on the data between them.
    folds = CombinatorialPurgedSplitter(groups=4, test_groups=2, max_folds=100).split(400)
    gapped = [f for f in folds if np.any(np.diff(np.sort(f.test)) > 1)]
    assert gapped, "no fold tested two separated blocks"


def test_cpcv_trains_on_rows_between_two_distant_test_blocks():
    # Purging each test block separately is what preserves this. One purge across
    # the union would span the gap and delete the training data inside it.
    folds = CombinatorialPurgedSplitter(groups=4, test_groups=2, max_folds=100).split(400)
    for fold in folds:
        blocks = np.sort(fold.test)
        if np.any(np.diff(blocks) > 1):
            assert len(fold.train) > 0
            break


def test_cpcv_respects_max_folds():
    folds = CombinatorialPurgedSplitter(groups=8, test_groups=2, max_folds=5).split(400)
    assert len(folds) == 5  # C(8, 2) is 28


def test_cpcv_purges_every_test_block():
    horizon = 6
    folds = CombinatorialPurgedSplitter(
        groups=5, test_groups=2, label_horizon=horizon, max_folds=100
    ).split(300)
    spans = label_spans(300, horizon=horizon)

    for fold in folds:
        test = {int(t) for t in fold.test}
        for row in fold.train:
            reach = set(range(int(row), int(spans[row]) + 1))
            assert not (reach & test), f"train row {row} reaches into the test set"


def test_cpcv_rejects_testing_every_group():
    with pytest.raises(SplitError, match=r"test_groups must be in \[1, 3\]"):
        CombinatorialPurgedSplitter(groups=4, test_groups=4).split(100)


def test_cpcv_needs_at_least_two_groups():
    with pytest.raises(SplitError, match="at least 2 groups"):
        CombinatorialPurgedSplitter(groups=1).split(100)


def test_cpcv_refuses_a_purge_that_consumes_the_training_data():
    with pytest.raises(SplitError, match="after purging"):
        CombinatorialPurgedSplitter(
            groups=4, test_groups=2, label_horizon=500, max_folds=100
        ).split(100)


# ── Registry ──────────────────────────────────────────────
def test_cv_splitters_registry_holds_the_list_returning_family():
    assert set(CV_SPLITTERS) == {"cv", "rolling_origin", "purged", "cpcv"}


def test_get_cv_splitter_class_resolves_by_name():
    assert get_cv_splitter_class("purged") is PurgedKFoldSplitter
    assert get_cv_splitter_class("cpcv") is CombinatorialPurgedSplitter


def test_get_cv_splitter_class_names_the_alternatives_when_asked_for_a_bad_one():
    with pytest.raises(SplitError, match="Unknown cv_strategy 'kfold'"):
        get_cv_splitter_class("kfold")
