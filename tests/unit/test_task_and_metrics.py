"""core/task.py + core/metrics.py — the task table and array-based metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ml_framework.core import metrics as m
from ml_framework.core.task import (
    TaskSpec,
    UnknownTaskError,
    available_tasks,
    get_task_spec,
    has_task_spec,
    register_task_spec,
)


# ── The table ─────────────────────────────────────────────
@pytest.mark.unit
def test_the_three_currently_runnable_tasks_have_rows():
    assert available_tasks() == ["binary", "multiclass", "regression"]


@pytest.mark.unit
def test_unregistered_task_says_so_instead_of_defaulting():
    """A task in the Literal but without a row must fail loudly.

    `forecasting` gets its row in P6; until then silently treating it as
    regression would produce plausible-looking wrong metrics.
    """
    assert not has_task_spec("forecasting")
    with pytest.raises(UnknownTaskError, match="No TaskSpec for task 'forecasting'"):
        get_task_spec("forecasting")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("task", "primary", "direction", "postprocess"),
    [
        ("binary", "acc", "max", "sigmoid"),
        ("multiclass", "acc", "max", "softmax"),
        ("regression", "mae", "min", "identity"),
    ],
)
def test_task_rows_carry_metric_direction_and_head_postprocessing(
    task, primary, direction, postprocess
):
    spec = get_task_spec(task)
    assert spec.primary_metric == primary
    assert spec.direction == direction
    assert spec.postprocess == postprocess


@pytest.mark.unit
def test_monitor_reproduces_todays_early_stopping_configuration():
    """EarlyStopping/ModelCheckpoint currently watch val/loss with mode=min."""
    for task in available_tasks():
        spec = get_task_spec(task)
        assert (spec.monitor, spec.monitor_mode) == ("val/loss", "min")


@pytest.mark.unit
def test_monitor_and_primary_metric_are_separate_axes():
    """What you monitor during training is not what you optimise for."""
    spec = get_task_spec("multiclass")
    assert spec.monitor == "val/loss" and spec.monitor_mode == "min"
    assert spec.primary_metric == "acc" and spec.direction == "max"


@pytest.mark.unit
def test_metric_fns_resolve_to_callables():
    fns = get_task_spec("regression").metric_fns
    assert set(fns) == {"mae", "rmse", "r2"}
    assert all(callable(fn) for fn in fns.values())


@pytest.mark.unit
def test_register_task_spec_refuses_to_clobber_a_row():
    spec = TaskSpec(
        name="binary",
        primary_metric="acc",
        direction="max",
        output_kind="labels",
        postprocess="sigmoid",
        metric_names=("acc",),
    )
    with pytest.raises(ValueError, match="already registered"):
        register_task_spec(spec)


@pytest.mark.unit
def test_task_specs_are_frozen():
    with pytest.raises(AttributeError):
        get_task_spec("binary").primary_metric = "f1"  # type: ignore[misc]


# ── Metrics ───────────────────────────────────────────────
@pytest.mark.unit
def test_accuracy_matches_the_current_evaluate_computation():
    """evaluate.py computes float(np.mean(preds == labels)); keep that exact value."""
    # Arrange
    y_true = np.array([0, 1, 2, 2, 1])
    y_pred = np.array([0, 1, 2, 1, 1])

    # Act / Assert
    assert m.accuracy(y_true, y_pred) == pytest.approx(4 / 5)
    assert m.accuracy(y_true, y_pred) == pytest.approx(float(np.mean(y_pred == y_true)))


@pytest.mark.unit
def test_mae_and_rmse_match_the_current_evaluate_computation():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.0, 4.0])
    assert m.mae(y_true, y_pred) == pytest.approx(float(np.abs(y_pred - y_true).mean()))
    assert m.rmse(y_true, y_pred) == pytest.approx(float(np.sqrt(((y_pred - y_true) ** 2).mean())))


@pytest.mark.unit
def test_perfect_predictions_score_perfectly():
    y = np.array([0, 1, 1, 0])
    assert m.accuracy(y, y) == 1.0
    assert m.f1(y, y) == 1.0
    assert m.mae(y, y) == 0.0
    assert m.rmse(y, y) == 0.0
    assert m.r2(y.astype(float), y.astype(float)) == 1.0


@pytest.mark.unit
def test_r2_returns_nan_for_a_constant_target_rather_than_zero():
    y_true = np.array([2.0, 2.0, 2.0])
    assert np.isnan(m.r2(y_true, np.array([2.0, 2.1, 1.9])))


@pytest.mark.unit
def test_roc_auc_returns_nan_when_undefined_instead_of_raising():
    """A metrics dict must never be the thing that fails a training run."""
    y_true = np.array([1, 1, 1])  # single class present
    y_prob = np.array([[0.2, 0.8], [0.1, 0.9], [0.3, 0.7]])
    assert np.isnan(m.roc_auc(y_true, y_prob))


@pytest.mark.unit
def test_f1_does_not_raise_on_an_absent_class():
    """zero_division=0, matching the existing classification_report call."""
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 0, 0])
    assert m.f1(y_true, y_pred) == pytest.approx(1 / 3, abs=0.2)


@pytest.mark.unit
def test_compute_metrics_uses_the_task_row_and_prefixes_keys():
    # Arrange
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])

    # Act
    out = m.compute_metrics("binary", y_true, y_pred, y_prob, prefix="test_")

    # Assert
    assert set(out) == {"test_acc", "test_f1_binary", "test_roc_auc"}
    assert out["test_acc"] == pytest.approx(0.75)


@pytest.mark.unit
def test_probability_metrics_are_skipped_when_a_backend_returns_no_probabilities():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    out = m.compute_metrics("binary", y_true, y_pred, None)
    assert "roc_auc" not in out
    assert set(out) == {"acc", "f1_binary"}


@pytest.mark.unit
def test_regression_metrics_are_the_documented_set():
    out = m.compute_metrics("regression", np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert set(out) == {"mae", "rmse", "r2"}


@pytest.mark.unit
def test_unknown_metric_name_fails_loudly():
    with pytest.raises(KeyError, match="Unknown metric"):
        m.metric_fn("not_a_metric")


@pytest.mark.unit
def test_metrics_accept_column_vectors_as_well_as_flat_arrays():
    """Backends differ on whether they return (n,) or (n, 1)."""
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert m.mae(y_true, y_pred) == 0.0
