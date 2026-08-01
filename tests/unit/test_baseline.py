"""The trivial model, and the comparison it exists to support.

The point of this module is not the arithmetic — it is that the arithmetic is
taken from the *right split* and compared in the *right direction*. Both are easy
to get backwards, and both produce a number that looks fine:

* Taking the majority class from the test labels makes the baseline stronger than
  anything achievable at training time, inverting the comparison.
* Comparing "larger is better" on MAE or MASE reports every regression model as
  beating a baseline it lost to.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_framework.core.baseline import PREFIX, baseline_metrics, baseline_predictions, compare
from ml_framework.core.protocols import Predictions


# ── What the trivial predictor says ───────────────────────
@pytest.mark.unit
def test_classification_predicts_the_majority_training_class():
    train = np.array([0, 0, 0, 1])
    guess = baseline_predictions("multiclass", np.array([1, 1, 1]), train)

    assert guess.tolist() == [0, 0, 0]


@pytest.mark.unit
def test_the_statistic_comes_from_train_not_from_the_split_being_scored():
    """The failure that would make the guard useless.

    The test set here is all 1s; taking the majority from it would give a baseline
    that scores 100% — stronger than any model could achieve honestly, so every
    model would appear to lose.
    """
    train, test = np.array([0, 0, 0, 0, 1]), np.array([1, 1, 1, 1])

    guess = baseline_predictions("binary", test, train)
    assert set(guess.tolist()) == {0}


@pytest.mark.unit
def test_regression_predicts_the_training_mean():
    guess = baseline_predictions("regression", np.zeros(3), np.array([1.0, 2.0, 3.0]))

    assert np.allclose(guess, 2.0)


@pytest.mark.unit
def test_forecasting_repeats_the_tail_of_the_history():
    guess = baseline_predictions("forecasting", np.zeros(4), np.array([5.0, 6.0, 7.0]))

    # Seasonality 1 by default: repeat the last observed value.
    assert np.allclose(guess, 7.0)


@pytest.mark.unit
def test_falling_back_to_the_scored_split_when_there_is_no_train_split():
    """Not ideal and not silent-wrong: with nothing else available the test's own
    distribution is the only statistic there is, and a baseline is better than
    no comparison at all."""
    guess = baseline_predictions("binary", np.array([1, 1, 0]), None)

    assert set(guess.tolist()) == {1}


# ── What it reports ───────────────────────────────────────
@pytest.mark.unit
def test_metrics_are_prefixed_so_they_cannot_be_confused_with_the_models():
    predictions = Predictions(y_true=np.array([0, 0, 1, 1]), y_pred=np.array([0, 1, 0, 1]))

    scores = baseline_metrics(predictions, "binary", train_y=np.array([0, 0, 0, 1]))
    assert scores and all(name.startswith(PREFIX) for name in scores)
    assert scores["baseline_acc"] == 0.5


@pytest.mark.unit
def test_probability_metrics_are_skipped_rather_than_invented():
    """A constant predictor has no calibrated confidence. Feeding it something
    made up would produce a `baseline_roc_auc` that means nothing but reads as a
    number worth comparing."""
    predictions = Predictions(y_true=np.array([0, 1, 0, 1]), y_pred=np.array([0, 1, 0, 1]))

    scores = baseline_metrics(predictions, "binary", train_y=np.array([0, 0, 1]))
    assert "baseline_roc_auc" not in scores


@pytest.mark.unit
def test_generation_gets_no_baseline_rather_than_a_meaningless_one():
    """ "Always emit the most common string" is not a baseline anybody would
    compare against, and inventing a number there would be worse than silence."""
    predictions = Predictions(
        y_true=np.array(["a", "b"], dtype=object), y_pred=np.array(["a", "a"], dtype=object)
    )

    assert baseline_metrics(predictions, "seq2seq") == {}


@pytest.mark.unit
def test_unlabelled_predictions_get_no_baseline():
    assert baseline_metrics(Predictions(y_true=None, y_pred=np.array([1])), "binary") == {}


# ── The comparison ────────────────────────────────────────
@pytest.mark.unit
def test_a_higher_is_better_metric_compares_upward():
    assert compare({"test_acc": 0.9}, {"baseline_acc": 0.6}, "binary") is True
    assert compare({"test_acc": 0.5}, {"baseline_acc": 0.6}, "binary") is False


@pytest.mark.unit
def test_a_lower_is_better_metric_compares_downward():
    """MAE, RMSE and MASE all improve by getting smaller. Comparing them the same
    way as accuracy would report every regression model as beating a baseline it
    actually lost to."""
    assert compare({"test_mae": 1.0}, {"baseline_mae": 4.0}, "regression") is True
    assert compare({"test_mae": 9.0}, {"baseline_mae": 4.0}, "regression") is False


@pytest.mark.unit
def test_failing_to_beat_the_baseline_warns_but_does_not_raise(caplog):
    """A model that ties the baseline on a genuinely unpredictable target is an
    honest result; failing the run would be pretending otherwise. But it has to be
    said, because this is the shape of a pipeline that learned nothing."""
    import logging

    with caplog.at_level(logging.WARNING):
        beat = compare({"test_acc": 0.4}, {"baseline_acc": 0.9}, "binary")

    assert beat is False
    assert "does NOT beat" in caplog.text


@pytest.mark.unit
def test_no_baseline_is_not_reported_as_a_loss():
    """An absent comparison is not a failed one."""
    assert compare({"test_acc": 0.4}, {}, "binary") is True
