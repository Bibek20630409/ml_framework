"""
core/baseline.py
────────────────
The trivial model, scored beside the real one.

**This is the single best guard against zero-config quietly shipping something
useless.** When the framework picks the model itself, the user has no intuition
for whether 0.83 is good — they did not choose the model, the features or the
split. A number with nothing to compare it against is not evidence. So every
auto-selected run also scores the dumbest possible predictor:

    binary / multiclass     always the majority training class
    regression              always the mean of the training target
    forecasting             repeat the last observed season
    token_classification    always the most common tag

If the model does not beat that, it is reported at WARNING. Not an error: a model
that ties the baseline on a genuinely unpredictable target is an honest result,
and failing the run would be pretending otherwise. But it must be *said*, because
a `test_acc` of 0.91 on a dataset that is 91% one class is the most common way an
ML pipeline looks successful while having learned nothing.

Costs milliseconds — no fitting, just a summary statistic over the training
labels. No torch, no sklearn beyond what `metrics` already uses.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .protocols import Predictions
from .task import get_task_spec

log = logging.getLogger(__name__)

# Prefix for every baseline figure in `metrics.json`, so they sort together and
# can never be confused with the model's own.
PREFIX = "baseline_"


def _constant(value: Any, n: int, dtype: Any) -> np.ndarray:
    return np.full(n, value, dtype=dtype)


def baseline_predictions(task: str, y_true: np.ndarray, train_y: np.ndarray | None) -> np.ndarray:
    """What the trivial predictor would have said for these rows.

    ``train_y`` is what the statistic is taken from — the training split, never
    the split being scored. Taking the majority class from the *test* labels would
    make the baseline stronger than anything achievable at training time, which
    inverts the comparison it exists to support.
    """
    spec = get_task_spec(task)
    labels = np.asarray(train_y if train_y is not None and len(train_y) else y_true)
    n = len(y_true)

    if spec.output_kind in ("labels", "probabilities", "token_labels"):
        values, counts = np.unique(labels, return_counts=True)
        return _constant(values[int(counts.argmax())], n, labels.dtype)

    if spec.output_kind == "series":
        # Seasonal naive: repeat the tail of the history. Season 1 (repeat the
        # last value) is the honest default when nothing declared a period.
        history = np.asarray(labels, dtype="float64")
        # `meta` is typed `object` so it cannot become a typed dumping ground;
        # narrowing belongs at the one place that reads this key.
        declared = spec.meta.get("seasonality")
        season = max(1, declared if isinstance(declared, int) else 1)
        season = min(season, len(history))
        tail = history[-season:]
        return np.resize(tail, n)

    return _constant(float(np.asarray(labels, dtype="float64").mean()), n, "float64")


def baseline_metrics(
    predictions: Predictions,
    task: str,
    *,
    train_y: np.ndarray | None = None,
) -> dict[str, float]:
    """``{"baseline_acc": …}`` — the same metrics, computed for the trivial model.

    Returns an empty dict rather than raising when the task has no meaningful
    trivial predictor. ``seq2seq`` is the case: "always emit the most common
    string" is not a baseline anybody would compare against, and inventing a
    number there would be worse than the silence.
    """
    if predictions.y_true is None:
        return {}

    spec = get_task_spec(task)
    if spec.output_kind == "text":
        # No sensible constant answer, and a bad one would be quoted.
        return {}

    y_true = np.asarray(predictions.y_true)
    if not len(y_true):
        return {}

    guess = baseline_predictions(task, y_true, train_y)
    # No probabilities: a constant predictor has no calibrated confidence, so the
    # proba-only metrics are skipped rather than fed something invented.
    scores = spec.compute(y_true, guess, None, prefix=PREFIX)
    return {name: float(value) for name, value in scores.items()}


def compare(metrics: dict[str, float], baseline: dict[str, float], task: str) -> bool:
    """Log how the model did against the baseline; return whether it won.

    The comparison is on the task's **primary metric** and in its declared
    direction, so "better" means better rather than larger — which for MAE, RMSE
    and MASE is the opposite thing.
    """
    if not baseline:
        return True

    spec = get_task_spec(task)
    model_key, baseline_key = f"test_{spec.primary_metric}", f"{PREFIX}{spec.primary_metric}"
    if model_key not in metrics or baseline_key not in baseline:
        return True

    model_score, trivial = metrics[model_key], baseline[baseline_key]
    if np.isnan(model_score) or np.isnan(trivial):
        return True

    beat = model_score > trivial if spec.direction == "max" else model_score < trivial
    if beat:
        log.info(
            "%s %.4f beats the trivial baseline (%.4f)", spec.primary_metric, model_score, trivial
        )
    else:
        log.warning(
            "the model does NOT beat the trivial baseline: %s %.4f vs %.4f. "
            "A constant prediction scores this well, so the model has learned little "
            "or nothing from the features.",
            spec.primary_metric,
            model_score,
            trivial,
        )
    return beat


__all__ = ["PREFIX", "baseline_metrics", "baseline_predictions", "compare"]
