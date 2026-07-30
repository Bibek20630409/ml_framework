"""
core/evaluate.py
────────────────
Held-out evaluation, from **arrays**. Writes ``report.txt``, ``predictions.csv``
and (for classification) ``confusion_matrix.txt``; returns a metrics dict.

v1 ran its own torch inference loop here with a three-way ``config.task`` branch
inside it — the third copy of postprocessing logic that also existed in
``lit_model._shared_step`` and ``inference.py``. The loop now belongs to the
backend (``predict_split`` returns :class:`Predictions`) and the branching to the
estimator, which leaves this module with the part that was always
framework-agnostic: turning predictions into numbers and files.

**No torch, no Lightning.** That is what lets a GBDT and a Prophet model be
evaluated by this same code path.

The output contract is preserved exactly, because these files are consumed by DVC
metrics, the Airflow quality gate and humans comparing runs across the refactor:

    metrics         {"test_acc"} | {"test_mae", "test_rmse"}
    report.txt      "Accuracy: …" + sklearn classification_report(digits=4)
    predictions.csv label, prediction, prob_class_{i} (multiclass) | probability (binary)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import accuracy, mae, rmse
from .protocols import Predictions
from .task import get_task_spec

log = logging.getLogger(__name__)

REPORT_NAME = "report.txt"
PREDICTIONS_NAME = "predictions.csv"
CONFUSION_NAME = "confusion_matrix.txt"


def evaluate(
    predictions: Predictions,
    task: str,
    *,
    output_dir: str | Path,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Score ``predictions`` for ``task`` and write the report files.

    Raises rather than guessing when the predictions carry no labels: an
    evaluation without ground truth is a prediction run, and silently returning an
    empty metrics dict would let a broken pipeline look successful.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if predictions.y_true is None:
        raise ValueError("evaluate() needs labelled predictions (Predictions.y_true is None)")

    spec = get_task_spec(task)
    labels = np.asarray(predictions.y_true)
    preds = np.asarray(predictions.y_pred)
    probs = None if predictions.y_prob is None else np.asarray(predictions.y_prob)

    if spec.is_classification:
        metrics = _write_classification(out, labels, preds, class_names)
    else:
        metrics = _write_regression(out, labels, preds)

    _write_predictions(out, task, labels, preds, probs)
    return metrics


def _write_classification(
    out: Path, labels: np.ndarray, preds: np.ndarray, class_names: list[str] | None
) -> dict[str, float]:
    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(
        labels, preds, target_names=class_names, digits=4, zero_division=0
    )
    acc = accuracy(labels, preds)
    log.info("Test accuracy: %.4f\n%s", acc, report)
    (out / REPORT_NAME).write_text(f"Accuracy: {acc:.4f}\n\n{report}", encoding="utf-8")
    (out / CONFUSION_NAME).write_text(str(confusion_matrix(labels, preds)), encoding="utf-8")
    return {"test_acc": acc}


def _write_regression(out: Path, labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    test_mae, test_rmse = mae(labels, preds), rmse(labels, preds)
    log.info("Test MAE: %.4f | RMSE: %.4f", test_mae, test_rmse)
    (out / REPORT_NAME).write_text(f"MAE: {test_mae:.4f}\nRMSE: {test_rmse:.4f}", encoding="utf-8")
    return {"test_mae": test_mae, "test_rmse": test_rmse}


def _write_predictions(
    out: Path,
    task: str,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray | None,
) -> None:
    """The v1 column schema, unchanged.

    Binary keeps a single ``probability`` column holding P(class 1), even though
    the estimator's canonical ``predict_proba`` output is two columns — the file
    is a published contract and the second column is redundant.
    """
    frame = pd.DataFrame({"label": labels, "prediction": preds})
    if probs is not None:
        if task == "multiclass" and probs.ndim == 2:
            for i in range(probs.shape[1]):
                frame[f"prob_class_{i}"] = probs[:, i]
        elif task == "binary":
            frame["probability"] = probs[:, 1] if probs.ndim == 2 else probs
    frame.to_csv(out / PREDICTIONS_NAME, index=False)
    log.info("predictions saved → %s", out / PREDICTIONS_NAME)
