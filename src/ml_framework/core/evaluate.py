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

    metrics         {"test_acc"} | {"test_mae", "test_rmse"} | {"test_mase", …}
    report.txt      "Accuracy: …" + sklearn classification_report(digits=4)
    predictions.csv [index,] label, prediction, prob_class_{i} (multiclass) | probability (binary)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import accuracy, mae, mase, rmse, smape
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
    elif spec.output_kind == "series":
        metrics = _write_forecast(out, labels, preds)
    else:
        metrics = _write_regression(out, labels, preds)

    _write_predictions(out, task, labels, preds, probs, predictions.index)
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


def _write_forecast(out: Path, labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """The scaled metrics first, because the unscaled ones are uninterpretable.

    "MAE 4.2" says nothing without knowing whether the series runs in single digits
    or millions, which is why MASE and sMAPE lead and MAE/RMSE sit underneath for
    anyone who wants them in the original units.

    The report deliberately renders **no verdict**. It is widely repeated that
    MASE < 1 means "better than naive", and over a multi-step horizon that reading
    is wrong — see :func:`ml_framework.core.metrics.mase`. Printing a judgement the
    number does not support would be worse than printing nothing, so the note
    points at the honest comparison instead: run ``ts.naive`` on the same split.
    """
    scores = {
        "test_mase": mase(labels, preds),
        "test_smape": smape(labels, preds),
        "test_mae": mae(labels, preds),
        "test_rmse": rmse(labels, preds),
    }
    log.info(
        "Forecast MASE: %.4f | sMAPE: %.2f%% | MAE: %.4f",
        scores["test_mase"],
        scores["test_smape"],
        scores["test_mae"],
    )
    report = "\n".join(
        [
            f"MASE:  {scores['test_mase']:.4f}",
            f"sMAPE: {scores['test_smape']:.2f}%",
            f"MAE:   {scores['test_mae']:.4f}",
            f"RMSE:  {scores['test_rmse']:.4f}",
            "",
            "MASE scales the error by the series' average step change, so it is",
            "comparable across series. It is NOT a pass mark: over a multi-step",
            "horizon values above 1 are normal. To judge whether this model earns",
            "its keep, train `ts.naive` on the same split and compare.",
        ]
    )
    (out / REPORT_NAME).write_text(report, encoding="utf-8")
    return scores


def _write_predictions(
    out: Path,
    task: str,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray | None,
    index: np.ndarray | None = None,
) -> None:
    """The v1 column schema, unchanged — plus ``index`` where it means something.

    Binary keeps a single ``probability`` column holding P(class 1), even though
    the estimator's canonical ``predict_proba`` output is two columns — the file
    is a published contract and the second column is redundant.

    Forecast rows lead with the timestamp: a forecast without the point in time it
    belongs to is not interpretable, which is why `Predictions` carries an index at
    all.
    """
    frame = pd.DataFrame({"label": labels, "prediction": preds})
    if index is not None and len(index) == len(labels):
        frame.insert(0, "index", _readable_index(index))
    if probs is not None:
        if task == "multiclass" and probs.ndim == 2:
            for i in range(probs.shape[1]):
                frame[f"prob_class_{i}"] = probs[:, i]
        elif task == "binary":
            frame["probability"] = probs[:, 1] if probs.ndim == 2 else probs
    frame.to_csv(out / PREDICTIONS_NAME, index=False)
    log.info("predictions saved → %s", out / PREDICTIONS_NAME)


def _readable_index(index: np.ndarray) -> np.ndarray:
    """Nanosecond epochs back to timestamps; anything else left alone.

    The time-series source stores a parsed datetime column as int64 epochs so the
    bundle stays numpy-only. Writing those raw would make the column unreadable by
    the humans the file exists for.
    """
    arr = np.asarray(index)
    if arr.dtype.kind == "i" and arr.size and int(arr.max()) > 10**15:
        return pd.to_datetime(arr).astype(str).to_numpy()
    return arr
