"""evaluate() consumes arrays, writes the v1 file contract, and imports no torch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_framework.core.evaluate import evaluate
from ml_framework.core.protocols import Predictions


@pytest.mark.unit
def test_evaluate_does_not_import_torch():
    """The whole point of the split: a GBDT or Prophet model is evaluated by this
    same module, and the serving image can drop torch entirely."""
    import ast
    import pathlib

    import ml_framework.core

    # Located by path, not by module attribute: `core.evaluate` is the re-exported
    # *function*, which shadows the submodule of the same name.
    source = pathlib.Path(ml_framework.core.__file__).with_name("evaluate.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not imported & {"torch", "pytorch_lightning", "torchmetrics"}


# ── Classification ────────────────────────────────────────
@pytest.mark.unit
def test_multiclass_writes_the_v1_metric_key_and_files(tmp_path):
    labels = np.array([0, 1, 2, 1, 0, 2])
    preds = np.array([0, 1, 2, 2, 0, 2])
    probs = np.eye(3)[preds]

    metrics = evaluate(
        Predictions(y_true=labels, y_pred=preds, y_prob=probs),
        "multiclass",
        output_dir=tmp_path,
    )

    assert set(metrics) == {"test_acc"}
    assert metrics["test_acc"] == pytest.approx(5 / 6)
    assert (tmp_path / "report.txt").read_text(encoding="utf-8").startswith("Accuracy: 0.8333")
    assert (tmp_path / "confusion_matrix.txt").exists()


@pytest.mark.unit
def test_multiclass_predictions_csv_keeps_a_column_per_class(tmp_path):
    labels = np.array([0, 1, 2])
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    evaluate(
        Predictions(y_true=labels, y_pred=probs.argmax(axis=1), y_prob=probs),
        "multiclass",
        output_dir=tmp_path,
    )
    frame = pd.read_csv(tmp_path / "predictions.csv")
    assert list(frame.columns) == [
        "label",
        "prediction",
        "prob_class_0",
        "prob_class_1",
        "prob_class_2",
    ]
    assert frame["prob_class_1"].tolist() == pytest.approx([0.2, 0.8, 0.2])


@pytest.mark.unit
def test_binary_predictions_csv_keeps_one_probability_column(tmp_path):
    """The estimator's canonical predict_proba is two columns; the published file
    schema is one, holding P(class 1)."""
    labels = np.array([0, 1, 1, 0])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]])
    evaluate(
        Predictions(y_true=labels, y_pred=(probs[:, 1] > 0.5).astype(int), y_prob=probs),
        "binary",
        output_dir=tmp_path,
    )
    frame = pd.read_csv(tmp_path / "predictions.csv")
    assert list(frame.columns) == ["label", "prediction", "probability"]
    assert frame["probability"].tolist() == pytest.approx([0.1, 0.8, 0.6, 0.3])


@pytest.mark.unit
def test_class_names_reach_the_sklearn_report(tmp_path):
    labels = np.array([0, 1, 0, 1])
    evaluate(
        Predictions(y_true=labels, y_pred=labels),
        "binary",
        output_dir=tmp_path,
        class_names=["negative", "positive"],
    )
    report = (tmp_path / "report.txt").read_text(encoding="utf-8")
    assert "negative" in report and "positive" in report


# ── Regression ────────────────────────────────────────────
@pytest.mark.unit
def test_regression_writes_mae_and_rmse_and_no_confusion_matrix(tmp_path):
    labels = np.array([1.0, 2.0, 3.0])
    preds = np.array([1.5, 2.0, 2.0])

    metrics = evaluate(Predictions(y_true=labels, y_pred=preds), "regression", output_dir=tmp_path)

    assert set(metrics) == {"test_mae", "test_rmse"}
    assert metrics["test_mae"] == pytest.approx(0.5)
    assert metrics["test_rmse"] == pytest.approx(np.sqrt((0.25 + 0 + 1) / 3))
    assert (tmp_path / "report.txt").read_text(encoding="utf-8").startswith("MAE: 0.5000")
    assert not (tmp_path / "confusion_matrix.txt").exists()


@pytest.mark.unit
def test_regression_predictions_csv_has_no_probability_column(tmp_path):
    labels = np.array([1.0, 2.0])
    evaluate(
        Predictions(y_true=labels, y_pred=np.array([1.1, 2.2])), "regression", output_dir=tmp_path
    )
    assert list(pd.read_csv(tmp_path / "predictions.csv").columns) == ["label", "prediction"]


# ── Guards ────────────────────────────────────────────────
@pytest.mark.unit
def test_unlabelled_predictions_are_refused_not_silently_scored(tmp_path):
    """A prediction run is not an evaluation; returning empty metrics would let a
    broken pipeline report success."""
    with pytest.raises(ValueError, match="labelled predictions"):
        evaluate(Predictions(y_true=None, y_pred=np.array([1, 0])), "binary", output_dir=tmp_path)


@pytest.mark.unit
def test_an_unregistered_task_says_so(tmp_path):
    from ml_framework.core.task import UnknownTaskError

    with pytest.raises(UnknownTaskError):
        evaluate(
            Predictions(y_true=np.array([1.0]), y_pred=np.array([1.0])),
            "multilabel",
            output_dir=tmp_path,
        )
