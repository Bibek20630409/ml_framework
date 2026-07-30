import pandas as pd
import pytest

from ml_framework.monitoring.model_quality import evaluate_against_labels


def _write(tmp_path, preds, labels):
    p = tmp_path / "preds.csv"
    lbl = tmp_path / "labels.csv"
    pd.DataFrame(preds).to_csv(p, index=False)
    pd.DataFrame(labels).to_csv(lbl, index=False)
    return str(p), str(lbl)


@pytest.mark.unit
def test_classification_accuracy(tmp_path):
    p, lbl = _write(
        tmp_path,
        {"id": [1, 2, 3, 4], "prediction": [0, 1, 2, 0]},
        {"id": [1, 2, 3, 4], "label": [0, 1, 2, 1]},  # last one wrong → 3/4
    )
    m = evaluate_against_labels(p, lbl, "multiclass")
    assert m["n"] == 4
    assert m["live_accuracy"] == 0.75


@pytest.mark.unit
def test_regression_errors(tmp_path):
    p, lbl = _write(
        tmp_path,
        {"id": [1, 2], "prediction": [1.0, 2.0]},
        {"id": [1, 2], "label": [1.0, 4.0]},  # errors 0 and 2
    )
    m = evaluate_against_labels(p, lbl, "regression")
    assert m["n"] == 2
    assert m["live_mae"] == 1.0


@pytest.mark.unit
def test_no_overlap_raises(tmp_path):
    p, lbl = _write(
        tmp_path,
        {"id": [1], "prediction": [0]},
        {"id": [99], "label": [0]},
    )
    with pytest.raises(ValueError):
        evaluate_against_labels(p, lbl, "multiclass")
