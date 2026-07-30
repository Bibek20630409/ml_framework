from pathlib import Path

import numpy as np
import pytest

from ml_framework.core import Inferencer
from ml_framework.pipeline import train


@pytest.mark.integration
@pytest.mark.parametrize(
    "task,csv_fixture",
    [
        ("multiclass", "tabular_csv"),
        ("binary", "binary_csv"),
        ("regression", "regression_csv"),
    ],
)
def test_train_produces_artifacts_and_infers(task, csv_fixture, make_config, request):
    csv = request.getfixturevalue(csv_fixture)
    cfg = make_config(csv, task)
    metrics = train(cfg)
    assert isinstance(metrics, dict) and metrics

    out = Path(cfg.output_dir)
    for artifact in ("model.ckpt", "scaler.pkl", "metadata.json", "report.txt", "predictions.csv"):
        assert (out / artifact).exists(), f"missing {artifact}"

    # Artifact-based inference round-trip.
    inf = Inferencer.from_artifacts(out)
    x = np.random.default_rng(0).normal(size=(5, inf.model.input_dim)).astype("float32")
    preds = inf.predict(x)
    assert len(preds) == 5

    if task != "regression":
        probs = inf.predict_proba(x)
        assert probs.shape[0] == 5
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


@pytest.mark.integration
def test_regression_small_dataset_does_not_crash(regression_csv, make_config):
    # The original framework crashed here (StratifiedKFold on continuous y).
    cfg = make_config(regression_csv, "regression")
    metrics = train(cfg)
    assert "test_mae" in metrics
