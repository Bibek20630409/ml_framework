import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_mlflow_tracking_registry_and_reload(tabular_csv, make_config, tmp_path):
    """Train with the mlflow backend → a run is logged, a model version is
    registered, and it can be loaded back from the registry and predict."""
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    from ml_framework.core import Inferencer
    from ml_framework.pipeline import train

    # SQLite backend (registry-capable; file store is deprecated in MLflow 3.x).
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    artifacts = (tmp_path / "mlartifacts").as_uri()
    cfg = make_config(
        tabular_csv,
        "multiclass",
        **{
            "logging.backend": "mlflow",
            "logging.mlflow_tracking_uri": uri,
            "logging.mlflow_artifact_location": artifacts,
            "logging.mlflow_experiment": "test-exp",
            "logging.registered_model_name": "test-model",
        },
    )

    metrics = train(cfg)
    assert isinstance(metrics, dict) and metrics

    # A registered model version now exists.
    client = MlflowClient(tracking_uri=uri)
    versions = client.search_model_versions("name='test-model'")
    assert len(versions) >= 1
    version = versions[0].version

    # Load it back from the registry (by version) and predict.
    inf = Inferencer.from_registry("test-model", version, tracking_uri=uri)
    x = np.random.default_rng(0).normal(size=(4, inf.n_features)).astype("float32")
    preds = inf.predict(x)
    assert len(preds) == 4
