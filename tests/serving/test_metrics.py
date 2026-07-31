import pytest
from fastapi.testclient import TestClient

from ml_framework.pipeline import train
from ml_framework.serving.api import create_app


@pytest.fixture
def trained_app(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")
    train(cfg)
    return create_app(cfg.runtime.output_dir)


@pytest.mark.serving
def test_metrics_endpoint_exposed(trained_app):
    pytest.importorskip("prometheus_fastapi_instrumentator")
    with TestClient(trained_app) as client:
        # generate some traffic so counters are non-empty
        client.post("/predict", json={"instances": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]})
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.text
        # request metrics from the instrumentator + our custom prediction counter
        assert "http_request" in body
        assert "mlf_predictions_total" in body


@pytest.mark.serving
def test_drift_endpoint(trained_app):
    with TestClient(trained_app) as client:
        # feed enough traffic to fill the drift window past min_samples
        for _ in range(40):
            client.post("/predict", json={"instances": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]})
        r = client.get("/drift")
        assert r.status_code == 200
        psi = r.json()["psi"]
        # reference has 6 features → per-feature PSI reported
        assert set(psi) == {f"f{i}" for i in range(6)}
