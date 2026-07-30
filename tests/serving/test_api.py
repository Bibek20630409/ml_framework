import pytest
from fastapi.testclient import TestClient

from ml_framework.pipeline import train
from ml_framework.serving.api import create_app


@pytest.fixture
def trained_app(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")
    train(cfg)
    app = create_app(cfg.output_dir)
    return app, cfg


@pytest.mark.serving
def test_health(trained_app):
    app, _ = trained_app
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["task"] == "multiclass"


@pytest.mark.serving
def test_predict(trained_app):
    app, _ = trained_app
    with TestClient(app) as client:
        body = {"instances": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]}
        r = client.post("/predict", json=body)
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 1


@pytest.mark.serving
def test_predict_proba_sums_to_one(trained_app):
    app, _ = trained_app
    with TestClient(app) as client:
        body = {"instances": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]}
        r = client.post("/predict_proba", json=body)
        assert r.status_code == 200
        probs = r.json()["probabilities"][0]
        assert abs(sum(probs) - 1.0) < 1e-4


@pytest.mark.serving
def test_wrong_feature_count_rejected(trained_app):
    app, _ = trained_app
    with TestClient(app) as client:
        r = client.post("/predict", json={"instances": [[0.1, 0.2]]})
        assert r.status_code == 422
