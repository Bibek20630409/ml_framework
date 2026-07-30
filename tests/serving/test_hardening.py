import pytest
from fastapi.testclient import TestClient

from ml_framework.pipeline import train
from ml_framework.serving.api import create_app

BODY = {"instances": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]}


@pytest.mark.serving
def test_auth_required_when_key_set(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")
    train(cfg)
    app = create_app(cfg.output_dir, api_key="secret123", rate_limit=None)
    with TestClient(app) as client:
        # no key → 401
        assert client.post("/predict", json=BODY).status_code == 401
        # wrong key → 401
        assert client.post("/predict", json=BODY, headers={"X-API-Key": "nope"}).status_code == 401
        # right key → 200
        r = client.post("/predict", json=BODY, headers={"X-API-Key": "secret123"})
        assert r.status_code == 200
        # health stays open (no auth)
        assert client.get("/health").status_code == 200


@pytest.mark.serving
def test_auth_disabled_when_no_key(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")
    train(cfg)
    app = create_app(cfg.output_dir, api_key=None, rate_limit=None)
    with TestClient(app) as client:
        assert client.post("/predict", json=BODY).status_code == 200


@pytest.mark.serving
def test_rate_limit_returns_429(tabular_csv, make_config):
    pytest.importorskip("slowapi")
    cfg = make_config(tabular_csv, "multiclass")
    train(cfg)
    app = create_app(cfg.output_dir, api_key=None, rate_limit="3/minute")
    with TestClient(app) as client:
        codes = [client.post("/predict", json=BODY).status_code for _ in range(5)]
        assert 429 in codes  # burst beyond 3/minute is throttled


@pytest.mark.serving
def test_request_size_cap(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")
    train(cfg)
    app = create_app(cfg.output_dir, api_key=None, rate_limit=None, max_instances=2)
    with TestClient(app) as client:
        big = {"instances": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] * 3}
        assert client.post("/predict", json=big).status_code == 413
