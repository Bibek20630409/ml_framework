"""The serving layer after P3: manifest-derived, backend-agnostic, and
safe to build more than once per process.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml_framework.pipeline import train
from ml_framework.serving.api import create_app

pytestmark = pytest.mark.serving

FEATURES = [f"f{i}" for i in range(6)]
ROW = dict.fromkeys(FEATURES, 0.25)


@pytest.fixture
def trained(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")
    train(cfg)
    return cfg


@pytest.fixture
def client(trained):
    with TestClient(create_app(trained.runtime.output_dir)) as c:
        yield c


# ── The Prometheus fix ────────────────────────────────────
def test_create_app_twice_in_one_process(trained):
    """v1 built its Counter and Gauge at module scope, so the *second*
    `create_app` raised "Duplicated timeseries in CollectorRegistry". That became
    load-bearing the moment anything created apps repeatedly — tuning, a test
    suite, or serving more than one model from one process.
    """
    first = create_app(trained.runtime.output_dir)
    second = create_app(trained.runtime.output_dir)  # must not raise
    with TestClient(first) as c1, TestClient(second) as c2:
        assert c1.get("/health").status_code == 200
        assert c2.get("/health").status_code == 200


def test_collectors_are_the_same_objects_across_calls():
    from ml_framework.serving.metrics import get_collectors

    pytest.importorskip("prometheus_client")
    a, b = get_collectors(), get_collectors()
    assert a.predictions is b.predictions
    assert a.drift is b.drift


def test_collectors_survive_a_cache_reset():
    """Recreating after a reset must find the existing collector, not duplicate it."""
    from ml_framework.serving import metrics

    pytest.importorskip("prometheus_client")
    first = metrics.get_collectors()
    metrics.reset_for_tests()
    second = metrics.get_collectors()  # goes down the lookup path this time
    assert second.predictions is first.predictions


# ── /health is manifest-derived ───────────────────────────
def test_health_keeps_its_v1_keys_and_adds_what_is_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"  # v1 key
    assert body["task"] == "multiclass"  # v1 key
    assert body["model"] == "mlp"
    assert body["backend"] == "lightning"
    assert body["data_kind"] == "tabular"
    assert body["n_features"] == 6
    assert body["bundle_version"] == 2


# ── The two tabular request shapes ────────────────────────
def test_positional_and_named_rows_agree(client):
    """Named is the documented production contract; positional is v1 and stays."""
    positional = client.post("/predict", json={"instances": [[0.25] * 6]})
    named = client.post("/predict", json={"inputs": [ROW]})
    assert positional.status_code == 200 and named.status_code == 200
    assert positional.json()["predictions"] == named.json()["predictions"]


def test_named_rows_are_order_independent(client):
    """The defect this prevents: a client sends columns in a different order and
    the server silently scores transposed data."""
    shuffled = {k: ROW[k] for k in reversed(FEATURES)}
    a = client.post("/predict", json={"inputs": [ROW]}).json()
    b = client.post("/predict", json={"inputs": [shuffled]}).json()
    assert a["predictions"] == b["predictions"]


def test_a_missing_named_feature_is_a_422_naming_the_column(client):
    body = {"inputs": [{k: v for k, v in ROW.items() if k != "f3"}]}
    res = client.post("/predict", json=body)
    assert res.status_code == 422
    assert "f3" in res.json()["detail"]


def test_the_wrong_feature_count_is_still_a_422(client):
    res = client.post("/predict", json={"instances": [[0.1, 0.2]]})
    assert res.status_code == 422
    assert "expected 6 features" in res.json()["detail"]


def test_a_body_with_neither_key_is_a_422(client):
    assert client.post("/predict", json={"rows": [[1.0]]}).status_code == 422


# ── Labels and confidence ─────────────────────────────────
def test_predictions_carry_class_labels_when_the_bundle_records_them(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass", **{"data.class_names": ["low", "mid", "high"]})
    train(cfg)
    with TestClient(create_app(cfg.runtime.output_dir)) as c:
        body = c.post("/predict", json={"inputs": [ROW]}).json()
        assert body["labels"][0] in {"low", "mid", "high"}
        assert c.post("/predict_proba", json={"inputs": [ROW]}).json()["classes"] == [
            "low",
            "mid",
            "high",
        ]


def test_predict_with_confidence_is_exposed(client):
    """`Inferencer.predict_with_confidence` survives P3 and gets a route."""
    res = client.post("/predict_with_confidence", json={"instances": [[0.25] * 6]})
    assert res.status_code == 200
    body = res.json()
    assert len(body["predictions"]) == 1
    assert 0.0 <= body["confidence"][0] <= 1.0


def test_regression_refuses_probabilities_from_the_manifest(regression_csv, make_config):
    """The 400 now comes from signature.output.kind, not from `task ==
    "regression"` — a check that was wrong for every task not yet invented."""
    cfg = make_config(regression_csv, "regression")
    train(cfg)
    with TestClient(create_app(cfg.runtime.output_dir)) as c:
        res = c.post("/predict_proba", json={"instances": [[0.25] * 5]})
        assert res.status_code == 400
        assert "not probabilities" in res.json()["detail"]
        assert c.post("/predict", json={"instances": [[0.25] * 5]}).status_code == 200


# ── Backend-agnostic ──────────────────────────────────────
def test_the_same_app_serves_a_gbdt_bundle(tabular_csv, make_config):
    """Nothing in the serving layer knows which backend produced the bundle."""
    pytest.importorskip("xgboost")
    cfg = make_config(
        tabular_csv,
        "multiclass",
        model="xgboost",
        **{"fit.params.n_estimators": 20, "fit.params.early_stopping_rounds": 0},
    )
    train(cfg)
    with TestClient(create_app(cfg.runtime.output_dir)) as c:
        assert c.get("/health").json()["backend"] == "gbdt"
        res = c.post("/predict", json={"inputs": [ROW]})
        assert res.status_code == 200
        probs = c.post("/predict_proba", json={"inputs": [ROW]}).json()["probabilities"]
        assert np.isclose(sum(probs[0]), 1.0, atol=1e-4)
