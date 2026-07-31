"""serving/schemas.py — manifest-derived request/response models.

The named-input path is the one that carries weight: silent column reordering is
the most common serving defect there is, and feature names are in the signature
precisely so the server can prevent it rather than score transposed data.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from ml_framework.serving.schemas import (
    ForecastRequest,
    ImageRequest,
    PayloadError,
    TabularRequest,
    TextRequest,
    request_model,
    response_model,
    supports_drift,
)

FEATURES = ["f0", "f1", "f2"]


# ── The table ─────────────────────────────────────────────
@pytest.mark.unit
@pytest.mark.parametrize(
    ("kind", "model"),
    [
        ("tabular", TabularRequest),
        ("text", TextRequest),
        ("image", ImageRequest),
        ("timeseries", ForecastRequest),
    ],
)
def test_the_request_schema_is_chosen_by_data_kind(kind, model):
    """v1 had one hardcoded pair for every model the framework could ever serve,
    so /docs described the framework rather than the deployed model."""
    assert request_model(kind) is model
    assert response_model(kind) is not None


@pytest.mark.unit
def test_an_unknown_data_kind_is_refused_by_name():
    with pytest.raises(PayloadError, match="no request schema for data kind 'audio'"):
        request_model("audio")


@pytest.mark.unit
def test_drift_is_declared_tabular_only():
    """PSI over token ids or pixel bytes is a number without a meaning."""
    assert supports_drift("tabular")
    assert not supports_drift("text")
    assert not supports_drift("image")
    assert not supports_drift("timeseries")


# ── Tabular: positional vs named ──────────────────────────
@pytest.mark.unit
def test_positional_instances_stay_supported():
    """The v1 contract. Existing clients send this and must keep working."""
    req = TabularRequest(instances=[[1.0, 2.0, 3.0]])
    assert np.array_equal(req.to_array(FEATURES), np.array([[1.0, 2.0, 3.0]], dtype="float32"))


@pytest.mark.unit
def test_named_inputs_are_reordered_into_training_feature_order():
    """The reason feature names are in the manifest signature at all."""
    req = TabularRequest(inputs=[{"f2": 3.0, "f0": 1.0, "f1": 2.0}])
    assert np.array_equal(req.to_array(FEATURES), np.array([[1.0, 2.0, 3.0]], dtype="float32"))


@pytest.mark.unit
def test_a_missing_named_feature_names_the_column():
    req = TabularRequest(inputs=[{"f0": 1.0, "f1": 2.0}])
    with pytest.raises(PayloadError, match=r"row 0 is missing features: \['f2'\]"):
        req.to_array(FEATURES)


@pytest.mark.unit
def test_an_unknown_named_feature_names_the_column():
    req = TabularRequest(inputs=[{"f0": 1.0, "f1": 2.0, "f2": 3.0, "f9": 4.0}])
    with pytest.raises(PayloadError, match=r"row 0 has unknown features: \['f9'\]"):
        req.to_array(FEATURES)


@pytest.mark.unit
def test_named_inputs_are_refused_when_the_bundle_records_no_feature_names():
    """Better than guessing dict order, which would be the silent-reordering bug
    this whole path exists to prevent."""
    req = TabularRequest(inputs=[{"a": 1.0}])
    with pytest.raises(PayloadError, match="records no feature names"):
        req.to_array([])


@pytest.mark.unit
def test_exactly_one_of_instances_or_inputs_is_required():
    with pytest.raises(ValidationError, match="exactly one"):
        TabularRequest(instances=[[1.0]], inputs=[{"f0": 1.0}])
    with pytest.raises(ValidationError, match="exactly one"):
        TabularRequest()


@pytest.mark.unit
def test_an_empty_body_is_rejected():
    with pytest.raises(ValidationError):
        TabularRequest(instances=[])


@pytest.mark.unit
def test_unknown_top_level_keys_are_rejected():
    with pytest.raises(ValidationError):
        TabularRequest(instances=[[1.0]], nonsense=True)


# ── Other kinds ───────────────────────────────────────────
@pytest.mark.unit
def test_text_requests_take_raw_strings():
    assert TextRequest(inputs=["hello", "world"]).n_rows == 2


@pytest.mark.unit
def test_image_items_require_exactly_one_source():
    ImageRequest(inputs=[{"b64": "aGk="}])
    ImageRequest(inputs=[{"url": "https://example.invalid/x.png"}])
    with pytest.raises(ValidationError, match="exactly one"):
        ImageRequest(inputs=[{"b64": "aGk=", "url": "https://example.invalid/x.png"}])


@pytest.mark.unit
def test_a_forecast_request_takes_a_horizon_not_a_feature_matrix():
    """`predict(X)` is a lying signature for forecasting, which is why the payload
    is parameterized by data kind rather than shared."""
    req = ForecastRequest(horizon=7, history=[1.0, 2.0, 3.0])
    assert req.n_rows == 7
    with pytest.raises(ValidationError):
        ForecastRequest(horizon=0)
