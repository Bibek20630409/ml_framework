"""
serving/schemas.py
──────────────────
Request/response models selected from the **bundle manifest**, keyed by
``data.kind``.

v1 had one hardcoded pair (``{"instances": [[float]]}`` → ``{"predictions": []}``)
for every model the framework could ever serve, so the OpenAPI docs described the
*framework* rather than the deployed model. Choosing from a table keyed by data
kind makes ``/docs`` describe this specific model — which is a real feature, not
tidiness: it is what makes a deployed endpoint self-documenting.

**Two tabular request shapes, and the named one is the recommendation.**
``{"instances": [[0.1, 0.2]]}`` is positional and stays supported (it is the v1
contract and existing clients send it). ``{"inputs": [{"f0": 0.1, "f1": 0.2}]}``
is keyed by feature name, and the manifest signature carries those names, so the
server can reorder columns to the training order instead of silently scoring
transposed data. Silent column reordering is the most common serving defect there
is; being able to prevent it is why feature names are in the signature at all.

Kinds beyond tabular have their request shapes declared here but reach a model
only when the corresponding source lands. Declaring them now keeps the table
honest about what a kind means rather than letting `/predict` accept anything.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from ..core.types import FrameworkError


class PayloadError(FrameworkError):
    """A request body could not be turned into model input."""


# ── Tabular ───────────────────────────────────────────────
class TabularRequest(BaseModel):
    """Positional ``instances`` or named ``inputs`` — exactly one of them."""

    model_config = {"extra": "forbid"}

    instances: list[list[float]] | None = Field(
        default=None,
        description="Positional rows, in training feature order. Back-compatible.",
    )
    inputs: list[dict[str, Any]] | None = Field(
        default=None,
        description="Named rows keyed by feature name. Recommended: order-independent.",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> TabularRequest:
        if (self.instances is None) == (self.inputs is None):
            raise ValueError("provide exactly one of 'instances' or 'inputs'")
        if not (self.instances or self.inputs):
            raise ValueError("the request contains no rows")
        return self

    @property
    def n_rows(self) -> int:
        return len(self.instances or self.inputs or [])

    def to_array(self, feature_names: list[str]) -> np.ndarray:
        """Build the model input matrix, reordering named rows to training order.

        The named path is the one that can fail usefully: a missing or unknown
        feature is a 422 naming the column, where the positional path can only
        check the count.
        """
        if self.instances is not None:
            arr = np.asarray(self.instances, dtype="float32")
            if arr.ndim != 2:
                raise PayloadError("'instances' must be a 2D array")
            return arr

        rows = self.inputs or []
        if not feature_names:
            raise PayloadError(
                "this bundle records no feature names, so named 'inputs' cannot be "
                "ordered; send positional 'instances' instead"
            )
        expected = set(feature_names)
        out = np.empty((len(rows), len(feature_names)), dtype="float32")
        for i, row in enumerate(rows):
            missing = expected - set(row)
            if missing:
                raise PayloadError(f"row {i} is missing features: {sorted(missing)}")
            unknown = set(row) - expected
            if unknown:
                raise PayloadError(f"row {i} has unknown features: {sorted(unknown)}")
            out[i] = [row[name] for name in feature_names]
        return out

    def to_model_input(self, feature_names: list[str]) -> np.ndarray:
        """The generic hook ``/predict`` calls. For tabular it is the matrix."""
        return self.to_array(feature_names)


class TextRequest(BaseModel):
    """Raw strings. Tokenization is the *bundle's* job, not the caller's.

    The alternative — asking clients to send token ids — would make every client
    responsible for using the right vocabulary, which is the train/serve skew the
    tokenizer-in-the-bundle design exists to prevent. It would also make the
    endpoint impossible to use from curl.
    """

    model_config = {"extra": "forbid"}

    inputs: list[str] = Field(..., min_length=1, description="Raw text, one per row.")

    @property
    def n_rows(self) -> int:
        return len(self.inputs)

    def to_model_input(self, feature_names: list[str]) -> list[str]:
        """Strings, unchanged.

        ``feature_names`` is accepted and ignored so the hook has one signature
        across kinds; a text bundle records no feature names, because a token
        sequence has no columns.
        """
        return list(self.inputs)


class ImageItem(BaseModel):
    model_config = {"extra": "forbid"}

    b64: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> ImageItem:
        if (self.b64 is None) == (self.url is None):
            raise ValueError("provide exactly one of 'b64' or 'url'")
        return self


class ImageRequest(BaseModel):
    model_config = {"extra": "forbid"}

    inputs: list[ImageItem] = Field(..., min_length=1)

    @property
    def n_rows(self) -> int:
        return len(self.inputs)


class ForecastRequest(BaseModel):
    """``predict(X)`` is a lying signature for forecasting — a horizon is not a
    feature matrix, which is why the payload varies by kind rather than pretending
    otherwise."""

    model_config = {"extra": "forbid"}

    horizon: int = Field(..., ge=1, description="Steps to forecast beyond the history.")
    history: list[float] | None = None
    exog: list[dict[str, Any]] | None = None

    @property
    def n_rows(self) -> int:
        return self.horizon


# ── Responses ─────────────────────────────────────────────
class PredictResponse(BaseModel):
    predictions: list[float]


class LabelledPredictResponse(BaseModel):
    """Adds the human-readable label when the bundle records class names."""

    predictions: list[float]
    labels: list[str] | None = None


class ProbaResponse(BaseModel):
    probabilities: list[list[float]]
    classes: list[str] | None = None


class ConfidenceResponse(BaseModel):
    predictions: list[int]
    confidence: list[float]
    labels: list[str] | None = None


class ForecastResponse(BaseModel):
    forecast: list[float]
    index: list[str] | None = None
    lower: list[float] | None = None
    upper: list[float] | None = None


# ── The table ─────────────────────────────────────────────
DataKindLiteral = Literal["tabular", "image", "text", "timeseries"]

REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "tabular": TabularRequest,
    "text": TextRequest,
    "image": ImageRequest,
    "timeseries": ForecastRequest,
}

RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "tabular": LabelledPredictResponse,
    "text": LabelledPredictResponse,
    "image": LabelledPredictResponse,
    "timeseries": ForecastResponse,
}

# Drift is a distributional comparison over named numeric features. Computing PSI
# over token ids or pixel bytes would produce a number with no meaning, so /drift
# answers 501 for those kinds instead — see api.py.
DRIFT_CAPABLE_KINDS: frozenset[str] = frozenset({"tabular"})


def request_model(data_kind: str) -> type[BaseModel]:
    try:
        return REQUEST_MODELS[data_kind]
    except KeyError:
        raise PayloadError(
            f"no request schema for data kind '{data_kind}'. Known: {sorted(REQUEST_MODELS)}"
        ) from None


def response_model(data_kind: str) -> type[BaseModel]:
    try:
        return RESPONSE_MODELS[data_kind]
    except KeyError:
        raise PayloadError(
            f"no response schema for data kind '{data_kind}'. Known: {sorted(RESPONSE_MODELS)}"
        ) from None


def supports_drift(data_kind: str) -> bool:
    return data_kind in DRIFT_CAPABLE_KINDS


__all__ = [
    "DRIFT_CAPABLE_KINDS",
    "REQUEST_MODELS",
    "RESPONSE_MODELS",
    "ConfidenceResponse",
    "ForecastRequest",
    "ForecastResponse",
    "ImageItem",
    "ImageRequest",
    "LabelledPredictResponse",
    "PayloadError",
    "PredictResponse",
    "ProbaResponse",
    "TabularRequest",
    "TextRequest",
    "request_model",
    "response_model",
    "supports_drift",
]
