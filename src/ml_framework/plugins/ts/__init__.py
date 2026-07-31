"""The time-series plugins: seasonal-naive, ARIMA, Prophet, LSTM.

**Three of them ride the `forecast` backend and one rides `lightning`**, and that
split is the payload declaration doing its job rather than an inconsistency:

* ``ts.naive``/``ts.arima``/``ts.prophet`` declare ``accepts={"series"}``. They are
  handed the ordered values and asked what comes next.
* ``ts.lstm`` declares ``accepts={"arrays"}``. It trains in mini-batches over
  epochs exactly as an MLP does, so it belongs on the loop that already exists.

The time-series source reads those declarations and produces the matching payload,
which is why one source serves both without a config flag.

``ts.naive`` deliberately has **no requirements**. MASE is defined against it — a
MASE of 1.0 *is* the seasonal-naive forecast — so it has to be available wherever
forecasting is, including an install with neither prophet nor statsmodels.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from ...core.plugins import ModelSpec
from ...core.protocols import Categorical, Float, Int
from ...core.registry import register_model_spec
from ...core.types import Capabilities, DataKind, Payload, Requirement, Task
from .arima import ArimaParams
from .arima import build as _build_arima
from .naive import NaiveParams
from .naive import build as _build_naive
from .params import LSTMParams
from .prophet import ProphetParams
from .prophet import build as _build_prophet

BUILTINS: tuple[str, ...] = ("ts.naive", "ts.arima", "ts.prophet", "ts.lstm")

_TASKS: frozenset[Task] = frozenset({"forecasting"})
_KINDS: frozenset[DataKind] = frozenset({"timeseries"})
_SERIES: frozenset[Payload] = frozenset({"series"})
_ARRAYS: frozenset[Payload] = frozenset({"arrays"})

# Shared by everything on the forecast backend. `produces_proba=False` is the one
# that does work: it is what makes `/predict_proba` return 400 from the manifest.
_FORECAST_CAPS: dict[str, Any] = {
    "needs_scaling": False,
    "native_categorical": False,
    "native_missing": False,
    "supports_sample_weight": False,
    "produces_proba": False,
    "supports_pruning": False,
    "supports_gpu": False,
    "supports_mixed_precision": False,
    "supports_lr_range_test": False,
    "supports_resume": False,
}


def _lazy_build(module: str) -> Callable[..., Any]:
    """Defer the import of a torch-importing plugin to build time."""

    def _build(ctx: Any) -> Any:
        return importlib.import_module(module, __name__).build(ctx)

    return _build


register_model_spec(
    ModelSpec(
        name="ts.naive",
        backend="forecast",
        build=_build_naive,
        tasks=_TASKS,
        data_kinds=_KINDS,
        # None. This is the baseline every other forecaster is measured against.
        requires=(),
        capabilities=Capabilities(accepts=_SERIES, **_FORECAST_CAPS),
        search_space={"model.params.seasons_averaged": Int(1, 4)},
        params_model=NaiveParams,
        # Highest priority of the four: on an unknown series the honest first move
        # is the baseline, and it costs milliseconds.
        auto_priority=5,
        description="Seasonal naive — repeat the last season. The MASE reference.",
    )
)

register_model_spec(
    ModelSpec(
        name="ts.arima",
        backend="forecast",
        build=_build_arima,
        tasks=_TASKS,
        data_kinds=_KINDS,
        requires=(Requirement("statsmodels", extra="timeseries", min_version="0.14"),),
        capabilities=Capabilities(accepts=_SERIES, **_FORECAST_CAPS),
        search_space={
            "model.params.p": Int(0, 5),
            "model.params.d": Int(0, 2),
            "model.params.q": Int(0, 5),
        },
        params_model=ArimaParams,
        auto_priority=15,
        description="ARIMA/SARIMA (statsmodels). Explicit order, prediction intervals.",
    )
)

register_model_spec(
    ModelSpec(
        name="ts.prophet",
        backend="forecast",
        build=_build_prophet,
        tasks=_TASKS,
        data_kinds=_KINDS,
        requires=(Requirement("prophet", extra="timeseries", min_version="1.1"),),
        capabilities=Capabilities(accepts=_SERIES, **_FORECAST_CAPS),
        search_space={
            # The knob most worth tuning and most able to overfit a short series.
            "model.params.changepoint_prior_scale": Float(0.001, 0.5, log=True),
            "model.params.seasonality_mode": Categorical(("additive", "multiplicative")),
        },
        params_model=ProphetParams,
        auto_priority=20,
        description="Prophet — interpretable trend + seasonality decomposition.",
    )
)

register_model_spec(
    ModelSpec(
        name="ts.lstm",
        # The Lightning backend, because an LSTM's fit loop *is* the Lightning loop.
        backend="lightning",
        build=_lazy_build(".lstm"),
        tasks=_TASKS,
        data_kinds=_KINDS,
        requires=(
            Requirement("torch", extra="lightning", min_version="2.0"),
            Requirement(
                "pytorch_lightning", extra="lightning", min_version="2.0", dist="pytorch-lightning"
            ),
        ),
        capabilities=Capabilities(
            accepts=_ARRAYS,
            needs_scaling=True,
            produces_proba=False,
            supports_pruning=True,
            supports_gpu=True,
            supports_mixed_precision=True,
            supports_lr_range_test=True,
            supports_resume=True,
            supports_sample_weight=False,
        ),
        search_space={
            "model.params.hidden_size": Int(16, 256, log=True),
            "model.params.num_layers": Int(1, 3),
            "model.params.dropout": Float(0.0, 0.5),
        },
        params_model=LSTMParams,
        auto_priority=10,
        description="Recurrent forecaster over sliding windows (Lightning backend).",
    )
)

__all__ = ["BUILTINS", "ArimaParams", "LSTMParams", "NaiveParams", "ProphetParams"]
