"""
plugins/ts/arima.py
───────────────────
ARIMA / SARIMA via statsmodels. Registered as ``"ts.arima"``.

The interpretable workhorse: an explicit ``(p, d, q)`` order says what the model
believes about the series — how much autoregression, how much differencing, how
much moving average — in a way a learned representation does not. ``d`` is where
non-stationarity is handled, which is why the source *reports* ADF/KPSS rather
than differencing on your behalf: the decision belongs in this order.

``statsmodels`` is imported inside :func:`build`, so this module stays importable
on an install without the ``timeseries`` extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel as PydanticModel
from pydantic import Field

from ...core.protocols import BuildContext


class ArimaParams(PydanticModel):
    """``model.params`` for ARIMA: the order, and whether it is seasonal."""

    model_config = {"frozen": True, "extra": "forbid"}

    p: int = Field(default=1, ge=0, le=12)  # autoregressive terms
    d: int = Field(default=1, ge=0, le=2)  # differencing — where trend is removed
    q: int = Field(default=1, ge=0, le=12)  # moving-average terms
    # Seasonal (P, D, Q); the period comes from the source's `seasonality`.
    seasonal: bool = False
    seasonal_p: int = Field(default=0, ge=0, le=2)
    seasonal_d: int = Field(default=0, ge=0, le=1)
    seasonal_q: int = Field(default=0, ge=0, le=2)
    trend: str | None = None


class ArimaForecaster:
    """A fitted ``SARIMAX`` result, presented through the backend's contract."""

    def __init__(
        self,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
        trend: str | None = None,
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.result: Any = None

    def fit(self, values: Any, index: Any = None) -> ArimaForecaster:  # noqa: ARG002
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        series = np.asarray(values, dtype="float64").reshape(-1)
        model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            # A short series rarely satisfies these; refusing to fit over it would
            # be less useful than fitting and letting the metrics judge.
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.result = model.fit(disp=False)
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        if self.result is None:
            raise ValueError("ArimaForecaster.forecast called before fit")
        steps = max(0, int(horizon))
        if steps == 0:
            return np.empty(0, dtype="float64")
        return np.asarray(self.result.forecast(steps=steps), dtype="float64").reshape(-1)

    def interval(self, horizon: int, alpha: float = 0.05):
        """Prediction bounds — the reason to reach for a statistical model."""
        if self.result is None:
            return None, None
        frame = self.result.get_forecast(steps=int(horizon)).conf_int(alpha=alpha)
        bounds = np.asarray(frame, dtype="float64")
        return bounds[:, 0], bounds[:, 1]


def build(ctx: BuildContext) -> ArimaForecaster:
    """The registry's entry point."""
    params = ArimaParams.model_validate(dict(ctx.params))
    period = int(dict(ctx.optim).get("seasonality", 1))
    seasonal_order = (
        (params.seasonal_p, params.seasonal_d, params.seasonal_q, period)
        if params.seasonal and period > 1
        else (0, 0, 0, 0)
    )
    return ArimaForecaster(
        order=(params.p, params.d, params.q),
        seasonal_order=seasonal_order,
        trend=params.trend,
    )


__all__ = ["ArimaForecaster", "ArimaParams", "build"]
