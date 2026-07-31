"""
plugins/ts/prophet.py
─────────────────────
Prophet. Registered as ``"ts.prophet"``.

Chosen for *interpretable seasonality*: it decomposes a series into trend,
weekly/yearly components and holidays, each of which can be inspected and argued
with. That is a different product from an LSTM that forecasts as well or better
and cannot tell you why.

Prophet insists on a ``DataFrame`` with ``ds``/``y`` columns and real timestamps.
The bundle carries an index that may be integer positions rather than dates, so
:meth:`ProphetForecaster.fit` synthesizes a daily date range in that case — Prophet
needs *a* time axis, and a regular one is the honest default when the data did not
supply one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel as PydanticModel
from pydantic import Field

from ...core.protocols import BuildContext


class ProphetParams(PydanticModel):
    """``model.params`` for Prophet: what it is allowed to believe."""

    model_config = {"frozen": True, "extra": "forbid"}

    # Larger → the trend bends more readily at changepoints. The single knob most
    # worth tuning, and the one most likely to overfit a short series.
    changepoint_prior_scale: float = Field(default=0.05, gt=0.0, le=1.0)
    seasonality_prior_scale: float = Field(default=10.0, gt=0.0)
    seasonality_mode: str = "additive"  # additive | multiplicative
    yearly_seasonality: bool | str = "auto"
    weekly_seasonality: bool | str = "auto"
    daily_seasonality: bool | str = "auto"
    interval_width: float = Field(default=0.8, gt=0.0, lt=1.0)


class ProphetForecaster:
    """A fitted Prophet model behind the backend's ``fit``/``forecast`` contract."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.model: Any = None
        self.freq: str = "D"
        self.last_timestamp: Any = None

    def fit(self, values: Any, index: Any = None) -> ProphetForecaster:
        import pandas as pd
        from prophet import Prophet

        series = np.asarray(values, dtype="float64").reshape(-1)
        stamps = self._timestamps(index, len(series))
        frame = pd.DataFrame({"ds": stamps, "y": series})

        self.model = Prophet(**self.kwargs)
        # Prophet logs a banner per fit through cmdstanpy; a training run should
        # not be drowned by it.
        import logging

        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        self.model.fit(frame)
        self.last_timestamp = stamps[-1]
        return self

    @staticmethod
    def _timestamps(index: Any, n: int):
        """Real dates, from the bundle's index if it holds any.

        The bundle's index is nanosecond epochs when the source parsed a datetime
        column, and positions otherwise. Prophet needs a time axis either way, so a
        synthetic daily range stands in — which affects the *labels* of the
        seasonality it finds, not whether it finds one.
        """
        import pandas as pd

        if index is not None:
            candidate = pd.to_datetime(np.asarray(index), errors="coerce")
            if not pd.isna(candidate).any():
                return candidate
        return pd.date_range("2000-01-01", periods=n, freq="D")

    def _future(self, horizon: int):
        import pandas as pd

        return pd.DataFrame(
            {"ds": pd.date_range(self.last_timestamp, periods=int(horizon) + 1, freq=self.freq)[1:]}
        )

    def forecast(self, horizon: int) -> np.ndarray:
        if self.model is None:
            raise ValueError("ProphetForecaster.forecast called before fit")
        steps = max(0, int(horizon))
        if steps == 0:
            return np.empty(0, dtype="float64")
        return np.asarray(self.model.predict(self._future(steps))["yhat"], dtype="float64").reshape(
            -1
        )

    def interval(self, horizon: int):
        """Prophet's own uncertainty bounds, at ``interval_width``."""
        if self.model is None:
            return None, None
        predicted = self.model.predict(self._future(int(horizon)))
        return (
            np.asarray(predicted["yhat_lower"], dtype="float64"),
            np.asarray(predicted["yhat_upper"], dtype="float64"),
        )


def build(ctx: BuildContext) -> ProphetForecaster:
    """The registry's entry point."""
    params = ProphetParams.model_validate(dict(ctx.params))
    return ProphetForecaster(
        changepoint_prior_scale=params.changepoint_prior_scale,
        seasonality_prior_scale=params.seasonality_prior_scale,
        seasonality_mode=params.seasonality_mode,
        yearly_seasonality=params.yearly_seasonality,
        weekly_seasonality=params.weekly_seasonality,
        daily_seasonality=params.daily_seasonality,
        interval_width=params.interval_width,
    )


__all__ = ["ProphetForecaster", "ProphetParams", "build"]
