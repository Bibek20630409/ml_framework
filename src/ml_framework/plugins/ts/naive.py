"""
plugins/ts/naive.py
───────────────────
Seasonal naive: the forecast for step *t+h* is the observation one season ago.

**No dependencies, and that is load-bearing.** This is the model MASE is measured
against — a MASE of 1.0 *is* this forecaster — so it has to be available wherever
forecasting is, including an install with neither prophet nor statsmodels. It is
also the always-on baseline the zero-config work needs: costing milliseconds, it
is the cheapest possible guard against shipping a model that is worse than
repeating last week's numbers.

With ``seasonality=1`` it degenerates to the random-walk forecast (repeat the last
observation), which is the right default when nothing is known about the period.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel as PydanticModel
from pydantic import Field

from ...core.protocols import BuildContext


class NaiveParams(PydanticModel):
    """``model.params`` for the seasonal-naive forecaster."""

    model_config = {"frozen": True, "extra": "forbid"}

    # None → take the seasonality the source derived from the data. An explicit
    # value overrides it, which is what makes this tunable.
    seasonality: int | None = Field(default=None, ge=1)
    # Average the last k seasons instead of taking the single most recent one.
    # k=1 is the textbook seasonal naive; k>1 trades responsiveness for noise
    # rejection on a jumpy series.
    seasons_averaged: int = Field(default=1, ge=1)


class SeasonalNaive:
    """Fitted state is the tail of the training series. That is the whole model."""

    def __init__(self, seasonality: int = 1, seasons_averaged: int = 1) -> None:
        self.seasonality = max(1, seasonality)
        self.seasons_averaged = max(1, seasons_averaged)
        self.pattern: np.ndarray | None = None
        self.last_value: float = 0.0

    def fit(
        self, values: Any, index: Any = None
    ) -> SeasonalNaive:  # noqa: ARG002 - uniform signature
        series = np.asarray(values, dtype="float64").reshape(-1)
        if series.size == 0:
            raise ValueError("SeasonalNaive.fit needs at least one observation")
        self.last_value = float(series[-1])

        period = min(self.seasonality, series.size)
        available = series.size // period
        k = max(1, min(self.seasons_averaged, available))
        if k == 1:
            self.pattern = series[-period:].copy()
        else:
            # Mean of the last k seasons, position by position.
            window = series[-k * period :].reshape(k, period)
            self.pattern = window.mean(axis=0)
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        if self.pattern is None:
            raise ValueError("SeasonalNaive.forecast called before fit")
        steps = max(0, int(horizon))
        if steps == 0:
            return np.empty(0, dtype="float64")
        # Tile the season forward. With period 1 this repeats the last value, which
        # is the random-walk forecast.
        repeats = int(np.ceil(steps / len(self.pattern)))
        return np.tile(self.pattern, repeats)[:steps].astype("float64")


def build(ctx: BuildContext) -> SeasonalNaive:
    """The registry's entry point."""
    params = NaiveParams.model_validate(dict(ctx.params))
    seasonality = params.seasonality or int(dict(ctx.optim).get("seasonality", 1))
    return SeasonalNaive(seasonality=seasonality, seasons_averaged=params.seasons_averaged)


__all__ = ["NaiveParams", "SeasonalNaive", "build"]
