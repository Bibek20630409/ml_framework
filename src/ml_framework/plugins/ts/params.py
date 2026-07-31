"""
plugins/ts/params.py
────────────────────
``model.params`` schemas for the time-series plugins that need torch.

Same split, same reason as ``plugins/params.py``: the config validator runs a
plugin's params model on **any** install, but ``lstm.py`` defines a
``LightningModule`` and therefore imports torch at module scope. Schemas live
here so a forecast-only install — prophet and statsmodels, no torch — can still
validate a config that names ``ts.lstm`` and refuse it with a pip command rather
than an ImportError.

The pure-forecast plugins (naive, arima, prophet) keep their schemas beside their
code: they import nothing heavier than pydantic at module scope.
"""

from __future__ import annotations

from pydantic import BaseModel as PydanticModel
from pydantic import Field


class LSTMParams(PydanticModel):
    """``model.params`` for the recurrent forecaster.

    ``window`` is deliberately absent: how many past observations a row contains is
    a property of the *data* the source produces, so it lives in
    ``data.params.window``. Duplicating it here would let the two disagree, and the
    network would then be built for a shape it never receives.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    hidden_size: int = Field(default=64, ge=1)
    num_layers: int = Field(default=1, ge=1, le=4)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    bidirectional: bool = False


__all__ = ["LSTMParams"]
