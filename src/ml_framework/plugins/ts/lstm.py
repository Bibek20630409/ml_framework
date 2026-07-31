"""
plugins/ts/lstm.py
──────────────────
Recurrent forecaster. Registered as ``"ts.lstm"``.

**It rides the Lightning backend, not the forecast one** — which is the clearest
demonstration of why backends are per fit-loop *shape* rather than per model
family. An LSTM forecaster trains exactly like an MLP: mini-batches, epochs, early
stopping, checkpoints. Only its input shape differs, and shape is the model's
business.

What makes that work is the payload split. This plugin declares
``accepts={"arrays"}``, so the time-series source hands it *windowed supervised
rows* (``x`` = the previous ``window`` observations, ``y`` = the next value) rather
than the raw series it gives Prophet. `Capabilities.accepts` was built for exactly
this decision, and it is made once, in the source.

The network reshapes ``(batch, window)`` to ``(batch, window, 1)`` internally, so
nothing upstream has to know it is recurrent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...core.lit_model import BaseModel
from ...core.protocols import BuildContext
from ...core.registry import register_model
from .params import LSTMParams


@register_model("ts.lstm")
class LSTMForecaster(BaseModel):
    @classmethod
    def params_model(cls) -> type[LSTMParams]:
        return LSTMParams

    def build_network(self) -> nn.Module:
        return _LSTMNet(
            window=self.input_dim,
            hidden_size=self.params.hidden_size,
            num_layers=self.params.num_layers,
            dropout=self.params.dropout,
            bidirectional=self.params.bidirectional,
        )


class _LSTMNet(nn.Module):
    """(batch, window) → (batch, 1).

    The reshape lives here rather than in the data layer because "this model reads
    its input as a sequence" is a property of the model. The windowed rows the
    source produces are the same rows an MLP would consume.
    """

    def __init__(
        self,
        window: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.window = window
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # torch ignores dropout on a single layer and warns about it; suppress
            # the warning by telling it the truth instead.
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.head = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:  # (batch, window) → (batch, window, 1)
            x = x.unsqueeze(-1)
        output, _ = self.lstm(x)
        # The last timestep summarizes the window; that is what predicts the next
        # value.
        return self.head(output[:, -1, :])


def build(ctx: BuildContext) -> LSTMForecaster:
    """The registry's entry point."""
    return LSTMForecaster(
        input_dim=ctx.input_dim,
        output_dim=1,
        task=ctx.task,
        params=ctx.params,
        optim=ctx.optim,
        class_weights=None,
    )


__all__ = ["LSTMForecaster", "LSTMParams", "build"]
