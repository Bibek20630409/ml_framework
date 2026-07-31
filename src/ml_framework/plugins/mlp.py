"""
plugins/mlp.py
──────────────
Default feed-forward network for tabular data. Registered as ``"mlp"``.

Architecture comes from :class:`MLPParams` (``model.params`` in the YAML), which
is the plugin's own frozen, ``extra="forbid"`` schema — so a typo in
``hidden_dims`` errors at config-load time even though core knows nothing about
this model's knobs. v1's ``ModelConfig._check_dims`` moves here with it: "all
hidden_dims must be positive" is a fact about the MLP, not about every model in
the framework.

Kaiming init for ReLU stacks, unchanged.
"""

from __future__ import annotations

import torch.nn as nn
from pydantic import BaseModel as PydanticModel
from pydantic import Field, model_validator

from ..core.lit_model import BaseModel
from ..core.protocols import BuildContext
from ..core.registry import register_model


class MLPParams(PydanticModel):
    model_config = {"frozen": True, "extra": "forbid"}

    hidden_dims: list[int] = Field(default_factory=lambda: [128, 64, 32])
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_dims(self) -> MLPParams:
        if any(d <= 0 for d in self.hidden_dims):
            raise ValueError("all hidden_dims must be positive")
        return self


@register_model("mlp")
class MLP(BaseModel):
    @classmethod
    def params_model(cls) -> type[MLPParams]:
        return MLPParams

    def build_network(self) -> nn.Module:
        layers: list[nn.Module] = []
        prev = self.input_dim
        for dim in self.params.hidden_dims:
            layers += [
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(self.params.dropout),
            ]
            prev = dim
        layers.append(nn.Linear(prev, self.output_dim))
        network = nn.Sequential(*layers)

        for m in network.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        return network


def build(ctx: BuildContext) -> MLP:
    """The registry's entry point: one :class:`BuildContext` in, a model out."""
    return MLP(
        input_dim=ctx.input_dim,
        output_dim=ctx.output_dim,
        task=ctx.task,
        params=ctx.params,
        optim=ctx.optim,
        class_weights=ctx.class_weights,
    )


__all__ = ["MLP", "MLPParams", "build"]
