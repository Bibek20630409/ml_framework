"""
plugins/mlp.py
──────────────
Default feed-forward network for tabular data. Registered as ``"mlp"``.

Architecture comes from :class:`~ml_framework.plugins.params.MLPParams`
(``model.params`` in the YAML). That schema lives in ``params.py`` rather than
here because the config validator must run it on installs without torch, while
*this* module defines a ``LightningModule`` subclass and therefore imports torch
at module scope. ``plugins/__init__.py`` registers the spec with a lazy ``build``,
so nothing imports this file until an MLP is actually constructed.

Kaiming init for ReLU stacks, unchanged.
"""

from __future__ import annotations

import torch.nn as nn

from ..core.lit_model import BaseModel
from ..core.protocols import BuildContext
from ..core.registry import register_model
from .params import MLPParams


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
