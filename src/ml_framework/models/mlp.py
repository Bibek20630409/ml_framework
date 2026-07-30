"""
models/mlp.py
─────────────
Default feed-forward network for tabular data. Registered as ``"mlp"``.

Reads architecture from ``self.config.model`` (hidden_dims, dropout), which are
tuned by ``mlf hpo``. Kaiming init for ReLU stacks.
"""

from __future__ import annotations

import torch.nn as nn

from ..core import BaseModel, register_model


@register_model("mlp")
class MLP(BaseModel):
    def build_network(self) -> nn.Module:
        layers: list[nn.Module] = []
        prev = self.input_dim
        for dim in self.config.model.hidden_dims:
            layers += [
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(self.config.model.dropout),
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
