"""
data/builders.py
────────────────
Factory helpers that turn a validated config into concrete datamodule / model
instances via the registry. Importing this module guarantees the built-in
datamodules and models are registered.

For custom data, register your own datamodule:

    from ml_framework.core import register_datamodule, TabularDataModule

    @register_datamodule("my_source")
    class MyDataModule(TabularDataModule):
        def setup(self, stage=None):
            ...            # load/merge, then reuse parent logic
            super().setup(stage)

...then set ``data.kind: my_source`` in the YAML.
"""

from __future__ import annotations

import torch

# Side-effect imports register the built-ins.
from .. import models as _models  # noqa: F401  (registers mlp/cnn)
from ..config import ExperimentConfig
from ..core import BaseModel, FrameworkDataModule  # registers tabular/image datamodules
from ..core.plugins import SourceSpec
from ..core.registry import get_datamodule_class, get_model_class, register_source
from ..core.types import Requirement


def build_datamodule(config: ExperimentConfig) -> FrameworkDataModule:
    return get_datamodule_class(config.data.kind)(config)


def build_model(
    config: ExperimentConfig,
    *,
    input_dim: int,
    output_dim: int,
    class_weights: torch.Tensor | None = None,
) -> BaseModel:
    return get_model_class(config.model.name)(
        input_dim=input_dim,
        output_dim=output_dim,
        config=config,
        class_weights=class_weights,
    )


# ── v2 source specs ───────────────────────────────────────
# Data *sources* are the extension point that replaces datamodules: from P1 there
# is exactly one datamodule (the Lightning adapter over a DataBundle), so
# registering datamodules stops making sense while registering sources starts to.
#
# `build` returns today's FrameworkDataModule; in P1 it returns a DataBundle. The
# spec metadata (data kind, payload, requirements) is already final.
register_source(
    SourceSpec(
        name="tabular",
        data_kind="tabular",
        build=build_datamodule,
        payload="arrays",
        requires=(),
        description="CSV/Parquet table with a target column.",
    )
)
register_source(
    SourceSpec(
        name="image",
        data_kind="image",
        build=build_datamodule,
        payload="dataset",
        requires=(
            Requirement("torchvision", extra="image", min_version="0.15"),
            Requirement("PIL", extra="image", min_version="9.0", dist="Pillow"),
        ),
        description="Directory-of-class-directories image folders (ImageFolder layout).",
    )
)
