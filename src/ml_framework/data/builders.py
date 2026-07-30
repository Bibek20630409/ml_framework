"""
data/builders.py
────────────────
Config → concrete objects, via the registries. Importing this module guarantees
the built-in sources, datamodules and models are registered.

``build_bundle`` is the entry point the training orchestrator uses: it returns a
:class:`~ml_framework.data.types.DataBundle`, which is framework-agnostic, rather
than a ``LightningDataModule``, which is not. ``build_datamodule`` remains for the
Lightning path (and for the LR finder), now as a thin wrapper.

For custom data, register a source:

    from ml_framework.core.registry import register_source
    from ml_framework.core.plugins import SourceSpec

    register_source(SourceSpec(name="my_source", data_kind="tabular",
                               build=my_build_fn, payload="arrays"))

...then set ``data.kind: my_source`` in the YAML.

**Import discipline:** everything below imports core *submodules*
(``..core.lit_model``, ``..core.registry``) rather than the ``..core`` package
surface. ``core`` re-exports the data layer's helpers, so reaching for
``from ..core import X`` here would be a circular import that only shows up in
whichever module happens to be imported first.
"""

from __future__ import annotations

from typing import Any

import torch

# Side-effect imports register the built-ins.
from .. import models as _models  # noqa: F401  (registers mlp/cnn)
from ..config import ExperimentConfig
from ..core.lit_model import BaseModel
from ..core.plugins import SourceSpec
from ..core.registry import get_datamodule_class, get_model_class, register_source
from ..core.types import Requirement
from .lightning_adapter import (  # noqa: F401  (registers tabular/image datamodules)
    BundleDataModule,
    ImageDataModule,
    TabularDataModule,
)
from .sources import build_image_bundle, build_tabular_bundle
from .types import DataBundle

_BUNDLE_BUILDERS = {
    "tabular": build_tabular_bundle,
    "image": build_image_bundle,
}


def build_bundle(config: ExperimentConfig) -> DataBundle:
    """Materialize the configured data source as a :class:`DataBundle`.

    Dispatches through ``SOURCES`` so a third-party source is reachable by the
    same ``data.kind`` mechanism as the built-ins.
    """
    from ..core.registry import SOURCES

    if config.data.kind in SOURCES:
        spec = SOURCES.get(config.data.kind)
        return spec.build(config)
    builder = _BUNDLE_BUILDERS.get(config.data.kind)
    if builder is None:
        raise KeyError(f"Unknown data kind '{config.data.kind}'. Known: {SOURCES.names()}")
    return builder(config)


def build_datamodule(config: ExperimentConfig) -> Any:
    """The Lightning adapter for the configured data kind."""
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
# `build` now returns a DataBundle, which is what `SourceSpec.build`'s docstring
# promised from P1. The spec metadata (data kind, payload, requirements) is
# unchanged from P0.
register_source(
    SourceSpec(
        name="tabular",
        data_kind="tabular",
        build=build_tabular_bundle,
        payload="arrays",
        requires=(),
        description="CSV/Parquet table with a target column.",
    )
)
register_source(
    SourceSpec(
        name="image",
        data_kind="image",
        build=build_image_bundle,
        payload="dataset",
        requires=(
            Requirement("torchvision", extra="image", min_version="0.15"),
            Requirement("PIL", extra="image", min_version="9.0", dist="Pillow"),
        ),
        description="Directory-of-class-directories image folders (ImageFolder layout).",
    )
)
