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
(``..core.protocols``, ``..core.registry``) rather than the ``..core`` package
surface. ``core`` re-exports the data layer's helpers, so reaching for
``from ..core import X`` here would be a circular import that only shows up in
whichever module happens to be imported first. ``ExperimentConfig`` is imported
under ``TYPE_CHECKING`` for the same reason from the other direction: the config
validator resolves plugins, so a module-scope import here would make validating a
config depend on the data package being importable first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Side-effect import: registers mlp/cnn in both registries.
from .. import plugins as _plugins  # noqa: F401
from ..core.plugins import SourceSpec
from ..core.protocols import BuildContext
from ..core.registry import MODELS, get_datamodule_class, register_source
from ..core.types import Requirement
from .lightning_adapter import (  # noqa: F401  (registers tabular/image datamodules)
    BundleDataModule,
    ImageDataModule,
    TabularDataModule,
)
from .sources import build_image_bundle, build_tabular_bundle
from .types import DataBundle

if TYPE_CHECKING:
    from ..config import ExperimentConfig

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
    class_weights: Any = None,
) -> Any:
    """Build the configured model through its plugin spec.

    The config is unpacked into a :class:`BuildContext` *here* rather than handed
    to the model, which is the point of the v2 change: a model knows its
    architecture and its forward pass, not the shape of the experiment around it.
    """
    spec = MODELS.get(config.model.name)
    return spec.build(
        BuildContext(
            task=config.task,
            input_dim=input_dim,
            output_dim=output_dim,
            class_weights=class_weights,
            params=config.model.params,
            optim=config.fit.params,
            seed=config.runtime.seed,
        )
    )


# ── v2 source specs ───────────────────────────────────────
# `build` returns a DataBundle, which is what `SourceSpec.build`'s docstring
# promises. Source-specific knobs come from `data.params` and are validated by
# the source's own frozen params model.
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
