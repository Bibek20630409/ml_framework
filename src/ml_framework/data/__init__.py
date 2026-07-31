"""The data layer: ingestion, splitting, preprocessing, and one Lightning adapter.

``build_bundle`` is the framework-agnostic entry point (arrays + a schema);
``build_datamodule`` is the Lightning-specific wrapper over it.

**The Lightning-bound names resolve lazily**, for the same reason ``core`` does
it: the serving path reaches into this package for ``preprocess.load_preprocessor``,
and an ``__init__`` that eagerly imported ``builders`` would pull the Lightning
adapter — and therefore torch — into every serving image. The split follows the
dependency exactly: :class:`DataBundle`, :class:`Split` and
:class:`FeatureSchema` are plain dataclasses and stay eager; ``build_bundle``,
``build_datamodule``, ``build_model`` and ``BundleDataModule`` resolve on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import DataBundle, FeatureSchema, Split

if TYPE_CHECKING:  # so type checkers and IDEs still see the lazy names
    from .builders import build_bundle, build_datamodule, build_model
    from .lightning_adapter import BundleDataModule

_LAZY: dict[str, str] = {
    "build_bundle": ".builders",
    "build_datamodule": ".builders",
    "build_model": ".builders",
    "BundleDataModule": ".lightning_adapter",
}


def __getattr__(name: str) -> Any:
    """Resolve a builder/adapter export on first use (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "build_bundle",
    "build_datamodule",
    "build_model",
    "BundleDataModule",
    "DataBundle",
    "FeatureSchema",
    "Split",
]
