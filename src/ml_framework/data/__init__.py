"""The data layer: ingestion, splitting, preprocessing, and one Lightning adapter.

``build_bundle`` is the framework-agnostic entry point (arrays + a schema);
``build_datamodule`` is the Lightning-specific wrapper over it.
"""

from .builders import build_bundle, build_datamodule, build_model
from .lightning_adapter import BundleDataModule
from .types import DataBundle, FeatureSchema, Split

__all__ = [
    "build_bundle",
    "build_datamodule",
    "build_model",
    "BundleDataModule",
    "DataBundle",
    "FeatureSchema",
    "Split",
]
