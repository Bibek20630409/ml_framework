"""Data sources: ingestion → :class:`~ml_framework.data.types.DataBundle`.

A *source* is the extension point that replaces datamodules. From P1 there is
exactly one datamodule (the Lightning adapter over a bundle), so registering
datamodules stopped making sense while registering sources started to.

``build_image_bundle`` is imported here for symmetry with ``build_tabular_bundle``
— safe because its torchvision imports live inside the function, per the rule that
a module must be importable with zero optional dependencies installed.
"""

from .image import build_image_bundle
from .tabular import build_tabular_bundle, read_table

__all__ = ["build_tabular_bundle", "build_image_bundle", "read_table"]
