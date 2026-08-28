"""Data sources: ingestion → :class:`~ml_framework.data.types.DataBundle`.

A *source* is the extension point that replaces datamodules. From P1 there is
exactly one datamodule (the Lightning adapter over a bundle), so registering
datamodules stopped making sense while registering sources started to.

``build_image_bundle`` is imported here for symmetry with ``build_tabular_bundle``
— safe because its torchvision imports live inside the function, per the rule that
a module must be importable with zero optional dependencies installed.
"""

from .audio import audio_labels, build_audio_bundle
from .image import build_image_bundle, train_labels
from .tabular import build_tabular_bundle, read_table
from .text import build_text_bundle, text_labels
from .timeseries import build_timeseries_bundle
from .video import build_video_bundle, video_labels

__all__ = [
    "train_labels",
    "audio_labels",
    "video_labels",
    "build_audio_bundle",
    "build_video_bundle",
    "text_labels",
    "build_text_bundle",
    "build_tabular_bundle",
    "build_image_bundle",
    "build_timeseries_bundle",
    "read_table",
]
