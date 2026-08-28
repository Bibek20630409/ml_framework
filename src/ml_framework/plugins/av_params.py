"""
plugins/av_params.py
────────────────────
Params schemas for the audio and video models.

Separate from ``params.py`` only because these two arrived with P13 and grouping
them keeps the diff legible; separate from ``audio.py``/``video.py`` for the
reason every plugin follows — those modules define ``LightningModule`` subclasses,
so importing them imports torch, and the config validator has to check
``model.params`` on an install that may not have it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class AudioCNNParams(BaseModel):
    """``model.params`` for ``audio.cnn``."""

    model_config = _FROZEN

    # A torchvision classification backbone. resnet18 is the default for the same
    # reason it is the image default: it transfers well and it is small enough to
    # fine-tune on one GPU.
    backbone: str = "resnet18"
    # ImageNet weights on a spectrogram are a weaker prior than on a photograph,
    # but a measurably better starting point than random -- the low-level edge and
    # texture filters transfer to time-frequency structure.
    pretrained: bool = True
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)


class VideoR3DParams(BaseModel):
    """``model.params`` for ``video.r3d``."""

    model_config = _FROZEN

    # `r3d_18` is the 3-D ResNet torchvision ships. Named rather than freely
    # configurable because the video backbones differ in their expected input
    # geometry, and silently feeding one the wrong clip length produces a shape
    # error several layers deep.
    backbone: str = "r3d_18"
    # KINETICS400_V1. Video pretraining matters far more than image pretraining
    # does: 400 classes of motion is a much better prior for a clip than nothing.
    pretrained: bool = True
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
