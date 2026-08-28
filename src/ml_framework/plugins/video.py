"""
plugins/video.py
────────────────
3-D ResNet for video clip classification. Registered as ``"video.r3d"``.

``torchvision.models.video.r3d_18`` with KINETICS400_V1 weights, consuming
``(B, C, T, H, W)`` clips. Two things about that are worth stating:

**It needs no new extra.** ``torchvision.models.video`` ships in the same package
the image models already require, so ``video.r3d`` costs an install nothing beyond
``[image]``. Only the *decoder* needs ``[video]`` (PyAV) — the model and the codec
have genuinely different dependencies, and conflating them would make the pip hint
for either one wrong.

**Video pretraining matters much more than image pretraining does.** 400 classes
of motion is a far better prior for a clip than ImageNet is for a photograph,
because a randomly-initialized 3-D convolution has to learn temporal structure
from scratch on a corpus that is almost always small. Training this from scratch
is possible and rarely a good idea.

*Rejected: TimeSformer/VideoMAE via ``transformers``.* Better ceiling, much
heavier, and a first video model in a general framework should be the one that
fine-tunes on one GPU.

Rides the **lightning** backend, like every other neural plugin here.
"""

from __future__ import annotations

import torch.nn as nn

from ..core.lit_model import BaseModel
from ..core.protocols import BuildContext
from ..core.registry import register_model
from .av_params import VideoR3DParams

# torchvision's video weights are named per architecture rather than by a shared
# "DEFAULT" alias in every release, so the mapping is explicit. A backbone absent
# from here is refused by name rather than silently trained from scratch.
_VIDEO_WEIGHTS: dict[str, str] = {
    "r3d_18": "R3D_18_Weights",
    "mc3_18": "MC3_18_Weights",
    "r2plus1d_18": "R2Plus1D_18_Weights",
}


@register_model("video.r3d")
class VideoR3D(BaseModel):
    """3-D ResNet over ``(B, C, T, H, W)`` clips."""

    @classmethod
    def params_model(cls) -> type[VideoR3DParams]:
        return VideoR3DParams

    def build_network(self) -> nn.Module:
        from torchvision.models import video as video_models

        name = self.params.backbone
        factory = getattr(video_models, name, None)
        if factory is None or name not in _VIDEO_WEIGHTS:
            raise ValueError(
                f"Unknown video backbone '{name}'; expected one of {sorted(_VIDEO_WEIGHTS)}. "
                "The video architectures differ in expected clip geometry, so this is a "
                "closed set rather than any torchvision attribute."
            )

        weights = None
        if self.params.pretrained:
            enum = getattr(video_models, _VIDEO_WEIGHTS[name], None)
            weights = getattr(enum, "DEFAULT", None) if enum is not None else None
        net = factory(weights=weights)

        # Every torchvision video model ends in a single Linear `fc`.
        if not (hasattr(net, "fc") and isinstance(net.fc, nn.Linear)):
            raise ValueError(f"Unsupported video backbone head: {name}")
        net.fc = nn.Sequential(
            nn.Dropout(self.params.dropout),
            nn.Linear(net.fc.in_features, self.output_dim),
        )
        return net


def build(ctx: BuildContext) -> VideoR3D:
    """The registry's entry point: one :class:`BuildContext` in, a model out."""
    return VideoR3D(
        input_dim=ctx.input_dim,
        output_dim=ctx.output_dim,
        task=ctx.task,
        params=ctx.params,
        optim=ctx.optim,
        class_weights=ctx.class_weights,
    )


__all__ = ["VideoR3D", "VideoR3DParams", "build"]
