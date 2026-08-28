"""
plugins/audio.py
────────────────
Log-mel spectrogram CNN for audio classification. Registered as ``"audio.cnn"``.

**A spectrogram CNN rather than a raw-waveform network**, and that is the whole
design decision. A 1-D CNN over raw samples (M5 and friends) is simpler and needs
no front-end, but it scores materially worse on every small-corpus classification
task and would misrepresent what this framework can do on audio. The mel
front-end is a fixed, cheap transform that turns the problem into one the image
stack already solves well.

**Rejected: a pretrained AudioSet/PANNs checkpoint.** It scores better still, and
it is a ~300 MB download as a hard dependency of a *default* model. That is the
same trade the NLP plugins refuse.

**One input channel, not three.** A spectrogram is not an RGB image. Stacking it
into three identical channels to satisfy a pretrained stem is the common shortcut
and it wastes two thirds of the first convolution; :meth:`build_network` reshapes
the stem instead, summing the pretrained RGB weights so the transferred filters
still see the same total signal.

Rides the **lightning** backend — no fourth training backend, for the same reason
``nlp.hf_text`` and ``ts.lstm`` do not have one: the fit loop is an epoch loop over
batches, which is exactly what that backend already is.
"""

from __future__ import annotations

import torch.nn as nn

from ..core.lit_model import BaseModel
from ..core.protocols import BuildContext
from ..core.registry import register_model
from .av_params import AudioCNNParams


@register_model("audio.cnn")
class AudioCNN(BaseModel):
    """torchvision backbone over a single-channel log-mel spectrogram."""

    @classmethod
    def params_model(cls) -> type[AudioCNNParams]:
        return AudioCNNParams

    def build_network(self) -> nn.Module:
        from torchvision import models

        factory = getattr(models, self.params.backbone, None)
        if factory is None:
            raise ValueError(f"Unknown torchvision backbone: {self.params.backbone}")

        net = factory(weights="DEFAULT" if self.params.pretrained else None)
        _to_single_channel(net, pretrained=self.params.pretrained)

        if hasattr(net, "fc") and isinstance(net.fc, nn.Linear):  # resnet family
            net.fc = nn.Sequential(
                nn.Dropout(self.params.dropout),
                nn.Linear(net.fc.in_features, self.output_dim),
            )
        elif hasattr(net, "classifier"):  # vgg / densenet / mobilenet
            classifier = net.classifier
            if isinstance(classifier, nn.Sequential):
                in_features: int = classifier[-1].in_features  # type: ignore[assignment,union-attr]
                classifier[-1] = nn.Linear(in_features, self.output_dim)
            elif isinstance(classifier, nn.Linear):
                net.classifier = nn.Linear(classifier.in_features, self.output_dim)
        else:
            raise ValueError(f"Unsupported backbone head: {self.params.backbone}")
        return net


def _to_single_channel(net: nn.Module, *, pretrained: bool) -> None:
    """Reshape the stem convolution from 3 input channels to 1, in place.

    The pretrained kernel is **summed** across the RGB axis rather than averaged or
    sliced. Summing preserves the response magnitude a grey input would have
    produced through the original three-channel filter, so the transferred features
    arrive at the next layer at the scale batch norm was calibrated for; averaging
    divides every activation by three and slicing discards two thirds of the
    learned filter.
    """
    stem_name = "conv1" if hasattr(net, "conv1") else "features"
    stem = getattr(net, stem_name, None)
    if isinstance(stem, nn.Sequential):
        stem = stem[0]
    if not isinstance(stem, nn.Conv2d) or stem.in_channels == 1:
        return

    replacement = nn.Conv2d(
        1,
        stem.out_channels,
        kernel_size=stem.kernel_size,  # type: ignore[arg-type]
        stride=stem.stride,  # type: ignore[arg-type]
        padding=stem.padding,  # type: ignore[arg-type]
        bias=stem.bias is not None,
    )
    if pretrained:
        import torch

        with torch.no_grad():
            replacement.weight.copy_(stem.weight.sum(dim=1, keepdim=True))
            if stem.bias is not None and replacement.bias is not None:
                replacement.bias.copy_(stem.bias)

    if isinstance(getattr(net, stem_name, None), nn.Sequential):
        getattr(net, stem_name)[0] = replacement
    else:
        setattr(net, stem_name, replacement)


def build(ctx: BuildContext) -> AudioCNN:
    """The registry's entry point: one :class:`BuildContext` in, a model out."""
    return AudioCNN(
        input_dim=ctx.input_dim,
        output_dim=ctx.output_dim,
        task=ctx.task,
        params=ctx.params,
        optim=ctx.optim,
        class_weights=ctx.class_weights,
    )


__all__ = ["AudioCNN", "AudioCNNParams", "build"]
