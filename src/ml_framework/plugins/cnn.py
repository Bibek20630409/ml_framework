"""
plugins/cnn.py
──────────────
Transfer-learning CNN for image classification. Registered as ``"cnn"``.

Uses a torchvision backbone (default resnet18) with the classifier head resized
to ``output_dim``. Consumes 4D image tensors (B, C, H, W) directly — unlike the
MLP, which is why images need this model rather than the flat network.

``torchvision`` is imported inside :meth:`CNN.build_network`, and this module is
itself imported lazily by ``plugins/__init__.py`` — defining a ``LightningModule``
subclass needs torch, which is optional. :class:`~ml_framework.plugins.params.CNNParams`
therefore lives in ``params.py``, so the config validator can check
``model.params`` without either dependency present.
"""

from __future__ import annotations

import torch.nn as nn

from ..core.lit_model import BaseModel
from ..core.protocols import BuildContext
from ..core.registry import register_model
from .params import CNNParams


@register_model("cnn")
class CNN(BaseModel):
    @classmethod
    def params_model(cls) -> type[CNNParams]:
        return CNNParams

    def build_network(self) -> nn.Module:
        from torchvision import models

        backbone_name = self.params.backbone
        pretrained = self.params.pretrained
        factory = getattr(models, backbone_name, None)
        if factory is None:
            raise ValueError(f"Unknown torchvision backbone: {backbone_name}")

        weights = "DEFAULT" if pretrained else None
        net = factory(weights=weights)

        # Resize the final classifier layer to output_dim across common backbones.
        if hasattr(net, "fc") and isinstance(net.fc, nn.Linear):  # resnet family
            net.fc = nn.Linear(net.fc.in_features, self.output_dim)
        elif hasattr(net, "classifier"):  # vgg / densenet / mobilenet
            classifier = net.classifier
            if isinstance(classifier, nn.Sequential):
                in_features: int = classifier[-1].in_features  # type: ignore[assignment,union-attr]
                classifier[-1] = nn.Linear(in_features, self.output_dim)
            elif isinstance(classifier, nn.Linear):
                net.classifier = nn.Linear(classifier.in_features, self.output_dim)
        else:
            raise ValueError(f"Unsupported backbone head: {backbone_name}")
        return net


def build(ctx: BuildContext) -> CNN:
    """The registry's entry point: one :class:`BuildContext` in, a model out."""
    return CNN(
        input_dim=ctx.input_dim,
        output_dim=ctx.output_dim,
        task=ctx.task,
        params=ctx.params,
        optim=ctx.optim,
        class_weights=ctx.class_weights,
    )


__all__ = ["CNN", "CNNParams", "build"]
