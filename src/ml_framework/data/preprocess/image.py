"""
data/preprocess/image.py
────────────────────────
The image transform stack — resize, augmentation, ImageNet normalization —
extracted from ``ImageDataModule.setup``.

Unlike the tabular preprocessor this fits nothing: the normalization constants
belong to the pretrained backbones and the augmentation policy is a choice, not a
statistic. What still has to round-trip through the bundle is the *configuration*
(``img_size``), because serving must resize exactly as training did. That is the
whole reason it is a preprocessor rather than a pair of module-level functions:
train/serve skew in image preprocessing is silent and produces a model that
merely looks bad.

``torchvision`` is imported inside the methods, so this module stays importable on
an install without the ``[image]`` extra — the rule every plugin module follows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BasePreprocessor

# The ImageNet statistics every torchvision pretrained backbone was trained with.
# Changing these silently degrades transfer learning, so they are named constants
# rather than inline literals.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImagePreprocessor(BasePreprocessor):
    """Builds the train (augmenting) and eval (deterministic) transform pipelines.

    Augmentation is applied to the training split only — augmenting evaluation
    data makes metrics noisy and non-reproducible.
    """

    def __init__(self, *, img_size: int = 224, augment: bool = True) -> None:
        self.img_size = int(img_size)
        self.augment = bool(augment)

    def params(self) -> dict[str, Any]:
        return {"img_size": self.img_size, "augment": self.augment}

    # ── transforms ──
    def _normalize(self) -> Any:
        from torchvision import transforms

        return transforms.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD))

    def train_transform(self) -> Any:
        from torchvision import transforms

        size = self.img_size
        if not self.augment:
            return self.eval_transform()
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                self._normalize(),
            ]
        )

    def eval_transform(self) -> Any:
        from torchvision import transforms

        size = self.img_size
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                self._normalize(),
            ]
        )

    # ── contract ──
    def transform(self, x: Any) -> Any:
        """Apply the eval pipeline to one PIL image or an iterable of them.

        Serving hands over decoded images; training hands the transform itself to
        ``ImageFolder``, so this path is inference-only.
        """
        tf = self.eval_transform()
        if isinstance(x, (list, tuple)):
            import torch

            return torch.stack([tf(item) for item in x])
        return tf(x)

    def _write(self, dest: Path) -> list[str]:
        # Everything needed to rebuild the pipeline is in `params()`, which
        # `save()` already records in preprocessor.json.
        return []
