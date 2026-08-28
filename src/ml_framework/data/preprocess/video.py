"""
data/preprocess/video.py
────────────────────────
Decoded clips → the ``(B, C, T, H, W)`` tensor a 3-D CNN consumes.

Like the image and audio preprocessors this fits nothing; the normalization
constants belong to the Kinetics-pretrained backbones. What has to round-trip
through the bundle is the geometry — ``clip_len``, ``img_size`` — because serving
must build exactly the clip training did.

**This is where the video pipeline's real cost lands, and it is a permute.** The
decoder returns ``THWC uint8``, which is what libavcodec produces and the honest
thing to record. A 3-D CNN wants ``CTHW float32`` normalized. That conversion is
a transpose plus a cast plus a divide — a copy with a 4x blowup — and it happens
here, **once per batch**, rather than per frame in a worker.

The same argument as the image path, one axis larger: a 16-frame 112x112 clip is
16x the pixels of one image, so doing this per sample rather than per batch is
16x the wasted work.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .base import BasePreprocessor

log = logging.getLogger(__name__)

# The Kinetics-400 statistics torchvision's video backbones were trained with.
# Different from ImageNet's, and using the wrong ones silently degrades transfer
# in exactly the way that is hardest to attribute.
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


class VideoPreprocessor(BasePreprocessor):
    """THWC uint8 clips → normalized CTHW float32 batches."""

    def __init__(self, *, clip_len: int = 16, img_size: int = 112) -> None:
        self.clip_len = int(clip_len)
        self.img_size = int(img_size)

    def params(self) -> dict[str, Any]:
        return {"clip_len": self.clip_len, "img_size": self.img_size}

    def fit_clip(self, clip: np.ndarray) -> np.ndarray:
        """Centre-crop or loop one clip to exactly ``clip_len`` frames.

        **Looped**, not zero-padded, when a clip is short. A zero-padded tail is
        black frames, which a motion model reads as a hard cut to darkness — a
        strong and entirely artificial signal. Repeating the clip is the standard
        answer and produces something the model can actually interpret.
        """
        have = int(clip.shape[0])
        if have == self.clip_len:
            return clip
        if have > self.clip_len:
            start = (have - self.clip_len) // 2
            return clip[start : start + self.clip_len]
        repeats = -(-self.clip_len // have)  # ceil
        return np.concatenate([clip] * repeats, axis=0)[: self.clip_len]

    @property
    def collate_fn(self) -> Callable[[Sequence[Any]], Any]:
        return self._collate

    def _collate(self, batch: Sequence[Any]) -> Any:
        """Decoded THWC clips → a normalized ``(B, C, T, H, W)`` tensor.

        The boundary the staged pipeline draws: everything before it is numpy
        buffers with a declared layout, everything after it is torch. The permute
        and the float cast are a copy, and doing them here means a *batch* pays.
        """
        import torch

        clips: list[np.ndarray] = []
        labels: list[int] = []
        for item in batch:
            sample, label = item if isinstance(item, tuple) else (item, None)
            array = np.asarray(getattr(sample, "array", sample))
            clips.append(self.fit_clip(array))
            if label is not None:
                labels.append(int(label))

        # (B, T, H, W, C) uint8 -> (B, C, T, H, W) float32 in [0, 1].
        stacked = torch.from_numpy(np.ascontiguousarray(np.stack(clips, axis=0)))
        x = stacked.permute(0, 4, 1, 2, 3).float().div_(255.0)

        mean = torch.tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1)
        std = torch.tensor(KINETICS_STD).view(1, 3, 1, 1, 1)
        x = (x - mean) / std

        if not labels:
            return x
        return x, torch.tensor(labels, dtype=torch.int64)

    # ── contract ──
    def transform(self, x: Any) -> Any:
        """Serving path, routed through the collate so there is one implementation.

        A second copy of the normalization is exactly how train/serve skew gets in.
        """
        items = x if isinstance(x, (list, tuple)) else [x]
        return self._collate(list(items))

    def _write(self, dest: Path) -> list[str]:
        # Nothing fitted; `params()` is recorded in preprocessor.json by `save()`.
        return []
