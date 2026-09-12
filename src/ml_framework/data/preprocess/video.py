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
**once per batch**, rather than per frame in a worker.

The same argument as the image path, one axis larger: a 16-frame 112x112 clip is
16x the pixels of one image, so doing this per sample rather than per batch is
16x the wasted work.

It is also split off from the stacking, into :meth:`VideoPreprocessor.build_tensor`
(construct) and :meth:`VideoPreprocessor.transform_batch` (transform). That is what
lets the transport layer run the 4x copy *after* the H2D transfer on a CUDA box —
so the bus carries the uint8 clip and the GPU does the arithmetic, instead of a
worker doing the arithmetic and the bus carrying four times as much.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .base import BasePreprocessor, host_array

log = logging.getLogger(__name__)

# The Kinetics-400 statistics torchvision's video backbones were trained with.
# Different from ImageNet's, and using the wrong ones silently degrades transfer
# in exactly the way that is hardest to attribute.
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


class VideoPreprocessor(BasePreprocessor):
    """THWC uint8 clips → normalized CTHW float32 batches, across two stages."""

    # See `AudioPreprocessor.stages`. The permute/cast/normalize is pure torch on
    # the input's own device, so it is correct on either side of the H2D copy —
    # and this is the row where deferring it also makes the copy 4x smaller.
    stages = frozenset({"construct", "transform", "gpu_transform"})

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

    # ── construct ──
    def build_tensor(self, batch: Sequence[Any]) -> tuple[Any, Any]:
        """Decoded THWC clips → a stacked ``(B, T, H, W, C)`` **uint8** tensor.

        The boundary the staged pipeline draws: everything before it is numpy
        buffers with a declared layout, everything after it is torch. Note what is
        deliberately *not* done here — no permute, no cast, no divide. Those are
        the 4x copy, and leaving them to :meth:`transform_batch` means the H2D copy
        moves uint8 rather than float32: a 16-frame 112x112 clip crosses the bus at
        600 kB instead of 2.4 MB.
        """
        import torch

        clips: list[np.ndarray] = []
        labels: list[int] = []
        for item in batch:
            sample, label = item if isinstance(item, tuple) else (item, None)
            clips.append(self.fit_clip(host_array(sample)))
            if label is not None:
                labels.append(int(label))

        x = torch.from_numpy(np.ascontiguousarray(np.stack(clips, axis=0)))
        y = torch.tensor(labels, dtype=torch.int64) if labels else None
        return x, y

    # ── transform / gpu_transform ──
    def transform_batch(self, x: Any) -> Any:
        """``(B, T, H, W, C)`` uint8 → normalized ``(B, C, T, H, W)`` float32.

        The permute plus the cast plus the divide — one copy with a 4x blowup, and
        the whole reason this stage is worth moving to a device. The normalization
        constants are built on ``x``'s own device so the arithmetic never forces a
        transfer back.
        """
        import torch

        x = x.permute(0, 4, 1, 2, 3).float().div_(255.0)
        mean = torch.tensor(KINETICS_MEAN, device=x.device).view(1, 3, 1, 1, 1)
        std = torch.tensor(KINETICS_STD, device=x.device).view(1, 3, 1, 1, 1)
        return (x - mean) / std

    # ── contract ──
    def transform(self, x: Any) -> Any:
        """Serving path, routed through the same two stages training uses.

        A second copy of the normalization is exactly how train/serve skew gets in.
        """
        items = x if isinstance(x, (list, tuple)) else [x]
        return self.collate_staged(list(items))

    def _write(self, dest: Path) -> list[str]:
        # Nothing fitted; `params()` is recorded in preprocessor.json by `save()`.
        return []
