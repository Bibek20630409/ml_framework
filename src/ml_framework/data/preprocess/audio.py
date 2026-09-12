"""
data/preprocess/audio.py
────────────────────────
Waveform → log-mel spectrogram, and the collate that batches ragged clips.

Like the image preprocessor this fits nothing — a mel filterbank is a fixed
function of ``(sample_rate, n_fft, n_mels)``, not a statistic learned from the
training set. What has to round-trip through the bundle is the *configuration*,
because serving must produce exactly the spectrogram training did. Train/serve
skew here is silent and produces a model that merely looks bad.

**This file is the tail of the staged pipeline for audio, and it is two stages,
not one.** Until now they were fused in a single ``_collate`` whose docstring
claimed the front-end ran "on the GPU when there is one" — it never did: the
collate builds a CPU tensor inside a DataLoader worker, so ``mel.to(x.device)``
resolved to CPU on every machine, every run. Splitting them makes the claim true
by making it a placement the transport layer can actually make:

* the *decoder* returns int16 or float32 PCM and nothing more — decode reverses a
  compression scheme and stops;
* :meth:`build_tensor` is **construct**: cast, length-fit and stack, once per
  batch, where a few MB is copied rather than a corpus;
* :meth:`transform_batch` is the mel filterbank — a matmul, and the only part
  worth putting on a device. It runs wherever its input already is, so the
  transport layer decides: in the collate on a CPU box, and after the H2D copy on
  a CUDA one.

Deferring it also shrinks the copy: what crosses the bus is a waveform, not a
spectrogram.

Clips are **fixed-length by construction**: a batch of ragged waveforms cannot be
stacked, and padding to the longest clip in each batch would make the input length
depend on batch composition. :meth:`AudioPreprocessor._span` centre-crops or
zero-pads every clip to exactly ``clip_seconds``, so every batch has the same shape
and a model can state its input size.

``torchaudio`` is imported inside the methods, so this module stays importable on
an install without the ``[audio]`` extra — the rule every plugin module follows.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .base import BasePreprocessor, host_array

log = logging.getLogger(__name__)

# Spectrogram geometry. 25 ms windows at 10 ms hop is the near-universal speech
# front-end, and stating it in samples-at-16 kHz rather than in milliseconds is
# what torchaudio's API wants.
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_N_FFT = 400  # 25 ms at 16 kHz
DEFAULT_HOP_LENGTH = 160  # 10 ms at 16 kHz
DEFAULT_N_MELS = 64

# Floor for the log, so silence maps to a large negative number rather than -inf.
# -inf propagates through a batch norm and produces NaN gradients several layers
# later, where the cause is no longer visible.
LOG_EPSILON = 1e-10

# Full-scale divisor per integer PCM width, applied to scale integer PCM into
# [-1, 1). 32768 rather than 32767, matching libsndfile — so a corpus read as int16
# and one read as float32 produce the same spectrogram instead of differing by one
# ULP. Both values are powers of two, so the division is exact in float32 and the
# order of "scale then average" versus "average then scale" cannot change a result;
# `build_tensor` relies on that when it averages a stereo clip before scaling it.
#
# A float dtype is absent rather than mapped to 1.0: "no scaling applies" and
# "scale by one" would compute the same answer, but only the first is true, and
# `build_tensor` skips the pass entirely for it.
_PCM_SCALE: dict[Any, float] = {np.int16: 32768.0, np.int32: 2147483648.0}


class AudioPreprocessor(BasePreprocessor):
    """Fixed-length PCM → log-mel spectrogram, split across the tail's stages."""

    # `gpu_transform` is a claim about `transform_batch`: it is a torch matmul
    # against a filterbank moved to the input's device, so it is correct on either
    # side of the H2D copy. That claim is what licenses the transport layer to
    # defer it.
    stages = frozenset({"construct", "transform", "gpu_transform"})

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        clip_seconds: float = 4.0,
        n_mels: int = DEFAULT_N_MELS,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.clip_seconds = float(clip_seconds)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self._mel: Any = None

    def params(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "clip_seconds": self.clip_seconds,
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
        }

    # ── derived geometry ──
    @property
    def clip_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))

    @property
    def n_frames(self) -> int:
        """Spectrogram frames per clip. What a model needs to size its head."""
        return self.clip_samples // self.hop_length + 1

    # ── the front-end ──
    def _transform(self) -> Any:
        """The mel filterbank, built once and reused.

        Cached because constructing it allocates the filter matrix, and doing that
        per batch would show up as a steady tax with no visible cause.
        """
        if self._mel is None:
            import torchaudio

            self._mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
        return self._mel

    def _span(self, have: int) -> tuple[slice, slice]:
        """``(source slice, destination slice)`` placing one clip in a fixed row.

        **Pure arithmetic — moves no data.** The whole crop/pad rule, stated once
        as offsets so that :meth:`build_tensor` can write straight into the batch
        buffer rather than building a per-sample array first.

        Fixed length by construction rather than per-batch padding: padding to the
        longest clip in each batch would make the input size depend on batch
        composition, so the same sample would produce different activations
        depending on what it was batched with.

        A **centre** crop rather than a leading one because the informative part of
        a clip is usually not at its start — a leading crop on a corpus with
        leading silence trains on silence. And a centre *pad* for the mirror
        reason: a short clip flush-left would put every one of them against the
        same edge, which a convolution can learn.
        """
        want = self.clip_samples
        if have >= want:
            start = (have - want) // 2
            return slice(start, start + want), slice(0, want)
        left = (want - have) // 2
        return slice(0, have), slice(left, left + have)

    @staticmethod
    def _mono(pcm: np.ndarray) -> np.ndarray:
        """``(channels, samples)`` -> mono, averaged in float.

        In float rather than in the source dtype to avoid the integer wrap that
        summing loud stereo in int16 would produce.
        """
        return pcm.mean(axis=0, dtype="float32")

    def log_mel(self, waveforms: Any) -> Any:
        """``(B, samples)`` float32 → ``(B, 1, n_mels, frames)`` log-mel.

        One channel, not three: a spectrogram is not an RGB image, and stacking it
        into three identical channels to satisfy a pretrained stem wastes two
        thirds of the first convolution. The audio model reshapes its stem instead.
        """
        import torch

        mel = self._transform().to(waveforms.device)
        spec = mel(waveforms)
        return torch.log(spec + LOG_EPSILON).unsqueeze(1)

    # ── construct ──
    def build_tensor(self, batch: Sequence[Any]) -> tuple[Any, Any]:
        """Decoded PCM → ``(B, samples)`` float32, plus the labels.

        The boundary the staged pipeline draws: everything before it produced numpy
        buffers and described them, everything after it is torch. The cast, the
        length fit and the stack all happen here so a *batch* pays for them — a few
        MB — rather than the corpus.

        No spectrogram yet. What this returns is what crosses the bus, and a
        waveform is smaller than its mel.

        **One allocation for the batch, written through directly.** The obvious
        spelling — widen each clip to float32, fit it to length, then ``np.stack``
        the results — copies twice and allocates a temporary per sample: once to
        widen, again to gather the scattered buffers into one block. Here the
        destination is allocated first and each sample is cast straight into its
        row, so the widen *is* the gather. The zero padding comes free from
        ``np.zeros``, and the geometry is :meth:`_span`.
        """
        import torch

        # The scale is read from the ORIGINAL dtype, before `_mono` turns a stereo
        # int16 clip into float32 -- reading it after would silently skip the
        # divide and hand the model raw sample values three orders too large.
        samples: list[tuple[np.ndarray, float | None]] = []
        labels: list[int] = []
        for item in batch:
            sample, label = item if isinstance(item, tuple) else (item, None)
            pcm = host_array(sample)
            scale = _PCM_SCALE.get(pcm.dtype.type)
            samples.append((self._mono(pcm) if pcm.ndim > 1 else pcm, scale))
            if label is not None:
                labels.append(int(label))

        out = np.zeros((len(samples), self.clip_samples), dtype="float32")
        for i, (pcm, scale) in enumerate(samples):
            src, dst = self._span(int(pcm.shape[-1]))
            # The int16 -> float32 cast happens in this assignment, into the final
            # buffer. Only the written span is scaled; the padding stays zero.
            out[i, dst] = pcm[src]
            if scale is not None:
                out[i, dst] /= scale

        x = torch.from_numpy(out)
        y = torch.tensor(labels, dtype=torch.int64) if labels else None
        return x, y

    # ── transform / gpu_transform ──
    def transform_batch(self, x: Any) -> Any:
        """The mel front-end, on whatever device ``x`` is already on.

        Device-agnostic by construction — :meth:`log_mel` moves the filterbank to
        the input rather than the other way round — which is the whole reason this
        may run either in the collate or after the H2D copy.
        """
        return self.log_mel(x)

    # ── contract ──
    def transform(self, x: Any) -> Any:
        """Serving path: one waveform (or several) → the same spectrogram.

        Routed through the same two methods training uses rather than
        reimplemented, because a second implementation of the front-end is exactly
        how train/serve skew gets in.
        """
        items = x if isinstance(x, (list, tuple)) else [x]
        return self.collate_staged(list(items))

    def _write(self, dest: Path) -> list[str]:
        # Everything needed to rebuild the front-end is in `params()`, which
        # `save()` already records in preprocessor.json. Nothing is fitted.
        return []
