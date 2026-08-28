"""
data/preprocess/audio.py
────────────────────────
Waveform → log-mel spectrogram, and the collate that batches ragged clips.

Like the image preprocessor this fits nothing — a mel filterbank is a fixed
function of ``(sample_rate, n_fft, n_mels)``, not a statistic learned from the
training set. What has to round-trip through the bundle is the *configuration*,
because serving must produce exactly the spectrogram training did. Train/serve
skew here is silent and produces a model that merely looks bad.

**The front-end runs in the collate, per batch, on the GPU when there is one.**
That is deliberate and it is the point at which this file participates in the
staged pipeline:

* the *decoder* returns int16 or float32 PCM and nothing more — decode reverses a
  compression scheme and stops;
* **tensor construction** happens here, once per batch, where a few MB is cast and
  stacked rather than a corpus;
* the mel transform is a matmul against a fixed filterbank, so running it on a
  batch is meaningfully faster than running it per sample in a worker, and it
  keeps the workers doing IO rather than FFTs.

Clips are **fixed-length by construction**: a batch of ragged waveforms cannot be
stacked, and padding to the longest clip in each batch would make the input length
depend on batch composition. :meth:`fit_length` centre-crops or zero-pads to
exactly ``clip_seconds``, so every batch has the same shape and a model can state
its input size.

``torchaudio`` is imported inside the methods, so this module stays importable on
an install without the ``[audio]`` extra — the rule every plugin module follows.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .base import BasePreprocessor

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


class AudioPreprocessor(BasePreprocessor):
    """Fixed-length PCM → log-mel spectrogram, with a batching collate."""

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

    def fit_length(self, pcm: np.ndarray) -> np.ndarray:
        """Centre-crop or zero-pad one waveform to exactly ``clip_samples``.

        Fixed length by construction rather than per-batch padding: padding to the
        longest clip in each batch would make the input size depend on batch
        composition, so the same sample would produce different activations
        depending on what it was batched with.

        A **centre** crop rather than a leading one because the informative part of
        a clip is usually not at its start — a leading crop on a corpus with
        leading silence trains on silence.
        """
        if pcm.ndim > 1:
            # (channels, samples) -> mono. Averaged in float to avoid the integer
            # wrap that summing loud stereo in int16 would produce.
            pcm = pcm.mean(axis=0, dtype="float32")

        want = self.clip_samples
        have = int(pcm.shape[-1])
        if have == want:
            return pcm
        if have > want:
            start = (have - want) // 2
            return pcm[start : start + want]
        pad = want - have
        left = pad // 2
        return np.pad(pcm, (left, pad - left), mode="constant")

    def to_float32(self, pcm: np.ndarray) -> np.ndarray:
        """Scale integer PCM into [-1, 1); pass float through untouched.

        Divided by 32768 rather than 32767, matching libsndfile — so a corpus read
        as int16 and one read as float32 produce the same spectrogram instead of
        differing by one ULP.
        """
        if pcm.dtype == np.int16:
            return pcm.astype("float32") / 32768.0
        if pcm.dtype == np.int32:
            return pcm.astype("float32") / 2147483648.0
        return np.ascontiguousarray(pcm, dtype="float32")

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

    # ── the collate: where tensor construction happens ──
    @property
    def collate_fn(self) -> Callable[[Sequence[Any]], Any]:
        return self._collate

    def _collate(self, batch: Sequence[Any]) -> Any:
        """Decoded PCM → a batched spectrogram tensor.

        This is the boundary the staged pipeline draws. Everything before it
        produced numpy buffers and described them; everything after it is torch.
        Doing the cast and the stack here means a *batch* pays for them — a few MB
        — rather than the corpus.
        """
        import torch

        waves: list[np.ndarray] = []
        labels: list[int] = []
        for item in batch:
            sample, label = item if isinstance(item, tuple) else (item, None)
            pcm = getattr(sample, "array", sample)
            waves.append(self.fit_length(self.to_float32(np.asarray(pcm))))
            if label is not None:
                labels.append(int(label))

        # One stack, one H2D-ready contiguous buffer.
        x = torch.from_numpy(np.ascontiguousarray(np.stack(waves, axis=0), dtype="float32"))
        features = self.log_mel(x)
        if not labels:
            return features
        return features, torch.tensor(labels, dtype=torch.int64)

    # ── contract ──
    def transform(self, x: Any) -> Any:
        """Serving path: one waveform (or several) → the same spectrogram.

        Routed through :meth:`collate_fn` rather than reimplemented, because a
        second implementation of the front-end is exactly how train/serve skew
        gets in.
        """
        items = x if isinstance(x, (list, tuple)) else [x]
        return self._collate(list(items))

    def _write(self, dest: Path) -> list[str]:
        # Everything needed to rebuild the front-end is in `params()`, which
        # `save()` already records in preprocessor.json. Nothing is fitted.
        return []
