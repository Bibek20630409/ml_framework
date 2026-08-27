"""
data/streaming/decoders/audio_file.py
─────────────────────────────────────
FLAC (and any other self-describing file libsndfile reads) via ``soundfile``.

**The only format in this pipeline with real built-in integrity.** FLAC carries a
per-frame CRC-16 and libsndfile raises on a mismatch, so damage surfaces at the
point of the read rather than as a quiet degradation. Every other row in the table
either has no checksum, or has one that only covers structure. That is why this
decoder's ``integrity`` is ``"checked"`` and why ``verify_checksums: "auto"``
deliberately does *not* re-hash its samples: paying for a second digest over bytes
the codec already validated is waste.

**The float64 trap.** ``soundfile.read()`` defaults to ``dtype="float64"``. On a
speech corpus that is a silent 4x memory blowup on every sample, and nothing
reports it because the result is perfectly valid audio. This decoder therefore
never calls ``read()`` bare — the dtype is always passed explicitly, defaulting to
``int16`` (the on-disk width for essentially every FLAC corpus) and overridable
through :class:`DecodeContext`. ``float64`` is refused outright.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np

from ..stages import DecodeContext, Decoded, Packet
from .base import BaseDecoder, as_bytes

# What libsndfile will hand back. `float64` is absent on purpose: it is `sf.read`'s
# default and always the wrong choice here, so making it unrepresentable is
# cheaper than documenting why not to ask for it.
_SUPPORTED_DTYPES: frozenset[str] = frozenset({"int16", "int32", "float32"})

_DEFAULT_DTYPE = "int16"


class SoundFileDecoder(BaseDecoder):
    """FLAC/WAV/AIFF → PCM through libsndfile. No demux; the container is the file."""

    name: ClassVar[str] = "audio.flac"

    def __init__(self, *, dtype: str = _DEFAULT_DTYPE, mono: bool = False) -> None:
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"audio.flac dtype must be one of {sorted(_SUPPORTED_DTYPES)}, got '{dtype}'. "
                "float64 is intentionally not offered: it is sf.read()'s default and a "
                "silent 4x blowup over the whole corpus."
            )
        self.dtype = dtype
        self.mono = mono

    def decode(self, packets: Iterable[Packet], *, ctx: DecodeContext) -> Decoded:
        import soundfile as sf

        dtype = ctx.dtype or self.dtype
        if dtype == "float64":
            raise ValueError(
                "audio.flac refuses dtype='float64': it quadruples the corpus in memory "
                "for no precision that survives the model. Use 'float32' or 'int16'."
            )
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"audio.flac cannot produce dtype '{dtype}'")

        payload = as_bytes(next(iter(packets)).data)
        # `always_2d=False` keeps mono as (n,) rather than (n, 1), matching what
        # the `wave` oracle produces so the equivalence test compares shapes too.
        pcm, rate = sf.read(io.BytesIO(payload), dtype=dtype, always_2d=False)

        if pcm.ndim == 2:
            # soundfile yields (samples, channels); the rest of this pipeline uses
            # (channels, samples). `ascontiguousarray` because a bare transpose is
            # a view with the wrong strides, and every downstream `from_numpy`
            # would fall off its zero-copy path.
            pcm = np.ascontiguousarray(pcm.T)
            if self.mono:
                pcm = _downmix(pcm, dtype)

        return Decoded(
            array=pcm,
            layout="pcm",
            dtype=str(pcm.dtype),
            lands_in="host",
            rate=int(rate),
            meta={"channels": 1 if pcm.ndim == 1 else int(pcm.shape[0])},
        )


def _downmix(pcm: np.ndarray, dtype: str) -> np.ndarray:
    """Average channels without wrapping.

    Integer PCM is summed in int32 before the divide; summing in the native width
    overflows on loud stereo material and produces a click rather than an error.
    """
    if dtype.startswith("float"):
        return pcm.mean(axis=0, dtype=dtype).astype(dtype)
    return pcm.mean(axis=0, dtype="int32").astype(dtype)


def build_decoder(**params: Any) -> SoundFileDecoder:
    """Factory named by :data:`DecoderSpec.factory`. Unknown keys raise."""
    unknown = set(params) - {"dtype", "mono"}
    if unknown:
        raise ValueError(
            f"audio.flac got unknown decoder_params {sorted(unknown)}; accepts: dtype, mono"
        )
    return SoundFileDecoder(
        dtype=str(params.get("dtype", _DEFAULT_DTYPE)),
        mono=bool(params.get("mono", False)),
    )
