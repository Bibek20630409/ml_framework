"""
data/streaming/decoders/wave_pcm.py
───────────────────────────────────
Uncompressed RIFF/WAVE through the standard library's :mod:`wave`.

**This decoder has no requirements, and that is the point.** It is the *oracle*
the cross-decoder equivalence tests compare against — the role ``local`` plays for
data backends. ``audio.flac`` must reproduce it sample-for-sample, and a claim
like that is only worth making if the reference path is guaranteed present on a
bare install.

It is also the honest baseline for what "decode" costs. A WAV has no compression
to reverse: this reads a header and reinterprets the frame block. Everything the
other audio decoders charge over this number is the codec.

Integrity: ``loud``. :mod:`wave` raises on a malformed RIFF header, so structural
damage surfaces — but there is no per-frame check, so a bit flip inside the data
chunk is an audible click that nothing reports.
"""

from __future__ import annotations

import io
import wave
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np

from ..stages import DecodeContext, Decoded, Packet
from .base import BaseDecoder, as_bytes

# WAVE stores signed PCM little-endian for every width but 8-bit, which is
# unsigned. Only 16-bit is accepted here: it is what every corpus in practice
# uses, and silently widening 24-bit into int32 would make the oracle disagree
# with libsndfile for reasons that have nothing to do with the codec.
_SUPPORTED_SAMPLE_WIDTH = 2


class WavePcmDecoder(BaseDecoder):
    """RIFF/WAVE → int16 PCM. No demux stage; decode is a reinterpretation."""

    name: ClassVar[str] = "audio.pcm"

    def __init__(self, *, mono: bool = False) -> None:
        self.mono = mono

    def decode(self, packets: Iterable[Packet], *, ctx: DecodeContext) -> Decoded:
        payload = as_bytes(next(iter(packets)).data)
        # `wave` wants a seekable file object; BytesIO over the blob avoids a
        # temp file and keeps the read in one place.
        with wave.open(io.BytesIO(payload), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            if width != _SUPPORTED_SAMPLE_WIDTH:
                raise ValueError(
                    f"wave: expected 16-bit PCM, got {width * 8}-bit. "
                    "Convert the corpus, or use the 'audio.flac' decoder."
                )
            frames = handle.readframes(handle.getnframes())

        pcm = np.frombuffer(frames, dtype="<i2")
        if channels > 1:
            # (channels, samples), C-contiguous — a transposed view would make
            # every downstream `from_numpy` fall off its zero-copy path.
            pcm = np.ascontiguousarray(pcm.reshape(-1, channels).T)
            if self.mono:
                # Mean in int32 so the sum cannot wrap before the divide.
                pcm = pcm.mean(axis=0, dtype="int32").astype("int16")

        pcm = _cast(pcm, ctx.dtype)
        return Decoded(
            array=pcm,
            layout="pcm",
            dtype=str(pcm.dtype),
            lands_in="host",
            rate=rate,
            meta={"channels": 1 if pcm.ndim == 1 else int(pcm.shape[0])},
        )


def _cast(pcm: np.ndarray, dtype: str | None) -> np.ndarray:
    """Honour ``DecodeContext.dtype``, scaling when the target is float.

    ``float32`` divides by 32768 rather than by 32767, matching libsndfile — so
    the oracle and ``audio.flac`` agree bit-for-bit instead of by one ULP, which
    is the difference between an equivalence test that means something and one
    that needs a tolerance.
    """
    if dtype is None or dtype == str(pcm.dtype):
        return pcm
    if dtype in ("float32", "float64"):
        return (pcm.astype(dtype) / 32768.0).astype(dtype)
    return pcm.astype(dtype)


def build_decoder(**params: Any) -> WavePcmDecoder:
    """Factory named by :data:`DecoderSpec.factory`.

    Unknown keys raise here rather than being ignored — ``data.decoder_params`` is
    validated *by the decoder*, so this function is the one owner of that dict for
    this decoder, and a silently dropped knob is how a run ends up not doing what
    the config says.
    """
    unknown = set(params) - {"mono"}
    if unknown:
        raise ValueError(f"audio.pcm got unknown decoder_params {sorted(unknown)}; accepts: mono")
    return WavePcmDecoder(mono=bool(params.get("mono", False)))
