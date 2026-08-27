"""
data/streaming/decoders/audio_compressed.py
───────────────────────────────────────────
MP3 and Opus through PyAV (libavcodec / libavformat).

These two share a module because they share a decode path, but their **demux**
stages are genuinely different, and that difference is the reason demux is a stage
at all:

* **MP3** is usually stored as *raw frames with no container*. Finding frame
  boundaries is a **frame-sync scan** — real work, done by
  ``CodecContext.parse()``, not a header read.
* **Opus** arrives wrapped in Ogg (or WebM). Demux is a **page parse**, done by
  ``libavformat`` opening the container.

Their integrity stories differ just as sharply, and this is the pairing that makes
the ``silent`` classification concrete:

* **Opus/Ogg has a per-page CRC and raises.** Damage surfaces.
* **MP3 resyncs silently past damage.** It finds the next valid frame header and
  carries on. You get *shorter audio and no error* — a perfectly well-formed array
  of the wrong length, which no exception handler will ever see. The only way to
  catch it is to compare the decoded duration against what the frame count implies,
  which is what ``mlf materialize`` does and why an MP3 corpus is refused until it
  has been through that pass.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator
from typing import Any, ClassVar

import numpy as np

from ..stages import Blob, DecodeContext, Decoded, Packet
from .base import BaseDecoder, as_bytes

# libavcodec hands audio back as planar or packed float; float32 is the useful
# common denominator and what every model front-end wants.
_DEFAULT_DTYPE = "float32"

# Codec name -> whether the bytes arrive inside a container. Drives which demux
# stage runs, and nothing else.
_CONTAINERIZED: dict[str, bool] = {"mp3": False, "opus": True}


class CompressedAudioDecoder(BaseDecoder):
    """Raw-frame or containerized compressed audio → float32 PCM.

    Holds libav state between ``demux`` and ``decode`` (a codec context, or an
    open container). That is why :meth:`close` exists on the protocol: an
    elementary-stream packet is not decodable outside the context that produced
    it, so the two stages cannot be stateless without re-parsing.
    """

    name: ClassVar[str] = "audio.compressed"

    def __init__(self, *, codec: str, dtype: str = _DEFAULT_DTYPE, mono: bool = False) -> None:
        if codec not in _CONTAINERIZED:
            raise ValueError(
                f"unsupported codec '{codec}'; expected one of {sorted(_CONTAINERIZED)}"
            )
        self.codec = codec
        self.dtype = dtype
        self.mono = mono
        self._container: Any = None
        self._ctx: Any = None

    # ── stage 2: demux ──
    def demux(self, blob: Blob) -> Iterator[Packet]:
        """Frame-sync scan (MP3) or Ogg page parse (Opus).

        Yields one :class:`Packet` per elementary-stream unit, carrying both the
        bytes (so materialization can count and hash them without decoding) and
        the native handle in ``opaque`` (so decode can actually use them).
        """
        import av

        payload = as_bytes(blob.data)
        self.close()  # a decoder instance is reused across samples

        if _CONTAINERIZED[self.codec]:
            self._container = av.open(io.BytesIO(payload))
            streams = [s for s in self._container.streams if s.type == "audio"]
            if not streams:
                raise ValueError(f"{self.codec}: no audio stream in container")
            stream = streams[0]
            for packet in self._container.demux(stream):
                if packet.size == 0:  # the flush packet libav appends
                    continue
                yield Packet(
                    data=bytes(packet),
                    stream=stream.index,
                    keyframe=bool(packet.is_keyframe),
                    pts=packet.pts,
                    opaque=packet,
                )
        else:
            # Raw frames: `parse` IS the frame-sync scan. No container exists, so
            # there is no stream index and no keyframe concept.
            self._ctx = av.CodecContext.create(self.codec, "r")
            for packet in self._ctx.parse(payload):
                yield Packet(data=bytes(packet), pts=packet.pts, opaque=packet)

    # ── stage 3: decode ──
    def decode(self, packets: Iterable[Packet], *, ctx: DecodeContext) -> Decoded:
        dtype = ctx.dtype or self.dtype
        chunks: list[np.ndarray] = []
        rate: int | None = None
        n_packets = 0

        for packet in packets:
            n_packets += 1
            native = packet.opaque
            if native is None:
                raise ValueError(
                    "audio.compressed decode() needs the demuxer's native packet; "
                    "call demux() first rather than synthesizing packets"
                )
            for frame in native.decode():
                rate = int(frame.sample_rate)
                # to_ndarray gives (channels, samples) for planar formats, which
                # is already this pipeline's convention.
                chunks.append(frame.to_ndarray())

        self._flush(chunks)
        if not chunks:
            raise ValueError(f"{self.codec}: decoded no audio frames")

        pcm = np.concatenate(chunks, axis=-1) if len(chunks) > 1 else chunks[0]
        if pcm.ndim == 2 and pcm.shape[0] == 1:
            pcm = pcm[0]
        elif pcm.ndim == 2 and self.mono:
            pcm = pcm.mean(axis=0, dtype="float32")
        pcm = np.ascontiguousarray(pcm, dtype=dtype)

        return Decoded(
            array=pcm,
            layout="pcm",
            dtype=str(pcm.dtype),
            lands_in="host",
            rate=rate,
            meta={
                "channels": 1 if pcm.ndim == 1 else int(pcm.shape[0]),
                "n_packets": n_packets,
                # Materialization compares this against the duration the frame
                # count implies. For MP3 that comparison is the ONLY way a silent
                # resync is ever detected.
                "n_samples": int(pcm.shape[-1]),
            },
        )

    def _flush(self, chunks: list[np.ndarray]) -> None:
        """Drain the decoder's internal buffer.

        libavcodec holds frames back; without a flush the tail of every sample is
        silently missing — which would look exactly like the MP3 truncation this
        decoder is supposed to help detect.
        """
        if self._ctx is None:
            return
        try:
            for frame in self._ctx.decode(None):
                chunks.append(frame.to_ndarray())
        except (EOFError, ValueError):
            # Nothing buffered. Not an error, and not worth a branch upstream.
            pass

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
        self._ctx = None


def _build(codec: str, params: dict[str, Any]) -> CompressedAudioDecoder:
    unknown = set(params) - {"dtype", "mono"}
    if unknown:
        raise ValueError(
            f"audio.{codec} got unknown decoder_params {sorted(unknown)}; accepts: dtype, mono"
        )
    return CompressedAudioDecoder(
        codec=codec,
        dtype=str(params.get("dtype", _DEFAULT_DTYPE)),
        mono=bool(params.get("mono", False)),
    )


# Two entry points rather than one, because two specs share this module and the
# lazy factory addresses them by attribute name (`DecoderSpec.factory`'s `attr`).
def build_mp3_decoder(**params: Any) -> CompressedAudioDecoder:
    return _build("mp3", params)


def build_opus_decoder(**params: Any) -> CompressedAudioDecoder:
    return _build("opus", params)
