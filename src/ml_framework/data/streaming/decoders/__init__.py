"""
data/streaming/decoders/
────────────────────────
One decoder per media **format**, not per data kind.

Registration is a spec plus a lazy factory, never an import of the codec — the
same rule ``data/backends/__init__.py`` follows, and for the same payoff: ``mlf
decoders`` lists the H.264 path, says what it would cost to trust it, and names
the pip command, all on an install with no FFmpeg binding anywhere.

The specs carry three declared columns, each with exactly one consumer:

``stages``      which of read/demux/decode does real **work**. Every decoder *has*
                all three methods (``BaseDecoder`` supplies defaults), so this is
                documentation for ``--show`` and a hint to the materialization
                pass, never a capability check.

                ``read`` appears on every spec — bytes always have to come off a
                device. What actually varies is the other two, and ``text.tokens``
                is the row that makes the point: it declares ``{"read"}`` alone,
                because a memmap slice is the entire pipeline for a token shard.
``lands_in``    ``"host"`` or ``"device"``. A device decoder's output never
                touched host memory, so there is no array to wrap, no pinned
                buffer and no H2D copy — the transport layer branches on this.
``integrity``   what a *failure surfaces*. The load-bearing column: two of the
                formats below fail silently, and a run over them is refused
                unless it has been through ``mlf materialize``.

``integrity_note`` is prose rather than a flag because the useful thing to know is
*how* a format fails, and "resyncs silently past damage: shorter audio, no error"
does not reduce to a boolean.
"""

from __future__ import annotations

import importlib
from typing import Any

# Imported as submodules rather than `from ...core import X`, for the reason
# `data/backends/__init__.py` documents: `core/__init__` lazily re-exports names
# that reach back into the data layer.
from ....core.plugins import DecoderSpec
from ....core.registry import register_decoder
from ....core.types import Requirement


def _lazy_factory(module: str, attr: str = "build_decoder"):
    """Defer the import to call time, so registering never imports a codec."""

    def _build(**params: Any) -> Any:
        mod = importlib.import_module(module, package=__package__)
        return getattr(mod, attr)(**params)

    return _build


# ── audio ─────────────────────────────────────────────────
# No `requires`: the standard library's `wave` module. Zero-dependency on purpose
# — this is the ORACLE the cross-decoder equivalence tests compare against, the
# role `local` plays for data backends. A claim that FLAC decodes to the same
# samples is only worth making if the reference path is present on a bare install.
register_decoder(
    DecoderSpec(
        name="audio.pcm",
        data_kind="audio",
        factory=_lazy_factory(".wave_pcm"),
        media_types=("audio/wav", "audio/x-wav"),
        suffixes=(".wav",),
        stages=frozenset({"read", "decode"}),
        output_dtype="int16",
        output_layout="pcm",
        lands_in="host",
        integrity="loud",
        integrity_note=(
            "stdlib `wave` raises on a malformed RIFF header, but there is no "
            "per-frame check: a bit flip inside the data chunk is an audible click "
            "that nothing reports."
        ),
        requires=(),
        description="Uncompressed RIFF/WAVE via the standard library. The audio oracle.",
    )
)

register_decoder(
    DecoderSpec(
        name="audio.flac",
        data_kind="audio",
        factory=_lazy_factory(".audio_file"),
        media_types=("audio/flac", "audio/x-flac"),
        suffixes=(".flac",),
        stages=frozenset({"read", "decode"}),
        output_dtype="int16",
        output_layout="pcm",
        lands_in="host",
        integrity="checked",
        integrity_note=(
            "per-frame CRC-16; libsndfile raises on mismatch. The only format here "
            "with real built-in integrity."
        ),
        oracle="audio.pcm",
        requires=(Requirement("soundfile", extra="audio", min_version="0.12.1"),),
        description=(
            "FLAC via libsndfile. Reads int16 or float32 -- never float64, which is "
            "sf.read()'s default and a silent 4x blowup."
        ),
    )
)

register_decoder(
    DecoderSpec(
        name="audio.mp3",
        data_kind="audio",
        factory=_lazy_factory(".audio_compressed", "build_mp3_decoder"),
        media_types=("audio/mpeg", "audio/mp3"),
        suffixes=(".mp3",),
        # A frame-sync scan over raw MP3 frames is a real demux stage: there is no
        # container, so finding frame boundaries is work, not a header read.
        stages=frozenset({"read", "demux", "decode"}),
        output_dtype="float32",
        output_layout="pcm",
        lands_in="host",
        integrity="silent",
        integrity_note=(
            "RESYNCS SILENTLY past damage: you get shorter audio and no error. "
            "Caught only by the duration cross-check in `mlf materialize`."
        ),
        oracle="audio.pcm",
        requires=(Requirement("av", extra="video", min_version="12.0"),),
        description="MP3 via libavcodec. Frame-sync demux, then decode to float32 PCM.",
    )
)

register_decoder(
    DecoderSpec(
        name="audio.opus",
        data_kind="audio",
        factory=_lazy_factory(".audio_compressed", "build_opus_decoder"),
        media_types=("audio/opus", "audio/ogg"),
        suffixes=(".opus", ".ogg"),
        stages=frozenset({"read", "demux", "decode"}),
        output_dtype="float32",
        output_layout="pcm",
        lands_in="host",
        integrity="checked",
        integrity_note="Ogg page CRC; damage raises. Unlike MP3, Opus does not resync quietly.",
        oracle="audio.pcm",
        requires=(Requirement("av", extra="video", min_version="12.0"),),
        description="Ogg/WebM-wrapped Opus via libavcodec. Page-parse demux, then decode.",
    )
)

# ── image ─────────────────────────────────────────────────
# `extra="image"` rather than a new one: Pillow is already what that extra
# installs, and extra names are interpolated verbatim into pip hints. Inventing an
# `imagecodec` extra would make the hint less truthful, not more.
register_decoder(
    DecoderSpec(
        name="image.jpeg",
        data_kind="image",
        factory=_lazy_factory(".image_file", "build_jpeg_decoder"),
        media_types=("image/jpeg",),
        suffixes=(".jpg", ".jpeg"),
        stages=frozenset({"read", "decode"}),
        output_dtype="uint8",
        output_layout="hwc",
        lands_in="host",
        integrity="loud",
        integrity_note=(
            "a truncated file decodes to a PARTIAL IMAGE with a warning by default; "
            "this decoder promotes that warning to an error, because the default "
            "trains you on grey bottoms."
        ),
        requires=(Requirement("PIL", extra="image", min_version="9.0", dist="Pillow"),),
        description="JPEG via libjpeg-turbo. YCbCr->RGB is fused into the scanline kernels.",
    )
)

register_decoder(
    DecoderSpec(
        name="image.png",
        data_kind="image",
        factory=_lazy_factory(".image_file", "build_png_decoder"),
        media_types=("image/png",),
        suffixes=(".png",),
        stages=frozenset({"read", "decode"}),
        output_dtype="uint8",
        output_layout="hwc",
        lands_in="host",
        integrity="checked",
        integrity_note="Adler-32 per zlib block and CRC-32 per chunk; damage raises.",
        requires=(Requirement("PIL", extra="image", min_version="9.0", dist="Pillow"),),
        description=(
            "PNG via libpng: DEFLATE plus per-scanline unfilter. No GPU decode path "
            "exists, and it is ~5-10x slower than JPEG."
        ),
    )
)

# ── video ─────────────────────────────────────────────────
register_decoder(
    DecoderSpec(
        name="video.h264",
        data_kind="video",
        factory=_lazy_factory(".video_file"),
        media_types=("video/mp4",),
        suffixes=(".mp4", ".m4v", ".mov"),
        # Demux is unambiguously real here: box traversal, the `stss` keyframe
        # index, and a seek to a clip boundary. It also stays on the CPU even when
        # decode does not, which is why it is a separate stage rather than folded
        # into decode.
        stages=frozenset({"read", "demux", "decode"}),
        output_dtype="uint8",
        output_layout="thwc",
        lands_in="host",
        integrity="loud",
        integrity_note=(
            "a missing `moov` box fails to open, loudly. Mid-stream corruption is "
            "worse: libavcodec logs and emits ARTIFACTED FRAMES with no exception."
        ),
        requires=(Requirement("av", extra="video", min_version="12.0"),),
        description="MP4/H.264 via libavformat + libavcodec on CPU threads.",
    )
)

# ── text ──────────────────────────────────────────────────
# The row with no demux, no decode, and the worst integrity story of all. See the
# module docstring in `tokens.py` for the uint16-vs-int32 storage arithmetic.
register_decoder(
    DecoderSpec(
        name="text.tokens",
        data_kind="text",
        factory=_lazy_factory(".tokens"),
        media_types=("application/octet-stream",),
        suffixes=(".bin",),
        stages=frozenset({"read"}),
        output_dtype="uint16",
        output_layout="tokens",
        lands_in="host",
        integrity="none",
        integrity_note=(
            "NO INTEGRITY CHECK AT ALL -- a flipped bit is a valid token id. The "
            "digest recorded in the shard index is the only defence."
        ),
        requires=(),
        description=(
            "Flat uint16/uint32 .bin + .idx via memmap. A read is a page fault; "
            "there is nothing to decode."
        ),
    )
)

# ── the device seam ───────────────────────────────────────
# No CUDA anywhere in this decoder. It exists so the `lands_in="device"` branch is
# exercised in CI on a CPU box: pinning skipped, `num_workers` forced to 0, and no
# tensor construction. A real NVDEC/nvJPEG decoder registers exactly like this one
# and needs no change to the transport layer.
#
# Its integrity is `silent` because that is the truth about hardware decode: NVDEC
# emits green or garbage frames with nothing surfaced, which is the worst row in
# the whole table and undetectable at training time. `oracle` names the host
# decoder that materialization cross-checks it against.
register_decoder(
    DecoderSpec(
        name="fake.device",
        data_kind="image",
        factory=_lazy_factory(".fake_device"),
        media_types=(),
        suffixes=(),
        stages=frozenset({"read", "decode"}),
        output_dtype="uint8",
        output_layout="hwc",
        lands_in="device",
        integrity="silent",
        integrity_note=(
            "stand-in for a hardware decoder, which fails silently: garbage frames, "
            "nothing surfaced. Cross-checked against its `oracle` at materialization."
        ),
        oracle="image.png",
        requires=(),
        description="Test seam for device-landing decoders. Allocates no CUDA memory.",
    )
)

__all__ = ["DecoderSpec"]
