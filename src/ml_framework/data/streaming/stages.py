"""
data/streaming/stages.py
────────────────────────
The vocabulary of the staged read path's **head** — read → demux → decode — and
the types that flow between them.

The whole pipeline is seven stages (``core.types.STAGES``); this module owns the
first three, and the tail — construct → transform → h2d → gpu_transform — is
declared by ``BasePreprocessor.stages`` and executed by the transport layer. The
seam is exactly where :class:`Decoded` stops.

Three head stages, not two. The distinction is not pedantry — it is the only way
to describe the formats honestly:

* a pre-tokenized ``.bin`` shard has **neither** demux nor decode (a memmap slice
  is a page fault, not a call),
* an MP3 has a real demux stage (frame-sync scan) *and* a real decode stage,
* a GPU decoder has decode but produces no host array to build a tensor from.

**Decode and tensor construction are independent axes.** Decode reverses a
compression scheme; tensor construction attaches a dtype, a shape and strides to
a pointer. Several formats have one without the other, so :class:`Decoded` carries
the buffer and *what it is* and deliberately builds nothing. Tensor construction
happens on the far side of a boundary, in the collate function — which is what
makes the independence a code location rather than a claim.

**No torch in this module.** numpy is a base dependency and is the currency of the
agnostic data layer (``data/types.py`` rule 1); torch enters at the Lightning
adapter and nowhere earlier.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Protocol, runtime_checkable

import numpy as np

# `Stage`/`Layout`/`LandsIn` are vocabulary, so they live in `core.types` beside
# `DataKind` and `Payload` rather than here — `core.plugins.DecoderSpec` is typed
# against them, and core may not import from the data layer. Re-exported so a
# decoder module has one obvious import.
from ...core.types import (
    DECODER_STAGES,
    LANDS_IN,
    LAYOUTS,
    STAGES,
    TAIL_STAGES,
    DecoderStage,
    LandsIn,
    Layout,
    Stage,
)

__all__ = [
    "DECODER_STAGES",
    "LANDS_IN",
    "LAYOUTS",
    "STAGES",
    "TAIL_STAGES",
    "Blob",
    "BlobSource",
    "DecodeContext",
    "Decoded",
    "DecoderStage",
    "LandsIn",
    "Layout",
    "Packet",
    "SampleRef",
    "Stage",
]


# ── What flows between the stages ─────────────────────────
@dataclass(frozen=True, slots=True)
class SampleRef:
    """Where one sample's bytes are. The unit a sampler resolves an index into.

    A positioned read — ``(shard, offset, nbytes)`` — rather than a path, because
    that is what a tar member or an object-store range request actually is. A
    one-file-per-sample layout is the degenerate case where ``offset`` is 0 and
    ``nbytes`` is the whole file.

    ``digest`` is the shard index's record of these bytes. It is the *only* defence
    a format with no integrity information has, which is why it lives on the ref
    rather than being recomputed by whoever happens to care.
    """

    index: int
    shard: str
    key: str
    offset: int = 0
    nbytes: int = 0
    media_type: str = ""
    digest: str = ""


@dataclass(frozen=True, slots=True)
class Blob:
    """Stage-1 output: the bytes, and where they came from.

    ``data`` is an ``np.ndarray`` for the memmap case — a *view*, not a copy — and
    ``bytes``/``memoryview`` otherwise. Typed as a union rather than normalized to
    ``bytes`` on purpose: normalizing is exactly the copy this whole path exists to
    avoid, and a 2-byte-per-token corpus cannot afford it.
    """

    data: bytes | memoryview | np.ndarray
    ref: SampleRef
    media_type: str = ""

    @property
    def nbytes(self) -> int:
        if isinstance(self.data, np.ndarray):
            return int(self.data.nbytes)
        return len(self.data)


@dataclass(frozen=True, slots=True)
class Packet:
    """Stage-2 output: one demuxed elementary-stream unit.

    Only the container formats produce more than one of these. ``keyframe`` and
    ``pts`` exist because seeking a video to a clip boundary is a keyframe search,
    not a byte offset — the demuxer is the only layer that can answer that.

    ``data`` admits ``np.ndarray`` only to carry the *degenerate* case:
    ``BaseDecoder.demux`` yields one packet wrapping the whole blob, and a memmap
    blob is already an array. Normalizing it to bytes there would be a copy for a
    stage that does not exist. A real container demuxer always yields bytes.
    """

    data: bytes | memoryview | np.ndarray
    stream: int = 0
    keyframe: bool = False
    pts: int | None = None
    time_base: tuple[int, int] | None = None
    opaque: Any = None
    """The demuxer's native packet, when decode needs more than the bytes.

    A real elementary-stream packet is only decodable inside the codec context
    that produced it — an H.264 slice means nothing without its SPS/PPS. Formats
    like that put the native handle here and decode consumes it; ``data`` still
    carries the bytes, so counting and hashing packets works uniformly.

    ``None`` for every format whose packets are self-contained, which is all of the
    single-frame ones.
    """


@dataclass(frozen=True, slots=True)
class Decoded:
    """Stage-3 output: a buffer, and what it is.

    **Nothing here builds a tensor.** ``dtype`` and ``layout`` are recorded as
    strings describing the buffer as the decoder left it, so the cost of turning
    it into a tensor is visible and paid in one known place (the collate function)
    instead of being smuggled into the decoder.

    That is what keeps the two axes separate in practice:

    * a token shard decodes to ``dtype="uint16"``, and the ``astype(int64)`` — a
      real copy, 4x — happens per *batch*, never per corpus;
    * a JPEG decodes to ``dtype="uint8", layout="hwc"``, and the permute +
      float32 + ``/255`` copy happens per batch too;
    * a device decoder sets ``lands_in="device"``, ``array`` is not numpy, and no
      tensor is constructed at all.
    """

    array: Any
    """``np.ndarray`` when ``lands_in == "host"``; an opaque device handle otherwise.

    ``Any`` rather than a union, for the reason ``Split.x`` is ``Any``: naming the
    alternative would force this module to import the library that defines it.
    """
    layout: Layout
    dtype: str
    lands_in: LandsIn = "host"
    # Sample rate for audio, frames per second for video. None when the notion
    # does not apply (images, tokens).
    rate: int | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """The buffer's shape, or ``()`` when the decoder landed on a device.

        A device handle has a shape too, but asking for it would mean importing
        the framework that owns it — so this answers only for what it can see.
        """
        array = self.array
        shape = getattr(array, "shape", None)
        return tuple(shape) if shape is not None else ()


@dataclass(frozen=True, slots=True)
class DecodeContext:
    """The narrow context a decoder is given.

    Never the whole ``ExperimentConfig`` — the same rule ``RunContext`` and
    ``BuildContext`` follow, and for the same reason: a decoder that can reach
    ``config.model`` will eventually read it.

    ``dtype`` is the load-bearing field. ``soundfile.read()`` defaults to
    **float64**, which is a silent 4x blowup on every audio sample in the corpus;
    pinning it here means the decision is made once, by the resolver, rather than
    forgotten once per decoder.
    """

    dtype: str | None = None
    layout: Layout | None = None
    target_rate: int | None = None
    seed: int = 42
    # Recompute the sample's digest and compare it against `SampleRef.digest`.
    # Off by default because it is only worth paying for formats that carry no
    # integrity information of their own — see `integrity.should_verify`.
    verify: bool = False


# ── Stage 1: where bytes come from ────────────────────────
@runtime_checkable
class BlobSource(Protocol):
    """Byte access for one storage layout, shared by every decoder.

    Stage 1 is the same operation for every format — a positioned read — so it is
    owned here once rather than reimplemented per codec. What differs between a
    directory of files, a tar shard and a ``.bin``/``.idx`` pair is *addressing*,
    not decoding, and that is exactly the seam this protocol draws.

    Implementations live in ``sources_io.py``.
    """

    def read_range(self, ref: SampleRef) -> bytes | memoryview | np.ndarray:
        """The bytes for one sample.

        May return a *view* (a memmap slice) rather than a copy. Callers must not
        assume the buffer is writeable or that it outlives the source.
        """
        ...

    def open(self, shard: str) -> BinaryIO:
        """A file object for a whole shard, for decoders that need to seek."""
        ...

    def close(self) -> None:
        """Release every open handle. Idempotent."""
        ...
