"""
data/streaming/decoders/base.py
───────────────────────────────
The :class:`Decoder` protocol and the base class that gives every decoder a
working ``read`` and ``demux``.

**Why defaults rather than optional methods.** Read, demux and decode are three
distinct stages, but most formats do not have all three: a JPEG has no demux, and
a pre-tokenized shard has neither demux nor decode. The obvious design — let a
decoder omit the stages it does not implement — pushes a ``hasattr`` check into
every caller, and callers get it wrong.

So the methods **always exist**. ``read`` delegates to the source; ``demux``
yields a single packet wrapping the whole blob; only ``decode`` is abstract. A
caller runs all three unconditionally and the ones that are no-ops cost a
function call and no copy.

What is *real* is therefore **declared, not inferred**: ``DecoderSpec.stages``
says which stages do work, and its two consumers (``mlf decoders --show`` and the
materialization pass, which only times a declared stage) read that declaration
instead of probing.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, ClassVar, Protocol, runtime_checkable

from ..stages import Blob, BlobSource, DecodeContext, Decoded, Packet, SampleRef


@runtime_checkable
class Decoder(Protocol):
    """One media format's path from a byte range to a buffer.

    Structural, not an ABC — the same choice :class:`DataBackend` makes, so a
    third-party decoder never has to import this package to satisfy it.
    """

    name: ClassVar[str]

    def read(self, ref: SampleRef, *, source: BlobSource) -> Blob: ...

    def demux(self, blob: Blob) -> Iterator[Packet]: ...

    def decode(self, packets: Iterable[Packet], *, ctx: DecodeContext) -> Decoded: ...

    def close(self) -> None: ...


class BaseDecoder:
    """Working defaults for the two stages most formats do not have.

    Subclasses override only what is genuinely a stage for their format, and
    declare the same set on their :class:`DecoderSpec`. The two must agree; a test
    pins that they do.
    """

    name: ClassVar[str] = "base"

    def read(self, ref: SampleRef, *, source: BlobSource) -> Blob:
        """Stage 1, for everyone: a positioned read through the source.

        Addressing is the source's job, not the codec's — what differs between a
        directory, a tar shard and a memmap is *where the bytes are*, which is
        orthogonal to how they decompress. That is the seam :class:`BlobSource`
        draws, and it is why no decoder here opens a file itself.
        """
        return Blob(data=source.read_range(ref), ref=ref, media_type=ref.media_type)

    def demux(self, blob: Blob) -> Iterator[Packet]:
        """Stage 2, degenerate: one packet, the whole blob, no copy.

        Overridden only by the container formats — MP3 (frame-sync scan), Ogg/Opus
        (page parse) and MP4 (box traversal plus the keyframe index).
        """
        yield Packet(data=blob.data)

    def decode(self, packets: Iterable[Packet], *, ctx: DecodeContext) -> Decoded:
        """Stage 3. No default — every decoder answers this one."""
        raise NotImplementedError(f"{type(self).__name__} must implement decode()")

    def close(self) -> None:
        """Release any handle the decoder holds. Idempotent; most hold none."""

    # ── convenience ──
    def decode_ref(self, ref: SampleRef, *, source: BlobSource, ctx: DecodeContext) -> Decoded:
        """Run all three stages for one sample.

        The single entry point ``StagedDataset`` uses, so the stage order lives
        here once rather than being re-spelled by every caller.
        """
        return self.decode(self.demux(self.read(ref, source=source)), ctx=ctx)

    def __enter__(self) -> BaseDecoder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def as_bytes(data: Any) -> bytes:
    """A ``bytes`` copy of whatever a packet is carrying.

    Used only by decoders that must hand a buffer to a C library expecting a
    contiguous byte string. Named explicitly because it **is a copy** — the one
    place in this path where that is unavoidable, and worth being able to grep for.
    """
    if isinstance(data, bytes):
        return data
    return bytes(memoryview(data))
