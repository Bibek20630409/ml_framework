"""
data/streaming/
───────────────
The staged read path: **materialize → index/sample → read → demux → decode**, and
the corrupt-sample policy that makes it safe to run under DDP.

Three things this package holds that the rest of the data layer deliberately does
not:

**1. Read, demux and decode are three stages, not two.** The formats disagree
about which of them exist — a pre-tokenized ``.bin`` shard has only a read, a JPEG
has only a decode, an MP4 has all three — so they are named separately and each
decoder *declares* which it performs. See ``decoders/base.py`` for why the methods
always exist anyway.

**2. Decode and tensor construction are independent axes.** Decode reverses a
compression scheme; tensor construction attaches a dtype and strides to a pointer.
:class:`~ml_framework.data.streaming.stages.Decoded` carries a buffer and a
description of it, and builds nothing. The tensor is built on the far side of a
boundary, in the collate function — which is what keeps a ``uint16`` token corpus
from being cast to ``int64`` in bulk, and a ``uint8`` image from being turned into
float32 anywhere but per batch.

**3. A corrupt sample is substituted, never skipped.** Under DDP every rank must
produce an identical number of batches or the next collective hangs with no error
message. ``continue`` is the one response that cannot preserve that, so it is not
a representable option — see ``integrity.py``.

Registration follows ``data/backends/__init__.py`` exactly: a spec plus a lazy
factory, never an import of the codec. ``mlf decoders`` therefore lists the H.264
path and names its pip command on an install with no FFmpeg binding.
"""

from __future__ import annotations

import os
from typing import Any

from ...core.registry import DECODERS, get_decoder

# Registration is an import side effect, exactly as `data/backends` does it. This
# module is the one place that has to happen, so `get_decoder` and `mlf decoders`
# both reach it by importing this package and nothing has to remember to.
from . import decoders as _decoders  # noqa: F401
from .integrity import (
    REQUIRES_MATERIALIZATION,
    RUNTIME_DETECTABLE,
    CorruptSampleError,
    FaultLog,
    SampleFault,
    ShardUnusableError,
    UnverifiedCorpusError,
    should_verify,
)
from .stages import Blob, BlobSource, DecodeContext, Decoded, Packet, SampleRef


def decoder_for(
    *,
    media_type: str = "",
    suffix: str = "",
    explicit: str | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    """The :class:`Decoder` for one sample, with ``data.decoder_params`` applied.

    One place resolves this, mirroring ``data/backends.engine_for``, so a source
    never has to remember to pass the params — forgetting would silently drop
    ``dtype="float32"`` and land float64 PCM, a 4x blowup that nothing reports.

    Resolution order, most specific first:

    1. ``explicit`` — what the config asked for (``data.params.decoder`` or
       ``--decoder``). Never second-guessed: if a corpus of ``.wav`` files should
       go through a different path, saying so must work.
    2. ``media_type`` — recorded per entry in the shard index at materialization,
       so it reflects what the bytes *are* rather than what they are named.
    3. ``suffix`` — the last resort, and the only one available before a corpus
       has been materialized.
    """
    name = explicit or _match(media_type=media_type, suffix=suffix)
    if name is None:
        raise LookupError(
            f"no decoder matches media_type={media_type!r} suffix={suffix!r}. "
            f"Registered decoders: {sorted(DECODERS.names())}. "
            "Set data.params.decoder (or --decoder) to choose one explicitly."
        )
    return get_decoder(name, **(params or {}))


def _match(*, media_type: str = "", suffix: str = "") -> str | None:
    """The registered decoder name for a media type or suffix, or ``None``.

    Reads specs rather than instantiating anything, so this is safe on a bare
    install — matching a ``.mp4`` still resolves to ``video.h264`` and lets
    ``get_decoder`` produce the pip hint, instead of failing here with a
    less useful "no decoder found".
    """
    if media_type:
        for spec in DECODERS.specs():
            if media_type in spec.media_types:
                return spec.name
    if suffix:
        wanted = suffix.lower()
        if not wanted.startswith("."):
            wanted = "." + wanted
        for spec in DECODERS.specs():
            if wanted in spec.suffixes:
                return spec.name
    return None


def suffix_of(path: str) -> str:
    """The lowercased extension of ``path``, or ``""``.

    A helper rather than a ``Path`` call at each site because a shard *key* is not
    always a filesystem path — a tar member name is a string, and ``Path`` on it
    would do the wrong thing with a leading ``./``.
    """
    _, ext = os.path.splitext(path)
    return ext.lower()


__all__ = [
    "REQUIRES_MATERIALIZATION",
    "RUNTIME_DETECTABLE",
    "Blob",
    "BlobSource",
    "CorruptSampleError",
    "DecodeContext",
    "Decoded",
    "FaultLog",
    "Packet",
    "SampleFault",
    "SampleRef",
    "ShardUnusableError",
    "UnverifiedCorpusError",
    "decoder_for",
    "should_verify",
    "suffix_of",
]
