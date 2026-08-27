"""
data/streaming/decoders/tokens.py
─────────────────────────────────
Pre-tokenized text: a flat ``.bin`` of token ids plus a ``.idx`` of offsets
(Megatron's ``IndexedDataset``, MDS, and every corpus shaped like them).

**This decoder has no demux stage and no decode stage.** That is not a gap in the
implementation — there is nothing to reverse. The bytes on disk *are* the token
ids; a read is an ``np.memmap`` slice, which is a page fault serviced by a kernel
DMA into the page cache, not a call into a codec. ``DecoderSpec.stages`` says
``{"read"}`` and means it.

Keeping it in the decoder registry anyway is deliberate. The registry's job is to
describe *how a corpus becomes a buffer*, and "it costs nothing" is one of the
answers — one that only reads as informative next to the rows where decode
dominates. The alternative, special-casing token corpora outside the registry,
would hide the one row whose integrity story is the worst of all.

## Integrity: ``none``, and what that actually means

There is **no integrity check at any layer**. Not a weak one — none. A flipped bit
in a ``uint16`` token id yields a different, entirely valid token id. The read
succeeds, the shape is right, the loss is finite, and the model trains on a
corrupted corpus for as long as you let it.

The only defence is the digest recorded per entry in the shard index, which is why
``verify_checksums: "auto"`` turns verification **on** for this decoder and off
for the ones that already carry a CRC.

## The dtype trade-off, and why the cast is not here

With a vocabulary of 65,536 or fewer, the choice is:

* ``uint16`` on disk — 2 bytes/token, then ``.astype(int64)`` per batch. At 15T
  tokens that is **30 TB**, plus a CPU cast every step.
* ``int32`` on disk — skipping the cast is cheaper but *still not free*
  (``int32 -> int64`` is also a copy). **60 TB**, and 2x the IO and page-cache
  pressure, permanently.

``uint16`` wins for essentially everyone: the cast is a memcpy-bound operation
over a few MB per step, while 30 TB of extra IO is forever. So this decoder
returns ``uint16`` and **refuses to widen it**. torch has no usable ``uint16``
arithmetic, so the cast has to happen somewhere — it happens in the collate
function, per *batch*. Doing it here would mean casting the whole corpus and
quadrupling its footprint to save nothing, which is precisely the mistake storing
``int64`` on disk makes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np

from ..stages import DecodeContext, Decoded, Packet
from .base import BaseDecoder

# Token id widths that make sense on disk. `int64` is deliberately absent: it is
# always the wrong choice for storage, and accepting it here would make the
# framework complicit in a 4x storage bill. The cast to int64 belongs in the
# collate, which is where a batch — not a corpus — pays for it.
_TOKEN_DTYPES: frozenset[str] = frozenset({"uint16", "uint32", "int32"})

# A uint16 id space. Named because the number is the whole reason uint16 is the
# default rather than a micro-optimization.
UINT16_VOCAB_LIMIT = 65_536


class TokenDecoder(BaseDecoder):
    """A flat token shard → a ``uint16`` view. No demux, no decode, no copy."""

    name: ClassVar[str] = "text.tokens"

    def __init__(self, *, dtype: str = "uint16") -> None:
        if dtype not in _TOKEN_DTYPES:
            raise ValueError(
                f"text.tokens dtype must be one of {sorted(_TOKEN_DTYPES)}, got '{dtype}'. "
                "int64 is intentionally not offered: it quadruples storage to save a "
                "per-batch cast that is memcpy-bound either way."
            )
        self.dtype = dtype

    def decode(self, packets: Iterable[Packet], *, ctx: DecodeContext) -> Decoded:
        """Reinterpret the blob as token ids. A view, never a copy.

        ``ctx.dtype`` is deliberately **ignored**: the on-disk width is a property
        of the corpus, not of the run, and honouring a request to widen it here
        would perform exactly the whole-corpus cast this module exists to avoid.
        """
        payload = next(iter(packets)).data

        if isinstance(payload, np.ndarray):
            # The memmap path: already an array of the right width, because the
            # source memmapped it with this dtype. `.view` rather than `.astype`
            # so the page cache stays the only copy that exists.
            tokens = payload if payload.dtype == self.dtype else payload.view(self.dtype)
        else:
            # `frombuffer` wraps the buffer without copying it. The result is
            # read-only when the buffer is, which is correct and is why the
            # tensor-construction path checks WRITEABLE before `from_numpy`.
            tokens = np.frombuffer(payload, dtype=self.dtype)

        return Decoded(
            array=tokens,
            layout="tokens",
            dtype=self.dtype,
            lands_in="host",
            rate=None,
            meta={"n_tokens": int(tokens.size), "cast_to_int64_in": "collate_fn"},
        )


def build_decoder(**params: Any) -> TokenDecoder:
    """Factory named by :data:`DecoderSpec.factory`.

    Unknown keys raise rather than being ignored: this function is the sole owner
    of ``data.decoder_params`` for this decoder, and a silently dropped ``dtype``
    would reinterpret the entire corpus at the wrong width — which, for a format
    with no integrity check, produces valid-looking garbage.
    """
    unknown = set(params) - {"dtype"}
    if unknown:
        raise ValueError(
            f"text.tokens got unknown decoder_params {sorted(unknown)}; accepts: dtype"
        )
    return TokenDecoder(dtype=str(params.get("dtype", "uint16")))
