"""
data/streaming/dataset.py
─────────────────────────
:class:`StagedDataset` — index → read → demux → decode → one item, with a corrupt
sample substituted rather than skipped.

## The one property everything rests on

    ``__len__`` is a constant read from the shard index, and ``__getitem__`` is
    **total**: it returns exactly one item for every ``i in range(len(self))``,
    whatever the bytes say.

That is the whole DDP-safety argument, and it needs no rank-aware code to hold —
which is good, because there is none anywhere in ``src/``. The chain:

1. ``__len__`` returns ``index.n_samples``, parsed from ``shards.json``: a file
   byte-identical on every rank, and **never** derived from what decodes.
2. Lightning injects ``DistributedSampler`` (its default), which derives per-rank
   counts as ``ceil(N / W)`` with wrap-around padding — a pure function of that one
   integer, so identical on every rank by construction.
3. ``drop_last`` is a function of the same constant and ``batch_size``.
4. A fault is resolved *inside* ``__getitem__`` by substitution, which returns a
   different **index**. The number of items produced is invariant to the number of
   faults.

Nothing in that chain knows what a rank is. That is the point, and
``tests/data/test_staged_dataset.py`` pins each link.

## Why the fault ceiling never raises at runtime

``max_fault_rate`` is a *content-dependent* abort: rank 0 could trip it while rank
1 does not, and a rank-divergent abort is precisely the hang this module exists to
prevent. So at runtime the ceiling is counted, logged and reported — and it is
enforced hard in ``mlf materialize``, which is a single process, runs before any
collective exists, and can therefore fail safely.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from .integrity import (
    REQUIRES_MATERIALIZATION,
    CorruptSampleError,
    FaultKind,
    FaultLog,
    SampleFault,
    ShardUnusableError,
    UnverifiedCorpusError,
    should_verify,
)
from .shards import ShardIndex, digest_bytes
from .stages import DecodeContext, Decoded, SampleRef
from .substitute import SubstitutionPolicy

log = logging.getLogger(__name__)

# How many substitutions to chain before giving up on one index. A substitute that
# is itself corrupt is normal; four in a row means the shard is gone, and looping
# further would turn a data problem into a hang.
_MAX_SUBSTITUTIONS = 4


class StagedDataset:
    """A map-style dataset over a shard index.

    Duck-typed rather than subclassing ``torch.utils.data.Dataset``: the base class
    is empty (it defines ``__getitem__`` as a stub and nothing else), and not
    importing torch keeps this module usable from ``mlf materialize``, which has no
    reason to load a deep-learning runtime to check a corpus.
    """

    def __init__(
        self,
        index: ShardIndex,
        *,
        source: Any,
        decoder: Any,
        ctx: DecodeContext | None = None,
        on_corrupt: str = "substitute",
        substitute: str = "redraw",
        verify_checksums: str = "auto",
        integrity: str = "none",
        allow_unverified: bool = False,
        seed: int = 42,
        fault_dir: str | Path | None = None,
        transform: Any = None,
    ) -> None:
        if on_corrupt not in ("substitute", "raise"):
            raise ValueError(f"on_corrupt must be substitute|raise, got '{on_corrupt}'")

        # A corpus whose decoder cannot report its own damage must have been
        # probed offline, or the run is training on unknown data while believing
        # otherwise. The escape hatch exists and costs typing -- the same friction
        # idiom as `allow_temporal_leakage`.
        if integrity in REQUIRES_MATERIALIZATION and not index.is_materialized:
            if not allow_unverified:
                raise UnverifiedCorpusError(
                    f"this corpus decodes with integrity='{integrity}', which means damage "
                    "produces a valid-shaped wrong result and no error -- nothing at training "
                    "time can see it. Run `mlf materialize` to decode-probe the corpus, or "
                    "set data.integrity.allow_unverified: true to proceed anyway."
                )
            log.warning(
                "training on an unverified integrity='%s' corpus: damage in it is "
                "undetectable at this layer and will silently degrade the model",
                integrity,
            )

        self.index = index
        self.source = source
        self.decoder = decoder
        self.ctx = ctx or DecodeContext()
        self.on_corrupt = on_corrupt
        self.integrity = integrity
        self.transform = transform
        self.verify = should_verify(integrity, verify_checksums)  # type: ignore[arg-type]
        self.policy = SubstitutionPolicy(index, mode=substitute, seed=seed)  # type: ignore[arg-type]
        self.epoch = 0
        self._faults = 0
        self._fault_dir = Path(fault_dir) if fault_dir is not None else None
        self._log: FaultLog | None = None

    @property
    def decoder_lands_in(self) -> str:
        """``"host"`` or ``"device"``, from the decoder's spec.

        Surfaced on the dataset so a source can put it in ``bundle.meta`` without
        reaching into the registry itself. The transport layer reads it there to
        decide pinning and worker count.
        """
        from ...core.registry import DECODERS

        try:
            return str(DECODERS.get_spec(self.decoder.name).lands_in)
        except Exception:  # noqa: BLE001 - a hand-built decoder need not be registered
            return "host"

    # ── the invariant ──
    def __len__(self) -> int:
        """The **declared** sample count, never the decodable one.

        Deriving this from what succeeds is the single change that would break
        DDP, so it is a straight read from the manifest.
        """
        return self.index.n_samples

    def __getitem__(self, i: int) -> Any:
        """Exactly one item for every valid ``i``. Total by construction."""
        if not 0 <= i < len(self):
            raise IndexError(f"index {i} out of range for {len(self)} samples")

        index = i
        for attempt in range(_MAX_SUBSTITUTIONS + 1):
            try:
                decoded = self._decode(index)
            except ShardUnusableError:
                # Shard-level and deterministic across ranks, so raising cannot
                # desynchronize anything. Propagate it.
                raise
            except Exception as exc:  # noqa: BLE001 - any codec may raise anything
                fault = self._fault_for(index, exc)
                if self.on_corrupt == "raise":
                    self._record(fault)
                    raise CorruptSampleError(fault.summary()) from exc
                if attempt == _MAX_SUBSTITUTIONS:
                    self._record(fault)
                    raise ShardUnusableError(
                        f"gave up after {_MAX_SUBSTITUTIONS} substitutions starting from "
                        f"index {i}; the shard is unreadable rather than the samples"
                    ) from exc
                # Substitute FIRST, then record: `substituted_with` is the whole
                # record of the policy having worked -- an index went in and a
                # different index came out, so the sample *count* was unaffected.
                # Recording before choosing would leave that field null on every
                # fault and make the log unable to show the property that matters.
                replacement = self.policy.substitute(index, fault, epoch=self.epoch)
                self._record(replace(fault, substituted_with=replacement))
                index = replacement
                continue

            self.policy.note_success(index)
            return self._present(decoded, index)

        # Unreachable: the loop either returns or raises.
        raise AssertionError("substitution loop fell through")  # pragma: no cover

    # ── stages ──
    def ref(self, index: int) -> SampleRef:
        entry = self.index.entry(index)
        return SampleRef(
            index=entry.i,
            shard=entry.shard,
            key=entry.key,
            offset=entry.offset,
            nbytes=entry.nbytes,
            media_type=entry.media_type,
            digest=entry.digest,
        )

    def _decode(self, index: int) -> Decoded:
        ref = self.ref(index)
        blob = self.decoder.read(ref, source=self.source)
        if self.verify and ref.digest:
            self._verify(ref, blob)
        return self.decoder.decode(self.decoder.demux(blob), ctx=self.ctx)

    def _verify(self, ref: SampleRef, blob: Any) -> None:
        """Re-hash the bytes and compare against the index.

        Only reached for a decoder that cannot detect its own damage. For a flat
        token shard this is the entire defence: without it a flipped bit is a
        valid token id and nothing anywhere would notice.
        """
        actual = digest_bytes(blob.data)
        if actual != ref.digest:
            raise ValueError(
                f"checksum mismatch: index recorded {ref.digest[:12]}..., bytes hash to "
                f"{actual[:12]}.... The stored bytes changed since materialization."
            )

    def _present(self, decoded: Decoded, index: int) -> Any:
        """Hand back the decoded sample plus its label.

        The transform (and hence tensor construction) is applied by the caller's
        collate function, not here — decode and tensor construction are separate
        axes, and this is the boundary between them.
        """
        item = self.transform(decoded) if self.transform is not None else decoded
        label = self.index.entry(index).label
        return item if label is None else (item, label)

    # ── faults ──
    @property
    def faults(self) -> int:
        return self._faults

    @property
    def fault_rate(self) -> float:
        return self._faults / len(self) if len(self) else 0.0

    def set_epoch(self, epoch: int) -> None:
        """Keeps substitution deterministic per epoch rather than per run."""
        self.epoch = epoch

    def _fault_for(self, index: int, exc: Exception) -> SampleFault:
        entry = self.index.entry(index)
        return SampleFault(
            index=index,
            shard=entry.shard,
            key=entry.key,
            stage="verify" if isinstance(exc, ValueError) and "checksum" in str(exc) else "decode",
            kind=_classify(exc),
            detail=f"{type(exc).__name__}: {exc}",
            decoder=getattr(self.decoder, "name", type(self.decoder).__name__),
        )

    def _record(self, fault: SampleFault) -> None:
        self._faults += 1
        if self._fault_dir is None:
            return
        if self._log is None:
            # Per (rank, worker), opened on the first fault -- a healthy run
            # writes zero bytes, so anything under faults/ is itself the signal.
            self._log = FaultLog.for_worker(self._fault_dir)
        self._log.record(fault)

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None
        for closeable in (self.decoder, self.source):
            closer = getattr(closeable, "close", None)
            if closer is not None:
                closer()


def _classify(exc: Exception) -> FaultKind:
    """Map a codec's exception onto the fault vocabulary.

    Every codec raises its own types, so this is a best-effort bucketing for
    reporting — the fault is recorded either way, and ``detail`` keeps the real
    exception text. Guessing wrong costs a label in a report, never a decision.
    """
    if isinstance(exc, FileNotFoundError):
        return "missing"
    if isinstance(exc, EOFError):
        return "truncated"
    text = str(exc).lower()
    if "checksum" in text or "crc" in text:
        return "checksum"
    if "truncat" in text or "incomplete" in text:
        return "truncated"
    if "shape" in text or "dimension" in text:
        return "shape"
    return "decode_error"
