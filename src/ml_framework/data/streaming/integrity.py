"""
data/streaming/integrity.py
───────────────────────────
How much a decoder's output can be trusted, and what happens when it cannot.

The four classifications below are the honest summary of what each format
actually detects — not what its documentation implies. The distinction that
matters is not "does it ever fail" but **does a failure surface**:

    checked   the format carries a verified integrity primitive and damage RAISES
    loud      no checksum, but structural damage raises; late damage may slip
    silent    damage yields a valid-shaped, wrong-valued output and NO exception
    none      no integrity information exists at all

``silent`` is the dangerous one and the reason this module exists. An MP3 resyncs
past corruption and returns *shorter audio with no error*; a damaged H.264 stream
emits artifacted frames while libavcodec merely logs. Both produce tensors of
exactly the right shape, so they never trip any handler and simply degrade the
model. Nothing at training time can see them — which is what
:data:`REQUIRES_MATERIALIZATION` encodes, and why the offline pass earns its cost.

``none`` is worse in a quieter way: in a flat ``uint16`` token shard a flipped bit
is a *valid token id*. The only defence is a digest recorded in the shard index.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

# `Integrity` is vocabulary and lives in `core.types` beside `DataKind`, because
# `core.plugins.DecoderSpec` is typed against it and core may not import the data
# layer. Re-exported here so the policy code has one obvious import.
from ...core.types import INTEGRITIES, FrameworkError, Integrity
from .stages import Stage

__all__ = [
    "FAULT_KINDS",
    "INTEGRITIES",
    "INTEGRITY_SUMMARY",
    "REQUIRES_MATERIALIZATION",
    "RUNTIME_DETECTABLE",
    "CorruptSampleError",
    "FaultKind",
    "FaultLog",
    "Integrity",
    "SampleFault",
    "ShardUnusableError",
    "UnverifiedCorpusError",
    "should_verify",
]

# ── Classification ────────────────────────────────────────
# Damage in these classes raises during a normal training read, so the corrupt-
# sample policy can handle it where it happens.
RUNTIME_DETECTABLE: frozenset[Integrity] = frozenset({"checked", "loud"})

# Damage in these classes is INVISIBLE at training time. A run over such a corpus
# must have been through `mlf materialize`, which decode-probes every sample and
# cross-checks what the runtime cannot see (decoded duration against the frame
# count, decoded bytes against a recorded digest).
REQUIRES_MATERIALIZATION: frozenset[Integrity] = frozenset({"silent", "none"})

# One line each, for `mlf decoders --show` and for error messages. A spec may
# override with its own `integrity_note`; this is the fallback so a new decoder
# is never silently undocumented.
INTEGRITY_SUMMARY: dict[Integrity, str] = {
    "checked": "carries a verified checksum; damage raises",
    "loud": "no checksum, but structural damage raises",
    "silent": "damage yields a valid-shaped wrong result and no error",
    "none": "no integrity information exists at all",
}


def should_verify(integrity: Integrity, mode: str = "auto") -> bool:
    """Whether to recompute a sample's digest and compare it to the index.

    ``"auto"`` pays for it exactly where it buys something: a format that already
    validates its own frames (FLAC's CRC-16, PNG's Adler-32) gains nothing from a
    second hash, and hashing is ~1 GB/s of the read budget. The formats with no
    detection of their own get it.
    """
    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode != "auto":
        raise ValueError(f"verify_checksums must be auto|always|never, got '{mode}'")
    return integrity in REQUIRES_MATERIALIZATION


# ── Errors ────────────────────────────────────────────────
class CorruptSampleError(FrameworkError):
    """A sample could not be decoded and the policy said to stop.

    Only reachable under ``data.integrity.on_corrupt: "raise"``, which is refused
    for distributed runs — an abort that depends on which rank drew the bad sample
    is a rank-divergent abort, and that is a hang, not an error.
    """


class ShardUnusableError(FrameworkError):
    """Every sample in a shard failed, so substitution has nothing to draw.

    A shard-level failure rather than a sample-level one, and deterministic across
    ranks because the shard index is identical on all of them.
    """


class UnverifiedCorpusError(FrameworkError):
    """A corpus whose decoder cannot self-report damage was never materialized.

    Raised at bundle-build time. The escape hatch is
    ``data.integrity.allow_unverified: true`` — it exists, and it costs typing.
    """


# ── Faults ────────────────────────────────────────────────
FaultKind = Literal["missing", "truncated", "checksum", "decode_error", "shape", "duration"]

FAULT_KINDS: tuple[FaultKind, ...] = (
    "missing",
    "truncated",
    "checksum",
    "decode_error",
    "shape",
    "duration",
)


@dataclass(frozen=True, slots=True)
class SampleFault:
    """One sample that could not be trusted, and what was served instead.

    ``substituted_with`` is the whole record of the policy having worked: an index
    went in and a different index came out, so the *count* of samples produced was
    never affected. A fault with ``substituted_with is None`` was raised, not
    substituted.
    """

    index: int
    shard: str
    key: str
    stage: Stage | Literal["verify"]
    kind: FaultKind
    detail: str
    decoder: str
    substituted_with: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        where = f"{self.shard}:{self.key}"
        served = "" if self.substituted_with is None else f" -> served {self.substituted_with}"
        return f"[{self.kind}] {where} at {self.stage} via {self.decoder}: {self.detail}{served}"


class FaultLog:
    """Append-only per-worker fault record.

    A DataLoader worker is a separate process and cannot call the ``RunLogger``
    (which may hold an MLflow client). The channel is therefore the filesystem,
    the same one the repo already uses for per-run records — one JSON-lines file
    per (rank, worker), aggregated after the epoch by the training callback.

    **A healthy run writes zero bytes.** The file is opened on the first fault, so
    the presence of anything under ``faults/`` is itself the signal.
    """

    def __init__(self, directory: str | Path, *, rank: int = 0, worker: int = 0) -> None:
        self.directory = Path(directory)
        self.rank = rank
        self.worker = worker
        self._handle: Any = None
        self._count = 0

    @property
    def path(self) -> Path:
        return self.directory / f"rank{self.rank}-worker{self.worker}.jsonl"

    @property
    def count(self) -> int:
        return self._count

    def record(self, fault: SampleFault) -> None:
        if self._handle is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            # Line-buffered: a run killed mid-epoch still leaves every fault it
            # had already seen, which is when you most want them.
            self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self._handle.write(json.dumps(fault.to_dict(), sort_keys=True) + "\n")
        self._count += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> FaultLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @classmethod
    def for_worker(cls, directory: str | Path) -> FaultLog:
        """A log named for the current rank and DataLoader worker.

        Rank comes from the environment rather than ``torch.distributed`` on
        purpose: this module is imported inside worker processes, and reading an
        env var cannot fail, initialize CUDA, or pull torch into a path that has
        so far avoided it. Lightning sets these for every launcher it supports.
        """
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)
        worker = 0
        try:  # torch may legitimately be absent; a fault log must still work.
            import torch.utils.data as _torch_data

            info = _torch_data.get_worker_info()
            worker = 0 if info is None else int(info.id)
        except Exception:  # pragma: no cover - torch-free installs
            worker = 0
        return cls(directory, rank=rank, worker=worker)

    @staticmethod
    def aggregate(directory: str | Path) -> list[SampleFault]:
        """Every fault recorded under ``directory``, across ranks and workers.

        Returns an empty list when the directory does not exist — the healthy
        case, which must not be an error.
        """
        root = Path(directory)
        if not root.is_dir():
            return []
        faults: list[SampleFault] = []
        for path in sorted(root.glob("rank*-worker*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    faults.append(SampleFault(**json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    # A torn final line from a killed worker. Skipping it is
                    # right: this is a diagnostic channel, and refusing to report
                    # 900 faults because the 901st is half-written helps nobody.
                    continue
        return faults
