"""
data/streaming/state.py
───────────────────────
Where a run had got to in its data, so a resume continues rather than restarts.

Lightning already checkpoints the model, the optimizer and the epoch counter.
What it cannot checkpoint is *which samples this epoch had already served*, and
without that a mid-epoch resume silently replays them — an extra partial pass over
a subset of the corpus, which shows up as a small unexplained difference in the
loss curve and nothing else.

:class:`LoaderState` is the missing piece. It rides in the datamodule's
``state_dict``, which Lightning stores in the checkpoint, so ``mlf train --resume``
already carries it and there is no new vehicle to build.

## Why ``samples_seen`` is in global-order units

Not per-rank counts. The sampler emits a full-corpus order that is a function of
``(seed, epoch)`` and never of rank, and the distributed wrapper cuts it. So a
position in that order means the same thing at any world size: a resume at
``W = 8`` after a crash at ``W = 4`` lands at the same place in the corpus, and the
partition simply re-cuts around it.

What genuinely differs is *which rank* sees which sample, so ``world_size`` is
recorded and a mismatch warns. It does not refuse — the global position is still
correct, and refusing would make elastic resumption impossible for a difference
that is usually intended.

## What does refuse

A changed ``index_digest``. Resuming a checkpoint against a corpus that has been
re-materialized means "sample 41,000" now names different bytes, so the position
is meaningless and continuing would replay a different dataset while reporting it
as the same run. That is the same posture ``read_manifest`` takes on a version
mismatch: refuse, and say what changed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ...core.types import FrameworkError

log = logging.getLogger(__name__)

STATE_VERSION = 1
STATE_FILENAME = "loader_state.json"


class LoaderStateError(FrameworkError):
    """A checkpoint's data position cannot be honoured against this corpus."""


@dataclass(frozen=True, slots=True)
class LoaderState:
    """A run's position in its data. Frozen: advancing it produces a new one."""

    version: int = STATE_VERSION
    # Which corpus this position refers to. The field that makes a resume safe.
    index_digest: str = ""
    seed: int = 42
    shuffle: str = "block"
    epoch: int = 0
    # Within this epoch, in GLOBAL ORDER units -- not per-rank, see the module docs.
    samples_seen: int = 0
    global_step: int = 0
    # Recorded, not required to match. A mismatch warns.
    world_size: int = 1
    faults: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LoaderState:
        version = int(raw.get("version", 0))
        if version > STATE_VERSION:
            raise LoaderStateError(
                f"loader state version {version} is newer than this build understands "
                f"({STATE_VERSION}). Upgrade ml-framework, or start the run fresh."
            )
        known = {f for f in cls.__slots__ if f != "version"}
        return cls(version=version, **{k: v for k, v in raw.items() if k in known})

    def advance(self, *, samples: int, steps: int = 0, faults: int = 0) -> LoaderState:
        from dataclasses import replace

        return replace(
            self,
            samples_seen=self.samples_seen + samples,
            global_step=self.global_step + steps,
            faults=self.faults + faults,
        )

    def next_epoch(self) -> LoaderState:
        """A fresh epoch starts at zero. ``samples_seen`` is within-epoch."""
        from dataclasses import replace

        return replace(self, epoch=self.epoch + 1, samples_seen=0)

    def resume_skip(self, *, index_digest: str, world_size: int = 1) -> int:
        """How many entries of this epoch's order to skip, having validated the state.

        Returns the skip rather than mutating anything, so the caller decides what
        to do with it — and so the validation happens exactly once, at the point
        the position is actually used.
        """
        if index_digest and self.index_digest and index_digest != self.index_digest:
            raise LoaderStateError(
                "the shard index changed since this checkpoint was written "
                f"(was {self.index_digest[:12]}..., now {index_digest[:12]}...); resuming "
                "would replay a different corpus under the same run. Re-materialize and "
                "start fresh, or resume against the original index."
            )
        if world_size != self.world_size:
            # Warn, do not refuse: the global position is still correct, and the
            # per-rank data seen genuinely differs whatever we do about it.
            log.warning(
                "resuming at world_size=%d but the checkpoint was written at %d. The global "
                "position in the corpus is preserved; which rank sees which sample is not.",
                world_size,
                self.world_size,
            )
        return self.samples_seen

    # ── the human-readable copy ──
    def write(self, directory: str | Path) -> Path:
        """Write ``loader_state.json`` beside the model.

        The checkpoint is the authority; this is the audit record, the same way
        ``config.json`` sits beside the manifest. Always written, including at
        ``samples_seen: 0`` — an absent file would be indistinguishable from a run
        that predates the phase.
        """
        path = Path(directory) / STATE_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", "utf-8")
        return path

    @classmethod
    def read(cls, directory: str | Path) -> LoaderState | None:
        path = Path(directory) / STATE_FILENAME
        if not path.is_file():
            return None
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
