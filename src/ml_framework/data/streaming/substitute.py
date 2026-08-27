"""
data/streaming/substitute.py
────────────────────────────
What to serve instead when a sample cannot be decoded.

**An index goes in and a different index comes out.** Never ``None``, never a
variable number of items, never a raised exception in the default policy. That
signature *is* the DDP-safety argument: the number of samples a dataset produces
is invariant to how many of them are corrupt, so two ranks reading different
shards still build the same number of batches, and the next collective does not
hang waiting for a rank that quietly dropped three samples.

Which is why there is no ``skip``. Not a rejected option — an absent one. It is
the single response that cannot preserve the count, and making it unrepresentable
is cheaper than documenting why not to use it.

## The two policies

``redraw``  draws a different index **from the same shard**. Two reasons, and the
            second is the one that matters at scale: it is a fair-ish substitute,
            and it keeps the read inside an already-open shard rather than seeking
            across the corpus to service a failure.

``repeat``  serves the previous successfully-decoded sample. Cheaper still (it is
            usually in page cache) and completely predictable, at the cost of
            duplicating one sample rather than sampling a fresh one.

Both are deterministic in ``(seed, epoch, index)``: the same corrupt sample
substitutes to the same replacement on every rank and on every re-run, so a run
that is reproducible with a clean corpus stays reproducible with a damaged one.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from .integrity import SampleFault, ShardUnusableError
from .shards import ShardIndex

log = logging.getLogger(__name__)

SubstituteMode = Literal["redraw", "repeat"]

SUBSTITUTE_MODES: tuple[SubstituteMode, ...] = ("redraw", "repeat")

# How many fresh draws to try before falling back to `repeat`. Bounded because a
# shard that is mostly corrupt would otherwise turn every access into a scan, and
# a loader that quietly runs 40x slower is its own kind of failure.
_MAX_REDRAWS = 4


class SubstitutionPolicy:
    """Resolves a faulted index to a healthy one, deterministically."""

    def __init__(
        self,
        index: ShardIndex,
        *,
        mode: SubstituteMode = "redraw",
        seed: int = 42,
    ) -> None:
        if mode not in SUBSTITUTE_MODES:
            raise ValueError(f"substitute must be one of {list(SUBSTITUTE_MODES)}, got '{mode}'")
        self.index = index
        self.mode = mode
        self.seed = seed
        self._groups = index.shard_groups()
        # Per-worker memory of the last good index, for `repeat`. Not shared and
        # not persisted: it is a convenience, and its only correctness requirement
        # is that it names *some* healthy index.
        self._last_good: int | None = None
        # Indices known bad this epoch, so a redraw does not hand back a sample
        # that already failed. Bounded by the fault count, which the ceiling keeps
        # small in any run worth continuing.
        self._known_bad: set[int] = set()

    def note_success(self, index: int) -> None:
        """Record a healthy read, so ``repeat`` has something to repeat."""
        self._last_good = index

    def substitute(self, index: int, fault: SampleFault, *, epoch: int = 0) -> int:
        """The index to read **instead**. Total: always returns one."""
        self._known_bad.add(index)
        chosen = self._redraw(index, epoch=epoch) if self.mode == "redraw" else self._repeat(index)
        log.warning("substituting sample %d with %d: %s", index, chosen, fault.summary())
        return chosen

    def _redraw(self, index: int, *, epoch: int) -> int:
        """A different index from the same shard, chosen deterministically."""
        shard = self.index.entry(index).shard
        members = self._healthy_members(shard, exclude=index)
        if not members:
            # Every sample in this shard is known bad. That is a shard-level
            # failure, and it is deterministic across ranks because the index is —
            # so raising here cannot desynchronize anything.
            raise ShardUnusableError(
                f"every sample in shard '{shard}' failed to decode ({len(self._groups[shard])} "
                "samples). The shard is unreadable, not the samples; check the file rather "
                "than raising data.integrity.max_fault_rate."
            )
        # Seeded on (seed, epoch, index) so the replacement for a given corrupt
        # sample is identical on every rank and across re-runs.
        rng = np.random.default_rng([self.seed, epoch, index])
        for _ in range(_MAX_REDRAWS):
            candidate = int(members[rng.integers(len(members))])
            if candidate not in self._known_bad:
                return candidate
        # Bounded, so a mostly-corrupt shard degrades to the cheap policy rather
        # than to a scan.
        return self._repeat(index)

    def _repeat(self, index: int) -> int:
        """The previous good index, or the first healthy one in the same shard."""
        if self._last_good is not None and self._last_good not in self._known_bad:
            return self._last_good
        # A worker's very first index faulted, so there is nothing to repeat. Scan
        # the shard for anything not yet known bad -- linear, and it happens at
        # most once per worker.
        shard = self.index.entry(index).shard
        members = self._healthy_members(shard, exclude=index)
        if not members:
            raise ShardUnusableError(
                f"shard '{shard}' has no healthy sample to substitute with, and this "
                "worker has not yet read one."
            )
        return int(members[0])

    def _healthy_members(self, shard: str, *, exclude: int) -> list[int]:
        return [i for i in self._groups.get(shard, ()) if i != exclude and i not in self._known_bad]
