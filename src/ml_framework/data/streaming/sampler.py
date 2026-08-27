"""
data/streaming/sampler.py
─────────────────────────
Shuffle order over a sharded corpus.

**No torch import anywhere in this module.** ``DataLoader`` duck-types a sampler
on ``__iter__`` and ``__len__``, the same trick ``TextDataset`` uses, so the
ordering logic stays testable without torch and stays out of the agnostic layer's
way.

## Why "block" is the default rather than a true permutation

A global permutation over sharded object storage is one seek per sample. Shards
exist precisely to make reads sequential, and a uniform shuffle throws that away —
the shuffle would be perfect and the loader would be IO-bound at 2% of line rate.

``block`` shuffles the *shard order*, then shuffles *within* each shard, then
concatenates. Every read stays inside one already-open shard, and the order within
a shard is a full permutation. What you lose is cross-shard mixing within a batch:
consecutive samples come from the same shard, so a corpus whose shards are
correlated (all of one class per shard) will produce correlated batches. That is a
property of how the corpus was *written*, and the fix is to interleave classes at
materialization — not to make every epoch pay for random access.

``global`` is a true permutation and is the right choice on local SSD or when the
corpus fits in page cache. ``none`` is index order, for debugging and for a
deterministic replay.

## Determinism, and what changing the world size actually changes

Three separate claims, worth stating separately because conflating them is how
people get this wrong:

1. **The global order is world-size-independent by construction.** It is a
   function of ``(index, seed, epoch)`` and never of rank. This emits a
   full-corpus order and lets Lightning's ``DistributedSampler`` wrapper cut it.
2. **Which rank sees which sample changes with the world size.** Unavoidable, and
   not something to pretend otherwise about.
3. **Batch counts stay equal** because ``DistributedSampler`` pads to
   ``ceil(N / W)``. Changing ``W`` changes the ≤``W-1`` duplicated samples at the
   tail and nothing else.

Rank partitioning is deliberately **not** implemented here. Doing it would be the
first rank-aware code in ``src/`` and would duplicate a padding rule Lightning
already gets right.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import numpy as np

from .shards import ShardIndex

Shuffle = Literal["block", "global", "none"]

SHUFFLES: tuple[Shuffle, ...] = ("block", "global", "none")


class ShardShuffleSampler:
    """Emits the full-corpus visit order for one epoch.

    ``set_epoch`` must be called before each epoch or every epoch sees the same
    order. Lightning's ``DistributedSamplerWrapper`` calls ``set_epoch`` on
    *itself*, not on the sampler it wraps, so the training callback drives this
    one explicitly.
    """

    def __init__(
        self,
        index: ShardIndex,
        *,
        seed: int = 42,
        shuffle: Shuffle = "block",
        epoch: int = 0,
        skip: int = 0,
    ) -> None:
        if shuffle not in SHUFFLES:
            raise ValueError(f"shuffle must be one of {list(SHUFFLES)}, got '{shuffle}'")
        self.index = index
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = epoch
        # Resume position, in GLOBAL ORDER units. Consumed once and then cleared:
        # a resumed epoch is short, and every epoch after it is whole.
        self.skip = skip
        self._groups = index.shard_groups()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        # A new epoch starts at the beginning, whatever a resume asked for.
        self.skip = 0

    def __len__(self) -> int:
        """The declared corpus size, minus anything a resume is skipping.

        ``len`` reflecting ``skip`` is what makes a resumed epoch actually shorter
        rather than silently replaying the samples already seen.
        """
        return max(0, self.index.n_samples - self.skip)

    def order(self) -> np.ndarray:
        """This epoch's full visit order, before any resume skip is applied.

        A materialized array rather than a generator, on purpose: it is what lets
        a mid-epoch resume be a **slice** instead of a replay, which is the real
        advantage of a map-style dataset over an iterable stream.
        """
        rng = np.random.default_rng([self.seed, self.epoch])

        if self.shuffle == "none":
            return np.arange(self.index.n_samples, dtype="int64")

        if self.shuffle == "global":
            return rng.permutation(self.index.n_samples).astype("int64")

        # block: shuffle the shard order, then shuffle within each shard.
        names = list(self._groups)
        if not names:
            return np.empty(0, dtype="int64")
        shard_order = [names[i] for i in rng.permutation(len(names))]
        chunks = [
            rng.permutation(np.asarray(self._groups[name], dtype="int64")) for name in shard_order
        ]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype="int64")

    def __iter__(self) -> Iterator[int]:
        order = self.order()
        if self.skip:
            order = order[self.skip :]
        return iter(int(i) for i in order)
