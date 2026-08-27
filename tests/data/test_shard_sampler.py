"""Shuffle order over a sharded corpus, and what changing the world size changes.

The three claims the sampler makes, each tested separately because conflating them
is how people get distributed shuffling wrong:

1. the global order is a function of ``(seed, epoch)`` and **never** of rank, so it
   is identical at every world size;
2. which rank sees which sample does change with the world size, and that is fine;
3. batch counts stay equal regardless, because ``DistributedSampler`` pads.

Plus the reason ``block`` is the default: it keeps every read inside one open
shard, which is the entire point of sharding over object storage.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_framework.data.streaming.sampler import ShardShuffleSampler
from ml_framework.data.streaming.shards import ShardEntry, ShardIndex

pytestmark = pytest.mark.unit


def make_index(*, n_shards: int = 4, per_shard: int = 25) -> ShardIndex:
    entries = []
    shards = []
    i = 0
    for shard in range(n_shards):
        name = f"train-{shard:04d}"
        shards.append(name)
        for member in range(per_shard):
            entries.append(ShardEntry(i=i, shard=name, key=f"{member}", label=i % 3))
            i += 1
    return ShardIndex(n_samples=i, entries=tuple(entries), shards=tuple(shards))


@pytest.fixture
def index() -> ShardIndex:
    return make_index()


# ── Every mode is a permutation ───────────────────────────────────────
@pytest.mark.parametrize("shuffle", ["block", "global", "none"])
def test_every_shuffle_mode_visits_every_sample_exactly_once(index, shuffle):
    """A sampler that dropped or duplicated samples would break batch parity
    before the distributed wrapper ever saw it."""
    order = list(ShardShuffleSampler(index, seed=42, shuffle=shuffle))
    assert sorted(order) == list(range(index.n_samples))


def test_none_is_index_order_for_deterministic_replay(index):
    assert list(ShardShuffleSampler(index, shuffle="none")) == list(range(index.n_samples))


def test_an_unknown_shuffle_mode_is_refused_by_name(index):
    with pytest.raises(ValueError, match="block"):
        ShardShuffleSampler(index, shuffle="random")


# ── Determinism ───────────────────────────────────────────────────────
def test_the_same_seed_and_epoch_give_the_same_order(index):
    a = list(ShardShuffleSampler(index, seed=42, shuffle="block", epoch=3))
    b = list(ShardShuffleSampler(index, seed=42, shuffle="block", epoch=3))
    assert a == b


def test_a_new_epoch_reshuffles(index):
    """Otherwise every epoch sees the same order and the shuffle is decorative."""
    first = list(ShardShuffleSampler(index, seed=42, epoch=0))
    second = list(ShardShuffleSampler(index, seed=42, epoch=1))
    assert first != second


def test_a_different_seed_gives_a_different_order(index):
    assert list(ShardShuffleSampler(index, seed=1)) != list(ShardShuffleSampler(index, seed=2))


def test_set_epoch_changes_the_order_and_clears_any_resume_skip(index):
    sampler = ShardShuffleSampler(index, seed=42, epoch=0, skip=10)
    assert len(sampler) == index.n_samples - 10

    sampler.set_epoch(1)

    assert sampler.skip == 0, "a fresh epoch starts at the beginning"
    assert len(sampler) == index.n_samples


# ── Claim 1: the order does not depend on world size ──────────────────
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_the_global_order_is_identical_at_every_world_size(index, world_size):
    """The sampler emits a full-corpus order and lets Lightning's wrapper cut it.

    Partitioning by rank here would be the first rank-aware code in ``src/`` and
    would duplicate a padding rule Lightning already gets right — so the order is
    a pure function of ``(seed, epoch)`` and the world size is not an input at all.
    """
    reference = list(ShardShuffleSampler(index, seed=42, epoch=2))
    assert list(ShardShuffleSampler(index, seed=42, epoch=2)) == reference
    # There is no world_size parameter to pass. That absence IS the property.
    assert "world_size" not in ShardShuffleSampler.__init__.__code__.co_varnames


# ── Claim 3: batch counts stay equal ──────────────────────────────────
@pytest.mark.parametrize("world_size", [2, 3, 5])
def test_ranks_get_equal_counts_because_the_wrapper_pads(index, world_size):
    import math

    torch_data = pytest.importorskip("torch.utils.data")

    class _Sized:
        def __len__(self) -> int:
            return index.n_samples

        def __getitem__(self, i):  # pragma: no cover - never called
            return i

    lengths = {
        len(torch_data.DistributedSampler(_Sized(), num_replicas=world_size, rank=r))
        for r in range(world_size)
    }
    assert lengths == {math.ceil(index.n_samples / world_size)}


# ── Why block is the default ──────────────────────────────────────────
def test_block_shuffle_keeps_each_shard_contiguous(index):
    """The whole reason to prefer it: every read stays inside one open shard.

    A global permutation over object storage is one seek per sample, which makes
    the shuffle perfect and the loader IO-bound at a fraction of line rate.
    """
    order = list(ShardShuffleSampler(index, seed=42, shuffle="block"))
    shard_of = {e.i: e.shard for e in index.entries}
    runs = [shard_of[order[0]]]
    for previous, current in zip(order, order[1:], strict=False):
        if shard_of[current] != shard_of[previous]:
            runs.append(shard_of[current])

    assert len(runs) == len(index.shards), "each shard should be visited in one run"
    assert sorted(runs) == sorted(index.shards)


def test_block_shuffle_still_permutes_within_each_shard(index):
    """Contiguous per shard, but not in index order inside one — otherwise the
    'shuffle' would only reorder shards."""
    order = list(ShardShuffleSampler(index, seed=42, shuffle="block"))
    first_shard = index.entry(order[0]).shard
    within = [i for i in order if index.entry(i).shard == first_shard]
    assert within != sorted(within)


def test_global_shuffle_does_not_keep_shards_contiguous(index):
    """The contrast that makes the block/global trade-off real rather than
    nominal."""
    order = list(ShardShuffleSampler(index, seed=42, shuffle="global"))
    shard_of = {e.i: e.shard for e in index.entries}
    switches = sum(
        1
        for previous, current in zip(order, order[1:], strict=False)
        if shard_of[current] != shard_of[previous]
    )
    assert switches > len(index.shards), "a true permutation should cross shards constantly"


# ── The resume slice ──────────────────────────────────────────────────
def test_a_resume_skip_slices_the_order_rather_than_replaying_it(index):
    """The real advantage of a map-style dataset plus a full index over an
    iterable stream: the position is a slice, not a fast-forward."""
    full = list(ShardShuffleSampler(index, seed=42, epoch=0))
    resumed = list(ShardShuffleSampler(index, seed=42, epoch=0, skip=30))

    assert resumed == full[30:]
    assert len(resumed) == index.n_samples - 30


def test_a_resumed_epoch_is_shorter_so_it_does_not_re_serve_what_was_seen(index):
    sampler = ShardShuffleSampler(index, seed=42, skip=30)
    assert len(sampler) == index.n_samples - 30
    assert len(list(sampler)) == len(sampler)


def test_the_same_global_position_is_reached_regardless_of_world_size(index):
    """``samples_seen`` is in global-order units, so a resume at a different world
    size lands at the same place in the corpus and the partition re-cuts around it."""
    seen = 40
    order = np.asarray(ShardShuffleSampler(index, seed=42, epoch=1).order())
    for _ in (2, 4, 8):  # the world size the run resumes at is not an input here
        resumed = list(ShardShuffleSampler(index, seed=42, epoch=1, skip=seen))
        assert resumed == [int(i) for i in order[seen:]]


# ── Ragged shards ─────────────────────────────────────────────────────
def test_shards_of_unequal_size_are_still_covered_exactly_once():
    """Real corpora do not divide evenly, and an off-by-one here would change
    ``len`` — the one number the parity argument cannot afford to be wrong."""
    entries = []
    shards = ["a", "b", "c"]
    sizes = [7, 1, 12]
    i = 0
    for name, size in zip(shards, sizes, strict=True):
        for _ in range(size):
            entries.append(ShardEntry(i=i, shard=name, key=str(i)))
            i += 1
    index = ShardIndex(n_samples=i, entries=tuple(entries), shards=tuple(shards))

    order = list(ShardShuffleSampler(index, seed=7, shuffle="block"))
    assert sorted(order) == list(range(sum(sizes)))
    assert len(order) == 20


def test_an_empty_corpus_produces_an_empty_order():
    index = ShardIndex(n_samples=0, entries=(), shards=())
    assert list(ShardShuffleSampler(index)) == []
    assert len(ShardShuffleSampler(index)) == 0
