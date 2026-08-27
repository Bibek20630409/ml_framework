"""Mid-epoch resume: continue where the run stopped, or refuse to pretend.

Lightning checkpoints the model, the optimizer and the epoch counter. What it
cannot checkpoint is *which samples this epoch had already served* — and without
that, a resume silently replays them. Nothing crashes; you get an extra partial
pass over a subset of the corpus and a small unexplained kink in the loss curve.

Two things are being pinned here. That a resume is a **slice** of the epoch's
order rather than a replay of it, and that a resume against a *different corpus*
is refused rather than performed on the wrong bytes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from ml_framework.data.streaming.sampler import ShardShuffleSampler
from ml_framework.data.streaming.shards import ShardEntry, ShardIndex
from ml_framework.data.streaming.state import (
    STATE_FILENAME,
    LoaderState,
    LoaderStateError,
)

pytestmark = pytest.mark.unit


def make_index(*, n_shards: int = 4, per_shard: int = 25) -> ShardIndex:
    entries, shards, i = [], [], 0
    for shard in range(n_shards):
        name = f"train-{shard:04d}"
        shards.append(name)
        for member in range(per_shard):
            entries.append(ShardEntry(i=i, shard=name, key=str(member)))
            i += 1
    return ShardIndex(
        n_samples=i, entries=tuple(entries), shards=tuple(shards), index_digest="abc123"
    )


# ── The slice ─────────────────────────────────────────────────────────
def test_a_resume_skips_exactly_the_samples_already_seen():
    index = make_index()
    state = LoaderState(index_digest="abc123", seed=42, epoch=1, samples_seen=37)

    skip = state.resume_skip(index_digest=index.index_digest)
    full = list(ShardShuffleSampler(index, seed=42, epoch=1))
    resumed = list(ShardShuffleSampler(index, seed=42, epoch=1, skip=skip))

    assert skip == 37
    assert resumed == full[37:]
    assert len(resumed) == index.n_samples - 37


def test_resuming_is_a_slice_not_a_replay():
    """The real advantage of a map-style dataset plus a full index over an
    iterable stream: the position is an offset into a materialized list, so
    nothing is decoded twice to get back to it."""
    index = make_index()
    full = list(ShardShuffleSampler(index, seed=42, epoch=0))
    resumed = list(ShardShuffleSampler(index, seed=42, epoch=0, skip=60))

    assert not set(resumed) & set(full[:60]), "a resumed epoch must not re-serve seen samples"
    assert set(resumed) | set(full[:60]) == set(range(index.n_samples))


def test_a_fresh_epoch_starts_from_the_beginning_again():
    """`samples_seen` is within-epoch, so it must reset — otherwise every epoch
    after a resume would be short by the resume offset."""
    state = LoaderState(epoch=1, samples_seen=37).next_epoch()
    assert state.epoch == 2
    assert state.samples_seen == 0


def test_advancing_accumulates_samples_steps_and_faults():
    state = LoaderState().advance(samples=32, steps=1).advance(samples=32, steps=1, faults=2)
    assert (state.samples_seen, state.global_step, state.faults) == (64, 2, 2)


def test_state_is_frozen_so_advancing_produces_a_new_one():
    original = LoaderState(samples_seen=10)
    advanced = original.advance(samples=5)
    assert original.samples_seen == 10
    assert advanced.samples_seen == 15


# ── World size: recorded, not required to match ───────────────────────
def test_a_resume_at_a_different_world_size_lands_at_the_same_global_position():
    """``samples_seen`` is in global-order units, never per-rank counts.

    The sampler emits a full-corpus order that is a function of ``(seed, epoch)``
    and never of rank, so a position in it means the same thing at any world size
    and the partition simply re-cuts around it.
    """
    index = make_index()
    state = LoaderState(index_digest="abc123", seed=42, epoch=1, samples_seen=37, world_size=4)

    order = list(ShardShuffleSampler(index, seed=42, epoch=1))
    for resumed_world_size in (1, 2, 8):
        skip = state.resume_skip(index_digest="abc123", world_size=resumed_world_size)
        assert list(ShardShuffleSampler(index, seed=42, epoch=1, skip=skip)) == order[37:]


def test_a_changed_world_size_warns_rather_than_refusing(caplog):
    """The global position is still correct; what genuinely differs is which rank
    sees which sample. Refusing would make elastic resumption impossible for a
    difference that is usually intended."""
    state = LoaderState(index_digest="abc123", world_size=4, samples_seen=10)

    with caplog.at_level(logging.WARNING):
        assert state.resume_skip(index_digest="abc123", world_size=8) == 10

    assert "world_size" in caplog.text
    assert "which rank sees which sample" in caplog.text


def test_an_unchanged_world_size_warns_about_nothing(caplog):
    state = LoaderState(index_digest="abc123", world_size=2, samples_seen=10)
    with caplog.at_level(logging.WARNING):
        state.resume_skip(index_digest="abc123", world_size=2)
    assert "world_size" not in caplog.text


# ── The refusal ───────────────────────────────────────────────────────
def test_resuming_against_a_changed_corpus_is_refused():
    """ "Sample 41,000" now names different bytes, so the position is meaningless.

    Continuing would replay a different dataset while reporting it as the same
    run — the same posture ``read_manifest`` takes on a version mismatch.
    """
    state = LoaderState(index_digest="abc123", samples_seen=37)

    with pytest.raises(LoaderStateError, match="shard index changed"):
        state.resume_skip(index_digest="def456")


def test_the_refusal_names_both_digests_so_the_change_is_identifiable():
    state = LoaderState(index_digest="abc123abc123", samples_seen=1)
    with pytest.raises(LoaderStateError) as excinfo:
        state.resume_skip(index_digest="def456def456")
    message = str(excinfo.value)
    assert "abc123abc123"[:12] in message
    assert "def456def456"[:12] in message


def test_a_state_with_no_digest_resumes_against_anything():
    """State written before an index existed. Permissive on purpose: refusing
    would break resumption for every run that predates the shard index."""
    assert LoaderState(samples_seen=5).resume_skip(index_digest="anything") == 5


def test_a_future_state_version_is_refused_rather_than_partially_read():
    """A newer state may carry fields whose absence changes behaviour silently."""
    with pytest.raises(LoaderStateError, match="newer than this build"):
        LoaderState.from_dict({"version": 99, "samples_seen": 10})


# ── The audit copy ────────────────────────────────────────────────────
def test_the_state_file_is_always_written_even_at_position_zero(tmp_path: Path):
    """An absent file would be indistinguishable from a run that predates the
    phase — the same rule ``hpo.json`` follows."""
    path = LoaderState().write(tmp_path)

    assert path.name == STATE_FILENAME
    assert json.loads(path.read_text())["samples_seen"] == 0


def test_the_state_round_trips_through_its_file(tmp_path: Path):
    original = LoaderState(
        index_digest="abc123",
        seed=7,
        epoch=3,
        samples_seen=99,
        global_step=12,
        world_size=4,
        shuffle="global",
        faults=2,
    )
    original.write(tmp_path)

    assert LoaderState.read(tmp_path) == original


def test_reading_a_missing_state_returns_none_rather_than_raising(tmp_path: Path):
    """A first run has no state, and that is not an error."""
    assert LoaderState.read(tmp_path) is None
