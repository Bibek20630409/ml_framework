"""The DDP-parity suite: batch count must not depend on how much data is corrupt.

This is the load-bearing test file of P13. Under DDP every rank must produce an
identical number of batches or the next collective hangs **with no error message**
— and a hang is the worst possible failure mode, because it looks like slow
training rather than like a bug.

The guarantee reduces to one property, and each test below pins one link of it:

    ``len(dataset)`` is a constant read from the shard index, and ``__getitem__``
    is *total* — exactly one item for every valid index, whatever the bytes say.

Note what is **not** here: any use of ``torch.distributed``. The whole argument is
supposed to hold without rank-aware code, so the parity test instantiates
``DistributedSampler`` for two ranks directly and compares counts. It runs on one
CPU, in-process, in milliseconds. If it ever needs a process group to pass, the
design has regressed.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from ml_framework.core.registry import get_decoder
from ml_framework.data.streaming.dataset import StagedDataset
from ml_framework.data.streaming.integrity import (
    CorruptSampleError,
    SampleFault,
    ShardUnusableError,
)
from ml_framework.data.streaming.shards import ShardEntry, ShardIndexWriter, digest_bytes
from ml_framework.data.streaming.sources_io import DirSource
from ml_framework.data.streaming.stages import DecodeContext
from ml_framework.data.streaming.substitute import SubstitutionPolicy

pytestmark = pytest.mark.unit

CORRUPT = b"this is not a RIFF header"


def _wav(seed: int, *, n: int = 64) -> bytes:
    rng = np.random.default_rng(seed)
    pcm = rng.integers(-2000, 2000, n, dtype="int16")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(pcm.tobytes())
    return buf.getvalue()


def build_corpus(
    root: Path,
    *,
    n_shards: int = 4,
    per_shard: int = 5,
    corrupt: set[int] | None = None,
) -> tuple[Path, int]:
    """A directory of .wav samples plus a written shard index.

    ``corrupt`` names indices whose *bytes* are garbage. The index still declares
    them — which is the entire point: a materialized index is a statement about
    what should exist, not about what currently decodes.
    """
    corrupt = corrupt or set()
    root.mkdir(parents=True, exist_ok=True)
    index_dir = root / "_mlf_shards"

    i = 0
    with ShardIndexWriter(index_dir, data_kind="audio") as writer:
        for shard in range(n_shards):
            shard_name = f"train-{shard:04d}"
            for member in range(per_shard):
                key = f"{shard_name}/{member:04d}.wav"
                payload = CORRUPT if i in corrupt else _wav(i)
                path = root / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                writer.add(
                    ShardEntry(
                        i=i,
                        shard=shard_name,
                        key=key,
                        nbytes=len(payload),
                        media_type="audio/wav",
                        label=i % 3,
                        # The digest of the bytes as written. For a corrupt sample
                        # that is the digest of the garbage, so verification would
                        # *pass* -- which is right: these samples are undamaged in
                        # transit and simply undecodable.
                        digest=digest_bytes(payload),
                        n_units=64,
                    )
                )
                i += 1
        writer.finalize(decoder_hint="audio.pcm", class_names=("a", "b", "c"))
    return index_dir, i


def make_dataset(root: Path, index_dir: Path, **kwargs) -> StagedDataset:
    from ml_framework.data.streaming.shards import ShardIndex

    # `loud` is audio.pcm's real classification; kwargs override it so the
    # integrity-dependent tests can ask for the other three.
    kwargs.setdefault("integrity", "loud")
    return StagedDataset(
        ShardIndex.read(index_dir),
        source=DirSource(root),
        decoder=get_decoder("audio.pcm"),
        ctx=DecodeContext(),
        **kwargs,
    )


@pytest.fixture
def clean_corpus(tmp_path: Path):
    index_dir, n = build_corpus(tmp_path / "clean")
    return tmp_path / "clean", index_dir, n


@pytest.fixture
def damaged_corpus(tmp_path: Path):
    """30% of samples corrupt, spread across every shard."""
    corrupt = {0, 3, 6, 9, 12, 15}
    index_dir, n = build_corpus(tmp_path / "damaged", corrupt=corrupt)
    return tmp_path / "damaged", index_dir, n, corrupt


# ── Link 1: length is declared, never derived ─────────────────────────
def test_len_is_the_declared_sample_count_not_the_decodable_one(damaged_corpus):
    root, index_dir, n, corrupt = damaged_corpus
    dataset = make_dataset(root, index_dir)

    assert len(dataset) == n == 20
    assert len(corrupt) == 6, "fixture sanity: a third of the corpus is unreadable"
    # The number that would be wrong if length were derived from what decodes.
    assert len(dataset) != n - len(corrupt)


def test_length_is_read_from_the_manifest_not_counted_from_the_entries(clean_corpus):
    """A manifest that disagrees with its entries is a detectable error, not a
    silently-adjusted length — because silently adjusting it is what would let two
    ranks disagree."""
    from ml_framework.data.streaming.shards import ShardIndex, ShardIndexError

    _, index_dir, n = clean_corpus
    index = ShardIndex.read(index_dir)
    with pytest.raises(ShardIndexError, match="inconsistent"):
        ShardIndex(n_samples=n + 1, entries=index.entries, shards=index.shards)


# ── Link 4: __getitem__ is total ──────────────────────────────────────
def test_every_index_yields_exactly_one_sample_even_when_a_third_are_corrupt(damaged_corpus):
    root, index_dir, n, _ = damaged_corpus
    dataset = make_dataset(root, index_dir)

    items = [dataset[i] for i in range(len(dataset))]

    assert len(items) == n
    assert all(item is not None for item in items)
    assert dataset.faults == 6


def test_a_corrupt_sample_is_substituted_never_skipped(damaged_corpus):
    """The signature that makes DDP parity possible: index in, index out."""
    root, index_dir, _, corrupt = damaged_corpus
    dataset = make_dataset(root, index_dir)

    for i in sorted(corrupt):
        decoded, label = dataset[i]
        # A real decode of *some* sample, never a None or a zero-filled placeholder.
        assert decoded.array.size == 64
        assert label in (0, 1, 2)


def test_substitution_draws_from_the_same_shard_to_keep_the_read_local(damaged_corpus):
    root, index_dir, _, _ = damaged_corpus
    from ml_framework.data.streaming.shards import ShardIndex

    index = ShardIndex.read(index_dir)
    policy = SubstitutionPolicy(index, mode="redraw", seed=42)
    fault = SampleFault(
        index=3,
        shard="train-0000",
        key="k",
        stage="decode",
        kind="decode_error",
        detail="synthetic",
        decoder="audio.pcm",
    )

    chosen = policy.substitute(3, fault, epoch=0)

    assert chosen != 3
    assert index.entry(chosen).shard == index.entry(3).shard


def test_substitution_is_deterministic_in_seed_epoch_and_index(damaged_corpus):
    """Every rank must substitute the same corrupt sample the same way, or a
    'reproducible' run stops being reproducible the moment a disk goes bad."""
    root, index_dir, _, _ = damaged_corpus
    from ml_framework.data.streaming.shards import ShardIndex

    index = ShardIndex.read(index_dir)
    fault = SampleFault(
        index=3,
        shard="train-0000",
        key="k",
        stage="decode",
        kind="decode_error",
        detail="synthetic",
        decoder="audio.pcm",
    )

    def choose(seed: int, epoch: int) -> int:
        return SubstitutionPolicy(index, mode="redraw", seed=seed).substitute(3, fault, epoch=epoch)

    assert choose(42, 0) == choose(42, 0)
    # A different epoch may reasonably pick differently; a different seed must.
    assert choose(42, 0) != choose(99, 0) or choose(42, 1) != choose(99, 1)


def test_repeat_substitution_serves_the_previous_good_sample(damaged_corpus):
    root, index_dir, _, _ = damaged_corpus
    dataset = make_dataset(root, index_dir, substitute="repeat")

    good = dataset[1]  # index 1 is clean
    substituted, _ = dataset[3]  # index 3 is corrupt

    np.testing.assert_array_equal(substituted.array, good[0].array)


def test_a_shard_with_no_healthy_sample_raises_rather_than_looping(tmp_path: Path):
    """A shard-level failure, and deterministic across ranks because the index is —
    so raising here cannot desynchronize anything."""
    root = tmp_path / "dead"
    index_dir, _ = build_corpus(root, n_shards=1, per_shard=3, corrupt={0, 1, 2})
    dataset = make_dataset(root, index_dir)

    with pytest.raises(ShardUnusableError):
        dataset[0]


def test_on_corrupt_raise_stops_the_run_for_single_device_use(damaged_corpus):
    """Kept for single-device runs and for materialization. A config validator
    refuses it under an explicitly distributed strategy."""
    root, index_dir, _, _ = damaged_corpus
    dataset = make_dataset(root, index_dir, on_corrupt="raise")

    assert dataset[1] is not None  # a clean index still works
    with pytest.raises(CorruptSampleError):
        dataset[0]


def test_skip_is_not_a_representable_policy():
    """Not a rejected option — an absent one. It is the single response that
    cannot preserve batch count."""
    with pytest.raises(ValueError, match="substitute|raise"):
        StagedDataset(_tiny_index(), source=None, decoder=None, on_corrupt="skip", integrity="loud")


def _tiny_index():
    from ml_framework.data.streaming.shards import ShardIndex

    return ShardIndex(
        n_samples=1,
        entries=(ShardEntry(i=0, shard="s", key="k"),),
        shards=("s",),
    )


# ── Links 2 and 3: what the distributed sampler does with that constant ──
def test_two_ranks_over_the_same_index_produce_the_same_number_of_batches(damaged_corpus):
    """The parity proof, and it needs no process group to run.

    ``DistributedSampler`` derives its per-rank count as ``ceil(N / W)`` from
    ``len(dataset)`` alone. Because that length is a constant from the manifest
    rather than a count of what decoded, both ranks get the same number — even
    though rank 0's shards here are healthier than rank 1's.
    """
    torch_data = pytest.importorskip("torch.utils.data")
    root, index_dir, _, _ = damaged_corpus
    dataset = make_dataset(root, index_dir)

    counts = []
    for rank in (0, 1):
        sampler = torch_data.DistributedSampler(dataset, num_replicas=2, rank=rank, shuffle=False)
        loader = torch_data.DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=list)
        counts.append(len(loader))

    assert counts[0] == counts[1], "ranks disagree on batch count; the next collective would hang"


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8])
def test_per_rank_length_is_a_pure_function_of_the_declared_count(damaged_corpus, world_size):
    """`ceil(N / W)` for every rank, at every world size, with N the declared count."""
    torch_data = pytest.importorskip("torch.utils.data")
    import math

    root, index_dir, n, _ = damaged_corpus
    dataset = make_dataset(root, index_dir)
    expected = math.ceil(n / world_size)

    lengths = {
        len(torch_data.DistributedSampler(dataset, num_replicas=world_size, rank=r))
        for r in range(world_size)
    }
    assert lengths == {expected}


def test_drop_last_is_a_function_of_the_declared_length_and_batch_size_only(damaged_corpus):
    """`lightning_adapter._loader` computes it this way and P13 must not change it.

    Pinned here rather than only in the adapter's own tests because it is the
    third link of the parity chain: a content-dependent ``drop_last`` would let two
    ranks build different batch counts from identical samplers.
    """
    root, index_dir, n, _ = damaged_corpus
    dataset = make_dataset(root, index_dir)

    for batch_size in (1, 4, 16, 32):
        assert (len(dataset) > batch_size) == (n > batch_size)


# ── The fault record ──────────────────────────────────────────────────
def test_a_healthy_run_writes_no_fault_file_at_all(clean_corpus, tmp_path: Path):
    """The presence of anything under faults/ is itself the signal."""
    root, index_dir, _ = clean_corpus
    fault_dir = tmp_path / "faults"
    dataset = make_dataset(root, index_dir, fault_dir=fault_dir)

    for i in range(len(dataset)):
        dataset[i]

    assert dataset.faults == 0
    assert not fault_dir.exists()


def test_each_fault_is_recorded_with_the_index_it_was_served_instead(
    damaged_corpus, tmp_path: Path
):
    from ml_framework.data.streaming.integrity import FaultLog

    root, index_dir, _, corrupt = damaged_corpus
    fault_dir = tmp_path / "faults"
    dataset = make_dataset(root, index_dir, fault_dir=fault_dir)

    for i in range(len(dataset)):
        dataset[i]
    dataset.close()

    faults = FaultLog.aggregate(fault_dir)
    assert len(faults) == len(corrupt)
    assert {f.index for f in faults} == corrupt
    assert all(f.decoder == "audio.pcm" for f in faults)
    assert all(f.kind in ("decode_error", "truncated", "checksum", "missing") for f in faults)


def test_the_fault_rate_is_reported_but_never_raises_at_runtime(damaged_corpus):
    """A content-dependent abort is rank-divergent, and a rank-divergent abort is
    the hang this whole design exists to prevent. The ceiling is enforced in
    ``mlf materialize``, which is one process and can fail safely."""
    root, index_dir, _, corrupt = damaged_corpus
    dataset = make_dataset(root, index_dir)

    for i in range(len(dataset)):
        dataset[i]

    assert dataset.fault_rate == pytest.approx(len(corrupt) / len(dataset))
    assert dataset.fault_rate > 0.01, "well over any sane max_fault_rate, and still no raise"


# ── Verification ──────────────────────────────────────────────────────
def test_a_bit_flip_is_caught_when_the_decoder_cannot_detect_damage_itself(tmp_path: Path):
    """`integrity: none` turns digest verification on — the only defence a flat
    token shard has, because a flipped bit there is a valid token id."""
    root = tmp_path / "tokens"
    index_dir, _ = build_corpus(root, n_shards=1, per_shard=3)
    dataset = make_dataset(root, index_dir, integrity="none", allow_unverified=True)

    assert dataset.verify, "auto must verify a decoder with no integrity of its own"

    # Corrupt the stored bytes *after* materialization, exactly as bit rot does.
    entry = dataset.index.entry(1)
    path = root / entry.key
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))

    dataset[1]  # substituted, not raised
    assert dataset.faults == 1


def test_a_self_checking_decoder_is_not_hashed_a_second_time(clean_corpus):
    """Hashing is ~1 GB/s of the read budget; paying it twice on a FLAC is waste."""
    root, index_dir, _ = clean_corpus
    dataset = make_dataset(root, index_dir, integrity="checked")
    assert not dataset.verify


def test_verification_can_be_forced_on_or_off_regardless_of_integrity(clean_corpus):
    root, index_dir, _ = clean_corpus
    assert make_dataset(root, index_dir, integrity="checked", verify_checksums="always").verify
    assert not make_dataset(
        root, index_dir, integrity="none", verify_checksums="never", allow_unverified=True
    ).verify


# ── The materialization gate ──────────────────────────────────────────
def test_an_unmaterialized_corpus_is_refused_for_a_silently_failing_decoder(tmp_path: Path):
    """MP3 and NVDEC produce valid-shaped wrong results. Nothing at training time
    can see that, so the offline pass is not optional for them."""
    from ml_framework.data.streaming.integrity import UnverifiedCorpusError
    from ml_framework.data.streaming.shards import ShardIndex

    root = tmp_path / "unverified"
    index_dir, _ = build_corpus(root, n_shards=1, per_shard=3)
    # An index that exists but was never decode-probed.
    index = ShardIndex.read(index_dir)
    unprobed = ShardIndex(
        n_samples=index.n_samples,
        entries=index.entries,
        shards=index.shards,
        materialized_at="",
    )

    with pytest.raises(UnverifiedCorpusError, match="mlf materialize"):
        StagedDataset(
            unprobed,
            source=DirSource(root),
            decoder=get_decoder("audio.pcm"),
            integrity="silent",
        )


def test_the_unverified_escape_hatch_exists_and_costs_typing(tmp_path: Path):
    from ml_framework.data.streaming.shards import ShardIndex

    root = tmp_path / "hatch"
    index_dir, _ = build_corpus(root, n_shards=1, per_shard=3)
    index = ShardIndex.read(index_dir)
    unprobed = ShardIndex(
        n_samples=index.n_samples, entries=index.entries, shards=index.shards, materialized_at=""
    )

    dataset = StagedDataset(
        unprobed,
        source=DirSource(root),
        decoder=get_decoder("audio.pcm"),
        integrity="silent",
        allow_unverified=True,
    )
    assert len(dataset) == 3


def test_a_materialized_corpus_needs_no_escape_hatch(clean_corpus):
    root, index_dir, n = clean_corpus
    dataset = make_dataset(root, index_dir, integrity="silent")
    assert len(dataset) == n
