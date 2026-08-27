"""The shard index: the manifest, the entries, and the digest that guards both.

The index is where the DDP-parity argument bottoms out. ``n_samples`` is
*declared* in a small JSON file that is byte-identical on every rank, and every
other number in the system derives from it. So the tests that matter here are
about the manifest being authoritative and self-consistent — not about
convenience.

The second theme is the digest. For a decoder that validates itself (FLAC, PNG)
it is redundant. For a flat ``uint16`` token shard it is the *entire* defence,
because a flipped bit there is a valid token id and nothing else in the stack can
possibly notice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml_framework.data.streaming.integrity import (
    REQUIRES_MATERIALIZATION,
    RUNTIME_DETECTABLE,
    should_verify,
)
from ml_framework.data.streaming.shards import (
    ENTRIES_NAME,
    INDEX_DIRNAME,
    MANIFEST_NAME,
    ShardEntry,
    ShardIndex,
    ShardIndexError,
    ShardIndexWriter,
    digest_bytes,
)

pytestmark = pytest.mark.unit


def write_index(root: Path, *, n: int = 6, shards: int = 2) -> ShardIndex:
    with ShardIndexWriter(root, data_kind="audio") as writer:
        for i in range(n):
            writer.add(
                ShardEntry(
                    i=i,
                    shard=f"train-{i % shards:04d}",
                    key=f"{i}.wav",
                    nbytes=100 + i,
                    media_type="audio/wav",
                    label=i % 3,
                    digest=digest_bytes(f"sample-{i}".encode()),
                    n_units=1600,
                )
            )
        return writer.finalize(decoder_hint="audio.pcm", class_names=("a", "b", "c"))


# ── Round trip ────────────────────────────────────────────────────────
def test_an_index_round_trips_through_its_two_files(tmp_path: Path):
    written = write_index(tmp_path)
    read = ShardIndex.read(tmp_path)

    assert read.n_samples == written.n_samples == 6
    assert read.entries == written.entries
    assert read.shards == written.shards
    assert read.decoder_hint == "audio.pcm"
    assert read.class_names == ("a", "b", "c")


def test_the_two_files_are_named_and_placed_predictably(tmp_path: Path):
    write_index(tmp_path)
    assert (tmp_path / MANIFEST_NAME).is_file()
    assert (tmp_path / ENTRIES_NAME).is_file()
    assert ShardIndex.exists(tmp_path)


def test_entries_are_json_lines_so_a_large_index_streams(tmp_path: Path):
    """Not one JSON document. A 15M-sample index appends during materialization,
    streams during verification, and is greppable when something has gone wrong."""
    write_index(tmp_path, n=6)
    lines = (tmp_path / ENTRIES_NAME).read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 6
    assert all(json.loads(line)["i"] == i for i, line in enumerate(lines))


def test_null_fields_are_omitted_so_the_file_does_not_carry_them_per_line(tmp_path: Path):
    """A 15M-line file should not repeat `"label": null` 15M times."""
    with ShardIndexWriter(tmp_path) as writer:
        writer.add(ShardEntry(i=0, shard="s", key="k"))
        writer.finalize()

    raw = json.loads((tmp_path / ENTRIES_NAME).read_text().strip())
    assert "label" not in raw
    assert "n_units" not in raw


def test_the_index_lives_beside_the_data_by_default(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    assert ShardIndex.location(corpus) == corpus / INDEX_DIRNAME


def test_the_index_location_is_overridable_for_a_read_only_corpus(tmp_path: Path):
    """The common case for a shared dataset mount."""
    elsewhere = tmp_path / "writable"
    assert ShardIndex.location(tmp_path / "corpus", elsewhere) == elsewhere


# ── The manifest is authoritative ─────────────────────────────────────
def test_len_comes_from_the_manifest_not_from_counting_entries(tmp_path: Path):
    """Deliberately not `len(self.entries)` even though a valid index makes them
    equal: this is the number that must be identical on every rank, so reading it
    from the manifest makes that true by construction rather than by coincidence."""
    index = write_index(tmp_path, n=6)
    assert len(index) == index.n_samples == 6


def test_a_manifest_that_disagrees_with_its_entries_is_refused(tmp_path: Path):
    """Silently adjusting the length is exactly what would let two ranks disagree."""
    index = write_index(tmp_path)
    with pytest.raises(ShardIndexError, match="inconsistent"):
        ShardIndex(n_samples=99, entries=index.entries, shards=index.shards)


def test_a_half_written_index_is_detectably_half_written(tmp_path: Path):
    """The manifest is written LAST, so it can never claim a count the entries
    file cannot satisfy."""
    writer = ShardIndexWriter(tmp_path)
    writer.__enter__()
    writer.add(ShardEntry(i=0, shard="s", key="k"))
    writer.__exit__()  # crash before finalize()

    assert (tmp_path / ENTRIES_NAME).is_file()
    assert not (tmp_path / MANIFEST_NAME).is_file()
    assert not ShardIndex.exists(tmp_path)


def test_a_missing_index_says_how_to_build_one(tmp_path: Path):
    with pytest.raises(ShardIndexError, match="mlf materialize"):
        ShardIndex.read(tmp_path)


def test_a_future_index_version_is_refused_rather_than_partially_read(tmp_path: Path):
    """A newer index may carry fields whose absence changes behaviour silently."""
    write_index(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    manifest["index_version"] = 99
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))

    with pytest.raises(ShardIndexError, match="newer than this build"):
        ShardIndex.read(tmp_path)


def test_a_corrupt_entry_line_names_the_line_number(tmp_path: Path):
    write_index(tmp_path)
    path = tmp_path / ENTRIES_NAME
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[2] = "{not json"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ShardIndexError, match=r":3"):
        ShardIndex.read(tmp_path)


# ── The digest ────────────────────────────────────────────────────────
def test_the_index_digest_changes_when_any_entry_changes(tmp_path: Path):
    """What makes a resume able to tell it is looking at a different corpus."""
    first = write_index(tmp_path / "a", n=6)
    same = write_index(tmp_path / "b", n=6)
    different = write_index(tmp_path / "c", n=7)

    assert first.index_digest == same.index_digest
    assert first.index_digest != different.index_digest


def test_a_sample_digest_detects_a_single_flipped_bit():
    payload = b"a token shard of some kind"
    flipped = bytearray(payload)
    flipped[3] ^= 0x01

    assert digest_bytes(payload) != digest_bytes(bytes(flipped))


def test_an_array_is_hashed_without_being_copied_to_bytes_first():
    """A memmap slice must be hashable in place, or hashing a token shard would
    defeat the point of memmapping it."""
    array = np.arange(64, dtype="uint16")
    assert digest_bytes(array) == digest_bytes(array.tobytes())


# ── Which decoders are worth verifying ────────────────────────────────
def test_auto_verifies_exactly_the_decoders_that_cannot_detect_damage():
    """Hashing is ~1 GB/s of the read budget. Paying it on a format that already
    validates its own frames buys nothing."""
    assert should_verify("none") and should_verify("silent")
    assert not should_verify("checked") and not should_verify("loud")


def test_the_two_integrity_groups_partition_the_classifications():
    from ml_framework.core.types import INTEGRITIES

    assert RUNTIME_DETECTABLE | REQUIRES_MATERIALIZATION == set(INTEGRITIES)
    assert not RUNTIME_DETECTABLE & REQUIRES_MATERIALIZATION


def test_verification_can_be_forced_either_way():
    assert should_verify("checked", "always")
    assert not should_verify("none", "never")


def test_an_unknown_verify_mode_is_refused_by_name():
    with pytest.raises(ValueError, match="auto|always|never"):
        should_verify("none", "sometimes")


# ── Reading labels without decoding ───────────────────────────────────
def test_labels_are_readable_without_decoding_anything(tmp_path: Path):
    """The analogue of reading ``ImageFolder.targets`` rather than opening every
    JPEG. For a corpus of 4-second clips that is a second versus an hour."""
    index = write_index(tmp_path, n=6)
    np.testing.assert_array_equal(index.labels(), [0, 1, 2, 0, 1, 2])


def test_partially_labelled_entries_yield_no_labels_at_all(tmp_path: Path):
    """A stratified split over half-known labels is worse than a refusal to
    stratify."""
    with ShardIndexWriter(tmp_path) as writer:
        writer.add(ShardEntry(i=0, shard="s", key="a", label=1))
        writer.add(ShardEntry(i=1, shard="s", key="b"))
        index = writer.finalize()

    assert index.labels() is None


# ── Shard grouping ────────────────────────────────────────────────────
def test_indices_group_by_shard_for_substitution_and_block_shuffling(tmp_path: Path):
    index = write_index(tmp_path, n=6, shards=2)

    assert index.indices_in_shard("train-0000") == (0, 2, 4)
    assert index.indices_in_shard("train-0001") == (1, 3, 5)
    assert index.shard_groups() == {"train-0000": (0, 2, 4), "train-0001": (1, 3, 5)}


def test_an_index_that_was_never_probed_says_so(tmp_path: Path):
    """The gate for a corpus whose decoder cannot report its own damage. Writing
    an index is cheap; probing every sample is not, so the two are distinct."""
    with ShardIndexWriter(tmp_path) as writer:
        writer.add(ShardEntry(i=0, shard="s", key="k"))
        index = writer.finalize(materialized=False)

    assert not index.is_materialized
    assert ShardIndex.read(tmp_path).is_materialized is False


def test_an_unknown_index_is_an_index_error_not_a_silent_none(tmp_path: Path):
    index = write_index(tmp_path, n=3)
    with pytest.raises(IndexError):
        index.entry(99)
