"""The offline pass, and the failures only it can see.

Two of the four integrity classes are invisible at training time *by
construction*: an MP3 resyncs past damage and returns shorter audio with no
error, a hardware decoder emits green frames with nothing surfaced. Both produce
correctly-shaped tensors, so no exception handler in the training loop will ever
run. They do not crash the run — they quietly make the model worse.

This is the pass that catches them, and these are the checks that justify its
cost. The fault ceiling is also enforced here and *only* here: at training time a
content-dependent abort would be rank-divergent, which is the collective hang the
whole phase exists to prevent.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from ml_framework.data.streaming.materialize import (
    MaterializeError,
    _infer_decoder,
    discover,
    materialize,
)
from ml_framework.data.streaming.shards import ShardIndex

pytestmark = pytest.mark.unit

GARBAGE = b"not a RIFF header at all"


def _wav(seed: int, *, n: int = 800) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(rng.integers(-2000, 2000, n, dtype="int16").tobytes())
    return buf.getvalue()


def build(root: Path, *, n: int = 10, corrupt: set[int] | None = None, shards: int = 2) -> Path:
    corrupt = corrupt or set()
    for i in range(n):
        path = root / f"class-{i % shards}" / f"{i:03d}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(GARBAGE if i in corrupt else _wav(i))
    return root


# ── Discovery ─────────────────────────────────────────────────────────
def test_a_loose_file_corpus_is_grouped_into_shards_by_top_level_directory(tmp_path: Path):
    """A corpus with no shard concept would make both substitution and the block
    shuffle degenerate, so directories become the grouping — which gives
    ImageFolder-style class dirs a sensible one for free."""
    build(tmp_path / "corpus", n=6, shards=2)
    candidates, _, shards = discover(tmp_path / "corpus")

    assert len(candidates) == 6
    assert sorted(shards) == ["class-0", "class-1"]


def test_discovery_ignores_a_previously_written_index(tmp_path: Path):
    """Otherwise `--force` would index the index, growing the corpus every run."""
    root = build(tmp_path / "corpus", n=4)
    materialize(root, probe=False)
    candidates, _, _ = discover(root)

    assert len(candidates) == 4
    assert all("_mlf_shards" not in c["key"] for c in candidates)


def test_a_missing_corpus_is_named_rather_than_yielding_an_empty_index(tmp_path: Path):
    with pytest.raises(MaterializeError, match="no such corpus"):
        materialize(tmp_path / "nope")


def test_an_empty_corpus_is_refused(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(MaterializeError, match="no readable samples"):
        materialize(tmp_path / "empty")


# ── Decoder inference ─────────────────────────────────────────────────
def test_the_decoder_is_inferred_by_majority_suffix(tmp_path: Path):
    """A majority vote, not first-match: a directory of audio with one stray
    README should materialize as audio, and picking by first-seen would make the
    result depend on filesystem ordering."""
    root = build(tmp_path / "corpus", n=6)
    (root / "README.md").write_text("notes")

    _, report = materialize(root, max_fault_rate=1.0)
    assert report.decoder == "audio.pcm"


def test_a_corpus_with_no_matching_decoder_says_what_is_registered(tmp_path: Path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.xyz").write_bytes(b"data")

    with pytest.raises(MaterializeError, match="audio.pcm"):
        _infer_decoder([{"key": "a.xyz"}])


def test_an_explicit_decoder_overrides_inference(tmp_path: Path):
    root = build(tmp_path / "corpus", n=4)
    _, report = materialize(root, decoder="audio.pcm")
    assert report.decoder == "audio.pcm"


# ── The invariant the whole phase rests on ────────────────────────────
def test_an_undecodable_sample_still_gets_an_index_entry(tmp_path: Path):
    """The index declares what SHOULD exist, not what happens to decode today.

    Dropping the entry would shrink ``n_samples``, and that is the one number
    that must not depend on the state of the bytes — every rank derives its batch
    count from it.
    """
    root = build(tmp_path / "corpus", n=10, corrupt={1, 4})
    index, report = materialize(root, max_fault_rate=1.0)

    assert index.n_samples == 10, "not 8"
    assert report.n_ok == 8
    assert report.n_faults == 2
    assert index.faults == 2


def test_a_faulted_entry_carries_no_digest_so_it_is_identifiable(tmp_path: Path):
    """A digest is a record that these bytes decoded once. A sample that never
    decoded has nothing to record, and the empty field is what says so."""
    root = build(tmp_path / "corpus", n=6, corrupt={2})
    index, _ = materialize(root, max_fault_rate=1.0)

    # Looked up by key, not by index: `i` is assigned in discovery order (sorted
    # paths), so it is deliberately NOT the number in the filename.
    by_key = {e.key: e for e in index.entries}
    corrupt_key = next(k for k in by_key if k.endswith("002.wav"))

    assert by_key[corrupt_key].digest == ""
    assert all(e.digest for k, e in by_key.items() if k != corrupt_key)


def test_the_index_position_is_discovery_order_not_the_filename(tmp_path: Path):
    """Worth pinning because it is the assumption a reader is most likely to make.

    ``i`` is assigned by walking the corpus in sorted-path order, so a corpus laid
    out as class directories interleaves the filename numbering. Everything
    downstream keys off ``i``, and only the index knows what it maps to.
    """
    root = build(tmp_path / "corpus", n=6, shards=2)
    index, _ = materialize(root)

    assert [e.key for e in index.entries] == [
        "class-0/000.wav",
        "class-0/002.wav",
        "class-0/004.wav",
        "class-1/001.wav",
        "class-1/003.wav",
        "class-1/005.wav",
    ]
    assert index.entry(1).key == "class-0/002.wav"


# ── The fault ceiling ─────────────────────────────────────────────────
def test_the_ceiling_raises_here_because_this_is_a_single_process(tmp_path: Path):
    """At training time the same check would be a rank-divergent abort: one rank
    stops, the others block on the next collective, and the run hangs with no
    error. Here there is one process and no collective, so failing is safe."""
    root = build(tmp_path / "corpus", n=10, corrupt={1, 2, 3})

    with pytest.raises(MaterializeError, match="30.00%"):
        materialize(root, max_fault_rate=0.01)


def test_the_ceiling_message_names_the_files_and_the_knob(tmp_path: Path):
    root = build(tmp_path / "corpus", n=10, corrupt={1})
    with pytest.raises(MaterializeError) as excinfo:
        materialize(root, max_fault_rate=0.0)

    message = str(excinfo.value)
    assert "max_fault_rate" in message
    assert "001.wav" in message


def test_a_deliberately_raised_ceiling_lets_a_damaged_corpus_through(tmp_path: Path):
    root = build(tmp_path / "corpus", n=10, corrupt={1, 2})
    _, report = materialize(root, max_fault_rate=0.5)
    assert report.fault_rate == pytest.approx(0.2)


def test_the_index_is_written_even_when_the_ceiling_trips(tmp_path: Path):
    """So a second look does not have to re-probe the whole corpus to see what
    went wrong."""
    root = build(tmp_path / "corpus", n=10, corrupt={1, 2, 3})
    with pytest.raises(MaterializeError):
        materialize(root, max_fault_rate=0.01)

    assert ShardIndex.exists(ShardIndex.location(root))
    assert (ShardIndex.location(root) / "faults.jsonl").read_text().count("\n") == 3


def test_a_clean_corpus_still_writes_an_empty_faults_file(tmp_path: Path):
    """An absent file would be indistinguishable from an index written before this
    check existed — the same rule ``hpo.json`` follows."""
    root = build(tmp_path / "corpus", n=4)
    materialize(root)

    faults = ShardIndex.location(root) / "faults.jsonl"
    assert faults.is_file()
    assert faults.read_text() == ""


# ── Probing marks the corpus as verified ──────────────────────────────
def test_probing_records_that_the_corpus_was_actually_decoded(tmp_path: Path):
    """The gate a `silent` or `none` decoder checks before training."""
    root = build(tmp_path / "corpus", n=4)
    index, _ = materialize(root, probe=True)
    assert index.is_materialized


def test_skipping_the_probe_leaves_the_corpus_unverified(tmp_path: Path):
    """Writing an index is cheap; decoding every sample is not. A caller that only
    wants the file listing must not get a corpus that claims to have been checked."""
    root = build(tmp_path / "corpus", n=4)
    index, report = materialize(root, probe=False)

    assert not index.is_materialized
    assert report.n_faults == 0, "nothing was probed, so nothing can have failed"
    assert all(e.digest == "" for e in index.entries)


def test_probing_records_the_decoded_unit_count_for_later_cross_checks(tmp_path: Path):
    """`n_units` is what a duration or frame-count comparison is made against."""
    root = build(tmp_path / "corpus", n=3)
    index, _ = materialize(root)
    assert all(e.n_units == 800 for e in index.entries)


# ── Sampling ──────────────────────────────────────────────────────────
def test_a_sample_rate_probes_a_deterministic_fraction(tmp_path: Path):
    root = build(tmp_path / "corpus", n=20)
    first, _ = materialize(root, sample_rate=0.25)
    second, _ = materialize(root, sample_rate=0.25)

    assert first.n_samples == 5
    assert [e.key for e in first.entries] == [e.key for e in second.entries]


def test_a_sampled_run_says_so_in_its_report(tmp_path: Path):
    """A 5% fault rate over 5% of the corpus is a different claim from one over
    all of it, and the report must not let those be confused."""
    root = build(tmp_path / "corpus", n=20)
    _, report = materialize(root, sample_rate=0.25)
    assert report.sampled
    assert "(sampled)" in report.render()


def test_an_out_of_range_sample_rate_is_refused(tmp_path: Path):
    root = build(tmp_path / "corpus", n=4)
    with pytest.raises(MaterializeError, match="sample-rate"):
        materialize(root, sample_rate=0.0)


# ── The report ────────────────────────────────────────────────────────
def test_the_report_buckets_faults_by_kind(tmp_path: Path):
    root = build(tmp_path / "corpus", n=10, corrupt={1, 2})
    _, report = materialize(root, max_fault_rate=1.0)

    assert report.by_kind() == {"decode_error": 2}
    assert "decode_error" in report.render()


def test_the_report_truncates_a_long_fault_list_but_says_how_many(tmp_path: Path):
    """Enough to act on, not so many that the ceiling message scrolls away."""
    root = build(tmp_path / "corpus", n=30, corrupt=set(range(15)))
    _, report = materialize(root, max_fault_rate=1.0)

    rendered = report.render()
    assert "and 5 more" in rendered
    assert rendered.count("  - [") == 10
