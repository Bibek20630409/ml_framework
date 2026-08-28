"""``audio`` and ``video`` as real data kinds, end to end through the staged path.

These two are the first sources that consume P13's machinery for real: a shard
index supplies the sample list *and the labels*, a positioned read gets the bytes,
a registered decoder does demux and decode, and a corrupt sample is substituted
rather than skipped.

The tests worth having here are the ones about the seams, not about waveforms:

* labels are read from the index **without decoding** — the property that makes
  cross-validation affordable;
* imbalance is corrected by loss weights and deliberately **not** by a sampler;
* a fold's index is **renumbered**, because ``__len__`` and the distributed
  sampler both derive from the entry positions;
* an unmaterialized corpus is refused with the command that fixes it, rather than
  silently triggering an hour of decoding inside ``mlf train``.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from ml_framework.config import ExperimentConfig
from ml_framework.core.types import DATA_KINDS, FrameworkError
from ml_framework.data.streaming.materialize import materialize
from ml_framework.data.streaming.shards import ShardIndex

pytestmark = pytest.mark.unit


def _wav(seed: int, *, n: int = 8000, rate: int = 16_000) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(rng.integers(-3000, 3000, n, dtype="int16").tobytes())
    return buf.getvalue()


def build_corpus(root: Path, *, per_class: tuple[int, ...] = (6, 6)) -> Path:
    """A class-directory corpus of WAVs, materialized."""
    i = 0
    for ci, count in enumerate(per_class):
        directory = root / f"class-{ci}"
        directory.mkdir(parents=True, exist_ok=True)
        for k in range(count):
            (directory / f"{k:03d}.wav").write_bytes(_wav(i))
            i += 1
    materialize(root)
    return root


def audio_config(tmp_path: Path, root: Path, **overrides) -> ExperimentConfig:
    payload = {
        "task": "binary",
        "data": {
            "kind": "audio",
            "path": str(root),
            "params": {"test_dir": str(root), "clip_seconds": 0.5},
        },
        "model": {"name": "audio.cnn"},
        "runtime": {"output_dir": str(tmp_path / "out"), "seed": 42},
    }
    for key, value in overrides.items():
        payload["data"]["params"][key] = value  # type: ignore[index]
    return ExperimentConfig.model_validate(payload)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return build_corpus(tmp_path / "clips")


# ── Vocabulary ────────────────────────────────────────────────────────
def test_audio_and_video_are_real_data_kinds():
    assert "audio" in DATA_KINDS
    assert "video" in DATA_KINDS


def test_both_kinds_map_to_the_dataset_payload_and_need_no_new_one():
    """A `dataset` payload already means "a lazily-decoding corpus". Audio and
    video differ in what one *item* is, not in the shape of the container — so
    `Capabilities` needs no new flag, and a flag with no consumer would be the
    decoration its docstring calls a review failure."""
    from ml_framework.core.protocols import DEFAULT_PAYLOAD, KIND_PAYLOADS

    for kind in ("audio", "video"):
        assert DEFAULT_PAYLOAD[kind] == "dataset"
        assert KIND_PAYLOADS[kind] == frozenset({"dataset"})


def test_both_kinds_have_a_registered_source_and_model():
    import ml_framework.data.builders  # noqa: F401  (registration side effect)
    from ml_framework.core.registry import MODELS, SOURCES

    assert {"audio", "video"} <= set(SOURCES.names())
    assert {"audio.cnn", "video.r3d"} <= set(MODELS.names())
    assert MODELS.get_spec("audio.cnn").data_kinds == frozenset({"audio"})
    assert MODELS.get_spec("video.r3d").data_kinds == frozenset({"video"})


def test_the_video_model_needs_no_extra_beyond_image():
    """torchvision.models.video ships r3d_18, so the MODEL costs nothing beyond
    `[image]`. Only the DECODER needs PyAV — conflating them would make the pip
    hint for one of them wrong."""
    import ml_framework.plugins  # noqa: F401
    from ml_framework.core.registry import DECODERS, MODELS

    model_extras = {r.extra for r in MODELS.get_spec("video.r3d").requires}
    assert "video" not in model_extras
    assert {r.extra for r in DECODERS.get_spec("video.h264").requires} == {"video"}


# ── Labels without decoding ───────────────────────────────────────────
def test_materialize_infers_labels_from_the_class_directories(corpus):
    """The layout already states the labels; recording them is what makes fold
    planning a JSON scan instead of a decode pass over the corpus."""
    index = ShardIndex.read(ShardIndex.location(corpus))

    assert index.class_names == ("class-0", "class-1")
    np.testing.assert_array_equal(index.labels(), [0] * 6 + [1] * 6)


def test_the_class_order_is_sorted_so_it_is_stable_across_machines(tmp_path: Path):
    """A class order that depended on filesystem iteration would silently relabel
    the corpus when it was copied, and every metric would still look plausible."""
    root = build_corpus(tmp_path / "z", per_class=(2, 2, 2))
    index = ShardIndex.read(ShardIndex.location(root))
    assert index.class_names == ("class-0", "class-1", "class-2")


def test_a_single_class_corpus_gets_no_labels_rather_than_all_zeros(tmp_path: Path):
    """One class is not a classification corpus, and labelling everything 0 would
    be a claim rather than an inference."""
    root = build_corpus(tmp_path / "one", per_class=(4,))
    assert ShardIndex.read(ShardIndex.location(root)).labels() is None


def test_audio_labels_reads_the_index_without_touching_a_decoder(corpus, tmp_path):
    from ml_framework.data.sources.audio import audio_labels

    config = audio_config(tmp_path, corpus)
    assert audio_labels(config) == [0] * 6 + [1] * 6


# ── The bundle ────────────────────────────────────────────────────────
def test_an_audio_bundle_holds_lazy_datasets_not_decoded_waveforms(corpus, tmp_path):
    from ml_framework.data.sources.audio import build_audio_bundle
    from ml_framework.data.streaming.dataset import StagedDataset

    bundle = build_audio_bundle(audio_config(tmp_path, corpus))

    assert bundle.payload == "dataset"
    assert bundle.data_kind == "audio"
    assert isinstance(bundle.train.x, StagedDataset)
    assert isinstance(bundle.test.x, StagedDataset)


def test_the_bundle_records_where_the_decoder_lands_for_the_transport_layer(corpus, tmp_path):
    from ml_framework.data.sources.audio import build_audio_bundle

    bundle = build_audio_bundle(audio_config(tmp_path, corpus))
    assert bundle.meta["lands_in"] == "host"
    assert bundle.meta["index_digest"]


def test_imbalance_is_corrected_by_loss_weights_and_never_by_a_sampler(tmp_path: Path):
    """A weighted random sampler over sharded storage is one seek per sample,
    which destroys the read locality this whole pipeline is built on — and it
    would fight `ShardShuffleSampler` for ownership of the visit order."""
    from ml_framework.data.sources.audio import build_audio_bundle

    root = build_corpus(tmp_path / "skew", per_class=(10, 2))
    bundle = build_audio_bundle(audio_config(tmp_path, root))

    assert bundle.class_weights is not None
    assert "sample_weights" not in bundle.meta, "the image source's mechanism, deliberately absent"


def test_class_weights_come_from_the_training_split_alone(corpus, tmp_path):
    """Not from the whole corpus.

    The corpus here is a balanced 6/6, but validation is carved out of it, so the
    *training* split is 5/6 and its correction is a real (if small) one. Computing
    weights from the full corpus instead would be the same category of leak as
    fitting a scaler before splitting: it lets the validation split influence the
    loss the model is trained with.
    """
    from ml_framework.data.sources.audio import build_audio_bundle

    bundle = build_audio_bundle(audio_config(tmp_path, corpus))
    train_labels = list(bundle.train.y)

    assert bundle.class_weights is not None
    assert len(train_labels) < 12, "validation was carved out of the corpus"
    expected = train_labels.count(0) / train_labels.count(1)
    np.testing.assert_allclose(bundle.class_weights, [expected])


def test_a_single_class_training_split_gets_no_weights():
    """Nothing to correct, and a weight vector over one class is meaningless."""
    from ml_framework.data.sources.staged_folder import class_weights

    assert class_weights([0, 0, 0], 2, "binary") is None
    assert class_weights([], 2, "binary") is None


def test_the_validation_split_is_carved_deterministically(corpus, tmp_path):
    from ml_framework.data.sources.audio import build_audio_bundle

    first = build_audio_bundle(audio_config(tmp_path, corpus))
    second = build_audio_bundle(audio_config(tmp_path, corpus))

    assert len(first.train.x) == len(second.train.x)
    assert len(first.train.x) + len(first.val.x) == 12
    np.testing.assert_array_equal(first.train.y, second.train.y)


# ── Folds ─────────────────────────────────────────────────────────────
def test_a_fold_index_is_renumbered_from_zero(corpus, tmp_path):
    """`__len__` and the distributed sampler both derive from the entry
    positions, so a sparse index would break the identity the batch-count-parity
    argument needs."""
    from ml_framework.data.sources.staged_folder import subset_index

    index = ShardIndex.read(ShardIndex.location(corpus))
    subset = subset_index(index, [7, 2, 9])

    assert subset.n_samples == 3
    assert [e.i for e in subset.entries] == [0, 1, 2]
    # ...and the entries still point at the originals' bytes.
    assert [e.key for e in subset.entries] == [
        index.entry(7).key,
        index.entry(2).key,
        index.entry(9).key,
    ]


def test_a_fold_keeps_the_index_digest_so_a_resume_can_still_identify_the_corpus(corpus):
    from ml_framework.data.sources.staged_folder import subset_index

    index = ShardIndex.read(ShardIndex.location(corpus))
    assert subset_index(index, [0, 1]).index_digest == index.index_digest


def test_class_weights_are_recomputed_per_fold():
    """Reusing one weight vector across folds would weight each fold by another
    fold's class balance — the same category of mistake as sharing a fitted
    scaler, and just as invisible in the result."""
    from ml_framework.data.sources.staged_folder import class_weights

    balanced = class_weights([0] * 5 + [1] * 5, 2, "multiclass")
    skewed = class_weights([0] * 9 + [1], 2, "multiclass")

    assert balanced is not None and skewed is not None
    assert not np.allclose(balanced, skewed)


def test_both_kinds_are_cross_validatable(corpus, tmp_path):
    from ml_framework.data.builders import _CV_BUILDERS

    assert "audio" in _CV_BUILDERS
    assert "video" in _CV_BUILDERS


# ── The materialization gate ──────────────────────────────────────────
def test_an_unmaterialized_corpus_names_the_command_that_fixes_it(tmp_path: Path):
    """Materialization is never implicit: it walks and decodes the whole corpus,
    which is minutes to hours that `mlf train` must not start by surprise."""
    from ml_framework.data.sources.audio import build_audio_bundle

    root = tmp_path / "raw"
    (root / "class-0").mkdir(parents=True)
    (root / "class-0" / "a.wav").write_bytes(_wav(0))

    with pytest.raises(FrameworkError, match="mlf materialize"):
        build_audio_bundle(audio_config(tmp_path, root))


def test_the_config_validator_requires_a_held_out_directory(tmp_path: Path):
    with pytest.raises(Exception, match="params.test_dir"):
        ExperimentConfig.model_validate(
            {
                "task": "binary",
                "data": {"kind": "audio", "path": str(tmp_path)},
                "model": {"name": "audio.cnn"},
            }
        )


# ── Zero-config detection ─────────────────────────────────────────────
def test_a_directory_of_audio_class_folders_sniffs_as_audio(corpus):
    from ml_framework.data.sniff import sniff

    found = sniff(corpus)
    assert found.kind == "audio"
    assert found.task == "binary"  # two class directories


def test_the_shard_index_directory_is_not_counted_as_a_class(corpus):
    """`_mlf_shards` lives inside the corpus. Counting it would make a two-class
    corpus look like three — which picks multiclass over binary, and therefore a
    different head, a different loss and a different metric."""
    from ml_framework.data.sniff import sniff

    assert (corpus / "_mlf_shards").is_dir(), "fixture sanity: the index is in the corpus"
    assert sniff(corpus).n_features == 2


def test_a_video_folder_sniffs_as_video(tmp_path: Path):
    from ml_framework.data.sniff import sniff

    for ci in range(3):
        directory = tmp_path / "vids" / f"class-{ci}"
        directory.mkdir(parents=True)
        (directory / "a.mp4").write_bytes(b"\x00" * 32)

    found = sniff(tmp_path / "vids")
    assert found.kind == "video"


def test_the_kind_census_survives_a_stray_file_of_another_family(tmp_path: Path):
    """A majority vote, not a first match: an audio corpus with cover art should
    not be classified by whatever the filesystem returned first."""
    from ml_framework.data.sniff import sniff

    directory = tmp_path / "mixed" / "class-0"
    directory.mkdir(parents=True)
    for i in range(5):
        (directory / f"{i}.wav").write_bytes(_wav(i))
    (directory / "cover.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "mixed" / "class-1").mkdir()
    for i in range(5):
        (tmp_path / "mixed" / "class-1" / f"{i}.wav").write_bytes(_wav(i + 100))

    assert sniff(tmp_path / "mixed").kind == "audio"


def test_zero_config_picks_a_default_model_for_both_kinds():
    from ml_framework.config.defaults import select_model

    assert select_model("audio", "binary", n_rows=100)[0] == "audio.cnn"
    assert select_model("video", "multiclass", n_rows=100)[0] == "video.r3d"


# ── A framework-wide bug this phase surfaced ──────────────────────────
@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_class_weights_never_reach_the_checkpoint(task):
    """`pos_weight` and `weight` are properties of the DATA, not learned parameters.

    torch registers both as buffers, so they land in ``state_dict`` — and every
    reload path deliberately passes ``class_weights=None`` (a loaded estimator
    predicts; it does not resume training), so loading then failed with:

        RuntimeError: Unexpected key(s) in state_dict: "criterion.pos_weight"

    Latent since the class-weight support landed, and unreachable until audio
    became the first source to emit binary weights from its own imbalance
    correction. Marking the buffers non-persistent matches what they are.
    """
    pytest.importorskip("torch")
    import numpy as np

    from ml_framework.plugins.mlp import MLP

    weights = (
        np.asarray([2.0], "float32") if task == "binary" else np.asarray([1.0, 3.0], "float32")
    )
    model = MLP(
        input_dim=4,
        output_dim=1 if task == "binary" else 2,
        task=task,
        params={"hidden_dims": [8], "dropout": 0.1},
        optim={"lr": 1e-3},
        class_weights=weights,
    )

    # The weight is live on the criterion...
    name = "pos_weight" if task == "binary" else "weight"
    assert getattr(model.criterion, name) is not None

    # ...and absent from what gets written to disk.
    assert not [k for k in model.state_dict() if k.startswith("criterion.")]


def test_a_weighted_model_round_trips_through_a_checkpoint(tmp_path):
    """The end-to-end shape of the bug: train with weights, reload without them."""
    torch = pytest.importorskip("torch")
    import numpy as np

    from ml_framework.plugins.mlp import MLP

    kwargs = {
        "input_dim": 4,
        "output_dim": 1,
        "task": "binary",
        "params": {"hidden_dims": [8], "dropout": 0.1},
        "optim": {"lr": 1e-3},
    }
    trained = MLP(**kwargs, class_weights=np.asarray([2.0], "float32"))
    path = tmp_path / "m.pt"
    torch.save(trained.state_dict(), path)

    # `class_weights=None` is exactly what `LightningBackend.fit` passes on reload.
    reloaded = MLP(**kwargs, class_weights=None)
    result = reloaded.load_state_dict(torch.load(path), strict=True)
    assert not result.unexpected_keys
    assert not result.missing_keys
