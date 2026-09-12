"""The pipeline's tail: construct → transform → h2d → gpu_transform.

Four of the seven stages had no name and no seam. ``read``/``demux``/``decode``
were declared per decoder and probed offline; everything after them was fused into
one opaque ``collate_fn`` that ran in a DataLoader worker — including
``audio.py``'s mel front-end, whose docstring claimed it ran "on the GPU when there
is one" while ``mel.to(x.device)`` resolved to CPU on every machine that has ever
run it.

What is tested here is therefore not "does a spectrogram come out" but the four
properties the split exists to create:

* the two halves are **separable** — ``build_tensor`` constructs and stops, and
  what it returns is what crosses the bus;
* they are **equivalent** — deferring the transform past H2D produces the same
  batch as running it in the collate, or the seam is a silent skew generator;
* the deferral is **conditional**, and every condition is a case where deferring
  would be wrong rather than merely unhelpful;
* the tail is **probeable offline**, which is the half of ``mlf materialize`` that
  a per-sample decode probe cannot reach.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from ml_framework.core.types import DECODER_STAGES, STAGES, TAIL_STAGES
from ml_framework.data.lightning_adapter import BundleDataModule
from ml_framework.data.preprocess.audio import AudioPreprocessor
from ml_framework.data.preprocess.base import BasePreprocessor, PreprocessorError, host_array
from ml_framework.data.preprocess.video import VideoPreprocessor
from ml_framework.data.streaming.stages import Decoded
from ml_framework.data.streaming.tail_probe import TailProbeError, probe_tail
from ml_framework.data.types import DataBundle, FeatureSchema, Split

pytestmark = pytest.mark.unit


# ── Fixtures ──────────────────────────────────────────────────────────
def pcm_sample(seed: int, *, n: int = 8000) -> Decoded:
    """One decoded audio sample, exactly as `wave.pcm` leaves it."""
    rng = np.random.default_rng(seed)
    return Decoded(
        array=rng.integers(-3000, 3000, n, dtype="int16"),
        layout="pcm",
        dtype="int16",
        rate=16_000,
    )


def clip_sample(seed: int, *, frames: int = 4, size: int = 8) -> Decoded:
    """One decoded video clip: THWC uint8, which is what libavcodec produces."""
    rng = np.random.default_rng(seed)
    return Decoded(
        array=rng.integers(0, 255, (frames, size, size, 3), dtype="uint8"),
        layout="thwc",
        dtype="uint8",
        rate=25,
    )


def audio_batch(n: int = 4) -> list[tuple[Decoded, int]]:
    return [(pcm_sample(i), i % 2) for i in range(n)]


def video_batch(n: int = 4) -> list[tuple[Decoded, int]]:
    return [(clip_sample(i), i % 2) for i in range(n)]


def audio_pre() -> AudioPreprocessor:
    # 0.5 s clips at the default 16 kHz: small enough to be fast, long enough that
    # the mel geometry is real rather than degenerate.
    return AudioPreprocessor(clip_seconds=0.5, n_mels=16)


def staged_bundle(preprocessor: Any, *, meta: dict[str, Any] | None = None) -> DataBundle:
    """A `payload="dataset"` bundle whose split is a plain list of decoded samples.

    A real `StagedDataset` is not needed to test the tail: everything under test
    happens strictly after `__getitem__` returns, and a list is the same shape.
    """
    items = audio_batch(8)
    split = Split(payload="dataset", x=items, y=np.asarray([label for _, label in items]))
    return DataBundle(
        train=split,
        val=split,
        test=split,
        schema=FeatureSchema(),
        task="binary",
        data_kind="audio",
        preprocessor=preprocessor,
        meta=meta or {},
    )


# ── The vocabulary ────────────────────────────────────────────────────
def test_the_pipeline_is_named_end_to_end():
    """Four stages had no name, so nothing could declare, time or probe them."""
    assert STAGES == (
        "read",
        "demux",
        "decode",
        "construct",
        "transform",
        "h2d",
        "gpu_transform",
    )
    assert DECODER_STAGES + TAIL_STAGES == STAGES


def test_no_decoder_claims_a_tail_stage():
    """`DecoderSpec.stages` is typed `DecoderStage` precisely so this cannot drift:
    a decoder claiming `h2d` would be a claim nothing anywhere executes."""
    from ml_framework.core.registry import DECODERS
    from ml_framework.data import streaming as _streaming  # noqa: F401  (registers them)

    for spec in DECODERS.specs():
        assert spec.stages <= set(DECODER_STAGES), f"{spec.name} declares a tail stage"


def test_the_staged_preprocessors_declare_the_tail_they_implement():
    for preprocessor in (audio_pre(), VideoPreprocessor(clip_len=4, img_size=8)):
        assert preprocessor.stages == frozenset({"construct", "transform", "gpu_transform"})


def test_a_preprocessor_that_never_split_its_tail_declares_nothing():
    """The v1 shape stays supported: an opaque `collate_fn` and no claims. The
    transport layer must not try to split what did not say it could be split."""
    from ml_framework.data.preprocess.tabular import TabularPreprocessor

    assert BasePreprocessor().stages == frozenset()
    assert TabularPreprocessor().stages == frozenset()


# ── construct is separable from transform ─────────────────────────────
def test_audio_construct_yields_a_waveform_not_a_spectrogram():
    """What `build_tensor` returns is what crosses the bus. A mel here would mean
    the H2D copy carried the transform's output rather than its input."""
    pre = audio_pre()
    x, y = pre.build_tensor(audio_batch(4))

    assert x.shape == (4, pre.clip_samples)
    assert x.dtype == torch.float32
    assert y.tolist() == [0, 1, 0, 1]


def test_audio_transform_is_the_mel_and_only_the_mel():
    pre = audio_pre()
    x, _ = pre.build_tensor(audio_batch(4))

    features = pre.transform_batch(x)

    assert features.shape == (4, 1, pre.n_mels, pre.n_frames)


def test_video_construct_leaves_uint8_so_the_copy_is_four_times_smaller():
    """The whole reason deferring video's transform pays twice: the permute+cast+
    divide is a 4x blowup, so doing it after H2D means the bus carries uint8."""
    pre = VideoPreprocessor(clip_len=4, img_size=8)
    x, y = pre.build_tensor(video_batch(3))

    assert x.shape == (3, 4, 8, 8, 3)
    assert x.dtype == torch.uint8
    assert y.tolist() == [0, 1, 0]


def test_video_transform_permutes_casts_and_normalizes():
    pre = VideoPreprocessor(clip_len=4, img_size=8)
    x, _ = pre.build_tensor(video_batch(3))

    out = pre.transform_batch(x)

    assert out.shape == (3, 3, 4, 8, 8)
    assert out.dtype == torch.float32
    # Normalized against Kinetics statistics, so [0,1] is no longer the range.
    assert out.min() < 0.0


def legacy_build(pre: AudioPreprocessor, batch) -> np.ndarray:
    """The pre-fusion construct, spelled out: cast, fit, stack.

    Kept here rather than deleted with the implementation because it is the only
    independent statement of what `build_tensor` must produce. The fused version
    is an *optimization*, and an optimization is only correct if it moves no
    sample — so the reference it has to match has to exist somewhere.
    """

    def fit(pcm: np.ndarray) -> np.ndarray:
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=0, dtype="float32")
        want = pre.clip_samples
        have = int(pcm.shape[-1])
        if have == want:
            return pcm
        if have > want:
            start = (have - want) // 2
            return pcm[start : start + want]
        pad = want - have
        left = pad // 2
        return np.pad(pcm, (left, pad - left), mode="constant")

    def to_f32(pcm: np.ndarray) -> np.ndarray:
        if pcm.dtype == np.int16:
            return pcm.astype("float32") / 32768.0
        if pcm.dtype == np.int32:
            return pcm.astype("float32") / 2147483648.0
        return np.ascontiguousarray(pcm, dtype="float32")

    waves = [fit(to_f32(np.asarray(s.array))) for s in batch]
    return np.stack(waves, axis=0).astype("float32")


@pytest.mark.parametrize("dtype", ["int16", "int32", "float32"])
@pytest.mark.parametrize("n", [4000, 8000, 12000])  # shorter, exact, longer
def test_the_fused_construct_is_bit_identical_to_the_two_copy_version(dtype, n):
    """The fusion must not move a single sample.

    `build_tensor` allocates the batch buffer once and casts each clip straight
    into its row, instead of widening to float32 and then stacking. That is two
    fewer passes over the data and one fewer allocation per sample — and it is
    only worth anything if the bytes are unchanged, which is what this asserts
    across every integer width and all three crop/pad cases.
    """
    pre = AudioPreprocessor(clip_seconds=0.5, n_mels=16)  # clip_samples == 8000
    rng = np.random.default_rng(n)
    if dtype == "float32":
        raw = rng.standard_normal(n).astype("float32")
    else:
        info = np.iinfo(dtype)
        raw = rng.integers(info.min // 2, info.max // 2, n, dtype=dtype)
    batch = [Decoded(array=raw, layout="pcm", dtype=dtype, rate=16_000) for _ in range(3)]

    x, _ = pre.build_tensor(batch)

    assert np.array_equal(x.numpy(), legacy_build(pre, batch)), "the fusion moved a sample"


def test_the_fused_construct_matches_for_stereo_too():
    """The scale is read from the ORIGINAL dtype. Reading it after `_mono` widened
    a stereo int16 clip to float32 would skip the divide entirely and hand the
    model sample values three orders of magnitude too large — loud, but only in
    the loss curve."""
    pre = AudioPreprocessor(clip_seconds=0.5, n_mels=16)
    raw = np.random.default_rng(0).integers(-30000, 30000, (2, 9000), dtype="int16")
    batch = [Decoded(array=raw, layout="pcm", dtype="int16", rate=16_000)]

    x, _ = pre.build_tensor(batch)

    assert np.array_equal(x.numpy(), legacy_build(pre, batch))
    assert abs(float(x.abs().max())) <= 1.0, "stereo int16 was not scaled into [-1, 1)"


def test_the_clip_geometry_centres_in_both_directions():
    """`_span` is the whole crop/pad rule, so it is worth testing as arithmetic
    rather than only through the buffer it fills.

    Centred both ways on purpose: a **centre crop** because the informative part of
    a clip is usually not at its start, and a **centre pad** because a short clip
    flush-left would put every one of them against the same edge — which a
    convolution can learn, and which no shape check would ever catch.
    """
    pre = AudioPreprocessor(clip_seconds=0.5, n_mels=16)  # clip_samples == 8000

    assert pre._span(8000) == (slice(0, 8000), slice(0, 8000))  # exact: identity
    assert pre._span(10000) == (slice(1000, 9000), slice(0, 8000))  # crop, centred
    assert pre._span(6000) == (slice(0, 6000), slice(1000, 7000))  # pad, centred

    # Whatever the input length, the destination is always inside one fixed row.
    for have in (1, 7999, 8000, 8001, 40_000):
        _, dst = pre._span(have)
        assert 0 <= dst.start <= dst.stop <= pre.clip_samples


def test_the_written_span_is_scaled_and_the_padding_is_not():
    """Only the region a clip actually occupies gets the integer divisor; the
    zero padding around it must stay zero. Scaling the whole row would be
    harmless here (0/32768 == 0) and wrong the moment a non-zero fill is used."""
    pre = AudioPreprocessor(clip_seconds=0.5, n_mels=16)
    short = np.full(6000, 32767, dtype="int16")  # full-scale, easy to spot

    x, _ = pre.build_tensor([Decoded(array=short, layout="pcm", dtype="int16")])
    row = x.numpy()[0]

    _, dst = pre._span(6000)
    assert np.allclose(row[dst], 32767 / 32768.0)
    assert row[: dst.start].tolist() == [0.0] * dst.start
    assert row[dst.stop :].tolist() == [0.0] * (pre.clip_samples - dst.stop)


def test_an_unlabelled_batch_constructs_a_null_label_half():
    pre = audio_pre()
    x, y = pre.build_tensor([pcm_sample(0), pcm_sample(1)])

    assert x.shape[0] == 2
    assert y is None


# ── ...and composing them is still one operation ──────────────────────
@pytest.mark.parametrize(
    ("preprocessor", "batch"),
    [(audio_pre(), audio_batch(4)), (VideoPreprocessor(clip_len=4, img_size=8), video_batch(4))],
)
def test_collate_staged_is_construct_then_transform(preprocessor, batch):
    x, y = preprocessor.collate_staged(batch)
    expected, _ = preprocessor.build_tensor(batch)

    assert torch.allclose(x, preprocessor.transform_batch(expected))
    assert y.tolist() == [0, 1, 0, 1]


def test_collate_fn_is_still_the_whole_tail():
    """The contract every consumer outside the training loop depends on: serving,
    `predict_split` and the LR range test all ask for `collate_fn` and must get a
    finished batch, not a half-built one."""
    pre = audio_pre()
    x, _ = pre.collate_fn(audio_batch(4))

    assert x.shape == (4, 1, pre.n_mels, pre.n_frames)


def test_serving_transform_goes_through_the_same_two_stages():
    """A second implementation of the front-end is exactly how train/serve skew
    gets in, so `transform()` must route through the training path."""
    pre = audio_pre()
    sample = pcm_sample(7)

    served = pre.transform(sample)
    trained = pre.collate_staged([sample])

    assert torch.allclose(served, trained)


# ── The device seam ───────────────────────────────────────────────────
def test_the_transform_stays_in_the_collate_without_a_device():
    """Nothing to defer to. The pipeline behaves exactly as it did before the split."""
    dm = BundleDataModule(staged_bundle(audio_pre()), accelerator="cpu")
    dm.setup()

    assert not dm.defers_transform
    assert dm.collate == dm.preprocessor._collate_full


def test_the_transform_moves_past_h2d_when_cuda_is_selected():
    """`accelerator="gpu"` is what Lightning will resolve, so it is what this
    resolves — the same rule pinning follows, and it needs no actual GPU."""
    dm = BundleDataModule(staged_bundle(audio_pre()), accelerator="gpu")
    dm.setup()

    assert dm.defers_transform
    assert dm.collate == dm.preprocessor._collate_deferred


def test_deferring_and_not_deferring_produce_the_same_batch():
    """The property the whole seam rests on. If these two ever disagree, the
    device path is training on something the serving path never sees."""
    batch = audio_batch(4)

    eager = BundleDataModule(staged_bundle(audio_pre()), accelerator="cpu")
    eager.setup()
    deferred = BundleDataModule(staged_bundle(audio_pre()), accelerator="gpu")
    deferred.setup()

    x_eager, y_eager = eager.collate(batch)
    x_after_hook, y_after_hook = deferred.on_after_batch_transfer(deferred.collate(batch), 0)

    assert torch.allclose(x_eager, x_after_hook)
    assert y_eager.tolist() == y_after_hook.tolist()


def test_the_hook_is_a_no_op_when_the_transform_already_ran():
    """Applied twice, the mel of a mel is not a spectrogram of anything."""
    dm = BundleDataModule(staged_bundle(audio_pre()), accelerator="cpu")
    dm.setup()

    batch = dm.collate(audio_batch(4))

    assert dm.on_after_batch_transfer(batch, 0) is batch


def test_an_explicit_collate_fn_forbids_deferring():
    """That callable did some unknown part of the tail; adding a transform on top
    of it would apply the front-end twice. `predict_split` relies on this."""
    pre = audio_pre()
    dm = BundleDataModule(staged_bundle(pre), accelerator="gpu", collate_fn=pre.collate_fn)
    dm.setup()

    assert not dm.defers_transform
    assert dm.collate == pre.collate_fn


def test_a_device_landing_decoder_has_no_host_tail_to_move():
    """The bytes never touched host memory, so there is nothing to defer — the
    same reason pinning and workers are already forced off for one."""
    dm = BundleDataModule(
        staged_bundle(audio_pre(), meta={"lands_in": "device"}), accelerator="gpu"
    )
    dm.setup()

    assert dm.lands_in == "device"
    assert not dm.defers_transform


def test_a_preprocessor_that_makes_no_device_claim_is_not_deferred():
    """`gpu_transform` is a claim that the transform is device-agnostic. Assuming
    it of a preprocessor that never made it is the unstated assumption this whole
    pipeline exists to remove."""

    class HostOnly(AudioPreprocessor):
        stages = frozenset({"construct", "transform"})

    dm = BundleDataModule(staged_bundle(HostOnly(clip_seconds=0.5, n_mels=16)), accelerator="gpu")
    dm.setup()

    assert not dm.defers_transform
    # Still uses the split path -- it declared `construct` -- just without deferring.
    assert dm.collate == dm.preprocessor._collate_full


def test_device_transform_can_be_turned_off():
    dm = BundleDataModule(staged_bundle(audio_pre()), accelerator="gpu", device_transform=False)
    dm.setup()

    assert not dm.defers_transform


def test_set_device_transform_is_what_a_manual_loop_uses():
    """`mlf lr` iterates the loader itself, so Lightning never calls
    `on_after_batch_transfer` and a deferred transform would simply never run."""
    dm = BundleDataModule(staged_bundle(audio_pre()), accelerator="gpu")
    dm.setup()
    assert dm.defers_transform

    dm.set_device_transform(False)

    assert not dm.defers_transform
    assert dm.collate == dm.preprocessor._collate_full


def test_the_lr_finder_turns_it_off_before_setup():
    """Pinned as source, not behaviour: running the real LR finder needs an
    optional extra, and what matters is that the call is there and comes first."""
    import inspect

    from ml_framework.pipeline import lr_finder

    source = inspect.getsource(lr_finder.find_lr)
    assert "set_device_transform(False)" in source
    assert source.index("set_device_transform(False)") < source.index("dm.setup()")


def test_an_ordinary_bundle_is_untouched_by_any_of_this():
    """A tabular run has no preprocessor tail and must get no collate at all."""
    rng = np.random.default_rng(0)
    split = Split(payload="arrays", x=rng.standard_normal((16, 3)).astype("float32"))
    bundle = DataBundle(
        train=split,
        val=split,
        test=split,
        schema=FeatureSchema(),
        task="binary",
        data_kind="tabular",
    )
    dm = BundleDataModule(bundle, accelerator="gpu")
    dm.setup()

    assert dm.collate is None
    assert not dm.defers_transform
    marker = object()
    assert dm.on_after_batch_transfer(marker, 0) is marker


# ── The device-handle hole ────────────────────────────────────────────
def test_a_device_handle_is_refused_by_name_rather_than_becoming_an_object_array():
    """`np.asarray` on a `DeviceHandle` yields a 0-d object array — a valid-shaped
    wrong result, which is the exact failure class this pipeline exists to make
    impossible. It was reachable: `audio.py` called `np.asarray` unguarded."""
    from ml_framework.data.streaming.decoders.fake_device import DeviceHandle

    decoded = Decoded(
        array=DeviceHandle((4, 4, 3), "uint8"), layout="hwc", dtype="uint8", lands_in="device"
    )

    with pytest.raises(PreprocessorError, match="device memory"):
        host_array(decoded)

    with pytest.raises(PreprocessorError, match="device memory"):
        audio_pre().build_tensor([decoded])


def test_host_array_passes_an_ordinary_decoded_sample_straight_through():
    sample = pcm_sample(0)
    assert host_array(sample) is sample.array


# ── The offline tail probe ────────────────────────────────────────────
def test_the_probe_runs_every_stage_it_can_and_names_the_ones_it_cannot():
    """ "Skipped: no CUDA device" is a materially different claim from "passed",
    and the report must not conflate them."""
    report = probe_tail(audio_batch(4), audio_pre())

    assert report.ok
    assert "construct" in report.ran
    assert "transform" in report.ran
    if torch.cuda.is_available():
        assert report.ran == ["construct", "transform", "h2d", "gpu_transform"]
        assert report.device_agreement is not None
    else:
        assert report.skipped["h2d"] == "no CUDA device on this machine"
        assert report.skipped["gpu_transform"] == "no CUDA device on this machine"


def test_the_probe_catches_a_construct_failure_a_decode_probe_cannot_see():
    """Every sample here decodes perfectly. What fails is the geometry rule — a
    per-batch step, so no per-sample probe anywhere could reach it.

    The failure is injected at `_span` because that is now the single statement of
    the crop/pad rule: a `_span` whose two slices disagree cannot write a row, the
    same way a clip geometry that does not produce `clip_samples` cannot.
    """

    class BadSpan(AudioPreprocessor):
        def _span(self, have: int) -> tuple[slice, slice]:
            src, dst = super()._span(have)
            # Source and destination now differ in width: the assignment into the
            # batch row raises rather than silently truncating.
            return src, slice(dst.start, dst.stop - 3)

    report = probe_tail(audio_batch(4), BadSpan(clip_seconds=0.5, n_mels=16))

    assert not report.ok
    assert report.failure[0] == "construct"
    assert "construct" not in report.ran


def test_a_ragged_corpus_can_no_longer_produce_a_stacking_failure():
    """A property the fused construct gained, worth pinning because it removes a
    whole failure mode: the destination is a fixed-size batch buffer, so clips of
    wildly different lengths are cropped or padded into it rather than failing to
    stack. Variable-duration audio is now a shape the pipeline absorbs."""
    pre = AudioPreprocessor(clip_seconds=0.5, n_mels=16)
    ragged = [
        Decoded(array=np.zeros(n, dtype="int16"), layout="pcm", dtype="int16")
        for n in (500, 8000, 40_000)
    ]

    x, _ = pre.build_tensor(ragged)

    assert x.shape == (3, pre.clip_samples)


def test_the_probe_catches_a_transform_failure():
    class BadGeometry(AudioPreprocessor):
        def transform_batch(self, x: Any) -> Any:
            raise RuntimeError("filterbank does not divide the clip")

    report = probe_tail(audio_batch(4), BadGeometry(clip_seconds=0.5, n_mels=16))

    assert not report.ok
    assert report.failure[0] == "transform"
    assert "filterbank does not divide" in report.failure[1]
    # construct still ran and is still reported -- knowing how far it got is the
    # difference between "the corpus is wrong" and "the geometry is wrong".
    assert report.ran == ["construct"]


def test_forcing_the_device_runs_all_four_stages_without_a_gpu():
    """`device="cpu"` makes the h2d/gpu_transform path executable on any machine.
    The copy is a no-op, but every decision after it is the real one — which is
    what needs testing, since those decisions are what fail a run."""
    report = probe_tail(audio_batch(4), audio_pre(), device="cpu")

    assert report.ok
    assert report.ran == ["construct", "transform", "h2d", "gpu_transform"]
    assert report.device_agreement == pytest.approx(0.0)
    assert report.skipped == {}


def test_a_transform_that_is_not_device_agnostic_is_caught_by_shape():
    """`gpu_transform` is a *claim* that host and device are the same operation.
    An unchecked claim of that shape is what this pipeline refuses everywhere
    else, so the probe checks it rather than taking the declaration's word."""

    class Asymmetric(AudioPreprocessor):
        seen = 0

        def transform_batch(self, x: Any) -> Any:
            # Second call (the "device" one) returns a different shape.
            Asymmetric.seen += 1
            out = super().transform_batch(x)
            return out if Asymmetric.seen == 1 else out[:, :, :-1, :]

    report = probe_tail(audio_batch(4), Asymmetric(clip_seconds=0.5, n_mels=16), device="cpu")

    assert not report.ok
    assert report.failure[0] == "gpu_transform"
    assert "claims these are the same operation" in report.failure[1]


def test_a_device_transform_that_merely_drifts_is_caught_too():
    """The worse version of the same bug: right shape, wrong numbers. Training on
    one and serving on the other is train/serve skew nothing reports."""

    class Drifting(AudioPreprocessor):
        seen = 0

        def transform_batch(self, x: Any) -> Any:
            Drifting.seen += 1
            out = super().transform_batch(x)
            return out if Drifting.seen == 1 else out + 0.5

    report = probe_tail(audio_batch(4), Drifting(clip_seconds=0.5, n_mels=16), device="cpu")

    assert not report.ok
    assert report.failure[0] == "gpu_transform"
    assert "differ by up to" in report.failure[1]
    assert report.device_agreement == pytest.approx(0.5, abs=1e-4)


def test_a_preprocessor_that_makes_no_device_claim_stops_after_h2d():
    class HostOnly(AudioPreprocessor):
        stages = frozenset({"construct", "transform"})

    report = probe_tail(audio_batch(2), HostOnly(clip_seconds=0.5, n_mels=16), device="cpu")

    assert report.ok
    assert report.ran == ["construct", "transform", "h2d"]
    assert "does not declare gpu_transform" in report.skipped["gpu_transform"]


def test_h2d_copies_the_pre_transform_tensor_not_the_transformed_one():
    """What the transport layer actually moves when the transform is deferred.
    Copying the mel instead would measure a stage the run does not perform."""
    pre = audio_pre()
    report = probe_tail(audio_batch(4), pre, device="cpu")

    assert report.shapes["h2d"].startswith(f"(4, {pre.clip_samples})")
    assert report.shapes["gpu_transform"].startswith(f"(4, 1, {pre.n_mels}, {pre.n_frames})")


def test_the_probe_declines_a_preprocessor_with_no_staged_tail():
    """Not a failure. Probing "did the opaque collate run" is both weaker and
    something the first training step already tells you."""
    report = probe_tail(audio_batch(2), BasePreprocessor())

    assert report.ok
    assert report.ran == []
    assert set(report.skipped) == set(TAIL_STAGES)


def test_the_report_renders_every_stage_it_has_something_to_say_about():
    text = probe_tail(audio_batch(2), audio_pre()).render()

    assert "construct" in text
    assert "transform" in text
    assert "float32" in text


# ── ...wired into `mlf materialize` ───────────────────────────────────
def _wav_corpus(root: Path, *, per_class: int = 4) -> Path:
    rng = np.random.default_rng(0)
    for ci in range(2):
        (root / f"class-{ci}").mkdir(parents=True, exist_ok=True)
        for k in range(per_class):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(rng.integers(-3000, 3000, 8000, dtype="int16").tobytes())
            (root / f"class-{ci}" / f"{k:03d}.wav").write_bytes(buf.getvalue())
    return root


def test_materialize_probes_the_tail_when_asked(tmp_path: Path):
    from ml_framework.data.streaming.materialize import materialize

    root = _wav_corpus(tmp_path / "clips")
    _, report = materialize(root, tail_preprocessor=audio_pre())

    assert report.n_faults == 0
    assert report.tail is not None
    assert report.tail.ok
    assert "construct" in report.render()


def test_materialize_says_nothing_about_the_tail_unless_asked(tmp_path: Path):
    """`None` rather than an empty report: "not probed" and "probed and clean" are
    different claims, the same rule `stall.json` and `faults.json` follow."""
    from ml_framework.data.streaming.materialize import materialize

    _, report = materialize(_wav_corpus(tmp_path / "clips"))

    assert report.tail is None
    assert "tail probe" not in report.render()


def test_a_corpus_that_decodes_but_cannot_batch_is_refused_offline(tmp_path: Path):
    """The gap this closes: every sample read, demuxed and decoded cleanly, and
    the run would still have died on step 1."""
    from ml_framework.data.streaming.materialize import materialize

    class BadGeometry(AudioPreprocessor):
        def transform_batch(self, x: Any) -> Any:
            raise RuntimeError("filterbank does not divide the clip")

    root = _wav_corpus(tmp_path / "clips")
    with pytest.raises(TailProbeError, match="stage 'transform' fails on the first batch"):
        materialize(root, tail_preprocessor=BadGeometry(clip_seconds=0.5, n_mels=16))


def test_the_tail_probe_reuses_the_buffers_the_walk_already_decoded(tmp_path: Path):
    """Decoding the corpus twice to build one batch would double the cost of the
    whole pass to learn nothing new."""
    from ml_framework.data.streaming import materialize as materialize_mod

    root = _wav_corpus(tmp_path / "clips", per_class=6)
    seen: list[int] = []
    original = materialize_mod._probe_one

    def counting(*args, **kwargs):
        seen.append(1)
        return original(*args, **kwargs)

    materialize_mod._probe_one = counting
    try:
        _, report = materialize_mod.materialize(root, tail_preprocessor=audio_pre())
    finally:
        materialize_mod._probe_one = original

    assert sum(seen) == 12, "one decode per sample, not two"
    assert report.tail is not None and report.tail.batch_size == 8


def test_probe_full_needs_a_config_and_says_why(tmp_path: Path):
    """The tail belongs to a preprocessor, and which preprocessor is a property of
    the run rather than of the bytes on disk — so unlike the decode probe, it
    cannot be inferred from the corpus."""
    from ml_framework.cli import main

    root = _wav_corpus(tmp_path / "clips")
    assert main(["materialize", "--data", str(root), "--probe-full"]) == 1


def test_probe_full_refuses_a_kind_with_no_staged_tail(tmp_path: Path):
    from ml_framework.data.sources import preprocessor_for

    class Cfg:
        class data:  # noqa: N801 - a stand-in for the config's shape
            kind = "tabular"

    with pytest.raises(ValueError, match="no staged tail"):
        preprocessor_for(Cfg)


# ── The decoder params `materialize` was dropping ─────────────────────
def _cfg(kind: str, *, params: dict | None = None, decoder_params: dict | None = None):
    """A stand-in carrying only what `decoder_params_for` reads."""
    return SimpleNamespace(
        data=SimpleNamespace(kind=kind, params=params or {}, decoder_params=decoder_params or {})
    )


def test_a_video_corpus_materializes_at_the_configured_geometry():
    """The bug this closes: `mlf materialize` passed `data.decoder_params` raw, so a
    video decoder never heard the clip geometry and decoded whole files it then
    threw most of away. The geometry lives in `data.params` because it is also the
    preprocessor's, which is exactly why the raw dict was missing it."""
    from ml_framework.data.sources import decoder_params_for

    params = decoder_params_for(_cfg("video", params={"clip_len": 32, "frame_stride": 4}))

    assert params["clip_len"] == 32
    assert params["frame_stride"] == 4


def test_an_explicit_decoder_param_still_wins():
    """An explicit setting is a decision, and the derived geometry must not
    overwrite it — the same precedence `build_video_bundle` already used."""
    from ml_framework.data.sources import decoder_params_for

    params = decoder_params_for(
        _cfg("video", params={"clip_len": 32}, decoder_params={"clip_len": 8})
    )

    assert params["clip_len"] == 8


def test_a_kind_that_implies_nothing_passes_its_params_through():
    """Never raises, unlike `preprocessor_for`: for these kinds
    `data.decoder_params` as written IS the right answer, and refusing would push
    a per-kind branch back into the caller."""
    from ml_framework.data.sources import decoder_params_for

    assert decoder_params_for(_cfg("audio", decoder_params={"dtype": "float32"})) == {
        "dtype": "float32"
    }
    assert decoder_params_for(_cfg("tabular")) == {}


def test_the_bundle_builder_and_materialize_derive_the_same_params():
    """Two callers, one rule. They disagreed before this, and the symptom was a
    shard index whose `n_units` described clips no run would ever read."""
    from ml_framework.data.sources import decoder_params_for
    from ml_framework.data.sources.video import video_decoder_params

    cfg = _cfg("video", params={"clip_len": 16, "frame_stride": 2})

    assert decoder_params_for(cfg) == video_decoder_params(cfg)


# ── Real CUDA ─────────────────────────────────────────────────────────
# Everything above tests the device path with `device="cpu"`, which exercises every
# *decision* — shape agreement, the drift threshold, the skip reasons — while the
# copy itself is a no-op and no device kernel ever runs. These four are the part
# that needs actual hardware. They skip everywhere today, by design: a test that
# silently passes without running is worse than one that says it did not run.
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.mark.gpu
@needs_cuda
def test_the_tail_probe_runs_end_to_end_on_a_real_device():
    from ml_framework.data.streaming.tail_probe import DEVICE_AGREEMENT_TOLERANCE

    report = probe_tail(audio_batch(8), audio_pre())

    assert report.ok, report.failure
    assert report.ran == ["construct", "transform", "h2d", "gpu_transform"]
    assert report.skipped == {}
    assert "cuda" in report.shapes["h2d"]
    # The number DEVICE_AGREEMENT_TOLERANCE should be calibrated from -- print it
    # so a first GPU run leaves the evidence in the log rather than only a verdict.
    print(f"\nmeasured host-vs-device drift: {report.device_agreement:.3e}")
    assert report.device_agreement <= DEVICE_AGREEMENT_TOLERANCE


@pytest.mark.gpu
@needs_cuda
def test_the_mel_actually_runs_on_the_device():
    """The claim `audio.py`'s docstring made for a year and the code did not keep."""
    pre = audio_pre()
    x, _ = pre.build_tensor(audio_batch(4))

    out = pre.transform_batch(x.to("cuda"))

    assert out.device.type == "cuda"


@pytest.mark.gpu
@needs_cuda
def test_deferring_past_h2d_matches_the_eager_path_on_real_hardware():
    """The equivalence the whole seam rests on, against real kernels rather than a
    no-op copy. cuDNN and the CPU accumulate in different orders, so this is
    `allclose`, not `array_equal` -- and the tolerance here is the claim."""
    batch = audio_batch(4)

    eager = BundleDataModule(staged_bundle(audio_pre()), accelerator="cpu")
    eager.setup()
    deferred = BundleDataModule(staged_bundle(audio_pre()), accelerator="gpu")
    deferred.setup()

    x_eager, _ = eager.collate(batch)
    moved = [t.to("cuda") for t in deferred.collate(batch)]
    x_device, _ = deferred.on_after_batch_transfer(moved, 0)

    assert x_device.device.type == "cuda"
    assert torch.allclose(x_eager, x_device.cpu(), atol=1e-3)


@pytest.mark.gpu
@needs_cuda
def test_pinning_is_on_for_a_host_corpus_on_a_real_device():
    """`pin_memory` decides; torch performs. Worth one check that the decision and
    the hardware agree, since every other pinning test asserts against a
    `_cuda_selected` answer rather than a device."""
    dm = BundleDataModule(staged_bundle(audio_pre()), pin_memory=True, accelerator="auto")
    dm.setup()

    assert dm.pin_memory
