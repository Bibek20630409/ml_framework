"""The transport tail: pinning, prefetch, workers, and the device seam.

The stages after decode — wrap as tensor, collate, **pin**, **H2D** — were absent
from this framework entirely: no ``pin_memory``, ``persistent_workers`` or
``prefetch_factor`` appeared anywhere in the repo. This is where they land.

The theme is that each of them is *conditional*, and every condition is about
avoiding a cost that buys nothing:

* pinning host RAM when there is no CUDA device to copy to,
* pinning a buffer a decoder already left **on** the device,
* asking for prefetch depth when there are no workers to prefetch,
* spawning workers that would have to return a CUDA tensor across a fork.

The last is the one that would hang rather than merely waste, which is why it is
enforced with a warning rather than documented.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from ml_framework.data.lightning_adapter import BundleDataModule
from ml_framework.data.types import DataBundle, FeatureSchema, Split

pytestmark = pytest.mark.unit


def make_bundle(*, n: int = 64, meta: dict[str, Any] | None = None) -> DataBundle:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 4)).astype("float32")
    y = rng.integers(0, 2, n)
    split = Split(payload="arrays", x=x, y=y)
    return DataBundle(
        train=split,
        val=split,
        test=split,
        schema=FeatureSchema(feature_names=("a", "b", "c", "d"), target_name="y"),
        task="binary",
        data_kind="tabular",
        input_dim=4,
        output_dim=1,
        meta=meta or {},
    )


def loader_kwargs(dm: BundleDataModule, name: str = "train") -> dict[str, Any]:
    """The kwargs `_loader` would pass to DataLoader, captured rather than inferred."""
    captured: dict[str, Any] = {}

    def _capture(dataset, **kwargs):
        captured.update(kwargs)
        return object()

    with patch("ml_framework.data.lightning_adapter.DataLoader", _capture):
        dm._loader(name)
    return captured


# ── Pinning ───────────────────────────────────────────────────────────
def test_pinning_is_off_without_a_cuda_device_even_when_requested():
    """The only effect would be page-locking host RAM for a copy that never
    happens — a real cost, invisibly paid, on every CPU box."""
    dm = BundleDataModule(make_bundle(), pin_memory=True, accelerator="cpu")
    dm.setup()

    assert not dm.pin_memory
    assert "pin_memory" not in loader_kwargs(dm)


def test_pinning_is_on_when_cuda_is_selected_and_the_batch_lands_on_the_host():
    dm = BundleDataModule(make_bundle(), pin_memory=True, accelerator="gpu")
    dm.setup()

    assert dm.pin_memory
    assert loader_kwargs(dm)["pin_memory"] is True


def test_pinning_is_off_for_a_decoder_that_already_landed_on_the_device():
    """Pinning a buffer already in device memory is a no-op at best and a
    device-to-host round trip at worst."""
    dm = BundleDataModule(
        make_bundle(meta={"lands_in": "device"}), pin_memory=True, accelerator="gpu"
    )
    dm.setup()

    assert dm.lands_in == "device"
    assert not dm.pin_memory
    assert "pin_memory" not in loader_kwargs(dm)


def test_pinning_can_be_turned_off_explicitly_on_cuda():
    dm = BundleDataModule(make_bundle(), pin_memory=False, accelerator="gpu")
    dm.setup()
    assert not dm.pin_memory


def test_auto_accelerator_asks_torch_rather_than_guessing():
    """ "auto" is what Lightning will resolve, so this has to resolve it the same
    way — guessing permissively costs pinned RAM for nothing."""
    import torch

    dm = BundleDataModule(make_bundle(), pin_memory=True, accelerator="auto")
    dm.setup()
    assert dm.pin_memory == torch.cuda.is_available()


# ── Workers, prefetch, persistence ────────────────────────────────────
def test_prefetch_and_persistence_are_absent_without_workers():
    """torch raises for `persistent_workers=True` at `num_workers=0`, and requires
    `prefetch_factor` to be None there. Guarded, not passed and hoped for."""
    dm = BundleDataModule(make_bundle(), num_workers=0, persistent_workers=True)
    dm.setup()
    kwargs = loader_kwargs(dm)

    assert kwargs["num_workers"] == 0
    assert "persistent_workers" not in kwargs
    assert "prefetch_factor" not in kwargs


def test_prefetch_and_persistence_are_passed_when_there_are_workers():
    dm = BundleDataModule(make_bundle(), num_workers=2, persistent_workers=True, prefetch_factor=4)
    dm.setup()
    kwargs = loader_kwargs(dm)

    assert kwargs["num_workers"] == 2
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 4


def test_persistent_workers_defaults_off_because_it_changes_the_augmentation_stream():
    """The biggest throughput win on a many-worker run, and still not a default:
    `worker_init_fn` then runs once instead of per epoch. A decision, not an
    inheritance."""
    from ml_framework.config.schema import RuntimeConfig

    assert RuntimeConfig().persistent_workers is False
    assert RuntimeConfig().prefetch_factor == 2
    assert RuntimeConfig().pin_memory is True


def test_a_device_decoder_forces_workers_to_zero(caplog):
    """Enforced rather than documented: a CUDA tensor cannot be returned from a
    DataLoader worker across a fork/spawn boundary without CUDA IPC, and the
    failure mode is a hang, not an error."""
    import logging

    dm = BundleDataModule(make_bundle(meta={"lands_in": "device"}), num_workers=4)
    dm.setup()

    with caplog.at_level(logging.WARNING):
        assert dm._workers == 0

    assert "num_workers=0" in caplog.text
    assert "cannot be returned from a DataLoader worker" in caplog.text


def test_a_host_decoder_keeps_the_workers_it_asked_for(caplog):
    import logging

    dm = BundleDataModule(make_bundle(), num_workers=2)
    dm.setup()

    with caplog.at_level(logging.WARNING):
        assert dm._workers == 2
    assert "forcing num_workers" not in caplog.text


def test_lands_in_defaults_to_host_for_every_ordinary_source():
    """`bundle.meta` is the documented home for source-specific extras, so an
    absent key must mean the ordinary case rather than an error."""
    dm = BundleDataModule(make_bundle())
    dm.setup()
    assert dm.lands_in == "host"


def test_lands_in_is_host_before_setup_rather_than_raising():
    """Loader construction may read it before the bundle is materialized."""
    dm = BundleDataModule(bundle_factory=make_bundle)
    assert dm.lands_in == "host"


# ── What P13c must NOT have changed ───────────────────────────────────
def test_drop_last_is_still_a_function_of_length_and_batch_size_only():
    """Link 3 of the DDP-parity chain. A content-dependent `drop_last` would let
    two ranks build different batch counts from identical samplers."""
    captured: dict[str, Any] = {}

    def _capture(dataset, **kwargs):
        captured.update(kwargs)
        return object()

    dm = BundleDataModule(make_bundle(n=64), batch_size=32)
    dm.setup()
    with patch("ml_framework.data.lightning_adapter.DataLoader", _capture):
        dm._loader("train", shuffle=True)

    assert captured["drop_last"] is True  # 64 > 32

    dm_small = BundleDataModule(make_bundle(n=16), batch_size=32)
    dm_small.setup()
    with patch("ml_framework.data.lightning_adapter.DataLoader", _capture):
        dm_small._loader("train", shuffle=True)
    assert captured["drop_last"] is False  # 16 > 32 is False


def test_a_sampler_and_shuffle_stay_mutually_exclusive():
    captured: dict[str, Any] = {}

    def _capture(dataset, **kwargs):
        captured.update(kwargs)
        return object()

    dm = BundleDataModule(make_bundle(), batch_size=8)
    dm.setup()
    sentinel = object()
    with patch("ml_framework.data.lightning_adapter.DataLoader", _capture):
        dm._loader("train", shuffle=True, sampler=sentinel)

    assert captured["sampler"] is sentinel
    assert "shuffle" not in captured
    assert "drop_last" not in captured


def test_a_dataset_payload_is_returned_untouched_including_a_device_one():
    """No tensor construction for a lazily-decoding corpus — and none at all for a
    device-landing one, whose items are not numpy."""
    marker = object()
    bundle = DataBundle(
        train=Split(payload="dataset", x=marker),
        val=Split(payload="dataset", x=marker),
        test=Split(payload="dataset", x=marker),
        schema=FeatureSchema(),
        task="binary",
        data_kind="image",
        meta={"lands_in": "device"},
    )
    dm = BundleDataModule(bundle)
    dm.setup()

    assert dm._datasets["train"] is marker


# ── A pre-existing bug this phase closes ──────────────────────────────
def test_build_datamodule_works_for_kinds_with_no_v1_compat_class(tmp_path):
    """`mlf lr` on a text or time-series config raised `KeyError: Unknown
    datamodule 'text'` before P13c.

    Only `tabular` and `image` ever got a v1 compat class, and
    ``build_datamodule`` went straight through ``get_datamodule_class``. Fixed by
    falling back rather than by adding two more compat classes: datamodules
    stopped being an extension point in P1, and each of these is now a
    config-bound bundle factory and nothing else.
    """
    import pandas as pd

    # The v1 registry is genuinely missing them -- that is the bug's cause, and
    # the fallback is what makes it not matter.
    import ml_framework.data.lightning_adapter  # noqa: F401  (populates it)
    from ml_framework.config import ExperimentConfig
    from ml_framework.core.registry import _DATAMODULE_REGISTRY
    from ml_framework.data.builders import build_datamodule

    assert "timeseries" not in _DATAMODULE_REGISTRY

    path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=80, freq="D"),
            "value": np.sin(np.arange(80) / 5.0) + 10,
        }
    ).to_csv(path, index=False)

    config = ExperimentConfig.model_validate(
        {
            "task": "forecasting",
            "data": {
                "kind": "timeseries",
                "path": str(path),
                "target": "value",
                "split": {"time_col": "date"},
            },
            "model": {"name": "ts.naive"},
            "runtime": {"output_dir": str(tmp_path / "out")},
        }
    )

    dm = build_datamodule(config)
    assert isinstance(dm, BundleDataModule)


# ── The staged wiring P13e added ──────────────────────────────────────
def _staged_bundle(tmp_path):
    """A real staged corpus, small enough to be fast."""
    import io
    import wave

    from ml_framework.core.registry import get_decoder
    from ml_framework.data.streaming.dataset import StagedDataset
    from ml_framework.data.streaming.materialize import materialize
    from ml_framework.data.streaming.shards import ShardIndex
    from ml_framework.data.streaming.sources_io import DirSource

    root = tmp_path / "clips"
    rng = np.random.default_rng(0)
    for ci in range(2):
        (root / f"class-{ci}").mkdir(parents=True, exist_ok=True)
        for k in range(8):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as h:
                h.setnchannels(1)
                h.setsampwidth(2)
                h.setframerate(16_000)
                h.writeframes(rng.integers(-2000, 2000, 800, dtype="int16").tobytes())
            (root / f"class-{ci}" / f"{k:03d}.wav").write_bytes(buf.getvalue())
    materialize(root)

    index = ShardIndex.read(ShardIndex.location(root))
    dataset = StagedDataset(
        index, source=DirSource(root), decoder=get_decoder("audio.pcm"), integrity="loud"
    )
    split = Split(payload="dataset", x=dataset, y=np.asarray(index.labels()))
    return (
        DataBundle(
            train=split,
            val=split,
            test=split,
            schema=FeatureSchema(),
            task="binary",
            data_kind="audio",
            meta={"lands_in": "host", "index_digest": index.index_digest},
        ),
        index,
    )


def test_a_staged_corpus_gets_a_shard_sampler(tmp_path):
    """`ShardShuffleSampler` is what keeps every read inside one open shard — the
    entire reason the corpus is sharded in the first place."""
    from ml_framework.data.streaming.sampler import ShardShuffleSampler

    bundle, _ = _staged_bundle(tmp_path)
    dm = BundleDataModule(bundle, shuffle="block", seed=7)
    dm.setup()

    sampler = dm._train_sampler()
    assert isinstance(sampler, ShardShuffleSampler)
    assert sampler.seed == 7
    assert sampler.shuffle == "block"


def test_an_ordinary_bundle_gets_no_shard_sampler(tmp_path):
    """Inert for every non-staged source, whose train split has no shard index."""
    dm = BundleDataModule(make_bundle())
    dm.setup()
    assert dm._train_sampler() is None


def test_the_shard_sampler_is_built_once_so_the_epoch_survives(tmp_path):
    """Rebuilding it per `train_dataloader()` call would silently reset the epoch
    and any resume offset."""
    bundle, _ = _staged_bundle(tmp_path)
    dm = BundleDataModule(bundle)
    dm.setup()

    first = dm._train_sampler()
    dm.set_epoch(4)
    assert dm._train_sampler() is first
    assert first.epoch == 4


def test_set_epoch_reshuffles_a_staged_corpus(tmp_path):
    bundle, _ = _staged_bundle(tmp_path)
    dm = BundleDataModule(bundle, shuffle="block", seed=3)
    dm.setup()

    dm.set_epoch(0)
    first = list(dm._train_sampler())
    dm.set_epoch(1)
    second = list(dm._train_sampler())

    assert sorted(first) == sorted(second)
    assert first != second, "a new epoch must reorder, or the shuffle is decorative"


def test_the_shard_sampler_wins_over_a_weighted_one(tmp_path):
    """They never actually collide — the staged sources correct imbalance with
    loss weights precisely so they do not — but the order makes that explicit."""
    from ml_framework.data.streaming.sampler import ShardShuffleSampler

    bundle, _ = _staged_bundle(tmp_path)
    bundle = DataBundle(
        train=bundle.train,
        val=bundle.val,
        test=bundle.test,
        schema=bundle.schema,
        task=bundle.task,
        data_kind=bundle.data_kind,
        meta={**bundle.meta, "sample_weights": np.ones(16)},
    )
    dm = BundleDataModule(bundle)
    dm.setup()

    assert isinstance(dm._train_sampler(), ShardShuffleSampler)


# ── Mid-epoch resume ──────────────────────────────────────────────────
def test_the_datamodule_checkpoints_its_position(tmp_path):
    """Lightning checkpoints the model, the optimizer and the epoch. What it
    cannot checkpoint is which samples this epoch already served."""
    bundle, index = _staged_bundle(tmp_path)
    dm = BundleDataModule(bundle)
    dm.setup()
    dm.set_epoch(2)

    state = dm.state_dict()

    assert state["epoch"] == 2
    assert state["index_digest"] == index.index_digest


def test_resuming_skips_what_was_already_served(tmp_path):
    bundle, index = _staged_bundle(tmp_path)
    dm = BundleDataModule(bundle, seed=5)
    dm.setup()

    dm.load_state_dict(
        {"version": 1, "index_digest": index.index_digest, "seed": 5, "epoch": 0, "samples_seen": 6}
    )
    resumed = list(dm._train_sampler())

    assert len(resumed) == index.n_samples - 6


def test_resuming_against_a_changed_corpus_is_refused(tmp_path):
    """ "Sample 41,000" would name different bytes, so continuing would replay a
    different dataset while reporting it as the same run."""
    from ml_framework.data.streaming.state import LoaderStateError

    bundle, _ = _staged_bundle(tmp_path)
    dm = BundleDataModule(bundle)
    dm.setup()

    with pytest.raises(LoaderStateError, match="shard index changed"):
        dm.load_state_dict({"version": 1, "index_digest": "0" * 32, "samples_seen": 3})


def test_resume_state_can_be_turned_off(tmp_path):
    bundle, _ = _staged_bundle(tmp_path)
    dm = BundleDataModule(bundle, resume_state=False)
    dm.setup()
    assert dm.state_dict() == {}


def test_an_ordinary_bundle_checkpoints_a_position_that_costs_nothing(tmp_path):
    """Non-staged sources have no index digest, so the state is inert rather than
    absent — which keeps `state_dict` one shape for every kind."""
    dm = BundleDataModule(make_bundle())
    dm.setup()
    assert dm.state_dict()["index_digest"] == ""
