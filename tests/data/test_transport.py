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
