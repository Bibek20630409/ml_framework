"""BundleDataModule: one dataloader implementation, parameterized by sampler."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from ml_framework.data import BundleDataModule, build_bundle
from ml_framework.data.lightning_adapter import ImageDataModule, TabularDataModule
from ml_framework.data.types import DataBundle, FeatureSchema, Split


def _array_bundle(n: int = 64, d: int = 3, *, weights=None, meta=None) -> DataBundle:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, d)).astype("float32")
    y = np.array([i % 2 for i in range(n)])
    split = Split(x=x, y=y)
    return DataBundle(
        train=split,
        val=split,
        test=split,
        schema=FeatureSchema(feature_names=tuple(f"f{i}" for i in range(d))),
        task="binary",
        data_kind="tabular",
        input_dim=d,
        output_dim=1,
        class_weights=weights,
        meta=meta or {},
    )


@pytest.mark.unit
def test_derived_attributes_come_from_the_bundle():
    dm = BundleDataModule(_array_bundle(), batch_size=8, num_workers=0)
    dm.setup()
    assert dm.input_dim == 3
    assert dm.output_dim == 1
    assert dm.feature_cols == ["f0", "f1", "f2"]


@pytest.mark.unit
def test_derived_attributes_are_zero_before_setup_as_in_v1():
    dm = BundleDataModule(bundle_factory=_array_bundle, batch_size=8, num_workers=0)
    assert dm.input_dim == 0 and dm.output_dim == 0 and dm.feature_cols == []


@pytest.mark.unit
def test_lazy_factory_is_called_once_however_often_setup_runs():
    """Lightning calls setup() per stage; the pipeline calls it explicitly too."""
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _array_bundle()

    dm = BundleDataModule(bundle_factory=factory, num_workers=0)
    dm.setup("fit")
    dm.setup("test")
    assert calls["n"] == 1


@pytest.mark.unit
def test_construction_requires_a_bundle_or_a_factory():
    with pytest.raises(ValueError, match="bundle or a bundle_factory"):
        BundleDataModule()


@pytest.mark.unit
def test_bundle_access_before_setup_is_an_error_not_a_none():
    with pytest.raises(RuntimeError, match="call setup"):
        _ = BundleDataModule(bundle_factory=_array_bundle).bundle


@pytest.mark.unit
def test_numpy_class_weights_become_a_float32_tensor():
    """The conversion happens at this boundary and nowhere else."""
    dm = BundleDataModule(_array_bundle(weights=np.array([4.0], dtype="float32")), num_workers=0)
    dm.setup()
    weights = dm.class_weights
    assert isinstance(weights, torch.Tensor)
    assert weights.dtype == torch.float32 and weights.numel() == 1


@pytest.mark.unit
def test_class_weights_stay_none_when_the_bundle_has_none():
    dm = BundleDataModule(_array_bundle(), num_workers=0)
    dm.setup()
    assert dm.class_weights is None


# ── Loaders ───────────────────────────────────────────────
@pytest.mark.unit
def test_train_loader_shuffles_and_drops_a_ragged_last_batch():
    dm = BundleDataModule(_array_bundle(n=64), batch_size=10, num_workers=0)
    dm.setup()
    loader = dm.train_dataloader()
    assert loader.drop_last is True
    assert not isinstance(loader.sampler, WeightedRandomSampler)


@pytest.mark.unit
def test_drop_last_is_off_when_there_is_only_one_batch():
    """Dropping the only batch would train on nothing."""
    dm = BundleDataModule(_array_bundle(n=8), batch_size=32, num_workers=0)
    dm.setup()
    assert dm.train_dataloader().drop_last is False


@pytest.mark.unit
def test_eval_loaders_do_not_shuffle_or_drop():
    """Evaluation order must match the split order or predictions.csv rows stop
    lining up with their labels."""
    dm = BundleDataModule(_array_bundle(n=64), batch_size=10, num_workers=0)
    dm.setup()
    for loader in (dm.val_dataloader(), dm.test_dataloader(), dm.split_dataloader("train")):
        assert loader.drop_last is False
        batches = [batch[0] for batch in loader]
        assert sum(len(b) for b in batches) == 64


@pytest.mark.unit
def test_sample_weights_in_meta_produce_a_weighted_sampler():
    """Image runs correct imbalance by sampling. The source hands over plain
    weights; building the torch sampler is this adapter's job."""
    bundle = _array_bundle(n=32, meta={"sample_weights": np.ones(32)})
    dm = BundleDataModule(bundle, batch_size=8, num_workers=0)
    dm.setup()
    loader = dm.train_dataloader()
    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert loader.drop_last is False  # sampler and shuffle/drop_last are exclusive


@pytest.mark.unit
def test_labels_keep_their_dtype_through_the_tensor_dataset():
    """int64 labels feed CrossEntropyLoss; float32 targets feed MSELoss. Coercing
    either one breaks the loss rather than the loader, several frames away."""
    dm = BundleDataModule(_array_bundle(), batch_size=4, num_workers=0)
    dm.setup()
    _, y = next(iter(dm.val_dataloader()))
    assert y.dtype == torch.int64


# ── v1 datamodule surface ─────────────────────────────────
@pytest.mark.unit
def test_v1_datamodules_are_thin_subclasses_of_the_one_adapter():
    """The duplicated train/val/test_dataloader bodies are gone; these two only
    know which bundle to build."""
    assert issubclass(TabularDataModule, BundleDataModule)
    assert issubclass(ImageDataModule, BundleDataModule)
    for name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        assert name not in vars(TabularDataModule)
        assert name not in vars(ImageDataModule)


@pytest.mark.unit
def test_tabular_datamodule_wraps_the_bundle_source(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")
    dm = TabularDataModule(cfg)
    dm.setup()
    assert dm.bundle.input_dim == dm.input_dim == 6
    assert dm.batch_size == cfg.fit.batch_size


@pytest.mark.unit
def test_from_bundle_uses_the_configs_loader_settings(tabular_csv, make_config):
    cfg = make_config(tabular_csv, "multiclass")
    dm = BundleDataModule.from_bundle(build_bundle(cfg), cfg)
    dm.setup()
    assert dm.batch_size == cfg.fit.batch_size
    assert dm.test_dataloader().batch_size == cfg.fit.batch_size
