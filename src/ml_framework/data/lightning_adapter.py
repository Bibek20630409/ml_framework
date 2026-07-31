"""
data/lightning_adapter.py
─────────────────────────
:class:`BundleDataModule` — the **only** ``LightningDataModule`` in the framework.

v1 had one per data kind, and ``train_dataloader``/``val_dataloader``/
``test_dataloader`` plus the ``_workers`` property were copied verbatim into both
(``lit_data.py:241-262`` and ``:321-341``). That duplication is the reason a third
data kind would have meant a third copy. There is now one implementation,
parameterized by the two things that genuinely differ:

* an optional **sampler** (image runs correct imbalance by sampling), and
* an optional **collate_fn** from the preprocessor (text padding, later).

Datamodules stop being an extension point here; *sources* are the extension
point, and this class is a thin adapter that turns any bundle into loaders.

It also keeps the derived-attribute surface v1 exposed after ``setup()``
(``input_dim``, ``output_dim``, ``class_weights``, ``feature_cols``,
``reference_stats``), because the training pipeline, the LR finder and the HPO
driver all read those. They are properties over the bundle now rather than fields
written during setup, so they cannot disagree with the data.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ..core.registry import register_datamodule
from ..utils.seed import resolve_num_workers
from .types import DataBundle

log = logging.getLogger(__name__)


class BundleDataModule(pl.LightningDataModule):
    """Wraps a :class:`DataBundle` in Lightning's dataloader protocol.

    Construction is **lazy by default**: pass ``bundle_factory`` and the data is
    materialized in ``setup()``, matching v1's contract (and Lightning's, where
    ``setup`` runs per-process under DDP while ``__init__`` runs once). Passing an
    already-built ``bundle`` is the path a backend uses when the orchestrator has
    built it.
    """

    def __init__(
        self,
        bundle: DataBundle | None = None,
        *,
        bundle_factory: Callable[[], DataBundle] | None = None,
        batch_size: int = 32,
        num_workers: int = -1,
        output_dir: str | Path | None = None,
        collate_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__()
        if bundle is None and bundle_factory is None:
            raise ValueError("BundleDataModule needs either a bundle or a bundle_factory")
        self._bundle = bundle
        self._factory = bundle_factory
        self.batch_size = batch_size
        self.num_workers_setting = num_workers
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._collate_fn = collate_fn
        self._datasets: dict[str, Any] = {}

    # ── bundle access ──
    @property
    def bundle(self) -> DataBundle:
        if self._bundle is None:
            raise RuntimeError("call setup() before using the bundle")
        return self._bundle

    @property
    def is_ready(self) -> bool:
        return self._bundle is not None

    # ── Lightning hooks ──
    def prepare_data(self) -> None:  # noqa: D401 - Lightning hook
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup(self, stage: str | None = None) -> None:
        """Materialize the bundle (once) and build the torch datasets.

        Idempotent: Lightning calls ``setup`` for each stage, and the pipeline
        calls it explicitly before reading the derived dims.
        """
        if self._bundle is None:
            assert self._factory is not None  # guarded in __init__
            self._bundle = self._factory()
        if not self._datasets:
            self._datasets = {name: self._build_dataset(name) for name in ("train", "val", "test")}

    def _build_dataset(self, name: str) -> Any:
        split = self.bundle.split(name)
        if split.payload == "dataset":
            return split.x
        if split.x is None:
            raise ValueError(f"split '{name}' has no features to build a dataset from")
        x = torch.tensor(np.asarray(split.x, dtype="float32"), dtype=torch.float32)
        if split.y is None:
            return TensorDataset(x)
        # No explicit dtype: int64 labels stay int64 and float32 regression targets
        # stay float32, which is what the loss functions expect.
        return TensorDataset(x, torch.tensor(np.asarray(split.y)))

    # ── derived attributes (v1 surface) ──
    @property
    def input_dim(self) -> int:
        return self.bundle.input_dim if self.is_ready else 0

    @property
    def output_dim(self) -> int:
        return self.bundle.output_dim if self.is_ready else 0

    @property
    def feature_cols(self) -> list[str]:
        return list(self.bundle.schema.feature_names) if self.is_ready else []

    @property
    def reference_stats(self) -> dict | None:
        return self.bundle.reference_stats if self.is_ready else None

    @property
    def preprocessor(self) -> Any | None:
        return self.bundle.preprocessor if self.is_ready else None

    @property
    def class_weights(self) -> torch.Tensor | None:
        """The bundle's numpy weights as a float32 tensor.

        The conversion happens *here* — the agnostic data layer must not import
        torch, and this adapter is the boundary where torch legitimately enters.
        """
        if not self.is_ready or self.bundle.class_weights is None:
            return None
        return torch.tensor(np.asarray(self.bundle.class_weights), dtype=torch.float32)

    # ── loaders ──
    @property
    def _workers(self) -> int:
        return resolve_num_workers(self.num_workers_setting)

    @property
    def sampler_weights(self) -> np.ndarray | None:
        if not self.is_ready:
            return None
        weights = self.bundle.meta.get("sample_weights")
        return None if weights is None else np.asarray(weights)

    def _train_sampler(self) -> WeightedRandomSampler | None:
        weights = self.sampler_weights
        if weights is None:
            return None
        return WeightedRandomSampler(weights.tolist(), len(weights))

    def _loader(self, name: str, *, shuffle: bool = False, sampler: Any = None) -> DataLoader:
        dataset = self._datasets[name]
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self._workers,
        }
        if self._collate_fn is not None:
            kwargs["collate_fn"] = self._collate_fn
        if sampler is not None:
            # A sampler and shuffle=True are mutually exclusive in torch.
            kwargs["sampler"] = sampler
        elif shuffle:
            kwargs["shuffle"] = True
            # Dropping a ragged last batch keeps BatchNorm from seeing a
            # single-row batch, which raises. Only safe when there is more than
            # one batch to begin with.
            kwargs["drop_last"] = len(dataset) > self.batch_size
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True, sampler=self._train_sampler())

    def val_dataloader(self) -> DataLoader:
        return self._loader("val")

    def test_dataloader(self) -> DataLoader:
        return self._loader("test")

    def split_dataloader(self, name: str) -> DataLoader:
        """An unshuffled loader for any split, used by ``predict_split``."""
        return self._loader(name)

    @classmethod
    def from_bundle(cls, bundle: DataBundle, config) -> BundleDataModule:
        """Wrap an already-built bundle using a config's loader settings."""
        preprocessor = bundle.preprocessor
        return cls(
            bundle,
            batch_size=config.fit.batch_size,
            num_workers=config.runtime.num_workers,
            output_dir=config.runtime.output_dir,
            collate_fn=getattr(preprocessor, "collate_fn", None),
        )


# ── v1 compatibility ──────────────────────────────────────
# These three names are public API (exported from ``core``) and ``data.kind`` still
# selects between them through the v1 datamodule registry. They are now *thin*: a
# config-bound bundle factory and nothing else, so the dataloader logic above has
# exactly one implementation. ``SOURCES`` is the forward-looking extension point.
FrameworkDataModule = BundleDataModule


@register_datamodule("tabular")
class TabularDataModule(BundleDataModule):
    def __init__(self, config) -> None:
        from .sources.tabular import build_tabular_bundle

        self.config = config
        super().__init__(
            bundle_factory=lambda: build_tabular_bundle(config),
            batch_size=config.fit.batch_size,
            num_workers=config.runtime.num_workers,
            output_dir=config.runtime.output_dir,
        )


@register_datamodule("image")
class ImageDataModule(BundleDataModule):
    def __init__(self, config) -> None:
        from .sources.image import build_image_bundle

        self.config = config
        super().__init__(
            bundle_factory=lambda: build_image_bundle(config),
            batch_size=config.fit.batch_size,
            num_workers=config.runtime.num_workers,
            output_dir=config.runtime.output_dir,
        )
