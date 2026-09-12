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

**It also owns the last two stages of the staged read pipeline.** ``h2d`` is
Lightning's copy, but where it happens relative to the *transform* is this class's
decision: :attr:`BundleDataModule.defers_transform` moves the preprocessor's
``transform_batch`` from the collate to :meth:`on_after_batch_transfer` whenever
there is a CUDA device and the preprocessor has declared the work device-agnostic.
That is the difference between a worker CPU running an FFT and the GPU running it,
and — for video — between the bus carrying float32 and carrying uint8.

It also keeps the derived-attribute surface v1 exposed after ``setup()``
(``input_dim``, ``output_dim``, ``class_weights``, ``feature_cols``,
``reference_stats``), because the training pipeline, the LR finder and the HPO
driver all read those. They are properties over the bundle now rather than fields
written during setup, so they cannot disagree with the data.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ..core.registry import register_datamodule
from ..utils.seed import resolve_num_workers
from .streaming.state import LoaderState
from .types import DataBundle

log = logging.getLogger(__name__)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    """``from_numpy`` when the buffer is already what torch wants, a copy otherwise.

    Tensor construction is attaching a dtype, a shape and strides to a pointer. It
    only has to *copy* when the pointer is not already pointing at what torch
    needs — and the previous unconditional ``torch.tensor(np.asarray(x,
    dtype="float32"))`` copied every time, twice when ``x`` was float64.

    The three conditions are exactly:

    ``float32``        torch's compute dtype here; anything else must be cast.
    C-contiguous       a Fortran-order or strided view has the wrong layout.
    **writeable**      ``from_numpy`` on a read-only array (an ``np.memmap``
                       view, which is exactly what a token shard hands back)
                       produces a tensor whose in-place operations are undefined
                       behaviour. torch *warns* rather than raising, so relying
                       on it to complain is not an option.

    Note what is deliberately absent: any handling of ``uint16``. torch has no
    usable uint16 arithmetic, so a token corpus must be cast — and the cast
    belongs in the ``collate_fn``, per batch over a few MB, not here, over the
    whole corpus with a 4x blowup. At 15T tokens that is the difference between
    30 TB on disk and 60 TB, against a memcpy-bound operation per step.
    """
    if array.dtype == np.float32 and array.flags["C_CONTIGUOUS"] and array.flags["WRITEABLE"]:
        return torch.from_numpy(array)
    return torch.tensor(np.ascontiguousarray(array, dtype="float32"))


def _cuda_selected(accelerator: str) -> bool:
    """Whether this run will actually reach a CUDA device.

    ``"auto"`` has to ask torch, because that is precisely what Lightning will do.
    Guessing wrong in the permissive direction costs page-locked host RAM for a
    copy that never happens.
    """
    if accelerator in ("cpu", "mps"):
        return False
    if accelerator in ("gpu", "cuda"):
        return True
    return bool(torch.cuda.is_available())


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
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        accelerator: str = "auto",
        shuffle: str = "block",
        seed: int = 42,
        resume_state: bool = True,
        device_transform: bool = True,
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
        # `pin_memory` defaults to False here but True on `RuntimeConfig`. Not a
        # contradiction: the config default is "pin when it would help", and the
        # `pin_memory` property below decides whether it would. A direct caller
        # constructing this class outside a run has no accelerator context, so the
        # safe default for them is off.
        self._pin_memory_setting = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self._accelerator = accelerator
        self._datasets: dict[str, Any] = {}
        # Staged-corpus ordering and the mid-epoch resume position. Inert for
        # every non-staged source, whose train split has no shard index.
        self._shuffle = shuffle
        self._seed = seed
        self._resume_state = resume_state
        self._sampler: Any = None
        self._state = LoaderState(seed=seed, shuffle=shuffle)
        self._resume_skip = 0
        # Permission, not a decision: `defers_transform` decides. Off makes the
        # pipeline behave exactly as it did before the tail was split, which is
        # what a caller driving the loaders itself needs — see `set_device_transform`.
        self._device_transform = device_transform

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
            # Already a lazily-decoding corpus -- including a device-landing one,
            # whose items are not numpy at all. Returned untouched: there is no
            # tensor to construct here, and Lightning's `transfer_batch_to_device`
            # is a no-op for something already on the right device.
            return split.x
        if split.x is None:
            raise ValueError(f"split '{name}' has no features to build a dataset from")
        x = _to_tensor(np.asarray(split.x))
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
    def lands_in(self) -> str:
        """Where this bundle's decoder left its samples: ``"host"`` or ``"device"``.

        Read from ``bundle.meta`` — the documented home for source-specific extras
        the ``DataBundle`` contract should not name. Defaults to ``"host"``, which
        is what every non-staged source produces.
        """
        if not self.is_ready:
            return "host"
        return str(self.bundle.meta.get("lands_in", "host"))

    @property
    def _workers(self) -> int:
        requested = resolve_num_workers(self.num_workers_setting)
        if self.lands_in == "device" and requested != 0:
            # Enforced, not documented. A CUDA tensor cannot be returned from a
            # DataLoader worker across a fork/spawn boundary without CUDA IPC,
            # and the failure mode is a hang rather than an error.
            log.warning(
                "decoder lands in device memory; forcing num_workers=0 (a CUDA tensor "
                "cannot be returned from a DataLoader worker). Decode is already "
                "off-CPU, so worker parallelism buys nothing here."
            )
            return 0
        return requested

    @property
    def pin_memory(self) -> bool:
        """Page-locked staging buffers, so H2D is an async DMA.

        False in two cases, both of which would otherwise cost something for
        nothing:

        * the decoder already landed the sample in **device** memory — pinning a
          buffer that is already there is a no-op at best and a device-to-host
          round trip at worst;
        * there is no CUDA device to copy *to*, where the only effect is
          page-locking host RAM for a transfer that never happens.
        """
        if not self._pin_memory_setting:
            return False
        if self.lands_in == "device":
            log.debug("not pinning: the decoder already landed this batch on the device")
            return False
        if not _cuda_selected(self._accelerator):
            log.debug("not pinning: no CUDA device is selected, so there is no H2D copy")
            return False
        return True

    # ── the transform seam: before or after H2D ──
    @property
    def defers_transform(self) -> bool:
        """Whether the transform stage runs **after** the H2D copy.

        Four conditions, and each one is a case where deferring would be wrong
        rather than merely unhelpful:

        * ``device_transform`` is off — a caller that drives the loaders itself has
          no ``on_after_batch_transfer`` hook, so the transform must stay in the
          collate or the model is handed a raw waveform;
        * an explicit ``collate_fn`` was supplied — that callable did some unknown
          part of the tail, and adding a second transform on top of it would apply
          the front-end twice;
        * the preprocessor does not declare ``gpu_transform`` — it has not claimed
          its transform is device-agnostic, and assuming it is would be exactly the
          kind of unstated assumption this pipeline exists to remove;
        * the decoder landed the sample in **device** memory — there is no host
          tail at all, so there is nothing to move.

        Everything else reduces to "is there a device to defer *to*", which is the
        same question :func:`_cuda_selected` answers for pinning.
        """
        if not self._device_transform or self._collate_fn is not None:
            return False
        if "gpu_transform" not in self._preprocessor_stages:
            return False
        if self.lands_in == "device":
            return False
        return _cuda_selected(self._accelerator)

    @property
    def _preprocessor_stages(self) -> frozenset[str]:
        """The tail stages the bundle's preprocessor declares, or none.

        Defaults to empty for a preprocessor that predates the split (or a
        third-party one), which is what keeps such a preprocessor on the single
        opaque ``collate_fn`` path it was written for.
        """
        if not self.is_ready:
            return frozenset()
        return frozenset(getattr(self.bundle.preprocessor, "stages", frozenset()))

    def set_device_transform(self, enabled: bool) -> None:
        """Allow or forbid deferring the transform past the H2D copy.

        For a caller that iterates ``train_dataloader()`` itself instead of handing
        the datamodule to a ``Trainer``: Lightning is what calls
        :meth:`on_after_batch_transfer`, so without it a deferred transform would
        simply never run. ``mlf lr`` is the one such caller in this repo.
        """
        self._device_transform = enabled

    @property
    def collate(self) -> Callable[[Any], Any] | None:
        """The batching callable this run's loaders get.

        Three sources, in precedence order: an explicit ``collate_fn`` (a caller
        who knows better), the preprocessor's split tail with the transform kept or
        dropped, and finally the preprocessor's own opaque ``collate_fn``.
        """
        if self._collate_fn is not None:
            return self._collate_fn
        preprocessor = self.preprocessor
        if preprocessor is None:
            return None
        if "construct" in self._preprocessor_stages:
            return (
                preprocessor._collate_deferred
                if self.defers_transform
                else preprocessor._collate_full
            )
        return getattr(preprocessor, "collate_fn", None)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int = 0) -> Any:
        """**gpu_transform.** Lightning's hook for "the batch is on the device now".

        The one place the deferred transform runs, and the reason the split exists:
        the mel matmul and the video permute are arithmetic, and arithmetic belongs
        where the accelerator is rather than in a DataLoader worker that should be
        doing IO.

        A no-op whenever the transform already ran in the collate, so a CPU run and
        a run whose preprocessor never split its tail both pass straight through.
        """
        preprocessor = self.preprocessor
        if not self.defers_transform or preprocessor is None:
            # The `is None` half is unreachable -- `defers_transform` reads the
            # preprocessor's own `stages` and a missing one declares nothing -- but
            # stating it keeps the narrowing local instead of spread across two
            # properties.
            return batch
        transform = preprocessor.transform_batch
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
            return transform(x), y
        return transform(batch)

    @property
    def sampler_weights(self) -> np.ndarray | None:
        if not self.is_ready:
            return None
        weights = self.bundle.meta.get("sample_weights")
        return None if weights is None else np.asarray(weights)

    def _train_sampler(self) -> Any | None:
        """The training sampler: shard-aware, weighted, or neither.

        Exactly one can win, and the order is not arbitrary. A staged corpus reads
        through a shard index, and :class:`ShardShuffleSampler` is what keeps every
        read inside one open shard — the entire reason the corpus is sharded. A
        ``WeightedRandomSampler`` over that storage is one seek per sample and
        would throw the locality away, which is why the staged sources correct
        imbalance with loss weights instead and never emit ``sample_weights``.

        So the two never actually collide in practice; this order makes that
        explicit rather than leaving it to whichever check ran first.
        """
        staged = self._shard_sampler()
        if staged is not None:
            return staged
        weights = self.sampler_weights
        if weights is None:
            return None
        return WeightedRandomSampler(weights.tolist(), len(weights))

    def _shard_sampler(self) -> Any | None:
        """A :class:`ShardShuffleSampler` when the train split is a staged corpus.

        Built once and cached, because it carries the epoch and any resume offset
        — rebuilding it per ``train_dataloader()`` call would silently reset both.
        """
        if self._sampler is not None:
            return self._sampler
        dataset = self._datasets.get("train")
        index = getattr(dataset, "index", None)
        if index is None:
            return None

        from .streaming.sampler import ShardShuffleSampler

        self._sampler = ShardShuffleSampler(
            index,
            seed=self._seed,
            shuffle=self._shuffle,  # type: ignore[arg-type]
            epoch=self._state.epoch,
            skip=self._resume_skip,
        )
        # Consumed once: the resumed epoch is short, every epoch after it is whole.
        self._resume_skip = 0
        return self._sampler

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle for a new epoch, and advance the recorded position.

        Called by ``StagedDataCallback``, because Lightning's
        ``DistributedSamplerWrapper`` calls ``set_epoch`` on *itself* rather than
        on the sampler it wraps — so a wrapped sampler would otherwise serve the
        same order every epoch and the shuffle would be decorative.
        """
        sampler = self._shard_sampler()
        if sampler is not None:
            sampler.set_epoch(epoch)
        self._state = replace(self._state, epoch=epoch, samples_seen=0)

    # ── resume ──
    def state_dict(self) -> dict[str, Any]:
        """The loader's position, which Lightning stores in the checkpoint.

        Lightning already checkpoints the model, the optimizer and the epoch. What
        it cannot checkpoint is *which samples this epoch had already served*, and
        without that a mid-epoch resume silently replays them — an extra partial
        pass over a subset of the corpus, visible only as an unexplained kink in
        the loss curve.
        """
        if not self._resume_state:
            return {}
        return replace(self._state, index_digest=self._index_digest).to_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the position, or refuse if the corpus changed underneath it."""
        if not self._resume_state or not state:
            return
        from .streaming.state import LoaderState

        restored = LoaderState.from_dict(state)
        # Raises when the index digest no longer matches: "sample 41,000" would
        # name different bytes, so resuming would replay a different corpus while
        # reporting it as the same run.
        self._resume_skip = restored.resume_skip(index_digest=self._index_digest)
        self._state = restored

    @property
    def _index_digest(self) -> str:
        if not self.is_ready:
            return ""
        return str(self.bundle.meta.get("index_digest", ""))

    def _loader(self, name: str, *, shuffle: bool = False, sampler: Any = None) -> DataLoader:
        dataset = self._datasets[name]
        workers = self._workers
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": workers,
        }
        collate = self.collate
        if collate is not None:
            kwargs["collate_fn"] = collate
        if workers > 0:
            # Guarded rather than passed unconditionally: torch raises for
            # `persistent_workers=True` with `num_workers=0`, and requires
            # `prefetch_factor` to be None there. Both are meaningless without
            # workers to do the prefetching.
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor
        if self.pin_memory:
            kwargs["pin_memory"] = True
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
        """Wrap an already-built bundle using a config's loader settings.

        No ``collate_fn`` is passed: the batching rule is read off the bundle's own
        preprocessor by :attr:`collate`, which is what lets the transform stage be
        placed rather than baked in. Passing it here would pin the whole tail into
        the collate and make ``device_transform`` unreachable.
        """
        return cls(
            bundle,
            batch_size=config.fit.batch_size,
            num_workers=config.runtime.num_workers,
            output_dir=config.runtime.output_dir,
            device_transform=config.runtime.device_transform,
            pin_memory=config.runtime.pin_memory,
            persistent_workers=config.runtime.persistent_workers,
            prefetch_factor=config.runtime.prefetch_factor,
            accelerator=config.runtime.accelerator,
            shuffle=config.data.shards.shuffle,
            seed=config.runtime.seed,
            resume_state=config.data.shards.resume_state,
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
            pin_memory=config.runtime.pin_memory,
            persistent_workers=config.runtime.persistent_workers,
            prefetch_factor=config.runtime.prefetch_factor,
            accelerator=config.runtime.accelerator,
            shuffle=config.data.shards.shuffle,
            seed=config.runtime.seed,
            resume_state=config.data.shards.resume_state,
            device_transform=config.runtime.device_transform,
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
            pin_memory=config.runtime.pin_memory,
            persistent_workers=config.runtime.persistent_workers,
            prefetch_factor=config.runtime.prefetch_factor,
            accelerator=config.runtime.accelerator,
            shuffle=config.data.shards.shuffle,
            seed=config.runtime.seed,
            resume_state=config.data.shards.resume_state,
        )
