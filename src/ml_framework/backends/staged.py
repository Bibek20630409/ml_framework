"""
backends/staged.py
──────────────────
The one Lightning callback the staged pipeline needs.

Three jobs, grouped here because they share the same hooks and all three would
otherwise be scattered across the fit loop:

**1. Reshuffle each epoch.** :class:`ShardShuffleSampler` needs ``set_epoch``, and
Lightning's ``DistributedSamplerWrapper`` calls that on *itself* rather than on the
sampler it wraps — so a wrapped sampler silently serves the same order every epoch.
Driving it explicitly from ``on_train_epoch_start`` is what makes the shuffle real
under DDP as well as on one device.

**2. Measure dataloader starvation.** See ``core/stall.py`` for why the number is
worth having; the hooks land in exactly the right places for it.

**3. Aggregate faults.** A DataLoader worker is a separate process and cannot call
the ``RunLogger`` (which may hold an MLflow client), so workers append JSON lines
and this reads them back at epoch end. That indirection is the reason this class
exists rather than a few lines inside ``StagedDataset``.

Everything here is **best-effort by construction**: a callback that measured
throughput must never be the reason a finished training run fails, the same rule
``RunLogger._numeric`` follows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl

# `write_data_reports` lives in `core.stall`, not here: this module imports
# pytorch_lightning at module scope, and `pipeline/train.py` calls the writer on
# EVERY run -- including a GBDT one on an install with no torch at all. Importing
# it from here would put Lightning in the import path of a torch-free run, which
# `tests/integration/test_torch_free_serving.py` exists to prevent.
from ..core.stall import StallProbe, StallProfile, write_data_reports

log = logging.getLogger(__name__)


class StagedDataCallback(pl.Callback):
    """Epoch shuffling, stall measurement and fault aggregation for a staged run."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        run_logger: Any = None,
        warmup: int = 10,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.run_logger = run_logger
        self.probe = StallProbe(warmup=warmup)
        self._profile = StallProfile()
        self._faults = 0

    @property
    def profile(self) -> StallProfile:
        """The last completed epoch's measurement. Never ``None``."""
        return self._profile

    @property
    def faults(self) -> int:
        return self._faults

    # ── epoch boundaries ──
    def on_train_epoch_start(self, trainer: Any, module: Any) -> None:
        epoch = int(getattr(trainer, "current_epoch", 0))
        _set_epoch(trainer, epoch)
        self.probe.reset()

    def on_train_epoch_end(self, trainer: Any, module: Any) -> None:
        self._profile = self.probe.profile()
        self._faults = _count_faults(self.output_dir / "faults")
        self._report(trainer)

    def on_fit_end(self, trainer: Any, module: Any) -> None:
        """Write the two reports.

        The **measurer** writes them, because it is the only thing holding the
        numbers. ``pipeline/train.py`` guarantees they *exist* for backends that
        never run this callback (a GBDT fit has no epoch loop to measure), by
        writing zero-valued defaults only when the file is absent. An absent file
        would be indistinguishable from a bundle written before this phase — the
        same rule ``hpo.json`` follows.
        """
        write_data_reports(self.output_dir, profile=self._profile, faults=self._faults)

    # ── batch boundaries ──
    def on_train_batch_start(self, trainer: Any, module: Any, batch: Any, idx: int) -> None:
        # Fired AFTER the batch is fetched, which is what makes the gap since the
        # previous batch ended exactly dataloader starvation rather than a mix of
        # fetch and setup.
        self.probe.batch_start()

    def on_train_batch_end(
        self, trainer: Any, module: Any, outputs: Any, batch: Any, idx: int
    ) -> None:
        self.probe.batch_end()

    # ── reporting ──
    def _report(self, trainer: Any) -> None:
        profile = self._profile
        if profile.measured:
            log.info("%s", profile.summary())

        metrics: dict[str, float] = {}
        if profile.measured:
            metrics["stall/data_wait_pct"] = profile.data_wait_pct
            gpu = profile.gpu_stall_pct
            if gpu is not None:
                metrics["stall/gpu_stall_pct"] = gpu
        if self._faults:
            metrics["data/faults"] = float(self._faults)
            n = _dataset_length(trainer)
            if n:
                metrics["data/fault_rate"] = self._faults / n

        if metrics and self.run_logger is not None:
            try:
                self.run_logger.log_metrics(metrics, step=int(getattr(trainer, "current_epoch", 0)))
            except Exception as exc:  # noqa: BLE001
                # A tracker must never be the reason a finished run fails -- the
                # same rule `RunLogger._numeric` encodes one level down.
                log.debug("could not log stall metrics: %s", exc)

        if self._faults:
            n = _dataset_length(trainer)
            rate = f" ({self._faults / n:.2%})" if n else ""
            # WARNING, never a raise. A content-dependent abort is rank-divergent,
            # which is the collective hang this whole phase exists to prevent --
            # `mlf materialize` is where a fault ceiling can fail safely.
            log.warning(
                "%d sample(s) could not be decoded this epoch%s and were substituted; " "see %s",
                self._faults,
                rate,
                self.output_dir / "faults",
            )


def _set_epoch(trainer: Any, epoch: int) -> None:
    """Push the epoch down to whatever is actually ordering the samples.

    Three places have to be tried, because Lightning may hand back the sampler,
    a ``DistributedSamplerWrapper`` around it, or nothing at all:

    1. the datamodule, which owns the staged sampler;
    2. the loader's sampler, if it exposes ``set_epoch``;
    3. that sampler's wrapped ``dataset``/``sampler`` attribute.

    Best-effort: a run whose sampler cannot be reached still trains, it just does
    not reshuffle — and it says so.
    """
    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is not None and hasattr(datamodule, "set_epoch"):
        datamodule.set_epoch(epoch)
        return

    for loader in _train_loaders(trainer):
        sampler = getattr(loader, "sampler", None)
        for candidate in (sampler, getattr(sampler, "sampler", None)):
            if candidate is not None and hasattr(candidate, "set_epoch"):
                candidate.set_epoch(epoch)
                return


def _train_loaders(trainer: Any) -> list[Any]:
    loaders = getattr(trainer, "train_dataloader", None)
    if loaders is None:
        return []
    if isinstance(loaders, (list, tuple)):
        return list(loaders)
    return [loaders]


def _dataset_length(trainer: Any) -> int:
    for loader in _train_loaders(trainer):
        dataset = getattr(loader, "dataset", None)
        try:
            return len(dataset)  # type: ignore[arg-type]
        except TypeError:
            continue
    return 0


def _count_faults(directory: Path) -> int:
    """How many faults the workers recorded. Zero when the directory is absent.

    A healthy run writes no file at all, so an absent directory is the common case
    and must not be an error.
    """
    from ..data.streaming.integrity import FaultLog

    try:
        return len(FaultLog.aggregate(directory))
    except Exception as exc:  # noqa: BLE001
        log.debug("could not aggregate faults: %s", exc)
        return 0
