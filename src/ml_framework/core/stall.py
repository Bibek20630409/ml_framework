"""
core/stall.py
─────────────
How much of a training run was spent **waiting for data**.

The reference pipeline's closing note is that the failure modes which actually
break production are not table-shaped, and the last one it lists is "measuring GPU
stall percentage so you know which layer to optimize at all". This is that
measurement.

It matters because every optimization in P13 — the block shuffle, the prefetch
depth, pinned memory, a device decoder — is an answer to a question nobody asks
until they know the loader is the bottleneck. Without a number, the usual outcome
is a week spent making the model faster while the GPU idles 60% of the time.

## Why this is not in ``core/profile.py``

Two reasons, and both would be papered over by putting it there:

* that module's docstring promises **nothing here imports torch**, and a GPU
  utilization measurement cannot honour that;
* ``ModelProfile`` is the **bake-off** record — numbers comparable across
  candidates in one run — whereas stall is a property of the *data pipeline*,
  which is the same for every candidate. It would be a field with the wrong owner.

## What the numbers mean

``data_wait_pct``    fraction of wall time the training loop spent blocked in the
                     dataloader. Actionable directly: high means the loader is the
                     bottleneck, and P13's knobs are the levers.
``gpu_stall_pct``    fraction of wall time the device was **not** busy. ``None``
                     when there is no CUDA device — deliberately not ``0.0``,
                     because a measurement that did not happen must never read as
                     a measurement of zero.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Batches to discard before measuring. Covers prefetch-queue fill, worker
# spin-up, cuDNN autotuning and the first CUDA context allocation -- all one-time
# costs that would otherwise be attributed to the loader forever.
STALL_WARMUP_BATCHES = 10


@dataclass(frozen=True, slots=True)
class StallProfile:
    """Where a training epoch's wall time went.

    Frozen and slotted like every other profile here, and reported through
    ``to_dict`` so ``stall.json`` and the ``RunLogger`` read the same numbers.
    """

    n_batches: int = 0
    wall_ms: float = 0.0
    compute_ms: float = 0.0
    data_wait_ms: float = 0.0
    # None on a box with no CUDA. Not 0.0 -- see `gpu_stall_pct`.
    device_busy_ms: float | None = None
    # Set when timing could not be taken; the numbers above are then all zero and
    # must not be read as "no stalling".
    error: str = ""

    @property
    def measured(self) -> bool:
        return not self.error and self.n_batches > 0

    @property
    def data_wait_pct(self) -> float:
        """Percentage of wall time blocked in the dataloader."""
        if not self.measured or self.wall_ms <= 0:
            return 0.0
        return 100.0 * self.data_wait_ms / self.wall_ms

    @property
    def gpu_stall_pct(self) -> float | None:
        """``100 * (1 - device_busy / wall)``, or ``None`` when nothing was measured.

        ``None`` rather than ``0.0`` on purpose: a run on a CPU box has no device
        to stall, and reporting zero would be indistinguishable from a perfectly
        fed GPU — the single most misleading number this module could produce. The
        same rule ``LatencyProfile.measured`` encodes for an unmeasurable candidate.
        """
        if self.device_busy_ms is None or not self.measured or self.wall_ms <= 0:
            return None
        return max(0.0, 100.0 * (1.0 - self.device_busy_ms / self.wall_ms))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_wait_pct"] = round(self.data_wait_pct, 2)
        stall = self.gpu_stall_pct
        payload["gpu_stall_pct"] = None if stall is None else round(stall, 2)
        payload["measured"] = self.measured
        return payload

    def summary(self) -> str:
        """Two lines for ``report.txt``, or one honest line when unmeasured."""
        if not self.measured:
            return f"data pipeline: not measured{f' ({self.error})' if self.error else ''}"
        lines = [
            f"data pipeline: {self.data_wait_pct:.1f}% of wall time waiting for batches "
            f"({self.n_batches} batches)"
        ]
        stall = self.gpu_stall_pct
        lines.append(
            f"GPU stall:     {stall:.1f}%"
            if stall is not None
            else "GPU stall:     not measured (no CUDA device)"
        )
        return "\n".join(lines)


class StallProbe:
    """Accumulates the intervals a training loop spends waiting versus computing.

    Driven by Lightning's batch hooks, which happen to fall in exactly the right
    places:

    * ``on_train_batch_start`` fires **after** the batch has been fetched, so the
      gap since the previous batch ended is precisely dataloader starvation;
    * ``on_train_batch_end`` closes the compute interval.

    ``time.perf_counter`` explicitly, never ``time.time()``: the latter has ~16 ms
    granularity on Windows, which is the same order as the intervals being measured
    here. That is the same rule ``core/profile.py`` follows.
    """

    def __init__(self, *, warmup: int = STALL_WARMUP_BATCHES, sync: bool = True) -> None:
        self.warmup = max(0, warmup)
        self.sync = sync
        self.reset()

    def reset(self) -> None:
        self._seen = 0
        self._n = 0
        self._wall_start: float | None = None
        self._last_end: float | None = None
        self._batch_start: float | None = None
        self._data_wait = 0.0
        self._compute = 0.0
        self._device_busy: float | None = None
        self._events: list[tuple[Any, Any]] = []
        self._error = ""

    # ── hooks ──
    def batch_start(self) -> None:
        now = time.perf_counter()
        self._seen += 1
        if self._seen <= self.warmup:
            # Still warming up: reset the clock each time so the wall window
            # starts at the first *measured* batch rather than at epoch start.
            self._last_end = now
            self._wall_start = now
            return

        if self._wall_start is None:
            # First measured batch with no warmup at all. Without this the window
            # would start at 0 and `wall_ms` would be the raw perf_counter value —
            # an arbitrary number of hours, against which every percentage rounds
            # to zero.
            self._wall_start = now

        if self._last_end is not None:
            # The gap between the previous batch finishing and this one being
            # ready. Lightning fetches before firing this hook, so it is exactly
            # the time the loop was blocked on the loader.
            self._data_wait += now - self._last_end
        self._batch_start = now
        self._start_device_timer()

    def batch_end(self) -> None:
        if self._seen <= self.warmup or self._batch_start is None:
            return
        self._stop_device_timer()
        now = time.perf_counter()
        self._compute += now - self._batch_start
        self._last_end = now
        self._n += 1

    # ── device timing ──
    def _start_device_timer(self) -> None:
        if not self.sync:
            return
        try:
            import torch

            if not torch.cuda.is_available():
                return
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._events.append((start, end))
        except Exception as exc:  # noqa: BLE001 - measurement must never fail a run
            self._error = f"{type(exc).__name__}: {exc}"
            self.sync = False

    def _stop_device_timer(self) -> None:
        if not self.sync or not self._events:
            return
        try:
            self._events[-1][1].record()
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"
            self.sync = False

    def _drain_device_events(self) -> float | None:
        """Total device-busy milliseconds, or ``None`` when nothing was recorded.

        Synchronizes once, at the end, rather than per batch: a
        ``cuda.synchronize()`` inside the hot loop would serialize the very
        overlap this is trying to measure.
        """
        if not self._events:
            return None
        try:
            import torch

            torch.cuda.synchronize()
            return float(sum(start.elapsed_time(end) for start, end in self._events))
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"
            return None

    # ── result ──
    def profile(self) -> StallProfile:
        if self._n == 0:
            return StallProfile(
                error=self._error
                or (
                    f"fewer than {self.warmup + 1} batches ran, so every one was warmup"
                    if self._seen
                    else "no batches ran"
                )
            )
        wall = (self._last_end or 0.0) - (self._wall_start or 0.0)
        return StallProfile(
            n_batches=self._n,
            wall_ms=wall * 1000.0,
            compute_ms=self._compute * 1000.0,
            data_wait_ms=self._data_wait * 1000.0,
            device_busy_ms=self._drain_device_events(),
            error=self._error,
        )


STALL_FILE = "stall.json"
FAULTS_FILE = "faults.json"


def write_data_reports(
    output_dir: str | Path,
    *,
    profile: StallProfile | None = None,
    faults: int = 0,
    only_if_absent: bool = False,
) -> None:
    """Write ``stall.json`` and ``faults.json``.

    **Always written**, including at ``{"faults": 0}`` and ``{"n_batches": 0}``.
    An absent file would be indistinguishable from a bundle produced before this
    phase existed, and "no faults" is a materially different claim from "nobody
    looked" — the same rule ``hpo.json`` established.

    ``only_if_absent`` is how the training pipeline fills in for a backend with no
    epoch loop to measure (a GBDT fit runs no callback), without clobbering the
    real numbers this callback already wrote.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        STALL_FILE: (profile or StallProfile()).to_dict(),
        FAULTS_FILE: {"faults": int(faults), "faults_dir": str(root / "faults")},
    }
    for name, payload in payloads.items():
        path = root / name
        if only_if_absent and path.exists():
            continue
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
