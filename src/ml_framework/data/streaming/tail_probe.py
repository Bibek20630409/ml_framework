"""
data/streaming/tail_probe.py
────────────────────────────
The offline probe for the pipeline's **tail**: construct → transform → h2d →
gpu_transform.

``materialize`` walks a corpus and exercises read → demux → decode, which catches
everything a *codec* can get wrong. It catches nothing after that, and the gap is
not academic: a corpus can pass materialization clean and still fail on step 1 of
training, because the four stages that turn decoded buffers into a device tensor
were never run. The failures live in a different place than codec failures do:

    construct       ragged shapes that cannot stack (clips of differing length
                    that no clip-geometry rule was configured for),
                    a dtype torch has no arithmetic for, a device handle handed
                    to a host preprocessor
    transform       a filterbank geometry that does not divide the clip, a
                    channel count the normalization constants do not match
    h2d             a device that is not there, or not enough memory for a batch
    gpu_transform   a transform that is not actually device-agnostic — it works
                    on CPU and raises (or, worse, silently differs) on CUDA

The last is the one that only this probe can see, and the reason the comparison
below exists rather than a bare "did it run": a ``gpu_transform`` declaration is a
*claim* that host and device produce the same answer, and an unchecked claim of
that shape is exactly what this pipeline refuses elsewhere.

**Torch is imported inside the functions.** ``materialize`` must stay usable on an
install with no deep-learning runtime — checking a corpus is not a reason to load
one — so the tail probe is opt-in (``mlf materialize --probe-full``) and pays for
torch only when asked.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ...core.types import TAIL_STAGES, FrameworkError

log = logging.getLogger(__name__)

# How closely the device transform must match the host one. Not exact: cuDNN and a
# CPU kernel differ by accumulation order, and demanding equality would fail every
# real GPU. Loose enough for float32 FFT/matmul drift, far tighter than a wrong
# layout or a mismatched normalization, which are what this is looking for.
#
# **UNVALIDATED ON HARDWARE.** This value is an estimate, not a measurement — it
# was chosen on a CPU-only machine where the "device" path is a no-op copy and the
# observed drift is exactly 0. Saying so in the constant rather than letting it read
# like a measured threshold is the same rule the rest of this pipeline follows: a
# number nobody looked at must not be indistinguishable from one somebody did.
#
# To calibrate, once, on a CUDA box:
#   1. mlf materialize -c <config> --probe-full --force
#   2. read the `host vs device max abs diff` line the report already prints
#   3. set this an order of magnitude above it
# Until then the risk is one-sided and mild: too tight fails a healthy corpus with
# a named, actionable error, rather than passing a skewed one in silence.
DEVICE_AGREEMENT_TOLERANCE = 1e-3

# Samples pulled through the tail. One batch is the whole point — this probes the
# per-batch stages, and running more would just repeat the same code path at the
# cost of decoding the corpus twice.
DEFAULT_PROBE_BATCH = 8


class TailProbeError(FrameworkError):
    """A corpus decodes, but cannot be turned into a batch the model can consume."""


@dataclass
class TailReport:
    """What each tail stage produced, or why it could not run.

    ``skipped`` is a first-class outcome and never an error: there is no ``h2d``
    stage on a CPU box, and saying "skipped: no CUDA device" is a materially
    different claim from saying it passed.
    """

    batch_size: int = 0
    ran: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    shapes: dict[str, str] = field(default_factory=dict)
    failure: tuple[str, str] | None = None
    device_agreement: float | None = None

    @property
    def ok(self) -> bool:
        return self.failure is None

    def render(self) -> str:
        lines = [f"tail probe   {self.batch_size} samples"]
        for stage in TAIL_STAGES:
            if stage in self.shapes:
                lines.append(f"  {stage:<14} {self.shapes[stage]}")
            elif stage in self.skipped:
                lines.append(f"  {stage:<14} skipped: {self.skipped[stage]}")
            elif self.failure is not None and self.failure[0] == stage:
                lines.append(f"  {stage:<14} FAILED: {self.failure[1]}")
        if self.device_agreement is not None:
            lines.append(f"  host vs device max abs diff  {self.device_agreement:.2e}")
        return "\n".join(lines)


def _describe(tensor: Any) -> str:
    """``(B, 1, 64, 401) float32 cpu`` — shape, dtype and device in one line."""
    shape = tuple(getattr(tensor, "shape", ()) or ())
    dtype = str(getattr(tensor, "dtype", "?")).replace("torch.", "")
    device = str(getattr(tensor, "device", "")) or "?"
    return f"{shape} {dtype} {device}"


def probe_tail(
    samples: Sequence[Any],
    preprocessor: Any,
    *,
    device: str | None = None,
) -> TailReport:
    """Run one batch of decoded samples through every stage after decode.

    ``samples`` are what ``StagedDataset.__getitem__`` returns — a ``Decoded``, or
    a ``(Decoded, label)`` pair. ``device`` forces the H2D target; ``None`` uses
    CUDA when there is one and skips the two device stages when there is not.

    Returns a :class:`TailReport` rather than raising, so the caller decides
    whether a tail failure is fatal. It is not a fault ceiling: one batch either
    works or does not, and there is no rate to compare.
    """
    import torch

    report = TailReport(batch_size=len(samples))
    stages = frozenset(getattr(preprocessor, "stages", frozenset()))
    if "construct" not in stages:
        # Not a failure. A preprocessor that never split its tail does the whole
        # thing in one opaque `collate_fn`, and probing "did the collate run" is
        # both weaker and something a training step already tells you.
        for stage in TAIL_STAGES:
            report.skipped[stage] = f"{type(preprocessor).__name__} declares no staged tail"
        return report

    # ── construct ──
    try:
        x, _y = preprocessor.build_tensor(list(samples))
    except Exception as exc:  # noqa: BLE001 - a preprocessor may raise anything
        report.failure = ("construct", f"{type(exc).__name__}: {exc}")
        return report
    report.ran.append("construct")
    report.shapes["construct"] = _describe(x)

    # ── transform, on the host ──
    try:
        host = preprocessor.transform_batch(x)
    except Exception as exc:  # noqa: BLE001
        report.failure = ("transform", f"{type(exc).__name__}: {exc}")
        return report
    report.ran.append("transform")
    report.shapes["transform"] = _describe(host)

    target = device or ("cuda" if torch.cuda.is_available() else "")
    if not target:
        reason = "no CUDA device on this machine"
        report.skipped["h2d"] = reason
        report.skipped["gpu_transform"] = reason
        return report

    # ── h2d ──
    try:
        # The pre-transform tensor, because that is what the transport layer
        # actually copies when the transform is deferred. Copying `host` instead
        # would measure a stage the run does not perform.
        moved = x.to(target)
    except Exception as exc:  # noqa: BLE001
        report.failure = ("h2d", f"{type(exc).__name__}: {exc}")
        return report
    report.ran.append("h2d")
    report.shapes["h2d"] = _describe(moved)

    if "gpu_transform" not in stages:
        report.skipped["gpu_transform"] = (
            f"{type(preprocessor).__name__} does not declare gpu_transform"
        )
        return report

    # ── gpu_transform, and the claim it makes ──
    try:
        on_device = preprocessor.transform_batch(moved)
    except Exception as exc:  # noqa: BLE001
        report.failure = ("gpu_transform", f"{type(exc).__name__}: {exc}")
        return report
    report.ran.append("gpu_transform")
    report.shapes["gpu_transform"] = _describe(on_device)

    if tuple(on_device.shape) != tuple(host.shape):
        report.failure = (
            "gpu_transform",
            f"device transform produced {tuple(on_device.shape)}, host produced "
            f"{tuple(host.shape)} -- `gpu_transform` claims these are the same operation",
        )
        return report

    drift = float((on_device.detach().cpu().float() - host.detach().float()).abs().max())
    report.device_agreement = drift
    if drift > DEVICE_AGREEMENT_TOLERANCE:
        report.failure = (
            "gpu_transform",
            f"device and host transforms differ by up to {drift:.2e} (tolerance "
            f"{DEVICE_AGREEMENT_TOLERANCE:.0e}) -- training on one and serving on the "
            "other would be train/serve skew nothing reports",
        )
    return report
