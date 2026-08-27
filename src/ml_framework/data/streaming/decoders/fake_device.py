"""
data/streaming/decoders/fake_device.py
──────────────────────────────────────
A decoder that claims ``lands_in="device"`` and allocates no device memory.

**Why this exists.** The GPU decode rows (NVDEC via nvcuvid, nvJPEG via DALI) are
structurally different from every host row: they return a buffer that never
touched host memory, so there is **no tensor construction, no pinned staging
buffer and no H2D copy**. That difference has to be expressed somewhere, and
expressing it only in a comment means the code path is never executed.

So the seam is real and this exercises it, on any machine, with no CUDA:

1. ``pin_memory`` is forced off — page-locking a buffer that is already on the
   device is a no-op at best and a device-to-host round trip at worst.
2. ``num_workers`` is forced to 0 — a device tensor cannot be returned from a
   DataLoader worker across a fork/spawn boundary without CUDA IPC, which hangs.
3. Tensor construction is skipped entirely.

A real DALI or TorchCodec decoder registers exactly like this one and needs no
change to the transport layer. That is the claim this module makes testable.

**Its integrity is ``silent``, and that is not a convenience.** It is the truth
about hardware decode, and the worst row in the whole table: NVDEC emits green or
garbage frames with *nothing surfaced* — no exception, no status, no log. It is
undetectable at training time, which is why the spec names an ``oracle`` and why
materialization decodes both and compares.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np

from ..stages import DecodeContext, Decoded, Packet
from .base import BaseDecoder


class DeviceHandle:
    """Stand-in for a buffer living in device memory.

    Deliberately **not** an ``np.ndarray`` and deliberately not convertible to
    one. If this were array-like, every downstream branch that is supposed to skip
    host handling would accidentally work, and the test would prove nothing. It
    carries a shape so the pipeline can reason about the batch without touching
    the data — which is exactly what a real CUDA tensor offers.
    """

    __slots__ = ("shape", "dtype", "device", "_payload")

    def __init__(self, shape: tuple[int, ...], dtype: str, payload: bytes = b"") -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = "cuda:0"
        self._payload = payload

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    def __repr__(self) -> str:
        return f"DeviceHandle(shape={self.shape}, dtype={self.dtype!r}, device={self.device!r})"

    def to_host(self) -> np.ndarray:
        """The explicit D2H copy, for the materialization cross-check only.

        Named as a method rather than offered through the buffer protocol so that
        every copy off the device is a visible call. A real hardware decoder's
        equivalent is expensive; making it look free is how it ends up in a hot
        loop.
        """
        count = int(np.prod(self.shape)) if self.shape else 0
        if len(self._payload) >= count:
            return np.frombuffer(self._payload[:count], dtype=self.dtype).reshape(self.shape)
        return np.zeros(self.shape, dtype=self.dtype)


class FakeDeviceDecoder(BaseDecoder):
    """Decodes to a :class:`DeviceHandle`. Touches no GPU."""

    name: ClassVar[str] = "fake.device"

    def __init__(self, *, shape: tuple[int, ...] = (8, 8, 3)) -> None:
        self.shape = tuple(shape)

    def decode(self, packets: Iterable[Packet], *, ctx: DecodeContext) -> Decoded:
        payload = next(iter(packets)).data
        raw = payload if isinstance(payload, bytes) else bytes(memoryview(payload))
        return Decoded(
            array=DeviceHandle(self.shape, "uint8", raw),
            layout="hwc",
            dtype="uint8",
            lands_in="device",
            rate=None,
            # No `to_chw_float_in` key: there is no host-side collate step to name.
            # A device decoder's transforms stay in the decode graph.
            meta={"device": "cuda:0"},
        )


def build_decoder(**params: Any) -> FakeDeviceDecoder:
    """Factory named by :data:`DecoderSpec.factory`. Unknown keys raise."""
    unknown = set(params) - {"shape"}
    if unknown:
        raise ValueError(
            f"fake.device got unknown decoder_params {sorted(unknown)}; accepts: shape"
        )
    return FakeDeviceDecoder(shape=tuple(params.get("shape", (8, 8, 3))))
