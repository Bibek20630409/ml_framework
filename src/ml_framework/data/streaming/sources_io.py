"""
data/streaming/sources_io.py
────────────────────────────
Stage 1 for every format: getting a sample's bytes off a device.

Three storage layouts, one protocol. What differs between them is **addressing**,
not decoding — a tar member, a file, and a slice of a memmapped array are three
answers to "where are these bytes", and none of them is a codec's business. That
is the seam :class:`~ml_framework.data.streaming.stages.BlobSource` draws, and it
is why no decoder in this package opens a file itself.

    DirSource      one file per sample. The obvious layout, and the slowest at
                   scale: one open() per sample, and a directory listing that a
                   network filesystem will not enjoy.
    TarSource      WebDataset-style shards. A positioned read inside an already
                   open handle -- no per-sample open, and the reads within a shard
                   are sequential if you visit them in order, which is exactly what
                   the block shuffle arranges.
    MemmapSource   a flat .bin addressed by byte range. Returns a VIEW: the read is
                   a page fault serviced by the kernel into the page cache, and no
                   copy exists anywhere in the path.

## Handles are per-source, and a source is per-worker

Every implementation here caches open handles. A DataLoader worker is a separate
process, so each gets its own source and its own handles — which is correct, and
also why none of this needs a lock. A source must never be shared across the fork:
:meth:`close` exists so a worker can release them, and the file objects are opened
lazily so a source constructed in the parent carries nothing across.
"""

from __future__ import annotations

import tarfile
import threading
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .stages import SampleRef

# What a token shard's bytes mean. A memmap has to be told; there is no header.
_DEFAULT_MEMMAP_DTYPE = "uint16"


class DirSource:
    """One file per sample, addressed by ``ref.key`` relative to a root.

    ``offset``/``nbytes`` are honoured when set, so this also serves a corpus of
    large files carved into samples — but the common case is a whole file, which
    is ``offset=0, nbytes=0``.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def read_range(self, ref: SampleRef) -> bytes:
        path = self.root / (ref.key or ref.shard)
        if not path.is_file():
            raise FileNotFoundError(f"sample {ref.index} ({ref.shard}:{ref.key}) is not at {path}")
        with path.open("rb") as handle:
            if ref.offset:
                handle.seek(ref.offset)
            return handle.read(ref.nbytes) if ref.nbytes else handle.read()

    def open(self, shard: str) -> BinaryIO:
        return (self.root / shard).open("rb")

    def close(self) -> None:
        """No persistent handles: every read opens and closes its own."""


class TarSource:
    """WebDataset-style tar shards, read by byte offset into an open handle.

    **Not** ``tarfile.extractfile``. Once the index records a member's offset, a
    ``seek`` + ``read`` on a raw handle is the whole operation, and it skips
    tarfile's per-call member scan. ``tarfile`` is still used to *build* the index
    (see :func:`scan_tar`), which happens once.

    One handle per shard, kept open. A sequential pass through a shard is then
    genuinely sequential IO, which is the entire reason to shard over object
    storage rather than store loose files.
    """

    def __init__(self, root: str | Path, *, max_open: int = 8) -> None:
        self.root = Path(root)
        self.max_open = max(1, max_open)
        self._handles: dict[str, BinaryIO] = {}
        # Sources are per-worker and never shared across the fork, so this guards
        # only the case of two threads inside one worker -- cheap, and it removes a
        # whole class of "works until num_workers>0" bug reports.
        self._lock = threading.Lock()

    def _handle_for(self, shard: str) -> BinaryIO:
        with self._lock:
            handle = self._handles.get(shard)
            if handle is not None:
                return handle
            if len(self._handles) >= self.max_open:
                # Evict the oldest. A block shuffle visits one shard at a time, so
                # this almost never fires; when it does, FIFO is right because the
                # shard being read now is the one just inserted.
                oldest = next(iter(self._handles))
                self._handles.pop(oldest).close()
            path = self.root / shard
            if not path.is_file():
                raise FileNotFoundError(f"shard '{shard}' is not at {path}")
            handle = path.open("rb")
            self._handles[shard] = handle
            return handle

    def read_range(self, ref: SampleRef) -> bytes:
        if not ref.nbytes:
            raise ValueError(
                f"sample {ref.index} ({ref.shard}:{ref.key}) has no nbytes; a tar member "
                "cannot be read without one. Re-run `mlf materialize`."
            )
        handle = self._handle_for(ref.shard)
        with self._lock:
            handle.seek(ref.offset)
            payload = handle.read(ref.nbytes)
        if len(payload) != ref.nbytes:
            raise EOFError(
                f"sample {ref.index} ({ref.shard}:{ref.key}) wanted {ref.nbytes} bytes at "
                f"offset {ref.offset}, got {len(payload)}. The shard is truncated."
            )
        return payload

    def open(self, shard: str) -> BinaryIO:
        return self._handle_for(shard)

    def close(self) -> None:
        with self._lock:
            for handle in self._handles.values():
                handle.close()
            self._handles.clear()

    def __enter__(self) -> TarSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class MemmapSource:
    """A flat ``.bin`` of fixed-width values, addressed by byte range.

    Returns a **view**, never a copy. A read here is an ``np.memmap`` slice, which
    is a page fault the kernel services by DMA-ing into the page cache — there is
    no decode, no allocation, and no user-space copy in the path at all.

    ``ref.offset``/``ref.nbytes`` are in **bytes**, like every other source, and
    converted to element indices here. Keeping the unit uniform across sources is
    what lets one :class:`SampleRef` describe a tar member and a token span
    without a per-source convention to remember.

    The returned view is **read-only** (a memmap opened ``"r"``), which matters
    downstream: ``torch.from_numpy`` on a read-only array yields a tensor whose
    in-place operations are undefined, so the tensor-construction path checks
    ``WRITEABLE`` before taking its zero-copy branch.
    """

    def __init__(self, path: str | Path, *, dtype: str = _DEFAULT_MEMMAP_DTYPE) -> None:
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self._array: np.ndarray | None = None

    @property
    def array(self) -> np.ndarray:
        """The memmap, opened on first use.

        Lazy so a source built in the parent process carries no mapping across a
        fork — each worker maps the file itself, which is what keeps the page
        cache shared rather than duplicated.
        """
        if self._array is None:
            if not self.path.is_file():
                raise FileNotFoundError(f"token shard is not at {self.path}")
            self._array = np.memmap(self.path, dtype=self.dtype, mode="r")
        return self._array

    def read_range(self, ref: SampleRef) -> np.ndarray:
        itemsize = self.dtype.itemsize
        if ref.offset % itemsize or (ref.nbytes and ref.nbytes % itemsize):
            raise ValueError(
                f"sample {ref.index}: offset {ref.offset} / nbytes {ref.nbytes} do not align "
                f"to the {itemsize}-byte element width of {self.dtype}. The index was probably "
                "written for a different dtype."
            )
        start = ref.offset // itemsize
        stop = start + (ref.nbytes // itemsize) if ref.nbytes else self.array.size
        if stop > self.array.size:
            raise EOFError(
                f"sample {ref.index} runs to element {stop} but the shard holds "
                f"{self.array.size}. The shard is truncated."
            )
        return self.array[start:stop]

    def open(self, shard: str) -> BinaryIO:
        return self.path.open("rb")

    def close(self) -> None:
        # Dropping the reference unmaps it; the pages stay in the page cache, which
        # is the point. `._mmap.close()` would evict them and make the next epoch
        # pay the fault again.
        self._array = None


def scan_tar(path: str | Path) -> list[dict[str, Any]]:
    """Member name, offset and size for every regular file in a tar.

    Used once, by materialization, to turn a shard into index entries. The
    per-sample read path deliberately does not come back here: ``tarfile`` rescans
    to find a member, and paying that per sample is how a tar-backed loader ends up
    slower than loose files rather than faster.
    """
    members: list[dict[str, Any]] = []
    with tarfile.open(path, "r") as archive:
        for member in archive:
            if not member.isfile():
                continue
            members.append(
                {"key": member.name, "offset": member.offset_data, "nbytes": member.size}
            )
    return members


def source_for(
    path: str | Path,
    *,
    kind: str = "auto",
    dtype: str = _DEFAULT_MEMMAP_DTYPE,
) -> Any:
    """The :class:`BlobSource` for a corpus location.

    ``"auto"`` reads the path: a ``.bin`` is a token shard, a directory holding
    ``.tar`` files is sharded, anything else is a directory of samples. Explicit
    beats inference — pass ``kind`` when the layout is unusual.
    """
    root = Path(path)
    if kind == "memmap" or (kind == "auto" and root.suffix == ".bin"):
        return MemmapSource(root, dtype=dtype)
    if kind == "tar" or (kind == "auto" and root.is_dir() and any(root.glob("*.tar"))):
        return TarSource(root)
    if kind in ("auto", "dir"):
        return DirSource(root)
    raise ValueError(f"unknown source kind '{kind}'; expected auto|dir|tar|memmap")
