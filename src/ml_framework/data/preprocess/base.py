"""
data/preprocess/base.py
───────────────────────
The preprocessor contract and its bundle round-trip.

**One invariant, and everything else here exists to serve it: nothing outside a
preprocessor reads a preprocessor's files.** v1 broke this in two places — the
inferencer looked for a hardcoded ``scaler.pkl`` and the MLflow logger named that
file in a fixed list — so adding a tokenizer or a per-series scaler would have
meant editing both. Instead each preprocessor writes whatever it likes under
``bundle/preprocessor/`` and describes itself in ``preprocessor.json``:

    {"class": "ml_framework.data.preprocess.tabular:TabularPreprocessor",
     "dir": "preprocessor", "files": ["scaler.pkl"], "params": {...}}

:func:`load_preprocessor` resolves that dotted path and hands the directory back
to the class that wrote it. The loader never learns what is inside.

**The second thing this class owns is the tail of the staged read pipeline.** The
decoder stops at :class:`~ml_framework.data.streaming.stages.Decoded` — a buffer
and a description of it — and everything after that is here: ``build_tensor``
(construct), ``transform_batch`` (transform), and the composition of the two.
Splitting them is not decoration: it is what lets the transport layer run the
transform *after* the H2D copy, on the device, instead of in a DataLoader worker.
A preprocessor opts in by declaring :attr:`BasePreprocessor.stages`; one that does
not is treated as an opaque ``collate_fn``, exactly as before.

No torch here: this module is imported by the serving path.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...core.types import FrameworkError, Stage

MANIFEST_NAME = "preprocessor.json"


class PreprocessorError(FrameworkError):
    """A preprocessor could not be fitted, saved or restored."""


def host_array(sample: Any) -> Any:
    """The numpy buffer inside one decoded sample, refusing a device handle by name.

    ``Decoded.array`` is an ``np.ndarray`` only when ``lands_in == "host"``;
    otherwise it is an opaque device handle that is *deliberately* not
    array-like. ``np.asarray`` on one of those produces a 0-d object array — a
    valid-shaped wrong result, which is the exact failure class this whole
    pipeline exists to make impossible — so it is caught here and named.

    A device-landing decoder has no host tail at all: its transforms belong in the
    decode graph, and the transport layer already skips pinning and forces
    ``num_workers=0`` for one. Reaching this function with such a sample means a
    host preprocessor was wired to a device decoder, which is a configuration
    error and not something to paper over.
    """
    import numpy as np

    array = getattr(sample, "array", sample)
    if isinstance(array, np.ndarray):
        return array
    if getattr(sample, "lands_in", "host") == "device" or hasattr(array, "to_host"):
        raise PreprocessorError(
            f"this preprocessor builds host tensors, but the decoder landed the sample in "
            f"device memory ({type(array).__name__}). A device decoder's transforms belong "
            f"in its decode graph -- there is no host construct/transform stage to run."
        )
    return np.asarray(array)


class BasePreprocessor:
    """Default implementations for the boring half of the protocol.

    Subclasses override :meth:`fit`, :meth:`transform` and the two file hooks
    (:meth:`_write`/:meth:`_read`). ``save``/``load`` are final: they own the
    ``preprocessor.json`` format so every preprocessor round-trips identically.
    """

    # Overridden by subclasses that need a custom batch collation (text padding).
    _collate_fn: Callable[[Sequence[Any]], Any] | None = None

    # Which stages of the pipeline's TAIL this preprocessor implements, out of
    # `construct`, `transform`, `gpu_transform`. Empty by default, which is the v1
    # shape: a preprocessor that only offers `collate_fn` does the whole tail in one
    # opaque call, and the transport layer must not try to split it.
    #
    # Declaring `gpu_transform` is the load-bearing one — it is a claim that
    # `transform_batch` is device-agnostic, so the transport layer may run it AFTER
    # the H2D copy instead of in a worker. Single consumer: `BundleDataModule`.
    stages: frozenset[Stage] = frozenset()

    # ── contract ──
    def fit(self, split: Any, schema: Any) -> None:
        """Fit on the **training split only**. Fitting on val/test is leakage."""
        return None

    def transform(self, x: Any) -> Any:
        return x

    @property
    def collate_fn(self) -> Callable[[Sequence[Any]], Any] | None:
        """The batching callable a DataLoader gets. Always the **whole** tail.

        A caller that asks for `collate_fn` and nothing else gets construct plus
        transform, which is what every consumer outside the training loop wants —
        the serving path, `predict_split`, an LR range test. Only the transport
        layer, which knows there is a device to defer to, reaches past this.
        """
        if "construct" in self.stages:
            return self._collate_full
        return self._collate_fn

    # ── the tail: construct → transform ──
    def build_tensor(self, batch: Sequence[Any]) -> tuple[Any, Any]:
        """**construct.** Decoded samples → ``(x, y)``, on the host, untransformed.

        Attaching a dtype, a shape and strides to a pointer — and nothing else. The
        dtype/layout/scale change is :meth:`transform_batch`'s job, and keeping them
        apart is what lets the second one move to the far side of the H2D copy.

        ``y`` is ``None`` for an unlabelled batch. Only implemented by
        preprocessors that declare ``"construct"`` in :attr:`stages`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the staged tail; it declares "
            f"stages={sorted(self.stages)}"
        )

    def transform_batch(self, x: Any) -> Any:
        """**transform**, or **gpu_transform** — the same work, wherever ``x`` is.

        Must be device-agnostic: called on a CPU tensor when the transform runs in
        the collate, and on a device tensor when the transport layer defers it past
        the H2D copy. A preprocessor that cannot honour that must not declare
        ``"gpu_transform"``.
        """
        return x

    def collate_staged(self, batch: Sequence[Any], *, transform: bool = True) -> Any:
        """``build_tensor`` then, unless deferred, ``transform_batch``.

        The one place the two halves are composed, so "run the transform here" and
        "run it after H2D" cannot drift into two different pipelines.
        """
        x, y = self.build_tensor(batch)
        if transform:
            x = self.transform_batch(x)
        return x if y is None else (x, y)

    # Named methods rather than `partial(self.collate_staged, transform=…)` because
    # a DataLoader's collate is pickled to every worker under spawn, and a bound
    # method of a picklable object is the shape that survives that reliably.
    def _collate_full(self, batch: Sequence[Any]) -> Any:
        return self.collate_staged(batch, transform=True)

    def _collate_deferred(self, batch: Sequence[Any]) -> Any:
        """Construct only. The transform runs after H2D, in the transport layer."""
        return self.collate_staged(batch, transform=False)

    # ── file hooks ──
    def _write(self, dest: Path) -> list[str]:
        """Write state files under ``dest``; return their bundle-relative names."""
        return []

    def _read(self, src: Path, spec: Mapping[str, Any]) -> None:
        """Restore state written by :meth:`_write`."""
        return None

    def params(self) -> dict[str, Any]:
        """Constructor-shaped values needed to rebuild this instance.

        Recorded in the manifest fragment and passed back to ``__init__`` by
        :func:`load_preprocessor`, so a preprocessor's own configuration does not
        have to be re-derived from the training config at serving time.
        """
        return {}

    # ── round-trip ──
    @classmethod
    def class_path(cls) -> str:
        return f"{cls.__module__}:{cls.__qualname__}"

    def save(self, dest: str | Path) -> dict[str, Any]:
        """Write state under ``dest`` and return the manifest fragment.

        The fragment is what lands in ``manifest.preprocessor``; the same dict is
        also written to ``dest/preprocessor.json`` so the directory is
        self-describing even in isolation (an MLflow artifact download, say).
        """
        out = Path(dest)
        out.mkdir(parents=True, exist_ok=True)
        files = self._write(out)
        fragment: dict[str, Any] = {
            "class": self.class_path(),
            "dir": out.name,
            "files": files,
            "params": self.params(),
        }
        (out / MANIFEST_NAME).write_text(
            json.dumps(fragment, indent=2, default=str), encoding="utf-8"
        )
        return fragment

    @classmethod
    def load(cls, src: str | Path, spec: Mapping[str, Any] | None = None) -> Any:
        """Rebuild from a directory written by :meth:`save`.

        ``spec`` is the manifest fragment; when omitted it is read from
        ``preprocessor.json`` in ``src``, which is why a preprocessor directory
        stays loadable without its bundle.
        """
        source = Path(src)
        if spec is None:
            path = source / MANIFEST_NAME
            if not path.exists():
                raise PreprocessorError(f"No {MANIFEST_NAME} in {source}")
            spec = json.loads(path.read_text(encoding="utf-8"))
        obj = cls(**dict(spec.get("params") or {}))
        obj._read(source, spec)
        return obj


def load_preprocessor(src: str | Path, spec: Mapping[str, Any]) -> Any:
    """Resolve ``spec["class"]`` and let that class load itself from ``src``.

    The dotted path is stored rather than a symbolic name so a third-party
    preprocessor round-trips without registering anything.
    """
    dotted = spec.get("class")
    if not dotted:
        raise PreprocessorError(f"preprocessor spec has no 'class' key: {dict(spec)!r}")
    module_name, _, attr = str(dotted).partition(":")
    if not attr:
        module_name, _, attr = str(dotted).rpartition(".")
    try:
        module = importlib.import_module(module_name)
        klass = getattr(module, attr)
    except (ImportError, AttributeError) as exc:
        raise PreprocessorError(
            f"Cannot import preprocessor '{dotted}' recorded in the bundle: {exc}"
        ) from exc
    return klass.load(src, spec)


class IdentityPreprocessor(BasePreprocessor):
    """Fits nothing, transforms nothing.

    Not a placeholder: a source whose data needs no fitted state (an image folder
    whose transforms are fixed) still has to answer ``manifest.preprocessor``, and
    a real no-op is better than a ``None`` that every caller must branch on.
    """
