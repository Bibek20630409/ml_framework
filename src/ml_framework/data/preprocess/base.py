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

No torch here: this module is imported by the serving path.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...core.types import FrameworkError

MANIFEST_NAME = "preprocessor.json"


class PreprocessorError(FrameworkError):
    """A preprocessor could not be fitted, saved or restored."""


class BasePreprocessor:
    """Default implementations for the boring half of the protocol.

    Subclasses override :meth:`fit`, :meth:`transform` and the two file hooks
    (:meth:`_write`/:meth:`_read`). ``save``/``load`` are final: they own the
    ``preprocessor.json`` format so every preprocessor round-trips identically.
    """

    # Overridden by subclasses that need a custom batch collation (text padding).
    _collate_fn: Callable[[Sequence[Any]], Any] | None = None

    # ── contract ──
    def fit(self, split: Any, schema: Any) -> None:
        """Fit on the **training split only**. Fitting on val/test is leakage."""
        return None

    def transform(self, x: Any) -> Any:
        return x

    @property
    def collate_fn(self) -> Callable[[Sequence[Any]], Any] | None:
        return self._collate_fn

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
