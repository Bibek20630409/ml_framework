"""Pipeline stages: train, tune, LR range test.

``train`` is backend-agnostic and imports no ML runtime, so it stays eager.
``run_hpo`` and ``find_lr`` build a ``pl.Trainer`` and a ``torch_lr_finder``
respectively — Lightning-only tools, and torch is optional from P3 — so they
resolve on first use (PEP 562). Without this, ``mlf train --model xgboost`` would
fail at *import* time on an install with no deep-learning stack, which is exactly
the case P3 exists to make work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .train import train

if TYPE_CHECKING:  # so type checkers and IDEs still see the lazy names
    from .hpo import run_hpo
    from .lr_finder import find_lr

_LAZY: dict[str, str] = {"run_hpo": ".hpo", "find_lr": ".lr_finder"}


def __getattr__(name: str) -> Any:
    """Resolve a Lightning-only stage on first use (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = ["train", "run_hpo", "find_lr"]
