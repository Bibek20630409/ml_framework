"""Pipeline stages: train, tune, LR range test.

``train`` and ``tune`` are backend-agnostic and import no ML runtime, so they stay
eager. ``find_lr`` builds a ``torch_lr_finder`` — a Lightning-only tool, and torch
is optional from P3 — so it resolves on first use (PEP 562). Without that,
``mlf train --model xgboost`` would fail at *import* time on an install with no
deep-learning stack.

``run_hpo`` was ``pipeline/hpo.py``. It is gone: its search space was hardcoded to
the MLP's shape, its objective read a Lightning-only ``val/loss``, and it printed
the winner for copy-paste instead of applying it. :mod:`ml_framework.pipeline.tune`
does all three properly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .train import train
from .tune import TuneResult, tune

if TYPE_CHECKING:  # so type checkers and IDEs still see the lazy name
    from .lr_finder import find_lr

_LAZY: dict[str, str] = {"find_lr": ".lr_finder"}


def __getattr__(name: str) -> Any:
    """Resolve a Lightning-only stage on first use (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = ["TuneResult", "find_lr", "train", "tune"]
