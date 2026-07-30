"""
utils/seed.py
─────────────
Reproducibility helpers and platform-aware worker defaults.
"""

from __future__ import annotations

import os
import platform


def resolve_num_workers(requested: int) -> int:
    """Resolve DataLoader worker count.

    ``requested == -1`` → auto: 0 on Windows (spawn-recursion / overhead issues
    with the multiprocessing start method), 4 elsewhere. Any explicit value is
    returned as-is (clamped to >= 0).
    """
    if requested is not None and requested >= 0:
        return requested
    if platform.system() == "Windows":
        return 0
    return min(4, (os.cpu_count() or 2))


def seed_everything(seed: int, workers: bool = True) -> int:
    """Seed python, numpy, and torch via Lightning's helper when available."""
    try:
        import pytorch_lightning as pl

        return pl.seed_everything(seed, workers=workers)
    except Exception:  # pragma: no cover - fallback path
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except Exception:
            pass
        return seed
