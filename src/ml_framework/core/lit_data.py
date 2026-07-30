"""
core/lit_data.py
────────────────
**Compatibility surface.** The implementation moved to the ``data`` package in P1;
these names stay importable from here because they are public API with tests
against them.

    old                                     new home
    ─────────────────────────────────────────────────────────────────────────────
    read_table                              data.sources.tabular
    detect_imbalance, apply_smote           data.preprocess.tabular
    compute_class_weights                   data.preprocess.tabular.class_weights
    split_dataset                           data.splitters (RandomSplitter)
    TabularDataModule, ImageDataModule      data.lightning_adapter
    FrameworkDataModule                     data.lightning_adapter.BundleDataModule

Why the file survives at all: v1's two datamodules duplicated their DataLoader
methods verbatim, and the ingestion logic was welded to Lightning. Splitting that
apart is the point of P1 — but the *names* were exported, so deleting the module
would break importers for no benefit. The functions here are one-line adapters,
not a second implementation.

``compute_class_weights`` is the only one that is not a pure re-export: the
agnostic layer returns numpy (a ``DataBundle`` must not contain tensors), while
this signature has always returned a ``torch.Tensor`` and has a test asserting
``.numel() == 1`` for binary. It wraps the numpy version rather than duplicating
the arithmetic.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.lightning_adapter import (
    BundleDataModule,
    FrameworkDataModule,
    ImageDataModule,
    TabularDataModule,
)
from ..data.preprocess.tabular import apply_smote, class_weights, detect_imbalance
from ..data.sources.tabular import read_table
from ..data.splitters import split_dataset

__all__ = [
    "read_table",
    "detect_imbalance",
    "apply_smote",
    "compute_class_weights",
    "split_dataset",
    "BundleDataModule",
    "FrameworkDataModule",
    "TabularDataModule",
    "ImageDataModule",
]


def compute_class_weights(y: np.ndarray, task: str) -> torch.Tensor:
    """Balanced weights on the *pre-SMOTE* distribution, as a float32 tensor.

    binary     → 1-element tensor ``[n_neg / n_pos]`` (BCE ``pos_weight``)
    multiclass → per-class balanced weights ``n / (n_classes * count_c)``
    """
    return torch.tensor(class_weights(y, task), dtype=torch.float32)
