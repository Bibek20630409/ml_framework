"""
pipeline/lr_finder.py
─────────────────────
LR range test (run once before training). Saves a plot and prints a suggested
learning rate to paste into ``fit.params.lr`` in the YAML config.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from ..config import ExperimentConfig
from ..core.lit_model import OptimSettings
from ..data import build_datamodule, build_model

log = logging.getLogger(__name__)

_CRITERION = {
    "binary": nn.BCEWithLogitsLoss,
    "multiclass": nn.CrossEntropyLoss,
    "regression": nn.MSELoss,
}


def _check_supported(config: ExperimentConfig) -> None:
    """Refuse politely instead of crashing inside ``torch_lr_finder``.

    The consumer of ``Capabilities.supports_lr_range_test``. An LR range test
    sweeps the learning rate across mini-batches and watches the loss — there is
    no such thing for a model with no gradient descent in it, so ``mlf lr`` on an
    XGBoost config used to fail somewhere deep inside a library that had been
    handed something it could not use.
    """
    from ..core.registry import MODELS
    from ..core.types import UnsupportedCapability

    spec = MODELS.get_spec(config.model.name)
    if not spec.capabilities.supports_lr_range_test:
        raise UnsupportedCapability(
            f"model '{spec.name}' has no learning rate to range-test "
            f"(backend '{spec.backend}' does not train by gradient descent). "
            f"`mlf lr` applies to the lightning backend; use `mlf tune` instead."
        )


def find_lr(config: ExperimentConfig) -> float:
    from torch_lr_finder import LRFinder  # optional dep

    _check_supported(config)

    out = Path(config.runtime.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dm = build_datamodule(config)
    dm.setup()
    model = build_model(config, input_dim=dm.input_dim, output_dim=dm.output_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optim = OptimSettings.from_mapping(config.fit.params)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=optim.weight_decay)
    criterion = _CRITERION[config.task]()

    finder = LRFinder(model, optimizer, criterion, device=device)
    finder.range_test(dm.train_dataloader(), end_lr=10, num_iter=100, step_mode="exp")

    fig, _ = finder.plot(suggest_lr=True)
    fig.savefig(out / "lr_finder_plot.png", dpi=150, bbox_inches="tight")

    losses = finder.history["loss"]
    lrs = finder.history["lr"]
    suggested = lrs[losses.index(min(losses))] / 10
    log.info("suggested LR: %.2e (set fit.params.lr in your YAML)", suggested)
    finder.reset()
    return suggested
