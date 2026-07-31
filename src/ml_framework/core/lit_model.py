"""
core/lit_model.py
─────────────────
Base LightningModule. Handles train/val/test steps, metrics, loss, and the
optimizer for all three tasks: binary, multiclass, regression.

Subclass this and implement `build_network()` (see plugins/mlp.py). You never
edit the step/optimizer logic here.

**It takes its parameters, not the whole config.** v1's ``__init__`` took an
``ExperimentConfig`` and reached into ``config.model.hidden_dims`` /
``config.optim.lr``, which meant the model knew the shape of the entire
experiment — the same coupling the frozen config was introduced to remove, just
one level down. It also made the model un-buildable from a bundle without
reconstructing a config first. Now:

    BaseModel(input_dim, output_dim, task, params, optim, class_weights)

``params`` is the plugin's own architecture block (validated by
``cls.params_model()``, so ``self.params.hidden_dims`` still reads exactly as
before) and ``optim`` is the fit loop's optimizer settings. Nothing here imports
``config``, which is also what breaks the import cycle that would otherwise exist
between the config validator and the plugin registry.

Key fixes vs. the original framework, all preserved:
  * Binary loss uses a **scalar** ``pos_weight`` (n_neg / n_pos), not a
    2-element vector, which BCEWithLogitsLoss cannot broadcast correctly.
  * Class weights are passed in explicitly (from the data bundle) instead of
    being mutated onto a shared global config.
  * ``save_hyperparameters`` ignores the params/optim/weights objects; they are
    re-supplied at ``load_from_checkpoint`` time from the bundle manifest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pydantic import BaseModel as PydanticModel
from torchmetrics import Accuracy, F1Score, MeanAbsoluteError, MeanSquaredError


@dataclass(frozen=True, slots=True)
class OptimSettings:
    """The optimizer + scheduler knobs, with their defaults in one place.

    These are the authoritative defaults: ``LightningBackend.params_model()``
    builds its Pydantic schema from this dataclass rather than restating the
    numbers, so ``fit.params`` and a directly-constructed model cannot drift.

    ``lr_patience``/``lr_factor`` configure ``ReduceLROnPlateau``. They are listed
    explicitly because dropping them in the v1→v2 move would silently change the
    schedule rather than fail.

    ``optimizer``/``scheduler`` make v1's hardcoded Adam + ReduceLROnPlateau a
    *default* rather than the only option. Both keep their v1 values, so an
    existing config trains exactly as it did.
    """

    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_patience: int = 10
    lr_factor: float = 0.5
    optimizer: str = "adam"
    scheduler: str = "plateau"
    # Cosine/step need to know the horizon; supplied by the backend, which is the
    # only thing that knows how long the loop will run.
    max_epochs: int = 100

    @classmethod
    def from_mapping(cls, params: Any) -> OptimSettings:
        """Build from ``fit.params``, ignoring keys that belong to the loop.

        ``fit.params`` also carries ``gradient_clip_val``, which is a Trainer
        argument rather than an optimizer one; filtering here keeps the backend
        from having to split the dict before handing it over.
        """
        if isinstance(params, cls):
            return params
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(params or {}).items() if k in known})


def _as_weight_tensor(weights: Any) -> torch.Tensor | None:
    """Class weights as a float32 tensor, from numpy or torch.

    The framework-agnostic data layer carries numpy (a ``DataBundle`` must not
    contain tensors); direct callers and the existing tests pass tensors. Both
    arrive here, so this is where the conversion belongs.
    """
    if weights is None:
        return None
    if isinstance(weights, torch.Tensor):
        return weights
    return torch.tensor(np.asarray(weights, dtype="float32"), dtype=torch.float32)


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task: str,
        params: Any = None,
        optim: Any = None,
        class_weights: Any = None,
    ):
        super().__init__()
        # params/optim/class_weights are not pickled into the checkpoint; they are
        # re-supplied at load time from the bundle manifest.
        self.save_hyperparameters(ignore=["params", "optim", "class_weights"])
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = task
        self.params = self._coerce_params(params)
        self.optim = OptimSettings.from_mapping(optim)
        self.class_weights = _as_weight_tensor(class_weights)
        self.network = self.build_network()
        self.criterion = self._build_criterion()
        self._build_metrics()

    def build_network(self) -> nn.Module:
        raise NotImplementedError("Implement build_network() in a registered model")

    # ── Params ────────────────────────────────────────────
    @classmethod
    def params_model(cls) -> type[PydanticModel] | None:
        """The Pydantic model validating this architecture's ``model.params``.

        The same class the plugin's ``ModelSpec`` carries, so a model built from a
        config and one rebuilt from a manifest are validated identically.
        ``None`` means "no declared params", and a plain namespace is used.
        """
        return None

    @classmethod
    def _coerce_params(cls, params: Any) -> Any:
        """Accept a dict (from a manifest) or an already-validated params object.

        ``build_network`` reads attributes either way, which is why the plugin
        bodies changed only in the attribute path (``self.config.model.dropout`` →
        ``self.params.dropout``) and not in their logic.
        """
        model = cls.params_model()
        if model is not None:
            return params if isinstance(params, model) else model.model_validate(dict(params or {}))
        if params is None:
            return SimpleNamespace()
        if isinstance(params, Mapping):
            return SimpleNamespace(**dict(params))
        return params

    # ── Loss ──────────────────────────────────────────────
    def _build_criterion(self) -> nn.Module:
        w = self.class_weights
        if self.task == "binary":
            # pos_weight must be a scalar (weight applied to the positive class).
            pos_weight = None
            if w is not None:
                pos_weight = w.reshape(-1)[-1] if w.numel() > 1 else w.reshape(-1)[0]
                pos_weight = pos_weight.to(dtype=torch.float32)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if self.task == "multiclass":
            return nn.CrossEntropyLoss(weight=w)
        if self.task == "regression":
            return nn.MSELoss()
        raise ValueError(f"Unknown task: {self.task}")

    # ── Metrics ───────────────────────────────────────────
    def _build_metrics(self) -> None:
        if self.task == "binary":
            self.train_acc = Accuracy(task="binary")
            self.val_acc = Accuracy(task="binary")
            self.test_acc = Accuracy(task="binary")
            self.val_f1 = F1Score(task="binary")
            self.test_f1 = F1Score(task="binary")
        elif self.task == "multiclass":
            nc = self.output_dim
            self.train_acc = Accuracy(task="multiclass", num_classes=nc)
            self.val_acc = Accuracy(task="multiclass", num_classes=nc)
            self.test_acc = Accuracy(task="multiclass", num_classes=nc)
            self.val_f1 = F1Score(task="multiclass", num_classes=nc, average="macro")
            self.test_f1 = F1Score(task="multiclass", num_classes=nc, average="macro")
        elif self.task == "regression":
            self.train_mae = MeanAbsoluteError()
            self.val_mae = MeanAbsoluteError()
            self.test_mae = MeanAbsoluteError()
            self.val_rmse = MeanSquaredError(squared=False)
            self.test_rmse = MeanSquaredError(squared=False)

    # ── Forward ───────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    # ── Steps ─────────────────────────────────────────────
    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)

        if self.task == "binary":
            logits = logits.squeeze(1)
            loss = self.criterion(logits, y.float())
            preds = torch.sigmoid(logits)
            getattr(self, f"{stage}_acc")(preds, y.int())
            self.log(f"{stage}/loss", loss, prog_bar=True)
            self.log(f"{stage}/acc", getattr(self, f"{stage}_acc"), prog_bar=True)
            if stage in ("val", "test"):
                getattr(self, f"{stage}_f1")(preds, y.int())
                self.log(f"{stage}/f1", getattr(self, f"{stage}_f1"), prog_bar=True)

        elif self.task == "multiclass":
            loss = self.criterion(logits, y.long())
            preds = logits.argmax(dim=1)
            getattr(self, f"{stage}_acc")(preds, y.long())
            self.log(f"{stage}/loss", loss, prog_bar=True)
            self.log(f"{stage}/acc", getattr(self, f"{stage}_acc"), prog_bar=True)
            if stage in ("val", "test"):
                getattr(self, f"{stage}_f1")(preds, y.long())
                self.log(f"{stage}/f1", getattr(self, f"{stage}_f1"), prog_bar=True)

        elif self.task == "regression":
            logits = logits.squeeze(1)
            loss = self.criterion(logits, y.float())
            getattr(self, f"{stage}_mae")(logits, y.float())
            self.log(f"{stage}/loss", loss, prog_bar=True)
            self.log(f"{stage}/mae", getattr(self, f"{stage}_mae"), prog_bar=True)
            if stage in ("val", "test"):
                getattr(self, f"{stage}_rmse")(logits, y.float())
                self.log(f"{stage}/rmse", getattr(self, f"{stage}_rmse"), prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    # ── Optimizer ─────────────────────────────────────────
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Adam by default, because that is what v1 used and it is a fine default.

        The point of the table is that adding AdamW or SGD is a row, not a branch
        in every model.
        """
        name = self.optim.optimizer.lower()
        lr, decay = self.optim.lr, self.optim.weight_decay
        if name == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=decay)
        if name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=decay)
        if name == "sgd":
            # Momentum is not optional in practice: plain SGD at these learning
            # rates does not converge in a comparable number of epochs.
            return torch.optim.SGD(self.parameters(), lr=lr, weight_decay=decay, momentum=0.9)
        raise ValueError(f"Unknown optimizer '{self.optim.optimizer}'. Use adam | adamw | sgd")

    def _build_scheduler(self, opt: torch.optim.Optimizer) -> dict | None:
        """The LR schedule, in the dict shape Lightning expects, or ``None``.

        ``plateau`` is the v1 behaviour and stays the default. It is the only one
        that needs a ``monitor``, which is why the return shape is a dict rather
        than a bare scheduler.
        """
        name = self.optim.scheduler.lower()
        if name in ("none", "off"):
            return None
        if name == "plateau":
            plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                patience=self.optim.lr_patience,
                factor=self.optim.lr_factor,
            )
            # The only one that needs a `monitor`, which is why the return shape is
            # a dict rather than a bare scheduler.
            return {"scheduler": plateau, "monitor": "val/loss"}
        if name == "cosine":
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, self.optim.max_epochs)
            )
            return {"scheduler": cosine}
        if name == "step":
            # A third of the run per step is the conventional starting point when
            # nothing more specific is known about the schedule.
            step = torch.optim.lr_scheduler.StepLR(
                opt, step_size=max(1, self.optim.max_epochs // 3), gamma=self.optim.lr_factor
            )
            return {"scheduler": step}
        raise ValueError(
            f"Unknown scheduler '{self.optim.scheduler}'. Use plateau | cosine | step | none"
        )

    def configure_optimizers(self):
        opt = self._build_optimizer()
        sched = self._build_scheduler(opt)
        return {"optimizer": opt, "lr_scheduler": sched} if sched else opt

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["BaseModel", "OptimSettings"]
