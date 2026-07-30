"""
core/lit_model.py
─────────────────
Base LightningModule. Handles train/val/test steps, metrics, loss, and the
optimizer for all three tasks: binary, multiclass, regression.

Subclass this and implement `build_network()` (see models/mlp.py). You never
edit the step/optimizer logic here.

Key fixes vs. the original framework:
  * Binary loss uses a **scalar** ``pos_weight`` (n_neg / n_pos), not a
    2-element vector, which BCEWithLogitsLoss cannot broadcast correctly.
  * Class weights are passed in explicitly (from the DataModule) instead of
    being mutated onto a shared global config.
  * ``save_hyperparameters`` ignores the (non-serializable) config object; it is
    re-supplied at ``load_from_checkpoint`` time from the artifact metadata.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, MeanAbsoluteError, MeanSquaredError

from ..config import ExperimentConfig


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: ExperimentConfig,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        # config/class_weights are not pickled into the checkpoint; they are
        # re-supplied at load time from the artifact bundle.
        self.save_hyperparameters(ignore=["config", "class_weights"])
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.task = config.task
        self.class_weights = class_weights
        self.network = self.build_network()
        self.criterion = self._build_criterion()
        self._build_metrics()

    def build_network(self) -> nn.Module:
        raise NotImplementedError("Implement build_network() in a registered model")

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
    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            patience=self.config.optim.lr_patience,
            factor=self.config.optim.lr_factor,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val/loss"},
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
