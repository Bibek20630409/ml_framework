"""
core/inference.py
─────────────────
Production inference. Loads a self-contained **artifact bundle** produced by the
training pipeline:

    <artifact_dir>/
      model.ckpt        ← Lightning checkpoint
      scaler.pkl        ← StandardScaler (tabular only)
      metadata.json     ← task, dims, class_names, feature order, full config

Usage:
    inf = Inferencer.from_artifacts("outputs")
    preds = inf.predict(X_new)               # tabular: raw numpy (auto-scaled)
    probs = inf.predict_proba(X_new)         # classification only

The bundle is self-describing — no live training config object required.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn

from .lit_model import BaseModel
from .registry import get_model_class

if TYPE_CHECKING:  # `core` is imported *by* the config layer's plugin resolution,
    # so a module-scope import here is a cycle. P3 removes the dependency outright.
    from ..config import ExperimentConfig

log = logging.getLogger(__name__)


class Inferencer:
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        scaler=None,
        feature_cols: list[str] | None = None,
        reference_stats: dict | None = None,
    ):
        self.model = model
        self.config = config
        self.task = config.task
        self.scaler = scaler
        self.feature_cols = feature_cols or []
        self.reference_stats = reference_stats  # drift baseline (may be None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    # ── Loaders ───────────────────────────────────────────
    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path) -> Inferencer:
        import joblib

        # Ensure the built-in models/datamodules are registered — a standalone
        # serving process may not have imported them yet.
        import ml_framework.plugins  # noqa: F401

        from ..config import ExperimentConfig

        art = Path(artifact_dir)
        meta_path = art / "metadata.json"
        ckpt_path = art / "model.ckpt"
        if not meta_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(f"Missing model.ckpt / metadata.json in {art}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        config = ExperimentConfig.model_validate(meta["config"])
        model_cls = cast("type[BaseModel]", get_model_class(config.model.name))
        model = model_cls.load_from_checkpoint(
            str(ckpt_path),
            input_dim=meta["input_dim"],
            output_dim=meta["output_dim"],
            task=config.task,
            params=config.model.params,
            optim=config.fit.params,
            class_weights=None,
            map_location="cpu",
        )
        scaler = None
        scaler_path = art / "scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            log.info("scaler loaded ← %s", scaler_path)

        reference = None
        ref_path = art / "reference_stats.json"
        if ref_path.exists():
            reference = json.loads(ref_path.read_text(encoding="utf-8"))
        return cls(
            model,
            config,
            scaler=scaler,
            feature_cols=meta.get("feature_cols", []),
            reference_stats=reference,
        )

    @classmethod
    def from_registry(
        cls,
        name: str,
        stage_or_version: str = "Production",
        tracking_uri: str | None = None,
    ) -> Inferencer:
        """Load a model from the MLflow Model Registry.

        Downloads the registered bundle (model.ckpt · scaler.pkl · metadata.json)
        to a local dir and reuses ``from_artifacts`` — so registry-loaded and
        file-loaded models behave identically. Requires the ``[mlops]`` extra.
        """
        from ..tracking import download_bundle

        local_dir = download_bundle(name, stage_or_version, tracking_uri)
        return cls.from_artifacts(local_dir)

    # ── Internals ─────────────────────────────────────────
    def _tabular_tensor(self, x: np.ndarray) -> torch.Tensor:
        x = np.asarray(x, dtype="float32")
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return torch.tensor(x, dtype=torch.float32).to(self.device)

    def _to_input(self, x) -> torch.Tensor:
        # Images: caller passes an already-transformed tensor. Tabular: numpy.
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return self._tabular_tensor(x)

    # ── API ───────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, x) -> np.ndarray:
        out = self.model(self._to_input(x))
        if self.task == "binary":
            return (torch.sigmoid(out.squeeze(1)) > 0.5).long().cpu().numpy()
        if self.task == "multiclass":
            return out.argmax(dim=1).cpu().numpy()
        return out.squeeze(1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, x) -> np.ndarray:
        if self.task == "regression":
            raise ValueError("predict_proba is not available for regression")
        out = self.model(self._to_input(x))
        if self.task == "binary":
            p = torch.sigmoid(out.squeeze(1))
            return torch.stack([1 - p, p], dim=1).cpu().numpy()
        return torch.softmax(out, dim=1).cpu().numpy()

    @torch.no_grad()
    def predict_with_confidence(self, x) -> tuple[np.ndarray, np.ndarray]:
        probs = self.predict_proba(x)
        return probs.argmax(axis=1), probs.max(axis=1)
