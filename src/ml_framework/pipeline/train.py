"""
pipeline/train.py
─────────────────
Final training entry point (hardened successor to the old top-level run.py).

Produces a self-contained artifact bundle in ``config.output_dir``:
  model.ckpt · scaler.pkl · metadata.json · report.txt · predictions.csv
plus training.log and (optionally) a WandB dashboard.

All execution is inside a function so the DataLoader worker processes on Windows
have a proper ``__main__`` guard (via the CLI / console-script entry point).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from ..config import ExperimentConfig
from ..core import evaluate
from ..data import build_datamodule, build_model
from ..utils import seed_everything, setup_logging

log = logging.getLogger(__name__)


def _build_logger(config: ExperimentConfig):
    backend = config.logging.backend
    if backend == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        return WandbLogger(
            project=config.logging.wandb_project,
            name=config.logging.wandb_run,
            log_model=config.logging.log_model,
        )
    if backend == "mlflow":
        from ..tracking import get_mlflow_logger

        return get_mlflow_logger(config)
    if backend == "csv":
        from pytorch_lightning.loggers import CSVLogger

        return CSVLogger(save_dir=config.output_dir, name="metrics")
    return False


def _write_metadata(config: ExperimentConfig, dm, out: Path) -> None:
    meta = {
        "task": config.task,
        "input_dim": int(dm.input_dim),
        "output_dim": int(dm.output_dim),
        "feature_cols": list(getattr(dm, "feature_cols", []) or []),
        "class_names": config.data.class_names,
        "config": config.model_dump(),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def train(config: ExperimentConfig) -> dict:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    seed_everything(config.seed, workers=True)
    log.info("task=%s model=%s data=%s", config.task, config.model.name, config.data.kind)

    dm = build_datamodule(config)
    dm.prepare_data()
    dm.setup()
    log.info("input_dim=%d output_dim=%d", dm.input_dim, dm.output_dim)

    model = build_model(
        config,
        input_dim=dm.input_dim,
        output_dim=dm.output_dim,
        class_weights=dm.class_weights,
    )
    log.info("trainable params: %d", model.count_parameters())

    logger = _build_logger(config)
    callbacks = [
        EarlyStopping(monitor="val/loss", patience=config.train.patience, mode="min"),
        ModelCheckpoint(
            dirpath=str(out),
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
    ]
    # LearningRateMonitor requires an active logger.
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    if config.logging.backend == "wandb" and logger:
        logger.watch(model, log="gradients", log_freq=50)

    trainer = pl.Trainer(
        max_epochs=config.train.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.train.gradient_clip_val,
        deterministic=config.train.deterministic,
        log_every_n_steps=10,
    )

    log.info("training…")
    trainer.fit(model, dm)
    trainer.test(model, dm)

    best_path = getattr(trainer.checkpoint_callback, "best_model_path", "")
    log.info("best checkpoint: %s", best_path)
    if best_path and Path(best_path).exists():
        shutil.copyfile(best_path, out / "model.ckpt")

    best_model = type(model).load_from_checkpoint(
        best_path,
        input_dim=dm.input_dim,
        output_dim=dm.output_dim,
        config=config,
        class_weights=None,
    )
    metrics = evaluate(best_model, dm, config, output_dir=str(out))
    _write_metadata(config, dm, out)
    # Machine-readable metrics (consumed by DVC `metrics` + dashboards).
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Drift baseline for serving-time monitoring.
    if getattr(dm, "reference_stats", None):
        (out / "reference_stats.json").write_text(json.dumps(dm.reference_stats), encoding="utf-8")

    # MLflow: log the bundle to the run and register a model version.
    if config.logging.backend == "mlflow" and logger:
        from ..tracking import log_and_register

        version = log_and_register(config, logger, out, metrics)
        if version:
            log.info(
                "registered model '%s' version %s", config.logging.registered_model_name, version
            )

    log.info("final metrics: %s", metrics)
    log.info("artifacts written to %s (model.ckpt, scaler.pkl, metadata.json)", out)
    return metrics
