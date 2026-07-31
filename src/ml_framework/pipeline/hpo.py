"""
pipeline/hpo.py
───────────────
Optuna hyperparameter search over architecture + optimizer settings. Works with
any registered model. Prints best params to paste back into the YAML config.

Fix vs. original: the pruning callback is imported from ``optuna_integration``
(the current package) with a fallback to the legacy ``optuna.integration`` path.

**This module is on its way out.** Its search space is hardcoded to the MLP's
shape, its objective hardcodes ``val/loss`` from ``trainer.callback_metrics``
(which no non-Lightning backend produces), and it ``print()``s the winner for
manual copy-paste instead of writing it back. ``pipeline/tune.py`` replaces all
of that with plugin-declared spaces, a ``TaskSpec``-driven objective and
automatic write-back; the search-space and pruning machinery it needs already
exists on ``ModelSpec.suggest`` and ``LightningBackend.trial_hooks``. Until then
this keeps working, on v2 config paths.
"""

from __future__ import annotations

import logging

import pytorch_lightning as pl
import torch

from ..config import ExperimentConfig
from ..data import build_datamodule, build_model

log = logging.getLogger(__name__)


def run_hpo(config: ExperimentConfig) -> dict:
    # optuna (and its Lightning pruning callback) are optional deps, imported
    # lazily so the rest of the pipeline works without the `[hpo]` extra.
    import optuna

    try:  # current package
        from optuna_integration import PyTorchLightningPruningCallback
    except ImportError:  # pragma: no cover - legacy fallback
        from optuna.integration import PyTorchLightningPruningCallback

    def objective(trial: optuna.Trial) -> float:
        n_layers = trial.suggest_int("n_layers", 1, 4)
        hidden_dims = [
            trial.suggest_int(f"n_units_l{i}", 32, 512, log=True) for i in range(n_layers)
        ]
        overrides = {
            "model.params.hidden_dims": hidden_dims,
            "model.params.dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "fit.params.lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "fit.params.weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        }
        trial_cfg = config.with_overrides(overrides)

        dm = build_datamodule(trial_cfg)
        dm.setup()
        model = build_model(
            trial_cfg,
            input_dim=dm.input_dim,
            output_dim=dm.output_dim,
            class_weights=getattr(dm, "class_weights", None),
        )
        trainer = pl.Trainer(
            max_epochs=min(50, trial_cfg.fit.budget.max_epochs or 50),
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val/loss")],
            accelerator="auto",
        )
        trainer.fit(model, dm)
        return trainer.callback_metrics.get("val/loss", torch.tensor(float("inf"))).item()

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=config.tune.max_trials, timeout=config.tune.max_seconds)

    log.info("best val/loss=%.4f params=%s", study.best_value, study.best_params)
    print("\nCOPY INTO YOUR YAML CONFIG:")
    units = [v for k, v in sorted(study.best_params.items()) if k.startswith("n_units")]
    print(f"  model.params.hidden_dims: {units}")
    for key, path in (
        ("dropout", "model.params.dropout"),
        ("lr", "fit.params.lr"),
        ("weight_decay", "fit.params.weight_decay"),
    ):
        if key in study.best_params:
            print(f"  {path}: {study.best_params[key]}")
    return study.best_params
