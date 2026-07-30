"""
tracking/mlflow_utils.py
────────────────────────
MLflow integration: experiment tracking + Model Registry.

MLflow is the production system-of-record (self-hostable, open-source, includes a
registry with stage transitions). We keep the training/serving code unchanged and
wrap it here:

  * ``get_mlflow_logger``  → a Lightning ``MLFlowLogger`` (metrics + params).
  * ``log_and_register``   → after training, log the self-contained artifact bundle
                             (model.ckpt · scaler.pkl · metadata.json) to the run and
                             register it as a new version of the configured model.
  * ``download_bundle``    → pull a registered model's bundle back to a local dir so
                             ``Inferencer.from_artifacts`` can load it unchanged.

mlflow is an optional dependency (``pip install -e ".[mlops]"``); everything here
imports it lazily so the base framework works without it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import ExperimentConfig

log = logging.getLogger(__name__)


def resolve_tracking_uri(config: ExperimentConfig) -> str:
    """The tracking URI, defaulting to a local SQLite store.

    The MLflow file store (``file:./mlruns``) is deprecated in 3.x and refuses to
    run by default; SQLite is the lightweight local backend that still supports the
    model registry. Production overrides this with an HTTP tracking server.
    """
    return config.logging.mlflow_tracking_uri or "sqlite:///mlflow.db"


def get_mlflow_logger(config: ExperimentConfig):
    """Build a Lightning ``MLFlowLogger`` bound to the configured tracking URI."""
    from pytorch_lightning.loggers import MLFlowLogger

    return MLFlowLogger(
        experiment_name=config.logging.mlflow_experiment,
        tracking_uri=resolve_tracking_uri(config),
        artifact_location=config.logging.mlflow_artifact_location,
        run_name=config.logging.wandb_run,  # reuse the run-name field
        log_model=False,  # we log the self-contained bundle ourselves below
    )


def log_and_register(
    config: ExperimentConfig,
    logger,
    bundle_dir: str | Path,
    metrics: dict,
) -> str | None:
    """Log the artifact bundle + final metrics to the active run and, if
    ``registered_model_name`` is set, register it. Returns the new version string.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = resolve_tracking_uri(config)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    run_id = logger.run_id
    bundle = Path(bundle_dir)

    # Log only the portable bundle files (not the whole outputs/ dir).
    for name in ("model.ckpt", "scaler.pkl", "metadata.json", "reference_stats.json"):
        f = bundle / name
        if f.exists():
            client.log_artifact(run_id, str(f), artifact_path="bundle")
    for key, value in metrics.items():
        try:
            client.log_metric(run_id, key, float(value))
        except (TypeError, ValueError):
            pass

    model_name = config.logging.registered_model_name
    if not model_name:
        log.info("mlflow: run %s logged (no registered_model_name → not registered)", run_id)
        return None

    # MLflow 3's high-level register_model expects a logged *model* flavor; we
    # register the portable bundle directly as a version via the low-level API so
    # the same artifact bundle loads back through Inferencer.from_artifacts.
    from mlflow.exceptions import MlflowException

    try:
        client.create_registered_model(model_name)
    except MlflowException:
        pass  # already exists
    mv = client.create_model_version(
        name=model_name, source=f"runs:/{run_id}/bundle", run_id=run_id
    )
    log.info("mlflow: registered '%s' version %s", model_name, mv.version)
    return str(mv.version)


def download_bundle(
    name: str,
    stage_or_version: str = "Production",
    tracking_uri: str | None = None,
) -> str:
    """Download a registered model's bundle to a local dir and return its path.

    ``stage_or_version`` may be a stage ("Production"/"Staging") or a version number.
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{name}/{stage_or_version}"
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    log.info("mlflow: downloaded %s → %s", model_uri, local_path)
    return local_path
