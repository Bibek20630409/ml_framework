"""
pipeline/train.py
─────────────────
Training orchestration. **Zero torch, zero Lightning.**

    build the data → pick the backend → fit → predict → evaluate → write the bundle

Every step above is a call through a protocol, so the same function trains an MLP,
an XGBoost model and a Prophet model. The mechanical check on that claim is that
grepping this file for the Lightning package name finds nothing — v1 constructed
``pl.Trainer`` and its callbacks right here, which is precisely why no non-torch
estimator could enter the pipeline at any price. (The phase gate greps for the
literal module name, so this file must not spell it out, even in prose.)

Produces an artifact bundle v2 in ``config.runtime.output_dir``:

    manifest.json · config.json · model/ · preprocessor/ · metrics.json
    reference_stats.json · report.txt · confusion_matrix.txt · predictions.csv · training.log

That is the whole bundle. v1's three root files (``model.ckpt``, ``scaler.pkl``,
``metadata.json``) are gone: the loader is manifest-driven now, so mirroring them
would be dead weight in every bundle. Bundles already on disk still load — see
``Inferencer._from_v1_bundle``.

All execution is inside a function so the DataLoader worker processes on Windows
have a proper ``__main__`` guard (via the CLI / console-script entry point).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .. import backends as _backends  # noqa: F401  (registers the lightning backend)
from ..backends.base import resolve_budget
from ..config import ExperimentConfig
from ..core.bundle import (
    MODEL_DIR,
    PREPROCESSOR_DIR,
    InputSignature,
    Manifest,
    ModelRef,
    OutputSignature,
    PreprocessorRef,
    RequirementRef,
    Signature,
    write_bundle,
)
from ..core.evaluate import evaluate
from ..core.protocols import RunContext
from ..core.registry import get_backend, validate_combination
from ..core.task import get_task_spec
from ..data import build_bundle
from ..tracking import build_run_logger
from ..utils import seed_everything, setup_logging

log = logging.getLogger(__name__)


def train(config: ExperimentConfig) -> dict:
    out = Path(config.runtime.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    seed_everything(config.runtime.seed, workers=True)
    log.info("task=%s model=%s data=%s", config.task, config.model.name, config.data.kind)

    # Fails here — with a pip command or an explanation of why the combination
    # cannot work — rather than 40 seconds into data loading.
    spec = validate_combination(config.task, config.data.kind, config.model.name)
    backend = get_backend(spec.backend)

    bundle = build_bundle(config)
    log.info("input_dim=%d output_dim=%d", bundle.input_dim, bundle.output_dim)

    run_logger = build_run_logger(
        config.logging.backend,
        output_dir=out,
        experiment=config.logging.mlflow_experiment,
        tracking_uri=config.logging.mlflow_tracking_uri,
        run_name=config.logging.wandb_run,
        artifact_location=config.logging.mlflow_artifact_location,
        project=config.logging.wandb_project,
    )
    run = RunContext(
        output_dir=out,
        seed=config.runtime.seed,
        budget=resolve_budget(config),
        run_logger=run_logger,
        accelerator=config.runtime.accelerator,
        devices=config.runtime.devices,
        precision=config.runtime.precision,
        deterministic=config.runtime.deterministic,
    )

    try:
        result = backend.fit(spec, bundle, config, run=run)
        size = backend.model_size(result.estimator)
        if size:
            log.info("model size: %s", size)

        predictions = backend.predict_split(result.estimator, bundle, "test")
        metrics = evaluate(
            predictions,
            config.task,
            output_dir=out,
            class_names=config.data.class_names,
        )

        artifact = backend.save(result.estimator, out / MODEL_DIR)
        preprocessor_ref = _save_preprocessor(bundle, out)
        manifest = _build_manifest(config, bundle, spec, artifact, preprocessor_ref, metrics, size)
        write_bundle(out, manifest, config=config.model_dump(), metrics=metrics)
        if bundle.reference_stats:
            (out / "reference_stats.json").write_text(
                json.dumps(bundle.reference_stats), encoding="utf-8"
            )

        run_logger.log_params(_tracked_params(config))
        run_logger.log_metrics(metrics)
        run_logger.log_artifacts(out, artifact_path="bundle")
        _register_with_mlflow(config, run_logger, out, metrics)
    except Exception:
        run_logger.finish("FAILED")
        raise
    run_logger.finish("FINISHED")

    log.info("final metrics: %s", metrics)
    log.info("artifacts written to %s (manifest.json, %s/, %s/)", out, MODEL_DIR, PREPROCESSOR_DIR)
    return metrics


# ── Tracking ──────────────────────────────────────────────
def _tracked_params(config: ExperimentConfig) -> dict[str, Any]:
    """The config as flat dotted keys, which is the shape trackers accept."""
    flat: dict[str, Any] = {}

    def walk(node: Any, prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}{key}"
            if isinstance(value, dict):
                walk(value, f"{path}.")
            else:
                flat[path] = value

    walk(config.model_dump(), "")
    return flat


def _register_with_mlflow(
    config: ExperimentConfig, run_logger: Any, out: Path, metrics: dict[str, float]
) -> None:
    if config.logging.backend != "mlflow" or run_logger.run_id is None:
        return
    from ..tracking import log_and_register

    version = log_and_register(config, run_logger, out, metrics)
    if version:
        log.info("registered model '%s' version %s", config.logging.registered_model_name, version)


# ── Bundle assembly ───────────────────────────────────────
def _save_preprocessor(bundle: Any, out: Path) -> dict[str, Any] | None:
    """Write the preprocessor's own directory and return its manifest fragment.

    The orchestrator decides *where*; the preprocessor decides *what*. Nothing
    here knows that a scaler exists.
    """
    if bundle.preprocessor is None:
        return None
    return bundle.preprocessor.save(out / PREPROCESSOR_DIR)


def _build_manifest(
    config: ExperimentConfig,
    bundle: Any,
    spec: Any,
    artifact: Any,
    preprocessor_ref: dict[str, Any] | None,
    metrics: dict[str, float],
    size: dict[str, Any],
) -> Manifest:
    """The serving contract.

    Deliberately does **not** embed the training config: reconstructing an
    ``ExperimentConfig`` at serving time would require a populated plugin registry
    *and* every training extra. ``config.json`` sits beside it as the audit record.
    """
    task_spec = get_task_spec(config.task)
    class_names = config.data.class_names or (
        list(bundle.schema.class_names) if bundle.schema.class_names else None
    )
    return Manifest(
        task=config.task,
        data_kind=config.data.kind,
        model=ModelRef(
            name=spec.name,
            backend=spec.backend,
            artifact=artifact.path,
            format=artifact.format,
            # The architecture params only — post-defaults, so `backend.load()`
            # can rebuild the network from the manifest without config.json.
            params=dict(config.model.params),
            size=size or None,
        ),
        signature=Signature(
            input=InputSignature(
                payload=bundle.payload,
                features=list(bundle.schema.feature_names),
                n_features=bundle.input_dim,
            ),
            output=OutputSignature(
                kind=task_spec.output_kind,
                n_classes=bundle.n_classes,
                class_names=class_names,
            ),
        ),
        preprocessor=PreprocessorRef(**preprocessor_ref) if preprocessor_ref else None,
        requires=[RequirementRef.from_requirement(r) for r in spec.requires],
        metrics=metrics,
    )
