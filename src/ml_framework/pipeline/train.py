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
from ..core.protocols import FitResult, RunContext
from ..core.registry import get_backend, validate_combination
from ..core.task import get_task_spec
from ..data import build_bundle
from ..tracking import build_run_logger
from ..utils import seed_everything, setup_logging
from .select import select
from .tune import HPO_FILE

log = logging.getLogger(__name__)

CV_FILE = "cv.json"
# Named here rather than imported from the Lightning backend: this module must not
# import a backend, which is the whole point of the orchestration split. The
# backend writes the file; the orchestrator only needs to know what it is called.
LAST_CHECKPOINT = "last.ckpt"


def train(
    config: ExperimentConfig,
    *,
    emit_config: str | Path | None = None,
    resume: bool | str | Path = False,
    baseline: bool = False,
) -> dict:
    """Fit the configured model and write a bundle; return its metrics.

    ``baseline=True`` also scores the trivial predictor (majority class / mean /
    seasonal-naive) and records it under ``baseline_*``, with a WARNING when the
    model fails to beat it. The CLI sets it whenever the *framework* chose the
    model rather than the user: a zero-config score has nothing to be judged
    against, and `test_acc: 0.91` on a dataset that is 91% one class is the most
    common way a pipeline looks successful while having learned nothing.

    Off by default because a user who named the model has their own frame of
    reference, and the flag is about supplying one that is missing.
    """
    out = Path(config.runtime.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    seed_everything(config.runtime.seed, workers=True)
    log.info("task=%s model=%s data=%s", config.task, config.model.name, config.data.kind)

    # Fails here — with a pip command or an explanation of why the combination
    # cannot work — rather than 40 seconds into data loading.
    spec = validate_combination(config.task, config.data.kind, config.model.name)

    # Choose the family, then its hyperparameters, then fit the winner. Both
    # `select` and `tune` return a *config* rather than a fitted model precisely
    # so this stays one code path: whether a bake-off ran, only tuning ran, or
    # neither did, everything below fits a config and writes a bundle.
    #
    # `select` is a pass-through to `tune` when `select.enabled` is false, which
    # is the default — so this call is exactly the old `tune(config)` until
    # somebody asks for a comparison.
    selection = select(config)
    tuning = selection.tuning
    config = selection.config
    if selection.ran:
        # A bake-off can change the model, which changes the spec and the backend.
        spec = validate_combination(config.task, config.data.kind, config.model.name)
        log.info("model selection chose '%s': %s", selection.winner, selection.reason)
    backend = get_backend(spec.backend)

    if tuning.ran or selection.ran:
        seed_everything(config.runtime.seed, workers=True)  # trials advanced the RNG
    if emit_config is not None:
        _emit_config(config, emit_config)

    # k-fold *estimates* performance; it does not produce the shipped model. Run it
    # first, then fall through to the single fit that writes the bundle — so there
    # is one bundle-writing path regardless of how the score was estimated.
    cv_metrics = _cross_validate(config, spec, backend, out) if config.data.split.folds else {}

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
        strategy=config.runtime.strategy,
        deterministic=config.runtime.deterministic,
        resume_from=_resolve_resume(resume, out, spec, backend),
    )

    try:
        if tuning.ran and config.tune.refit == "reuse" and tuning.estimator is not None:
            # `reuse` keeps the winning trial's model. Cheap, but it was trained
            # under the *reduced* trial budget — which is why `best` (refit at full
            # budget) is the default and both are spelled out in the config.
            log.info("refit=reuse — keeping the winning trial's estimator")
            result = FitResult(estimator=tuning.estimator, val_metrics={})
        else:
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
        if baseline:
            trivial = _baseline_metrics(predictions, bundle, config.task)
            metrics.update(trivial)
            # After the merge, so the comparison reads the same dict that lands in
            # metrics.json rather than a copy that could drift from it.
            _compare_to_baseline(metrics, trivial, config.task)
        # The CV estimate sits beside the holdout score rather than replacing it:
        # they answer different questions, and a single held-out number on a small
        # dataset is exactly the one worth distrusting.
        metrics.update(cv_metrics)

        artifact = backend.save(result.estimator, out / MODEL_DIR)
        preprocessor_ref = _save_preprocessor(bundle, out)
        manifest = _build_manifest(
            config, bundle, spec, artifact, preprocessor_ref, metrics, size, tuning, selection
        )
        # `config` here is the *tuned* config, so config.json is the record of what
        # actually trained — which is what closes v1's copy-paste gap.
        write_bundle(out, manifest, config=config.model_dump(), metrics=metrics)
        (out / HPO_FILE).write_text(
            json.dumps(tuning.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        if bundle.reference_stats:
            (out / "reference_stats.json").write_text(
                json.dumps(bundle.reference_stats), encoding="utf-8"
            )
        _write_data_reports(out)

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


# ── Resume ────────────────────────────────────────────────
def _resolve_resume(resume: bool | str | Path, out: Path, spec: Any, backend: Any) -> Path | None:
    """The checkpoint to continue from, or ``None`` — with the reason logged.

    ``True`` means "the last checkpoint of the bundle in ``output_dir``"; a path
    means that file. Both are resolved and existence-checked *here* rather than in
    a backend, so a backend that cannot resume is never handed a path it would
    have to explain away, and the user hears about a missing file before training
    starts rather than after.
    """
    if not resume:
        return None

    caps = backend.capabilities
    if not caps.supports_resume:
        # The consumer of `Capabilities.supports_resume`. A one-shot fit(X, y) has
        # no partial state; saying so beats ignoring the flag.
        log.warning(
            "backend '%s' cannot resume (a one-shot fit has no partial state) — training fresh",
            spec.backend,
        )
        return None

    candidate = Path(resume) if not isinstance(resume, bool) else out / MODEL_DIR / LAST_CHECKPOINT
    if not candidate.exists():
        raise FileNotFoundError(
            f"--resume found no checkpoint at {candidate}. "
            f"A run only leaves one behind after completing at least one epoch."
        )
    log.info("resuming from %s", candidate)
    return candidate


# ── Cross-validation ──────────────────────────────────────
def _cross_validate(
    config: ExperimentConfig, spec: Any, backend: Any, out: Path
) -> dict[str, float]:
    """Fit every fold, and return the aggregate as ``cv_<metric>_mean|_std``.

    An **orchestration** mode, not a Lightning one: it drives the splitter and the
    same ``fit``/``predict_split`` protocol calls every backend implements, so
    GBDT gets cross-validation for free and forecasting will too.

    The per-fold detail goes to ``cv.json`` because the *spread* is the point. A
    mean of 0.85 across folds of 0.84/0.86 and a mean of 0.85 across 0.70/1.00 are
    the same number and completely different results; reporting only the mean
    hides which one you have.
    """
    from ..data.builders import build_cv_bundles

    folds = config.data.split.folds
    log.info("cross-validating: %d folds", folds)
    per_fold: list[dict[str, float]] = []

    for i, fold in enumerate(build_cv_bundles(config)):
        run = RunContext(
            output_dir=out / "cv" / f"fold_{i}",
            seed=config.runtime.seed,
            budget=resolve_budget(config),
            accelerator=config.runtime.accelerator,
            devices=config.runtime.devices,
            precision=config.runtime.precision,
            strategy=config.runtime.strategy,
            deterministic=config.runtime.deterministic,
        )
        result = backend.fit(spec, fold, config, run=run)
        predictions = backend.predict_split(result.estimator, fold, "test")
        scores = get_task_spec(config.task).compute(
            predictions.y_true, predictions.y_pred, predictions.y_prob
        )
        log.info("fold %d/%d: %s", i + 1, folds, scores)
        per_fold.append(scores)

    aggregate = _aggregate_folds(per_fold)
    (out / CV_FILE).write_text(
        json.dumps({"folds": folds, "per_fold": per_fold, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    log.info("cross-validation: %s", aggregate)
    return aggregate


def _aggregate_folds(per_fold: list[dict[str, float]]) -> dict[str, float]:
    """Mean and standard deviation per metric, prefixed ``cv_``."""
    import statistics

    if not per_fold:
        return {}
    out: dict[str, float] = {}
    for name in per_fold[0]:
        values = [f[name] for f in per_fold if name in f]
        if not values:
            continue
        out[f"cv_{name}_mean"] = float(statistics.fmean(values))
        out[f"cv_{name}_std"] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return out


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
def _compare_to_baseline(metrics: dict[str, float], trivial: dict[str, float], task: str) -> None:
    from ..core.baseline import compare

    compare(metrics, trivial, task)


def _baseline_metrics(predictions: Any, bundle: Any, task: str) -> dict[str, float]:
    """Score the trivial predictor and say whether the model beat it.

    Reads the training labels off the bundle so the statistic comes from the split
    it should — taking the majority class from the *test* labels would make the
    baseline stronger than anything achievable at training time, inverting the
    comparison.
    """
    from ..core.baseline import baseline_metrics

    scores = baseline_metrics(predictions, task, train_y=bundle.train.y)
    if not scores:
        log.info("no meaningful trivial baseline for task '%s'; skipping", task)
    return scores


def _write_data_reports(out: Path) -> None:
    """Guarantee ``stall.json`` and ``faults.json`` exist, for every backend.

    The Lightning callback writes the real numbers at ``on_fit_end``; this fills
    in zero-valued defaults for a backend with no epoch loop to measure — a GBDT
    fit is one ``fit(X, y)`` call and has no batches to be starved of.

    ``only_if_absent`` so the defaults never clobber a real measurement, and the
    files are always present either way: "no faults" is a materially different
    claim from "nobody looked", and an absent file cannot tell them apart.
    """
    from ..core.stall import FAULTS_FILE, STALL_FILE, write_data_reports

    write_data_reports(out, only_if_absent=True)

    # Two lines in report.txt, which is the file a human actually opens. Appended
    # here rather than written by `evaluate`: the numbers come from the fit loop,
    # and threading them through the evaluation path would couple two things that
    # otherwise share nothing.
    report = out / "report.txt"
    if not report.is_file():
        return
    try:
        stall = json.loads((out / STALL_FILE).read_text(encoding="utf-8"))
        faults = json.loads((out / FAULTS_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    lines = ["", "Data pipeline"]
    if stall.get("measured"):
        lines.append(f"  waiting for batches: {stall['data_wait_pct']:.1f}% of wall time")
        gpu = stall.get("gpu_stall_pct")
        # `None` is not `0.0`: a run with no CUDA device has no device to stall,
        # and printing zero would be indistinguishable from a perfectly fed GPU.
        lines.append(
            f"  GPU stall:           {gpu:.1f}%"
            if gpu is not None
            else "  GPU stall:           not measured (no CUDA device)"
        )
    else:
        lines.append("  not measured")
    lines.append(f"  corrupt samples:     {faults.get('faults', 0)} (substituted)")
    with report.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _save_preprocessor(bundle: Any, out: Path) -> dict[str, Any] | None:
    """Write the preprocessor's own directory and return its manifest fragment.

    The orchestrator decides *where*; the preprocessor decides *what*. Nothing
    here knows that a scaler exists.
    """
    if bundle.preprocessor is None:
        return None
    return bundle.preprocessor.save(out / PREPROCESSOR_DIR)


def _emit_config(config: ExperimentConfig, dest: str | Path) -> None:
    """Write the effective config as YAML, for committing back to ``configs/``.

    ``bundle/config.json`` is the audit record of one run; this is the file you
    keep. Writing it is what makes a tuned result reproducible without rerunning
    the search.
    """
    import yaml

    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Effective config after tuning, written by `mlf train --emit-config`.\n"
        + yaml.safe_dump(config.model_dump(), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    log.info("effective config written to %s", path)


def _build_manifest(
    config: ExperimentConfig,
    bundle: Any,
    spec: Any,
    artifact: Any,
    preprocessor_ref: dict[str, Any] | None,
    metrics: dict[str, float],
    size: dict[str, Any],
    tuning: Any,
    selection: Any = None,
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
        # The winner and the space it came from, so a served model can answer "how
        # were these numbers chosen?" without the training directory.
        hpo=tuning.to_dict() if tuning.ran else None,
        # And the same question one level up: "why this model family?" A served
        # bundle carrying the bake-off it won is the difference between an
        # architecture decision that can be audited and one that has to be
        # remembered.
        selection=(selection.to_dict() if selection is not None and selection.ran else None),
    )
