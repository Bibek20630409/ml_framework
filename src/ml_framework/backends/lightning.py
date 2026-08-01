"""
backends/lightning.py
─────────────────────
The Lightning fit loop. **All of it, and only here.**

``pl.Trainer``, its callbacks, the checkpoint recovery and the Optuna pruning
callback used to live in ``pipeline/train.py`` and ``pipeline/hpo.py``. They move
here verbatim so the orchestrator can drive an XGBoost or a Prophet fit through
the same three calls. The mechanical statement of that is
``grep pytorch_lightning src/ml_framework/pipeline/train.py`` returning nothing.

This is one backend for a fit-loop *shape*, not for a library: the ~50 lines of
Trainer setup are byte-identical for an MLP, a CNN, an LSTM, a TFT and a
HuggingFace transformer, which is why those five need one backend rather than
five ``fit()`` implementations.

:class:`LightningEstimator` is the other half of the split. It holds the module
and applies the task's canonical head postprocessing **once** — collapsing the
sigmoid/softmax/identity branching that v1 duplicated across
``lit_model._shared_step``, ``evaluate.py:41-52`` and ``inference.py:133-149``.
Being predict-only is what lets it cross into a serving process without dragging
the registry or the training config along.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import numpy as np
import torch
from pydantic import BaseModel as PydanticModel
from pydantic import Field

from ..core.export import (
    ONNX_REQUIREMENTS,
    ExportResult,
    UnsupportedExportError,
    example_input_shape,
    unsupported,
)
from ..core.lit_model import OptimSettings
from ..core.plugins import check_requirements
from ..core.protocols import (
    ArtifactRef,
    BuildContext,
    Categorical,
    FitResult,
    Float,
    Predictions,
    RunContext,
)
from ..core.task import get_task_spec
from ..core.types import Capabilities, UnsupportedCapability
from ..data.lightning_adapter import BundleDataModule
from .base import BaseBackend, clean_metrics, resolve_precision

log = logging.getLogger(__name__)

_OPTIM_DEFAULTS = OptimSettings()


class LightningFitParams(PydanticModel):
    """Schema for ``fit.params`` on this backend.

    The optimizer defaults are read off :class:`OptimSettings` rather than
    restated, so a model constructed directly and one configured through YAML
    cannot drift apart. ``lr_patience``/``lr_factor`` configure
    ``ReduceLROnPlateau``: dropping them in the v1→v2 move would silently change
    the schedule rather than fail. ``gradient_clip_val`` is a Trainer argument and
    therefore correctly a *backend* param — a GBDT has no use for it.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    lr: float = Field(default=_OPTIM_DEFAULTS.lr, gt=0.0)
    weight_decay: float = Field(default=_OPTIM_DEFAULTS.weight_decay, ge=0.0)
    lr_patience: int = Field(default=_OPTIM_DEFAULTS.lr_patience, ge=1)
    lr_factor: float = Field(default=_OPTIM_DEFAULTS.lr_factor, gt=0.0, lt=1.0)
    gradient_clip_val: float = Field(default=1.0, ge=0.0)
    # v1's hardcoded Adam + ReduceLROnPlateau become the *defaults*, so an existing
    # config trains exactly as it did while the choice is now expressible.
    optimizer: Literal["adam", "adamw", "sgd"] = _OPTIM_DEFAULTS.optimizer  # type: ignore[assignment]
    scheduler: Literal["plateau", "cosine", "step", "none"] = _OPTIM_DEFAULTS.scheduler  # type: ignore[assignment]
    # Simulates a larger batch than fits in memory: gradients accumulate over N
    # batches before stepping. The effective batch size is `batch_size * N`, which
    # is worth knowing when comparing runs.
    accumulate_grad_batches: int = Field(default=1, ge=1)


def _pruning_callback(trial: Any, monitor: str) -> Any | None:
    """``PyTorchLightningPruningCallback``, or ``None`` if unavailable.

    ``optuna_integration`` moved the callback out of ``optuna.integration``; both
    paths are tried. Returning ``None`` rather than raising is deliberate: tuning
    is on by default, so an install without the integration package should tune
    *without* pruning rather than refuse to train.
    """
    for module, attr in (
        ("optuna_integration.pytorch_lightning", "PyTorchLightningPruningCallback"),
        ("optuna_integration", "PyTorchLightningPruningCallback"),
        ("optuna.integration", "PyTorchLightningPruningCallback"),
    ):
        try:
            import importlib

            return getattr(importlib.import_module(module), attr)(trial, monitor=monitor)
        except (ImportError, AttributeError):
            continue
    log.debug("no Lightning pruning callback available; tuning without pruning")
    return None


CHECKPOINT_DIR = "checkpoints"
MODEL_FILE = "model.ckpt"
# The resumable state: last epoch's weights *plus* optimizer and scheduler.
LAST_FILE = "last.ckpt"
ARTIFACT_FORMAT = "lightning-checkpoint"
# Evaluation batching does not change results — the module is in eval mode, so
# BatchNorm uses running statistics and dropout is off — but a default keeps
# predict_split usable when no config is in scope.
DEFAULT_EVAL_BATCH_SIZE = 256
# 17 is broadly supported by onnxruntime and old enough that a deployment target a
# year behind still loads it. Pinned rather than left to torch's default, which
# moves between releases and would silently change what the artifact requires.
ONNX_OPSET = 17
# How closely the traced network must reproduce the module's own output before an
# artifact is written. Tight on purpose: this is the same computation twice, not
# two runtimes, so any real difference means `forward` does something extra.
TRACE_PARITY_ATOL = 1e-6


# ── Estimator ─────────────────────────────────────────────
class LightningEstimator:
    """Predict-only wrapper around a trained ``LightningModule``.

    The one place head postprocessing happens. ``TaskSpec.postprocess`` says which
    of sigmoid/softmax/identity applies, so adding a task adds a table row rather
    than a fourth copy of the same three-branch conditional.
    """

    def __init__(
        self,
        module: Any,
        task: str,
        *,
        checkpoint_path: str | Path | None = None,
        batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
        device: Any | None = None,
        preprocessor: Any = None,
    ) -> None:
        self.module = module
        self.task = task
        self.task_spec = get_task_spec(task)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Only the `generate` path uses it, and only to turn token ids back into
        # strings. Handing the estimator the *same* preprocessor the data went
        # through — rather than letting a generative model carry a second copy of
        # the tokenizer — is what stops the two from disagreeing about the
        # vocabulary. `Inferencer` attaches it after loading a bundle.
        self.preprocessor = preprocessor
        self.module.to(self.device).eval()

    def attach_preprocessor(self, preprocessor: Any) -> None:
        """Late-bind the preprocessor, for the load-from-bundle path.

        ``backend.load()`` sees only the manifest; the preprocessor is loaded
        separately by :class:`~ml_framework.core.inference.Inferencer`, which calls
        this once it has both.
        """
        self.preprocessor = preprocessor

    # ── internals ──
    def _to_device(self, inputs: Any) -> Any:
        """Move model input to the device, whatever shape it is.

        A ``Mapping`` passes through as a mapping — a tokenized text batch is
        ``input_ids`` plus ``attention_mask``, and flattening it into one float
        matrix would be meaningless. The model's ``forward`` is what understands
        the dict; this only relocates it. (An HF ``BatchEncoding`` is a Mapping, so
        it takes this branch.)
        """
        if isinstance(inputs, Mapping):
            return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        return torch.tensor(np.asarray(inputs, dtype="float32"), dtype=torch.float32).to(
            self.device
        )

    @torch.no_grad()
    def logits(self, inputs: Any) -> torch.Tensor:
        return self.module(self._to_device(inputs))

    def decode(self, logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray | None]:
        """(hard predictions, probabilities) for this task, from raw logits.

        The single implementation of head postprocessing. ``predict``,
        ``predict_proba`` and ``predict_split`` all route through it, so the three
        can never disagree about what a logit means.
        """
        post = self.task_spec.postprocess
        if self.task_spec.output_kind == "token_labels":
            # (B, T, C): argmax over the last axis leaves one label per position.
            # Deliberately *not* flattened here — the padding mask lives with the
            # labels, so dropping the padded positions is `predict_split`'s job.
            return logits.argmax(dim=-1).cpu().numpy(), torch.softmax(logits, dim=-1).cpu().numpy()
        if post == "sigmoid":
            prob = torch.sigmoid(logits.squeeze(1))
            preds = (prob > 0.5).long()
            return preds.cpu().numpy(), torch.stack([1 - prob, prob], dim=1).cpu().numpy()
        if post == "softmax":
            return logits.argmax(dim=1).cpu().numpy(), torch.softmax(logits, dim=1).cpu().numpy()
        return logits.squeeze(1).cpu().numpy(), None

    @torch.no_grad()
    def generate(self, inputs: Any) -> np.ndarray:
        """Autoregressive decode → an array of strings.

        The one prediction path that is not a function of the logits, which is
        what ``Postprocess.generate`` names. The model produces ids and the
        preprocessor turns them back into text, because the tokenizer that made
        them is the preprocessor's.
        """
        if self.preprocessor is None:
            raise UnsupportedCapability(
                f"task '{self.task}' generates text, which needs the bundle's tokenizer "
                f"to decode. This estimator was built without a preprocessor attached."
            )
        ids = self.module.generate_ids(self._to_device(inputs))
        return np.asarray(self.preprocessor.decode(ids), dtype=object)

    def batch_predictions(self, inputs: Any) -> tuple[np.ndarray, np.ndarray | None]:
        """``(predictions, probabilities)`` for one batch, whatever the task.

        The single routing point. ``predict`` and the backend's ``predict_split``
        both go through it, so a generative model cannot end up decoded one way by
        the serving path and another way by the evaluation path.
        """
        if self.task_spec.postprocess == "generate":
            return self.generate(inputs), None
        return self.decode(self.logits(inputs))

    # ── contract ──
    def predict(self, inputs: Any) -> np.ndarray:
        preds, _ = self.batch_predictions(inputs)
        return preds

    def predict_proba(self, inputs: Any) -> np.ndarray:
        if self.task_spec.postprocess == "generate":
            raise UnsupportedCapability(
                f"task '{self.task}' produces text; a generated string has no class "
                f"distribution behind it"
            )
        _, prob = self.decode(self.logits(inputs))
        if prob is None:
            raise UnsupportedCapability(
                f"task '{self.task}' produces {self.task_spec.output_kind}, not probabilities"
            )
        return prob

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.module.parameters() if p.requires_grad)


# ── Backend ───────────────────────────────────────────────
class LightningBackend(BaseBackend):
    """Iterative mini-batch training: epoch loop plus validation callbacks."""

    name: ClassVar[str] = "lightning"
    capabilities: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset({"arrays", "dataset"}),
        needs_scaling=True,
        produces_proba=True,
        supports_pruning=True,
        supports_gpu=True,
        supports_mixed_precision=True,
        supports_lr_range_test=True,
        # An epoch loop has partial state worth continuing from, which is what
        # makes `--resume` meaningful here and nowhere else yet.
        supports_resume=True,
        # Sample weights are not plumbed through the Lightning loop; imbalance is
        # handled by class weights in the loss or by resampling.
        supports_sample_weight=False,
    )

    # ── fit ──
    def fit(self, spec: Any, bundle: Any, cfg: Any, *, run: RunContext) -> FitResult:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import (
            EarlyStopping,
            LearningRateMonitor,
            ModelCheckpoint,
        )

        out = Path(run.output_dir)
        task_spec = get_task_spec(bundle.task)
        # The backend validates its own `fit.params` — the config layer resolves
        # only `model.params`, because doing the same for every backend would mean
        # importing all of them to validate a YAML file.
        fit_params = LightningFitParams.model_validate(dict(cfg.fit.params))

        dm = BundleDataModule.from_bundle(bundle, cfg)
        dm.setup()
        model = spec.build(
            BuildContext(
                task=bundle.task,
                input_dim=bundle.input_dim,
                output_dim=bundle.output_dim,
                n_classes=bundle.n_classes,
                feature_schema=bundle.schema,
                # numpy from the bundle; BaseModel converts. The adapter's tensor
                # property exists for the loss, not for the build contract.
                class_weights=bundle.class_weights,
                params=cfg.model.params,
                # `max_epochs` reaches the model so cosine/step schedules know
                # their horizon — the only thing that knows it is the loop.
                optim={
                    **fit_params.model_dump(),
                    "max_epochs": run.budget.max_epochs or cfg.fit.budget.max_epochs or 100,
                },
                seed=run.seed,
            )
        )
        log.info("trainable params: %d", model.count_parameters())

        logger = self._build_logger(cfg, run)
        callbacks: list[Any] = [
            EarlyStopping(
                monitor=task_spec.monitor,
                patience=run.budget.patience or cfg.fit.patience,
                mode=task_spec.monitor_mode,
            ),
            ModelCheckpoint(
                dirpath=str(out / CHECKPOINT_DIR),
                filename="best",
                monitor=task_spec.monitor,
                mode=task_spec.monitor_mode,
                save_top_k=1,
                # `last.ckpt` is what `--resume` continues from. The *best*
                # checkpoint is the wrong thing to resume: it holds the weights
                # from whichever epoch scored highest, not the optimizer and
                # scheduler state the loop stopped with.
                save_last=True,
            ),
        ]
        # LearningRateMonitor requires an active logger.
        if logger:
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))
        if cfg.logging.backend == "wandb" and logger:
            logger.watch(model, log="gradients", log_freq=50)
        if run.budget.max_seconds:
            # `fit.budget.max_seconds` enforced, not merely carried. Lightning's
            # Timer stops at an epoch boundary, so the cap is a floor on when
            # training ends rather than a hard kill — which is the right trade for
            # a budget: a half-finished epoch produces no usable checkpoint.
            from pytorch_lightning.callbacks import Timer

            callbacks.append(Timer(duration=timedelta(seconds=run.budget.max_seconds)))

        hooks = self.trial_hooks(run.trial, monitor=task_spec.monitor) if run.trial else None
        if hooks:
            callbacks.extend(hooks.callbacks)

        trainer = pl.Trainer(
            max_epochs=run.budget.max_epochs or cfg.fit.budget.max_epochs,
            accelerator=run.accelerator,
            devices=run.devices,
            strategy=run.strategy,
            # Lightning types this as a closed Literal; `resolve_precision` already
            # narrowed the string to one of those members.
            precision=self._precision(run),  # type: ignore[arg-type]
            accumulate_grad_batches=fit_params.accumulate_grad_batches,
            callbacks=callbacks,
            logger=logger,
            gradient_clip_val=fit_params.gradient_clip_val,
            deterministic=run.deterministic,
            log_every_n_steps=10,
        )

        log.info("training…")
        # `ckpt_path` restores weights *and* optimizer/scheduler/epoch state, which
        # is the difference between resuming and re-initialising from weights.
        resume = str(run.resume_from) if run.resume_from else None
        if resume:
            log.info("resuming from %s", resume)
        trainer.fit(model, dm, ckpt_path=resume)
        val_metrics = clean_metrics(trainer.callback_metrics)
        # Tests the in-memory (last) model, as v1 did — this populates the
        # tracker's test/* series. The reported metrics come from the *best*
        # checkpoint reloaded below, via the orchestrator's evaluate().
        trainer.test(model, dm)

        best_path = self._best_checkpoint(trainer, out)
        log.info("best checkpoint: %s", best_path)
        best_model = type(model).load_from_checkpoint(
            str(best_path),
            input_dim=bundle.input_dim,
            output_dim=bundle.output_dim,
            task=bundle.task,
            params=cfg.model.params,
            optim=fit_params.model_dump(),
            class_weights=None,
        )
        estimator = LightningEstimator(
            best_model,
            bundle.task,
            checkpoint_path=best_path,
            batch_size=cfg.fit.batch_size,
            # Generation needs the tokenizer to turn ids back into text, and this
            # is the one the data went through.
            preprocessor=bundle.preprocessor,
        )
        return FitResult(estimator=estimator, val_metrics=val_metrics)

    def _precision(self, run: RunContext) -> str:
        """Resolved once, here, so every downgrade is logged with its reason."""
        return resolve_precision(
            run.precision,
            self.capabilities,
            accelerator=run.accelerator,
            has_gpu=torch.cuda.is_available(),
        )

    def _best_checkpoint(self, trainer: Any, out: Path) -> Path:
        """The best checkpoint, or a freshly written one if there is none.

        ``best_model_path`` is empty when validation never ran (a zero-epoch or
        fast-dev run). Saving one there keeps ``save()`` total instead of making
        every caller handle a missing artifact.
        """
        best = getattr(trainer.checkpoint_callback, "best_model_path", "")
        if best and Path(best).exists():
            return Path(best)
        fallback = out / CHECKPOINT_DIR / "last.ckpt"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(fallback))
        log.warning("no best checkpoint recorded; saved the final weights to %s", fallback)
        return fallback

    def _build_logger(self, cfg: Any, run: RunContext) -> Any:
        """The Lightning logger for the training loop.

        Separate from ``run.run_logger`` on purpose. The backend-neutral
        ``RunLogger`` records run-level params, final metrics and the bundle; a
        Lightning logger records per-epoch metrics from inside the loop, which is a
        job only Lightning can do. For MLflow the two are bound to the **same run**
        via ``run_id`` — v1 had the Lightning logger own the run, which is why
        ``log_and_register`` had to reach into it for a ``run_id`` that a GBDT run
        would never have.
        """
        backend = cfg.logging.backend
        if backend == "wandb":
            from pytorch_lightning.loggers import WandbLogger

            return WandbLogger(
                project=cfg.logging.wandb_project,
                name=cfg.logging.wandb_run,
                log_model=cfg.logging.log_model,
            )
        if backend == "mlflow":
            from pytorch_lightning.loggers import MLFlowLogger

            from ..tracking.mlflow_utils import resolve_tracking_uri

            run_id = getattr(run.run_logger, "run_id", None)
            return MLFlowLogger(
                experiment_name=cfg.logging.mlflow_experiment,
                tracking_uri=resolve_tracking_uri(cfg),
                artifact_location=cfg.logging.mlflow_artifact_location,
                run_name=cfg.logging.wandb_run,  # reuse the run-name field
                run_id=run_id,
                log_model=False,  # the self-contained bundle is logged instead
            )
        if backend == "csv":
            from pytorch_lightning.loggers import CSVLogger

            return CSVLogger(save_dir=cfg.runtime.output_dir, name="metrics")
        return False

    # ── persistence ──
    def save(self, est: Any, dest: str | Path) -> ArtifactRef:
        """Copy the checkpoint into ``dest``; return its path relative to the bundle.

        ``dest`` is the bundle's model directory, so the returned path is relative
        to ``dest.parent`` — the bundle root, which is what the manifest records.
        """
        target_dir = Path(dest)
        target_dir.mkdir(parents=True, exist_ok=True)
        source = getattr(est, "checkpoint_path", None)

        # `last.ckpt` rides along beside the model so `--resume` has something to
        # continue from after the run directory is cleaned. It is *not* the
        # manifest's artifact — that stays the best weights — because a loader
        # wants those and only a resuming trainer wants the last optimizer state.
        # Written for every model, including one with its own format: resuming is
        # a property of the loop, not of the file the loop produced.
        if source and Path(source).parent.joinpath(LAST_FILE).exists():
            shutil.copyfile(Path(source).parent / LAST_FILE, target_dir / LAST_FILE)

        own_format = getattr(getattr(est, "module", None), "save_artifact", None)
        if own_format is not None:
            # A model may own its serialization — see `plugins/nlp/hf_text.py`,
            # where a Lightning checkpoint would round-trip the weights but need
            # the HuggingFace hub to rebuild the architecture to put them in. The
            # hook is discovered by name so the backend never learns which models
            # have one.
            return cast("ArtifactRef", own_format(target_dir))

        if source is None or not Path(source).exists():
            raise FileNotFoundError(
                "LightningEstimator has no checkpoint to save; it was not produced by fit()"
            )
        shutil.copyfile(source, target_dir / MODEL_FILE)
        return ArtifactRef(path=f"{target_dir.name}/{MODEL_FILE}", format=ARTIFACT_FORMAT)

    def load(self, bundle_dir: str | Path, manifest: Any) -> LightningEstimator:
        """Rebuild an estimator from a bundle — **manifest only**.

        ``load`` lives on the backend rather than the estimator because it needs
        registry access to reconstruct an architecture before weights can go into
        it; making every estimator a registry client would drag the training
        dependencies into the serving image.

        v1's ``BaseModel`` took a whole ``ExperimentConfig``, so P1's version of
        this method had to read ``config.json`` back and re-validate it — which
        made loading a bundle depend on a populated plugin registry *and* on the
        training config still being parseable by the current schema. Now the
        manifest carries everything the architecture needs (``model.params`` and
        the signature), and ``config.json`` is purely the audit record the plan
        says it is.
        """
        from ..core.lit_model import BaseModel
        from ..plugins import model_class

        root = Path(bundle_dir)
        # `model_class` imports the defining module first. The neural plugins are
        # registered with a *lazy* build (defining a LightningModule imports torch,
        # which is optional from P3), so their class may not be in the v1 registry
        # yet in a process that has only ever loaded bundles.
        model_cls = cast("type[BaseModel]", model_class(manifest.model.name))

        own_format = getattr(model_cls, "load_artifact", None)
        if own_format is not None:
            return LightningEstimator(
                own_format(root / manifest.model.artifact, manifest), manifest.task
            )

        module = model_cls.load_from_checkpoint(
            str(root / manifest.model.artifact),
            input_dim=manifest.signature.input.n_features,
            output_dim=self._output_dim(manifest),
            task=manifest.task,
            params=manifest.model.params,
            # Optimizer settings are a training concern; a loaded estimator only
            # predicts, so the defaults are never consulted.
            optim=None,
            class_weights=None,
            map_location="cpu",
        )
        return LightningEstimator(module, manifest.task)

    @staticmethod
    def _output_dim(manifest: Any) -> int:
        """The head width, recovered from the signature.

        Delegates to :func:`~ml_framework.core.bundle.head_width`: a
        self-serializing model rebuilds itself from the same manifest, and two
        copies of the binary-is-one-logit rule would eventually disagree.
        """
        from ..core.bundle import head_width

        return head_width(manifest)

    # ── prediction ──
    def predict_split(self, est: Any, bundle: Any, split: str) -> Predictions:
        """Run ``est`` over one split of ``bundle`` and return arrays.

        The result is what ``evaluate`` consumes, so the evaluator no longer holds
        a hand-rolled torch loop with per-task branching in it.
        """
        target = bundle.split(split)
        dm = BundleDataModule(
            bundle,
            batch_size=getattr(est, "batch_size", DEFAULT_EVAL_BATCH_SIZE),
            # 0 workers: an evaluation pass is not worth a process pool, and
            # spawning one mid-run is a reliable source of Windows surprises.
            num_workers=0,
            # The same batching rule training used. Without it a text split
            # arrives as a list of raw strings and the model is handed something
            # it has no way to read.
            collate_fn=getattr(bundle.preprocessor, "collate_fn", None),
        )
        dm.setup()

        spec = get_task_spec(bundle.task)
        preds: list[np.ndarray] = []
        probs: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        with torch.no_grad():
            for batch in dm.split_dataloader(split):
                x, y = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else (batch, None)
                batch_preds, batch_probs = est.batch_predictions(x)

                if spec.output_kind == "token_labels" and y is not None:
                    # Ragged by nature: batches are padded to their own longest
                    # sequence, so (B, T) arrays from different batches cannot be
                    # concatenated. Dropping the ignored positions — sub-word
                    # continuations, specials, padding — flattens both sides to 1-D
                    # over *real* tokens, which concatenates and is also the only
                    # granularity the metrics mean anything at.
                    keep = np.asarray(y.cpu()) != spec.ignore_index
                    labels.append(np.asarray(y.cpu())[keep])
                    preds.append(np.asarray(batch_preds)[keep])
                    if batch_probs is not None:
                        probs.append(np.asarray(batch_probs)[keep])
                    continue

                preds.append(batch_preds)
                if batch_probs is not None:
                    probs.append(batch_probs)
                if y is None:
                    continue
                if spec.output_kind == "text":
                    # The references are token ids too; they have to come back
                    # through the same tokenizer as the predictions or the string
                    # metrics compare text against numbers.
                    labels.append(np.asarray(est.preprocessor.decode(y), dtype=object))
                else:
                    labels.append(y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y))

        index = target.index
        if labels and index is not None and len(np.concatenate(labels)) != len(index):
            # Token tagging emits one row per *token* while `index` addresses
            # sentences, and a mismatched index silently mislabels every row of
            # predictions.csv. Dropping it is honest; padding it out would not be.
            index = None

        return Predictions(
            y_true=np.concatenate(labels) if labels else None,
            y_pred=np.concatenate(preds),
            y_prob=np.concatenate(probs) if probs else None,
            index=index,
        )

    # ── HPO ──
    def search_space(self) -> dict[str, Any]:
        """Knobs that belong to the *loop*, not to any architecture on it.

        Declared once here so every neural plugin stops repeating lr and
        batch_size. Keys are dotted v2 config paths, so applying a trial is
        ``config.with_overrides(values)``.
        """
        return {
            "fit.params.lr": Float(1e-4, 1e-2, log=True),
            "fit.params.weight_decay": Float(1e-5, 1e-2, log=True),
            "fit.batch_size": Categorical((16, 32, 64, 128)),
        }

    def trial_hooks(self, trial: Any, *, monitor: str = "val/loss") -> Any:
        """Optuna pruning as a Lightning callback.

        The dual-import fallback moves here from ``hpo.py``: ``optuna_integration``
        is the current home of the callback and ``optuna.integration`` the legacy
        one. Keeping it inside the backend is what lets the tuning driver stay free
        of integration packages entirely — it asks for hooks and never learns which
        package supplied them.

        Pruning matters most here of the three backends: a neural trial costs
        minutes, so abandoning a bad one early is the difference between exploring
        the space and sampling it once. A missing integration package degrades to
        no pruning rather than failing the run.

        ``monitor`` defaults to the value every registered ``TaskSpec`` currently
        uses; the driver passes ``TaskSpec.monitor`` explicitly once a task
        disagrees.
        """
        from ..core.protocols import TrialHooks

        if trial is None:
            return TrialHooks.empty()
        callback = _pruning_callback(trial, monitor)
        return TrialHooks(callbacks=(callback,) if callback else ())

    def params_model(self) -> type[PydanticModel]:
        """Pydantic schema for ``fit.params`` on this backend."""
        return LightningFitParams

    # A traced graph, and the same graph in torch's own format. Both need a
    # fixed-width float input, which is why a text bundle is refused rather than
    # traced -- see `core.export.example_input_shape`.
    export_formats: ClassVar[tuple[str, ...]] = ("onnx", "torchscript")

    def export(self, est: Any, dest: Path, fmt: str, *, manifest: Any = None) -> ExportResult:
        """Trace the module to ONNX or TorchScript.

        **Tracing exports the graph, not the preprocessing.** The bundle's scaler
        and image transforms stay behind, so whatever loads this artifact must
        apply them itself. Said in the returned notes and printed by the CLI,
        because an ONNX file fed raw unscaled features produces confident nonsense
        with no error -- the same shape of failure as a mismatched tokenizer.

        The example shape comes from the **manifest**, so export works against a
        bundle alone, which is the situation it exists for.
        """
        import torch

        if fmt not in self.export_formats:
            unsupported(self.name, fmt, self.export_formats)
        if manifest is None:
            raise UnsupportedExportError(
                "exporting a traced graph needs the bundle manifest to know the "
                "input shape; load the bundle rather than the estimator alone"
            )

        shape = example_input_shape(manifest)
        module = est.module.eval()
        example = torch.zeros(shape, dtype=torch.float32)

        # Trace the inner network, not the LightningModule around it. `jit.trace`
        # walks a module's attributes, and `LightningModule.trainer` raises when
        # the module is detached from a Trainer -- which a loaded bundle always is.
        #
        # That substitution assumes `forward` is exactly `self.network(x)`, which
        # is true for every model that can reach this line (text is refused
        # earlier). Assuming it silently would be the wrong kind of shortcut, so
        # it is checked below instead.
        graph = getattr(module, "network", module)
        with torch.no_grad():
            expected = module(example)
            actual = graph(example)
        if not torch.allclose(expected, actual, atol=TRACE_PARITY_ATOL):
            raise UnsupportedExportError(
                f"model '{manifest.model.name}' does something in forward() beyond "
                f"calling its network, so tracing the network alone would drop it. "
                f"Exporting would produce an artifact that scores differently from "
                f"the bundle."
            )

        dest.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "torchscript":
            # `trace` rather than `script`: the plugin bodies use ordinary Python
            # control flow that `script` rejects, and a traced graph is what a
            # torchscript consumer expects anyway.
            with torch.no_grad():
                traced = torch.jit.trace(graph, example)
            traced.save(str(dest))
            note = "traced graph; preprocessing is NOT included"
        else:
            check_requirements(ONNX_REQUIREMENTS, what="ONNX export")
            with torch.no_grad():
                torch.onnx.export(
                    graph,
                    # A 1-tuple, not a bare tensor: torch types the second argument
                    # as the *args tuple the module is called with, and a single
                    # tensor is the one-argument case spelled ambiguously.
                    (example,),
                    str(dest),
                    input_names=["input"],
                    output_names=["output"],
                    # Without this the batch size is frozen at whatever was traced
                    # and the artifact can only ever score one row at a time.
                    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                    opset_version=ONNX_OPSET,
                    # The dynamo exporter, now that `[export]` declares
                    # `onnxscript`. The TorchScript exporter this replaced is
                    # deprecated and unsupported on Python 3.14+, so `dynamo=False`
                    # was a pin with no future. Measured across 1/7/64 rows: same
                    # opset, same dynamic batch axis, and a max abs difference that
                    # stays four orders of magnitude inside ONNX_TOLERANCE.
                    dynamo=True,
                    # NOT cosmetic, and not a preference. The dynamo exporter writes
                    # a U+2705 to stdout when it succeeds; on Windows, where the
                    # console is cp1252, encoding it raises UnicodeEncodeError *from
                    # inside a successful export*. A correct artifact would be
                    # reported as a failure. This is the same class of bug as the
                    # UTF-8 stdout fix in `utils/logging.py`, arriving via a
                    # dependency rather than our own logging.
                    verbose=False,
                )
            note = f"opset {ONNX_OPSET}, dynamic batch; preprocessing is NOT included"

        log.info("exported %s -> %s", fmt, dest)
        return ExportResult(path=dest, format=fmt, notes=note)  # type: ignore[arg-type]

    def model_size(self, est: Any) -> dict[str, Any]:
        return {"trainable_parameters": int(est.count_parameters())}


def build_backend() -> LightningBackend:
    """Factory referenced by the registry's ``BackendSpec``."""
    return LightningBackend()


__all__ = ["LightningBackend", "LightningEstimator", "build_backend"]
