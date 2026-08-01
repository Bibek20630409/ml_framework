"""
config/schema.py
────────────────
Validated, immutable experiment configuration (Pydantic v2) — **schema v2**.

The v1 schema had one block per *implementation detail* (``optim``, ``train``)
and hardcoded every model's knobs on ``ModelConfig``, so adding XGBoost meant
adding ``max_depth``/``n_estimators`` to a class in core that every other model
also validated against. v2 splits the difference the plugin design needs:

* fixed blocks for what the **framework** owns — ``task``, ``runtime``, ``data``
  location/splitting, ``fit`` budget, ``tune``, ``logging``;
* free-form ``params`` sub-dicts for what a **plugin** owns — ``model.params``
  (the architecture), ``fit.params`` (the loop), ``data.params`` (the source).

``model.params`` is **not** a discriminated union, deliberately. A
``Literal[...]`` discriminator is a closed set living in core, so every
third-party plugin would have to edit this file; Pydantic would also have to
import every union member here, pulling xgboost/transformers/prophet eagerly and
defeating lazy extras. Instead the plugin ships its own frozen,
``extra="forbid"`` params model and the validator below runs it — so **typos
still error at config-load time**, and the defaulted values are written back so
``config.json`` records the fully-materialized effective params (a
reproducibility win over v1, which recorded only what the user typed).

Who validates which ``params`` block, and when:

    model.params   here, at load time, via ``ModelSpec.params_model``
    fit.params     by the backend, in ``fit()`` (``TrainingBackend.params_model``)
    data.params    by the source, in its ``build_*_bundle``

The asymmetry is not an oversight. Resolving a model spec costs one import of
``ml_framework.plugins``, whose modules are required to be importable with zero
optional dependencies. Resolving a *backend* and a *source* would mean importing
the whole data layer and every backend just to validate a YAML file — so those
two are validated by the code that consumes them, which is the same guarantee one
step later.

Load from YAML:
    cfg = ExperimentConfig.from_yaml("configs/example_tabular.yaml")

Override programmatically (returns a new copy — frozen):
    cfg = cfg.with_overrides({"fit.params.lr": 3e-4, "fit.budget.max_epochs": 5})

v1 files are mechanically convertible: see :mod:`ml_framework.config.migrate` and
``mlf migrate-config``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Task/DataKind now come from the dependency-free core vocabulary rather than
# being re-declared here. v1 kept its own narrow literals; sharing them is what
# lets a plugin declare `tasks=frozenset({"forecasting"})` against the same
# strings the config validates.
from ..core.types import DataKind, Task

if TYPE_CHECKING:
    from ..core.plugins import ModelSpec

SplitStrategy = Literal["auto", "random", "temporal", "group"]
Refit = Literal["best", "reuse"]
# `16`/`bf16` are the shorthand people type; the backend normalizes them to
# Lightning's `-mixed` spellings rather than rejecting them.
Precision = Literal["16", "16-mixed", "bf16", "bf16-mixed", "32", "32-true", "64"]

# Dotted prefixes under which `with_overrides` may *create* a key. Everything
# else keeps v1's "the key must already exist" rule, because a typo in a fixed
# block is a bug and silently creating `fit.epocs` would hide it. These four are
# plugin-owned dicts whose keys core cannot know, so the rule cannot apply.
_CREATABLE_PREFIXES: tuple[str, ...] = (
    "model.params.",
    "fit.params.",
    "data.params.",
    "tune.overrides.",
)

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class RuntimeConfig(BaseModel):
    """Where the run happens and how reproducible it is.

    v1 scattered these across the top level (``seed``, ``output_dir``) and
    ``train`` (``num_workers``, ``deterministic``). They are one concern: nothing
    here is a property of the model or the data.
    """

    model_config = _FROZEN

    seed: int = 42
    output_dir: str = "outputs"
    num_workers: int = -1  # -1 → auto (0 on Windows, else 4)
    accelerator: str = "auto"
    devices: str | int = "auto"
    # Honoured by backends that declare `supports_mixed_precision`, and **warned
    # about** by the ones that do not — silently training in fp32 after being asked
    # for fp16 is the kind of thing you discover from a wall-clock number months
    # later. `16`/`bf16` are accepted and normalized to Lightning's `-mixed` names.
    precision: Precision = "32"
    # Multi-device strategy. "auto" lets Lightning choose, which is right until you
    # have a reason it is not.
    strategy: str = "auto"
    deterministic: bool = True


class SplitConfig(BaseModel):
    """How train/val/test are cut.

    Splitting gets its own block because the *correct* split is a property of the
    data, not of the model, and because ``strategy`` is the config path that makes
    ``TemporalSplitter``/``GroupSplitter`` reachable at all. The leakage guard
    (refusing an explicitly shuffled split on time-series data) arrives with the
    forecasting work that gives it something to guard.
    """

    model_config = _FROZEN

    strategy: SplitStrategy = "auto"
    val_size: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_size: float = Field(default=0.15, gt=0.0, lt=1.0)
    # Chronological order column. Setting it flips `auto` to a temporal split.
    time_col: str | None = None
    # Entity column for GroupSplitter — repeated visits/sessions must not straddle
    # the split, a silent defect nothing in v1 prevented.
    group_col: str | None = None
    # Rows dropped between segments. Not cosmetic: with lag features the last
    # train rows and the first val rows share source observations.
    gap: int = Field(default=0, ge=0)
    # 0 or 1 → a single holdout split (the default). >= 2 → k-fold
    # cross-validation, which `train()` runs as an orchestration mode: k fits to
    # *estimate* performance honestly, then the usual single fit to produce the
    # bundle. It belongs in the split block because k-fold is a way of cutting the
    # data — and being driven by the Splitter is what gives GBDT and forecasting
    # cross-validation too, rather than it being a Lightning feature.
    folds: int = Field(default=0, ge=0)
    # Steps ahead each fold forecasts, for rolling-origin cross-validation.
    horizon: int = Field(default=1, ge=1)
    # Expanding window (a production retrain) vs sliding (old data is misleading).
    expanding: bool = True
    # The deliberate friction point. Shuffling a time series is the most damaging
    # silent failure in this domain — it produces a suspiciously good score rather
    # than an error — so the escape hatch requires typing the word "leakage".
    allow_temporal_leakage: bool = False

    @model_validator(mode="after")
    def _check_sizes(self) -> SplitConfig:
        if self.val_size + self.test_size >= 1.0:
            raise ValueError("val_size + test_size must be < 1.0")
        if self.folds == 1:
            raise ValueError("split.folds must be 0 (holdout) or >= 2; 1 fold is not a split")
        return self

    @property
    def cross_validate(self) -> bool:
        return self.folds >= 2

    def resolved_strategy(self, data_kind: str) -> str:
        """``auto`` made concrete: temporal whenever time ordering matters."""
        if self.strategy != "auto":
            return self.strategy
        if self.time_col or data_kind == "timeseries":
            return "temporal"
        if self.group_col:
            return "group"
        return "random"


class DataConfig(BaseModel):
    """Where the data is and what the target is. Source-specific knobs go in
    ``params``, validated by the source that reads them."""

    model_config = _FROZEN

    kind: DataKind = "tabular"
    # The dataset location: a CSV/Parquet path for tabular, the training image
    # folder for image. One field rather than v1's csv_path/train_dir pair —
    # "where the data is" does not change meaning per kind.
    path: str | None = None
    target: str | None = None
    class_names: list[str] | None = None
    split: SplitConfig = Field(default_factory=SplitConfig)
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_required_by_kind(self) -> DataConfig:
        """v1's ``_check_required_by_kind``, same errors, v2 field names.

        Only the built-in kinds are checked. A third-party source registers its
        own ``data.kind`` and validates its own ``params``; hardcoding a closed set
        of requirements here would re-create the problem the plugin registry exists
        to remove.
        """
        if self.kind == "tabular":
            if not self.path or not self.target:
                raise ValueError("tabular data requires 'path' and 'target'")
        elif self.kind == "image":
            if not self.path or not self.params.get("test_dir"):
                raise ValueError("image data requires 'path' (train dir) and 'params.test_dir'")
        elif self.kind == "timeseries":
            if not self.path or not self.target:
                raise ValueError("timeseries data requires 'path' and 'target' (the value column)")
        self._check_temporal_leakage()
        return self

    def _check_temporal_leakage(self) -> None:
        """Refuse a shuffled split on time-ordered data.

        The most damaging silent failure in this domain: a random split puts future
        rows in training and past rows in test, and reports a *better* score for it.
        Nothing crashes, so nothing tells you. Warning and proceeding would be the
        conventional choice and the wrong one — the whole point is that the number
        looks fine.

        The escape hatch exists because there are legitimate reasons (a
        cross-sectional model that happens to carry a date column), but it costs
        typing the word "leakage", which is roughly the amount of deliberation the
        decision deserves.
        """
        if self.kind != "timeseries" or self.split.strategy != "random":
            return
        if self.split.allow_temporal_leakage:
            return
        raise ValueError(
            "data.split.strategy: random on kind: timeseries shuffles the future into "
            "training and reports a score that is not an estimate of anything. Use "
            "strategy: temporal (or auto), or set "
            "data.split.allow_temporal_leakage: true if you genuinely mean it."
        )


class ModelConfig(BaseModel):
    """Which plugin, and its architecture knobs.

    ``params`` is free-form *here* and strict *there*: the plugin's
    ``params_model`` is frozen + ``extra="forbid"``, and
    :meth:`ExperimentConfig._resolve_plugin_params` runs it at load time. v1's
    ``_check_dims`` (positive ``hidden_dims``) now lives in the MLP plugin, where
    the only model it applies to can see it.
    """

    model_config = _FROZEN

    name: str = "mlp"  # key in the model registry
    params: dict[str, Any] = Field(default_factory=dict)


class BudgetConfig(BaseModel):
    """The training budget, one axis per field. ``None`` means "no limit"."""

    model_config = _FROZEN

    max_epochs: int | None = Field(default=200, ge=1)
    max_seconds: float | None = Field(default=None, gt=0.0)


class FitConfig(BaseModel):
    """How the fit loop runs, independent of which loop it is.

    ``budget``/``patience``/``batch_size`` are meaningful for every backend;
    ``params`` holds the loop's own knobs (lr, weight_decay, the
    ``ReduceLROnPlateau`` schedule, gradient clipping) and is validated by
    ``TrainingBackend.params_model()``.
    """

    model_config = _FROZEN

    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    patience: int = Field(default=20, ge=1)  # early stopping
    batch_size: int = Field(default=32, ge=1)
    params: dict[str, Any] = Field(default_factory=dict)


class TuneConfig(BaseModel):
    """Hyperparameter search, consumed by ``pipeline/tune.py``.

    **These defaults are mostly not what runs.** ``max_trials``/``max_seconds``
    carry values only so the fields have a type and a documented shape; the driver
    treats a value equal to the default as "unset" and substitutes the *per-backend*
    budget from ``config/defaults.py`` — 30 trials / 300 s for trees, 10 / 900 s for
    neural nets. A uniform default here would either waste the cheap case or make
    the expensive one feel broken.

    Setting either field to anything else makes it authoritative. That rule is what
    lets one schema serve backends whose per-trial cost differs by two orders of
    magnitude; the alternative, defaulting every field to ``None``, would push the
    same ambiguity into the YAML where it reads worse.
    """

    model_config = _FROZEN

    enabled: bool = True
    max_trials: int = Field(default=20, ge=1)
    max_seconds: float | None = Field(default=900.0, gt=0.0)
    # None → the task's primary metric, so the objective is never hardcoded to a
    # Lightning-only key the way v1's `val/loss` was.
    metric: str | None = None
    refit: Refit = "best"
    # Narrows a plugin's declared search space, keyed by the same dotted paths.
    overrides: dict[str, Any] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    model_config = _FROZEN

    backend: Literal["wandb", "csv", "mlflow", "none"] = "csv"
    wandb_project: str = "ml-framework"
    wandb_run: str | None = None
    log_model: bool = False

    # ── MLflow (tracking + model registry) ────────────────
    # tracking_uri: None → local "sqlite:///mlflow.db" (registry-capable; the file
    # store is deprecated in MLflow 3.x). In production point at "http://mlflow:5000".
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "ml-framework"
    # Where run/model artifacts are stored (None → MLflow default, e.g. ./mlartifacts
    # locally or an s3://… bucket in production).
    mlflow_artifact_location: str | None = None
    # If set, the trained bundle is registered as a version of this model.
    registered_model_name: str | None = None


class ExperimentConfig(BaseModel):
    model_config = _FROZEN

    task: Task

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    fit: FitConfig = Field(default_factory=FitConfig)
    tune: TuneConfig = Field(default_factory=TuneConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # ── Plugin resolution ─────────────────────────────────
    @model_validator(mode="after")
    def _resolve_plugin_params(self) -> ExperimentConfig:
        """Check the (task, kind, model) triple and materialize ``model.params``.

        Two things happen here that used to happen much later, or not at all:

        1. ``validate_combination`` fails an impossible or uninstalled combination
           **at config-load time** — with a pip command, or an explanation of why
           the combination could never work — instead of surfacing as a shape error
           40 seconds into data loading.
        2. The plugin's own params model validates ``model.params`` and its
           defaults are written back with ``model_copy(update=...)``, which does
           **not** re-run validation, so there is no recursion.

        The registry import is lazy (module scope would be a cycle: ``plugins``
        imports ``core``, which the config layer is imported by).
        """
        import ml_framework.plugins  # noqa: F401  (registration is an import side effect)

        from ..core.registry import validate_combination
        from ..core.task import get_task_spec

        # Rejects a task in the v2 vocabulary that has no TaskSpec row yet
        # (forecasting, seq2seq): the Literal is only a key, the table is what
        # makes a task runnable.
        get_task_spec(self.task)

        spec: ModelSpec = validate_combination(self.task, self.data.kind, self.model.name)
        updates: dict[str, Any] = {}
        if spec.fit_defaults:
            # The user's own keys win: `self.fit.params` holds only what was
            # written, since the backend's schema supplies the rest at fit time.
            # That is what makes "defaults the user did not set" answerable here
            # and *not* answerable for `batch_size` or `budget`, which are typed
            # fields whose defaults are indistinguishable from an explicit value.
            merged = {**dict(spec.fit_defaults), **dict(self.fit.params)}
            if merged != dict(self.fit.params):
                updates["fit"] = self.fit.model_copy(update={"params": merged})
        if spec.params_model is None:
            return self.model_copy(update=updates) if updates else self
        try:
            resolved = spec.params_model.model_validate(dict(self.model.params))
        except ValidationError as exc:
            # Re-raised as a ValueError so Pydantic reports it as a failure of
            # `model.params` on *this* config. Without the wrapper the message
            # names a bare key ("dropout") with no hint that it belongs to a
            # plugin's schema — which is exactly the question the user has.
            raise ValueError(
                f"model.params is not valid for model '{spec.name}': "
                + "; ".join(
                    f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
                    for e in exc.errors()
                )
            ) from exc
        updates["model"] = self.model.model_copy(update={"params": resolved.model_dump()})
        return self.model_copy(update=updates)

    # ── Loaders ───────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    def with_overrides(self, overrides: dict[str, Any]) -> ExperimentConfig:
        """Return a new config with dotted-key overrides applied.

        Example: ``{"fit.params.lr": 3e-4, "fit.budget.max_epochs": 5}``.

        This is also the HPO trial-application mechanism: a plugin's search space
        is keyed by these same dotted paths, so applying a trial is exactly
        ``config.with_overrides(trial_values)``.

        Unknown keys raise ``KeyError`` as in v1, **except** under the plugin-owned
        ``params`` dicts (see :data:`_CREATABLE_PREFIXES`), whose keys core cannot
        enumerate. A typo there is still caught — one layer down, by the plugin's
        ``extra="forbid"`` params model.
        """
        data = self.model_dump()

        # Changing the model invalidates the params that belong to the old one.
        # `_resolve_plugin_params` writes every default back into `model.params` at
        # load time, so a config validated once carries xgboost's `tree_method` --
        # and re-validating it as catboost fails with a wall of "extra inputs are
        # not permitted".
        #
        # Cleared *before* the overrides are applied rather than after, so that
        # `--set model.name=catboost --set model.params.depth=5` lands `depth` on an
        # empty dict and the rest of catboost's defaults materialize around it.
        # Clearing afterwards would either keep the stale keys or discard the depth
        # the user just set, and both are wrong.
        if "model.name" in overrides:
            data["model"]["params"] = {}

        for dotted, value in overrides.items():
            creatable = dotted.startswith(_CREATABLE_PREFIXES)
            keys = dotted.split(".")
            node = data
            for k in keys[:-1]:
                if k not in node or not isinstance(node[k], dict):
                    raise KeyError(f"Unknown config path: {dotted}")
                node = node[k]
            if keys[-1] not in node and not creatable:
                raise KeyError(f"Unknown config key: {dotted}")
            node[keys[-1]] = value

        return self.__class__.model_validate(data)


__all__ = [
    "BudgetConfig",
    "DataConfig",
    "ExperimentConfig",
    "FitConfig",
    "LoggingConfig",
    "ModelConfig",
    "RuntimeConfig",
    "SplitConfig",
    "TuneConfig",
]
