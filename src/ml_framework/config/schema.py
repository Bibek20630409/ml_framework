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

    model.params         here, at load time, via ``ModelSpec.params_model``
    fit.params           by the backend, in ``fit()`` (``TrainingBackend.params_model``)
    data.params          by the source, in its ``build_*_bundle``
    data.backend_params  by the data backend, at first use

The asymmetry is not an oversight. Resolving a model spec costs one import of
``ml_framework.plugins``, whose modules are required to be importable with zero
optional dependencies. Resolving a *backend*, a *source* or a *data backend*
would mean importing the whole data layer, every backend, or pyspark just to
validate a YAML file — so those three are validated by the code that consumes
them, which is the same guarantee one step later.

Load from YAML:
    cfg = ExperimentConfig.from_yaml("configs/example_tabular.yaml")

Override programmatically (returns a new copy — frozen):
    cfg = cfg.with_overrides({"fit.params.lr": 3e-4, "fit.budget.max_epochs": 5})

v1 files are mechanically convertible: see :mod:`ml_framework.config.migrate` and
``mlf migrate-config``.
"""

from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)

SplitStrategy = Literal["auto", "random", "temporal", "group"]
# How k-fold is *cut*, once `split.folds >= 2`. Orthogonal to `SplitStrategy`,
# which cuts a single holdout partition: a run can hold out temporally and
# cross-validate with rolling origins, and those are two separate statements.
# `auto` resolves from the data kind and task, reproducing the pre-P11 behaviour
# exactly, so an existing config keeps the folds it already had.
CVStrategy = Literal["auto", "stratified", "kfold", "rolling_origin", "purged", "cpcv"]
Refit = Literal["best", "reuse"]
# What a hyperparameter trial is scored on. `holdout` fits once against the
# single validation split — cheap, and the default. `cv` averages the objective
# across inner folds, which is what makes "select the model and its
# hyperparameters jointly, on cross-validation" literally true, at k times the
# cost per trial.
TuneObjective = Literal["holdout", "cv"]
# How a bake-off picks its winner from the per-criterion measurements.
SelectObjective = Literal["tolerance", "weighted"]
# `16`/`bf16` are the shorthand people type; the backend normalizes them to
# Lightning's `-mixed` spellings rather than rejecting them.
Precision = Literal["16", "16-mixed", "bf16", "bf16-mixed", "32", "32-true", "64"]

# Dotted prefixes under which `with_overrides` may *create* a key. Everything
# else keeps v1's "the key must already exist" rule, because a typo in a fixed
# block is a bug and silently creating `fit.epocs` would hide it. These are
# plugin-owned dicts whose keys core cannot know, so the rule cannot apply.
_CREATABLE_PREFIXES: tuple[str, ...] = (
    "model.params.",
    "fit.params.",
    "data.params.",
    "data.backend_params.",
    # Decoder-owned, like the two above. `data.integrity.*` and `data.shards.*` are
    # NOT here on purpose: they are typed fields, so the "key must already exist"
    # rule applies correctly and `--set data.integrity.on_corupt=raise` rightly
    # raises instead of silently creating a setting nothing reads.
    "data.decoder_params.",
    "tune.overrides.",
    # Criterion names are a closed set, but they are validated by
    # `SelectConfig._check_weights` rather than by key existence — a weights dict
    # starts empty, so the "key must already exist" rule would make every weight
    # unsettable from `--set`.
    "select.weights.",
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

    # ── Transport: how a batch gets from a worker to the device ──
    # These live here rather than on `fit` for the reason this class's docstring
    # gives: worker count, page-locking and prefetch depth are properties of
    # *where the run happens*, not of the model or the data. `fit` is budget,
    # patience and batch size.
    #
    # Page-locked staging buffers, so H2D is an async DMA rather than a bounce
    # through pageable memory. Honoured only on CUDA and only for a host-landing
    # decoder; auto-downgraded with a debug log otherwise, so the default is a
    # no-op on a CPU box rather than a waste of pinned RAM.
    pin_memory: bool = True
    # DEFAULT FALSE, deliberately. Keeping workers alive between epochs is the
    # single biggest throughput win on a many-worker image or audio run -- but
    # `worker_init_fn` then runs once instead of per epoch, which changes the
    # augmentation stream. Turning it on is a decision, not an inheritance.
    persistent_workers: bool = False
    # Batches each worker prefetches. torch's own default; raising it trades host
    # RAM for tolerance of a bursty decode.
    prefetch_factor: int = Field(default=2, ge=1)
    # Run the preprocessor's transform stage AFTER the H2D copy, on the device,
    # instead of in a DataLoader worker. Honoured only where it is both possible
    # and useful: a CUDA device, a host-landing decoder, and a preprocessor that
    # declares `gpu_transform` (audio's mel, video's permute+normalize). Inert
    # everywhere else, so the default costs nothing on a CPU box or a GBDT run.
    #
    # The knob exists because "the transform moved to the GPU" changes where a
    # slow run's time is going, and finding that out from a wall-clock number is
    # the failure this repo keeps designing against. Turn it off to compare.
    device_transform: bool = True


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
    # How those folds are cut. `auto` = the pre-P11 rule (temporal data gets
    # rolling origins, classification gets stratified k-fold, everything else
    # plain k-fold), so setting `folds` alone behaves exactly as it always did.
    cv_strategy: CVStrategy = "auto"
    # Steps ahead each fold forecasts, for rolling-origin cross-validation.
    horizon: int = Field(default=1, ge=1)
    # Expanding window (a production retrain) vs sliding (old data is misleading).
    expanding: bool = True
    # ── Purging (cv_strategy: purged | cpcv) ──────────────
    # How many rows ahead a row's label is computed from. A 10-step-ahead target
    # at row i is not known until row i+10, so i and i+10 share information and a
    # fold that trains on one while testing the other leaks. 0 → labels are known
    # at their own row and purging is a no-op.
    label_horizon: int = Field(default=0, ge=0)
    # The exact form of the same statement: a column of per-row label end times
    # (López de Prado's t1). Wins over `label_horizon` when both are set, because
    # a per-row span is strictly more information than one number for all rows.
    label_end_col: str | None = None
    # Rows dropped *after* each test window, to break serial correlation that
    # purging (which looks forward) cannot see. < 1.0 → a fraction of the
    # dataset; >= 1.0 → a literal row count.
    embargo: float = Field(default=0.0, ge=0.0)
    # ── CPCV (cv_strategy: cpcv) ──────────────────────────
    # Contiguous blocks the data is cut into, and how many are tested per fold.
    # Produces C(cpcv_groups, cpcv_test_groups) folds — 6 choose 2 is 15 fits.
    cpcv_groups: int = Field(default=6, ge=2)
    cpcv_test_groups: int = Field(default=2, ge=1)
    # The cost ceiling on that combinatorial explosion.
    cpcv_max_folds: int = Field(default=20, ge=1)
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
        if self.cpcv_test_groups >= self.cpcv_groups:
            raise ValueError(
                f"split.cpcv_test_groups ({self.cpcv_test_groups}) must be less than "
                f"split.cpcv_groups ({self.cpcv_groups}) — testing every group leaves "
                f"nothing to train on"
            )
        return self

    @property
    def cross_validate(self) -> bool:
        return self.folds >= 2

    @property
    def purges(self) -> bool:
        """Whether this configuration drops rows around the test window."""
        return self.label_horizon > 0 or self.label_end_col is not None or self.embargo > 0

    def resolved_strategy(self, data_kind: str) -> str:
        """``auto`` made concrete: temporal whenever time ordering matters."""
        if self.strategy != "auto":
            return self.strategy
        if self.time_col or data_kind == "timeseries":
            return "temporal"
        if self.group_col:
            return "group"
        return "random"

    def resolved_cv_strategy(self, data_kind: str, task: str) -> str:
        """``cv_strategy: auto`` made concrete.

        The resolution order encodes which mistake is worse. Time ordering wins
        first, because a shuffled fold on temporal data is the failure that
        reports a *better* score. Purging settings win next: someone who declared
        a label horizon has said their rows overlap, and honouring that with plain
        k-fold would ignore the one thing they told us. Stratification is the
        tie-break for row-labelled tasks, and plain k-fold is the floor.

        Explicitly *not* consulted: ``strategy``. A temporal holdout with
        stratified inner folds is a coherent thing to ask for, and inferring one
        from the other would make it unaskable.
        """
        if self.cv_strategy != "auto":
            return self.cv_strategy
        if self.time_col or data_kind == "timeseries":
            return "rolling_origin"
        if self.purges:
            return "purged"
        return "stratified" if task in ("binary", "multiclass") else "kfold"


class IntegrityConfig(BaseModel):
    """What happens to a sample that cannot be trusted.

    **There is no ``skip``.** Not a rejected option — an *absent* one. Under DDP
    every rank must produce an identical number of batches or the next collective
    hangs with no error message, and ``continue`` is the one action that cannot
    preserve that. Making it unrepresentable in the schema is cheaper, and more
    reliable, than documenting why not to use it.
    """

    model_config = _FROZEN

    # `substitute` serves a different sample; `raise` stops the run. The latter is
    # refused for an explicitly distributed strategy below, because an abort that
    # depends on which rank drew the bad sample is a hang, not an error.
    on_corrupt: Literal["substitute", "raise"] = "substitute"
    # `redraw` takes another sample from the SAME shard (fair-ish, and it keeps the
    # read inside an already-open shard); `repeat` re-serves the previous good one.
    substitute: Literal["redraw", "repeat"] = "redraw"
    # Enforced by `mlf materialize` ONLY. At runtime this is counted and logged but
    # never raised: a content-dependent abort is rank-divergent, which is the exact
    # failure this block exists to prevent. Materialization is one process, runs
    # before any collective exists, and can therefore fail safely.
    max_fault_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    # "auto" verifies digests for decoders whose integrity is `none` or `silent`,
    # and skips the ones already paying for their own CRC. Hashing is ~1 GB/s of
    # the read budget; paying it twice on a FLAC is waste.
    verify_checksums: Literal["auto", "always", "never"] = "auto"
    # Escape hatch for training on a corpus whose decoder cannot report its own
    # damage without probing it first. It exists, and it costs typing — the same
    # friction idiom as `allow_temporal_leakage`.
    allow_unverified: bool = False


class ShardConfig(BaseModel):
    """Where the shard index lives and how the corpus is ordered."""

    model_config = _FROZEN

    # None -> `<data.path>/_mlf_shards`. Overridable for a read-only corpus mount,
    # which is the common case for a shared dataset.
    index_dir: str | None = None
    # `block` shuffles shard order then within each shard, so every read stays
    # inside one open shard — the entire reason to shard over object storage.
    # `global` is a true permutation (right on local SSD); `none` is index order.
    shuffle: Literal["block", "global", "none"] = "block"
    # Carry the loader position in the checkpoint, so `--resume` continues
    # mid-epoch rather than silently replaying what this epoch already served.
    resume_state: bool = True


class DataConfig(BaseModel):
    """Where the data is and what the target is. Source-specific knobs go in
    ``params``, validated by the source that reads them; engine knobs go in
    ``backend_params``, validated by the data backend."""

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
    # Which engine reads and reduces the table: `local` (pandas) or `spark`.
    # A plain `str`, not a `Literal`: closing the set would put every third-party
    # engine in this file, the same reason `model.name` is not one. An unknown
    # name surfaces as UnknownPluginError from `get_data_backend`, which lists
    # what is registered.
    backend: str = "local"
    # Engine knobs, validated by the backend at first use. A *separate* dict from
    # `params` rather than a shared one: `params` is validated by the source with
    # `extra="forbid"` (see `TabularSourceParams`), so an engine key placed there
    # would make the source reject the config. One owner per dict.
    backend_params: dict[str, Any] = Field(default_factory=dict)
    integrity: IntegrityConfig = Field(default_factory=IntegrityConfig)
    shards: ShardConfig = Field(default_factory=ShardConfig)
    # Decoder knobs, validated by the decoder at first use. A *third* separate dict
    # for the identical reason `backend_params` is separate from `params`: `params`
    # is validated by the source with `extra="forbid"`, so a decoder key placed
    # there would make the source reject the config. One owner per dict.
    decoder_params: dict[str, Any] = Field(default_factory=dict)

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
        elif self.kind in ("audio", "video"):
            # Same shape as image: a training folder and an explicitly held-back
            # test folder. `target` is not required -- the label is the class
            # directory, recorded per entry in the shard index at materialization.
            if not self.path or not self.params.get("test_dir"):
                raise ValueError(
                    f"{self.kind} data requires 'path' (train dir) and 'params.test_dir'"
                )
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
    # What a trial is scored on. `holdout` fits once against the validation split.
    # `cv` averages the objective over `cv_folds` inner folds, cut by the same
    # `data.split.cv_strategy` the outer estimate uses — so a trial cannot win on
    # a lucky validation split, at k times the cost.
    objective: TuneObjective = "holdout"
    # Inner folds for `objective: cv`. Separate from `data.split.folds`, which is
    # the *outer* estimate: sharing one number would make the honest nested setup
    # (5 outer, 3 inner) unexpressible.
    cv_folds: int = Field(default=3, ge=2)
    # Optuna trials run concurrently. Threads, not processes — real speedup for
    # GBDT and sklearn (which release the GIL in fit), close to none for a Python
    # -bound loop. Leave at 1 on a single GPU: concurrent trials contend for it
    # and the wall-clock gets worse, not better.
    n_jobs: int = Field(default=1, ge=1)
    # Narrows a plugin's declared search space, keyed by the same dotted paths.
    overrides: dict[str, Any] = Field(default_factory=dict)


class ConstraintConfig(BaseModel):
    """Production limits a candidate must meet to be eligible to win.

    These are **hard** constraints, not preferences: a candidate that violates one
    is disqualified and reported with the reason, however good its score is. That
    is the whole point — a model that cannot answer inside the latency budget is
    not a better model that happens to be slow, it is a model that does not work.

    ``None`` means the axis is unconstrained, which is the default for all of
    them. A framework that shipped a default latency ceiling would fail runs on a
    machine slower than the one the number was picked on.
    """

    model_config = _FROZEN

    # Measured p95 of single-row inference, in milliseconds. The p95 rather than
    # the mean because a tail that misses the budget is a timeout in production,
    # and the mean hides it.
    max_latency_p95_ms: float | None = Field(default=None, gt=0.0)
    # Serialized artifact size on disk. The number that decides whether a model
    # fits in a serving image and a memory budget.
    max_model_mb: float | None = Field(default=None, gt=0.0)
    # Refuse a model whose predictions cannot be attributed to features. Drops
    # anything scoring below `min_explainability` on the tiered scale in
    # `core/explain.py` — 1.0 native importances, 0.8 SHAP, 0.5 permutation.
    min_explainability: float | None = Field(default=None, ge=0.0, le=1.0)
    # A floor on the primary metric. A bake-off where nothing clears the bar
    # should say so rather than crown the least bad candidate.
    min_performance: float | None = None


class SelectConfig(BaseModel):
    """Cross-family model selection, consumed by ``pipeline/select.py``.

    Off by default. Selection tunes and profiles *every* eligible candidate
    family, so turning it on multiplies the cost of a run by the number of
    candidates — that is a decision to make deliberately, not to inherit.
    """

    model_config = _FROZEN

    enabled: bool = False
    # The candidate pool. Empty → every installed model compatible with
    # (task, data.kind), ranked by `auto_priority`. Naming candidates explicitly
    # is how you narrow a bake-off to the families you would actually deploy.
    candidates: list[str] = Field(default_factory=list)
    # Cap on how many candidates are evaluated, applied after gating and in
    # priority order. A guard against a plugin-rich install turning one command
    # into forty fits.
    max_candidates: int = Field(default=8, ge=1)
    objective: SelectObjective = "tolerance"
    constraints: ConstraintConfig = Field(default_factory=ConstraintConfig)
    # `objective: tolerance` — how much primary-metric difference counts as noise.
    # None → one standard error of the CV mean, computed per candidate, which
    # adapts to how variable the data actually is. An explicit value (0.01) is a
    # fixed band in metric units.
    tolerance: float | None = Field(default=None, ge=0.0)
    # `objective: weighted` — normalized 0-1 per-criterion weights. Need not sum
    # to 1; they are normalized. Ignored under `objective: tolerance`.
    weights: dict[str, float] = Field(default_factory=dict)
    # Candidates evaluated concurrently, as separate processes. 1 → sequential,
    # which is the default because a bake-off on one GPU is not made faster by
    # running four fits on it at once.
    max_workers: int = Field(default=1, ge=1)
    # Measure inference latency and artifact size for each candidate. On by
    # default: two of the five selection criteria are unavailable without it, and
    # it costs a few hundred forward passes.
    profile: bool = True
    # Rows sampled for the latency benchmark. Enough to be stable, small enough
    # not to dominate the run.
    profile_samples: int = Field(default=128, ge=1)

    @model_validator(mode="after")
    def _check_weights(self) -> SelectConfig:
        known = {"performance", "latency", "cost", "explainability", "maintainability"}
        unknown = sorted(set(self.weights) - known)
        if unknown:
            raise ValueError(
                f"select.weights has unknown criteria {unknown}. Known: {sorted(known)}"
            )
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("select.weights must be non-negative")
        if self.objective == "weighted" and not self.weights:
            raise ValueError(
                "select.objective is 'weighted' but select.weights is empty — "
                "give at least one criterion a weight, or use objective: tolerance"
            )
        if self.weights and sum(self.weights.values()) <= 0:
            raise ValueError("select.weights must not sum to zero")
        return self


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
    select: SelectConfig = Field(default_factory=SelectConfig)
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

    @model_validator(mode="after")
    def _check_corrupt_policy_is_safe_for_the_strategy(self) -> ExperimentConfig:
        """Refuse ``on_corrupt: raise`` on an explicitly distributed run.

        Raising on a bad sample is a *content-dependent* abort: whether it fires
        depends on which samples a given rank happened to draw. Under DDP that is
        a rank-divergent abort — one rank stops, the others reach the next
        collective and wait forever. The failure presents as a hang with no error
        message, which is the worst diagnostic outcome available.

        Only an *explicit* strategy is refused. ``"auto"`` with more than one
        device is warned about instead: Lightning decides what to do there, and a
        config validator that guessed would refuse valid single-process runs.
        That asymmetry is the honest limit of what this layer can know.
        """
        if self.data.integrity.on_corrupt != "raise":
            return self
        strategy = str(self.runtime.strategy).lower()
        if strategy.startswith(("ddp", "fsdp", "deepspeed")):
            raise ValueError(
                f"data.integrity.on_corrupt: 'raise' is unsafe with runtime.strategy: "
                f"'{self.runtime.strategy}'. Whether it fires depends on which samples a "
                "rank drew, so one rank aborts while the others block on the next "
                "collective -- a hang, not an error. Use 'substitute' (the default), which "
                "keeps the batch count identical on every rank, and run `mlf materialize` "
                "if you want a corpus to fail loudly before training starts."
            )
        if strategy == "auto" and self.runtime.devices != 1:
            log.warning(
                "data.integrity.on_corrupt: 'raise' with runtime.devices=%r. If Lightning "
                "selects a distributed strategy, a corrupt sample will abort one rank and "
                "hang the rest. Prefer 'substitute' unless this is a single-device run.",
                self.runtime.devices,
            )
        return self

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
