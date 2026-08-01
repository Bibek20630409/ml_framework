"""
pipeline/tune.py
────────────────
The hyperparameter search driver. Replaces ``pipeline/hpo.py``, which had three
problems this module exists to fix:

1. **Its space was hardcoded to the MLP's shape.** Running ``mlf hpo`` on a CNN
   tuned ``hidden_dims`` — a parameter that model does not have — and reported a
   "best" result that meant nothing. Spaces are now *declared by the plugin and
   the backend*, so a model is tuned on its own knobs or not at all.
2. **Its objective hardcoded ``val/loss`` from ``trainer.callback_metrics``**, an
   object no non-Lightning backend produces. The objective now reads
   ``cfg.tune.metric or TaskSpec.primary_metric`` out of ``FitResult.val_metrics``,
   a plain dict, with the direction coming from the task table.
3. **It printed the winner for copy-paste.** The result is now applied:
   :func:`tune` hands back a config, ``train()`` fits it, and the tuned values land
   in ``bundle/config.json``, ``hpo.json`` and ``manifest.hpo``.

**Applying a trial is exactly ``config.with_overrides(values)``** — the mechanism
the config layer already had and already tested. That is the whole reason search
spaces are keyed by dotted config paths.

The effective space is ``backend.search_space() | model.search_space |
tune.overrides``, in that order, so ``lr`` and ``batch_size`` are declared once on
the Lightning backend rather than repeated in every neural plugin, a model that
disagrees about one of them wins, and a YAML ``tune.overrides`` block can narrow
any of it.

**Nothing here imports an Optuna integration package.** Pruning is per-backend,
reached through ``backend.trial_hooks(trial)``; this module knows only that hooks
exist.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..backends.base import resolve_budget
from ..config import ExperimentConfig
from ..config.defaults import TuneBudget, budget_for
from ..core.protocols import Budget, ParamSpec, RunContext
from ..core.registry import get_backend, validate_combination
from ..core.task import get_task_spec
from ..core.types import FrameworkError

log = logging.getLogger(__name__)

HPO_FILE = "hpo.json"


class TuningError(FrameworkError):
    """A search could not be run (bad space, unusable objective)."""


@dataclass(frozen=True, slots=True)
class TuneResult:
    """The outcome of a search, and the config that won it."""

    config: ExperimentConfig
    best_params: dict[str, Any] = field(default_factory=dict)
    best_value: float | None = None
    metric: str = ""
    direction: str = "max"
    n_trials: int = 0
    n_pruned: int = 0
    elapsed: float = 0.0
    trials: list[dict[str, Any]] = field(default_factory=list)
    space: dict[str, str] = field(default_factory=dict)
    # Why a search did not happen, when one did not. `None` means it ran.
    skipped: str | None = None
    # Set when `tune.refit: reuse` keeps the winning trial's fitted estimator.
    estimator: Any | None = None

    @property
    def ran(self) -> bool:
        return self.skipped is None

    def to_dict(self) -> dict[str, Any]:
        """The ``hpo.json`` payload — and ``manifest.hpo``.

        Deliberately records the *space* alongside the winner: "max_depth=6" is
        uninterpretable without knowing it was drawn from [3, 12], and a run whose
        best value sits at a boundary is telling you the range was wrong.
        """
        return {
            "ran": self.ran,
            "skipped": self.skipped,
            "metric": self.metric,
            "direction": self.direction,
            "best_value": self.best_value,
            "best_params": self.best_params,
            "n_trials": self.n_trials,
            "n_pruned": self.n_pruned,
            "elapsed_seconds": round(self.elapsed, 2),
            "space": self.space,
            "trials": self.trials,
        }


# ── Space assembly ────────────────────────────────────────
def effective_space(config: ExperimentConfig) -> dict[str, ParamSpec]:
    """``model | backend | tune.overrides``, keyed by dotted config path.

    The merge order is the precedence: a backend may not silently shadow a
    plugin's own knob, and the user's ``tune.overrides`` wins over both.
    """
    spec = validate_combination(config.task, config.data.kind, config.model.name)
    backend = get_backend(spec.backend)

    space: dict[str, ParamSpec] = {}
    # Backend first, model second: the backend declares the knobs that belong to
    # the *loop* (lr, batch_size) once instead of every neural plugin repeating
    # them, and a model that has an opinion about one of them overrides it. The
    # specific beats the general — `nlp.hf_text` narrows lr to the fine-tuning band
    # because the backend's from-scratch range would spend most trials destroying
    # a pretrained encoder.
    space.update(dict(backend.search_space()))
    space.update(dict(spec.search_space))
    space.update(_coerce_overrides(config.tune.overrides))
    return space


def _coerce_overrides(overrides: Mapping[str, Any]) -> dict[str, ParamSpec]:
    """Turn a YAML ``tune.overrides`` block into :class:`ParamSpec` objects.

    YAML cannot hold a dataclass, so the block is written as plain data and
    reconstructed here::

        tune:
          overrides:
            model.params.max_depth: {type: int, low: 3, high: 8}
            fit.params.learning_rate: {type: float, low: 0.05, high: 0.2, log: true}
            model.params.tree_method: {type: categorical, choices: [hist, exact]}
            fit.params.subsample: 0.9        # a bare value pins it (Const)
    """
    from ..core.protocols import Categorical, Const, Float, Int

    out: dict[str, ParamSpec] = {}
    for path, raw in dict(overrides).items():
        if not isinstance(raw, Mapping):
            out[path] = Const(raw)  # a bare value pins the parameter
            continue
        spec = dict(raw)
        kind = str(spec.pop("type", "")).lower()
        try:
            if kind == "float":
                out[path] = Float(**spec)
            elif kind == "int":
                out[path] = Int(**spec)
            elif kind == "categorical":
                out[path] = Categorical(tuple(spec["choices"]))
            elif kind == "const":
                out[path] = Const(spec["value"])
            else:
                raise TuningError(
                    f"tune.overrides['{path}']: unknown type '{kind}'. "
                    f"Use float | int | categorical | const, or a bare value to pin it."
                )
        except TuningError:
            raise
        except (TypeError, KeyError) as exc:
            raise TuningError(f"tune.overrides['{path}'] is malformed: {exc}") from exc
    return out


def describe_space(space: Mapping[str, ParamSpec]) -> dict[str, str]:
    """A serializable rendering of the space, for ``hpo.json``."""
    return {path: f"{type(p).__name__}{_spec_args(p)}" for path, p in space.items()}


def _spec_args(spec: ParamSpec) -> str:
    from dataclasses import asdict, is_dataclass

    if not is_dataclass(spec):  # pragma: no cover - ParamSpec members are dataclasses
        return ""
    return "(" + ", ".join(f"{k}={v!r}" for k, v in asdict(spec).items()) + ")"


# ── Budget resolution ─────────────────────────────────────
def resolve_tune_budget(config: ExperimentConfig, backend_name: str) -> TuneBudget:
    """The per-backend default, with the user's explicit settings layered on.

    ``TuneConfig`` carries non-None defaults, so "did the user set this?" cannot be
    read off the value alone. The rule is deliberate and documented: a value that
    differs from the schema default is treated as intentional. The alternative —
    making every ``tune`` field ``None`` by default — would push that ambiguity
    into the YAML, where it is worse.
    """
    from ..config.schema import TuneConfig

    schema_default = TuneConfig()
    budget = budget_for(backend_name)
    return budget.with_overrides(
        max_trials=(
            config.tune.max_trials if config.tune.max_trials != schema_default.max_trials else None
        ),
        max_seconds=(
            config.tune.max_seconds
            if config.tune.max_seconds != schema_default.max_seconds
            else None
        ),
    )


def _trial_budget(config: ExperimentConfig, budget: TuneBudget) -> Budget:
    """The fit budget for a single trial: the configured one, capped.

    Without the per-trial epoch cap a single Lightning trial can consume the whole
    wall budget and the "search" degenerates to one sample.
    """
    base = resolve_budget(config)
    max_epochs = base.max_epochs
    if budget.trial_max_epochs is not None:
        max_epochs = min(max_epochs or budget.trial_max_epochs, budget.trial_max_epochs)
    return Budget(
        max_epochs=max_epochs,
        max_seconds=base.max_seconds,
        patience=base.patience,
    )


# ── The driver ────────────────────────────────────────────
def tune(
    config: ExperimentConfig,
    *,
    bundle_factory: Any | None = None,
    output_dir: str | Path | None = None,
) -> TuneResult:
    """Search the effective space and return the winning config.

    ``bundle_factory`` lets the caller supply already-built data. Trials that do
    not touch ``data.*`` can reuse one bundle, which is the difference between
    re-reading a CSV thirty times and reading it once; a trial that *does* change
    the data block rebuilds, so the optimization cannot silently produce a model
    tuned against the wrong preprocessing.
    """
    task_spec = get_task_spec(config.task)
    spec = validate_combination(config.task, config.data.kind, config.model.name)
    backend_name = spec.backend
    metric = config.tune.metric or task_spec.primary_metric
    direction = task_spec.direction

    if not config.tune.enabled:
        return _skipped(config, "tune.enabled is false", metric, direction)

    space = effective_space(config)
    if not space and spec.suggest is None:
        # Not a failure: a plugin may legitimately declare nothing worth tuning
        # (the CNN deliberately does not tune its backbone).
        log.info("no search space for model '%s' — skipping tuning", spec.name)
        return _skipped(config, "the merged search space is empty", metric, direction)

    try:
        import optuna
    except ImportError:
        # Tuning is on by default, so a missing optional extra must not fail a
        # training run. It degrades to a plain fit and says so, loudly enough to
        # notice and quietly enough not to block.
        log.warning(
            "optuna is not installed — training without tuning. "
            "Install it with: pip install 'ml-framework[hpo]'"
        )
        return _skipped(config, "optuna is not installed", metric, direction)

    budget = resolve_tune_budget(config, backend_name)
    log.info(
        "tuning %s/%s on '%s' (%s): %d trials, %.0fs budget, %d parameters",
        spec.name,
        backend_name,
        metric,
        direction,
        budget.max_trials,
        budget.max_seconds,
        len(space),
    )

    out = Path(output_dir or config.runtime.output_dir)
    cache = _BundleCache(bundle_factory)
    backend = get_backend(backend_name)
    trial_budget = _trial_budget(config, budget)
    records: list[dict[str, Any]] = []

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize" if direction == "max" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=config.runtime.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    def objective(trial: Any) -> float:
        values = _suggest(trial, space, spec, config)
        trial_cfg = config.with_overrides(values)
        bundle = cache.get(trial_cfg)
        run = RunContext(
            output_dir=out / "trials" / f"trial_{trial.number}",
            seed=config.runtime.seed,
            budget=trial_budget,
            accelerator=config.runtime.accelerator,
            devices=config.runtime.devices,
            precision=config.runtime.precision,
            deterministic=config.runtime.deterministic,
            trial=trial,
        )
        result = backend.fit(spec, bundle, trial_cfg, run=run)
        value = result.metric(metric)
        if value is None:
            raise TuningError(
                f"backend '{backend_name}' did not report '{metric}'. "
                f"It reported: {sorted(result.val_metrics)}. "
                f"Set tune.metric to one of those, or fix the backend."
            )
        trial.set_user_attr("values", values)
        trial.set_user_attr("metrics", dict(result.val_metrics))
        if config.tune.refit == "reuse":
            cache.keep_estimator(trial.number, result.estimator)
        records.append(
            {"number": trial.number, "value": value, "params": values, "state": "COMPLETE"}
        )
        return value

    started = time.perf_counter()
    # No `catch=`: Optuna already marks a `TrialPruned` trial PRUNED and moves on,
    # and swallowing anything else would turn a genuine bug into a quietly worse
    # model. A trial that raises should stop the study and say why.
    study.optimize(objective, n_trials=budget.max_trials, timeout=budget.max_seconds)
    elapsed = time.perf_counter() - started

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        log.warning("no trial completed — training with the configured parameters")
        return _skipped(config, "no trial completed", metric, direction, elapsed=elapsed)

    best_values: dict[str, Any] = dict(study.best_trial.user_attrs.get("values", {}))
    pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
    for t in study.trials:
        if t.state.name != "COMPLETE":
            records.append(
                {
                    "number": t.number,
                    "value": None,
                    "params": t.user_attrs.get("values", {}),
                    "state": t.state.name,
                }
            )

    log.info(
        "best %s=%.4f after %d trials (%d pruned) in %.0fs: %s",
        metric,
        study.best_value,
        len(study.trials),
        pruned,
        elapsed,
        best_values,
    )

    return TuneResult(
        config=config.with_overrides(best_values),
        best_params=best_values,
        best_value=float(study.best_value),
        metric=metric,
        direction=direction,
        n_trials=len(study.trials),
        n_pruned=pruned,
        elapsed=elapsed,
        trials=sorted(records, key=lambda r: r["number"]),
        space=describe_space(space),
        estimator=(
            cache.estimator(study.best_trial.number) if config.tune.refit == "reuse" else None
        ),
    )


def _skipped(
    config: ExperimentConfig,
    reason: str,
    metric: str,
    direction: str,
    *,
    elapsed: float = 0.0,
) -> TuneResult:
    log.info("tuning skipped: %s", reason)
    return TuneResult(
        config=config, metric=metric, direction=direction, skipped=reason, elapsed=elapsed
    )


def _suggest(
    trial: Any, space: Mapping[str, ParamSpec], spec: Any, config: ExperimentConfig
) -> dict[str, Any]:
    """One trial's values, declarative space first then the plugin's escape hatch.

    ``ModelSpec.suggest`` exists because a conditional space — "n_layers, then that
    many widths" — cannot be written as a flat mapping. It runs *after* the
    declarative space and overwrites it, which is what makes it an escape hatch
    rather than a second, competing mechanism.
    """
    values = {path: p.suggest(trial, path.rsplit(".", 1)[-1]) for path, p in space.items()}
    if spec.suggest is not None:
        values.update(dict(spec.suggest(trial, config)))
    return values


class _BundleCache:
    """One built bundle per distinct data configuration.

    Keyed on the blocks that actually change the data — ``task``, the whole
    ``data`` block, and the seed. A trial that only moves ``max_depth`` reuses the
    bundle; one that moves ``data.params.imbalance_strategy`` gets a fresh one.
    Keying on the *whole* config would defeat the cache, and caching
    unconditionally would tune a model against preprocessing it never saw.
    """

    def __init__(self, factory: Any | None = None) -> None:
        self._factory = factory
        self._cache: dict[str, Any] = {}
        self._estimators: dict[int, Any] = {}

    @staticmethod
    def _key(config: ExperimentConfig) -> str:
        import json

        return json.dumps(
            {
                "task": config.task,
                "data": config.data.model_dump(),
                "seed": config.runtime.seed,
            },
            sort_keys=True,
            default=str,
        )

    def get(self, config: ExperimentConfig) -> Any:
        key = self._key(config)
        if key not in self._cache:
            if self._factory is not None:
                self._cache[key] = self._factory(config)
            else:
                from ..data import build_bundle

                self._cache[key] = build_bundle(config)
        return self._cache[key]

    def keep_estimator(self, number: int, estimator: Any) -> None:
        self._estimators[number] = estimator

    def estimator(self, number: int) -> Any | None:
        return self._estimators.get(number)


__all__ = [
    "HPO_FILE",
    "TuneResult",
    "TuningError",
    "describe_space",
    "effective_space",
    "resolve_tune_budget",
    "tune",
]
