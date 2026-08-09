"""
pipeline/select.py
──────────────────
Cross-family model selection: **tune every candidate, then choose on five
criteria**, not on the score alone.

What existed before this module: one model name in the config, tuned within
itself. `select_model()` in ``config/defaults.py`` picked that name from a rule
table by data kind and row count — a good default, and not a comparison. Nothing
in the framework ever fit an XGBoost model and an MLP against the same data and
said which one to ship.

The shape of the answer:

    gate  →  tune each survivor  →  profile each  →  decide  →  return a config

    ┌───────────────────────────────────────────────────────────┐
    │ candidates = models_for(task, data_kind)                   │  compatible
    │            ∩ installed  ∩ ModelRule.fits(n_rows)           │  + installed
    │            ∩ declared-capability constraints               │  + eligible
    ├───────────────────────────────────────────────────────────┤
    │ for each (sequential, or ProcessPoolExecutor):             │
    │     tuned  = tune(config[model=c])   ← its own space       │
    │     score  = cross-validated primary metric ± std          │
    │     profile= latency · artifact bytes · explainability     │
    ├───────────────────────────────────────────────────────────┤
    │ disqualify on measured hard constraints                    │
    │ decide among survivors → winning config                    │
    └───────────────────────────────────────────────────────────┘

Three design decisions worth stating, because each had a plausible alternative:

**1. ``select`` returns a config, exactly like ``tune``.** It does not return a
fitted model and does not write a bundle. That keeps ``train()`` at one
bundle-writing path whether selection ran, tuning ran, or neither did — the same
property that made ``tune`` return a config in the first place. The winner is
refit at full budget by the existing code below it.

**2. Gating happens in two phases, because it must.** Compatibility, availability
and data-size rules are answerable before training and are used to *skip* work.
Latency and artifact size cannot be known until a model exists, so they
disqualify after the fact. A candidate that trains and then fails its latency
budget is reported with the number it missed by — telling someone their model is
2 ms too slow is useful; silently omitting it is not.

**3. The default decision rule never trades accuracy for speed silently.** See
:func:`decide_tolerance`. The weighted alternative exists for people who need an
auditable weight table, and is opt-in.

Cost: a bake-off is ``n_candidates × tuning_budget``. That is why
``select.enabled`` defaults to False and ``max_candidates`` defaults to 8.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..core.profile import (
    CostProfile,
    LatencyProfile,
    MaintainabilityProfile,
    ModelProfile,
    fold_stability,
    measure_cost,
    measure_latency,
)
from ..core.protocols import RunContext
from ..core.registry import get_backend, models_for, validate_combination
from ..core.task import get_task_spec
from ..core.types import FrameworkError
from .tune import TuneResult, tune

log = logging.getLogger(__name__)

SELECTION_FILE = "selection.json"

# Criteria the weighted objective knows, and the direction each improves in.
# `+1` means larger is better after normalization.
CRITERIA: tuple[str, ...] = (
    "performance",
    "latency",
    "cost",
    "explainability",
    "maintainability",
)


class SelectionError(FrameworkError):
    """A bake-off could not produce a winner."""


# ── Results ───────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CandidateReport:
    """One candidate's outcome — measured, or the reason it has no measurements.

    Carries no estimator and no config object, only plain data, because these
    cross a process boundary when ``select.max_workers > 1`` and an estimator is
    frequently unpicklable. The winning candidate is *refit* from its parameters
    rather than shipped back through a pipe.
    """

    model: str
    backend: str = ""
    profile: ModelProfile | None = None
    best_params: dict[str, Any] = field(default_factory=dict)
    n_trials: int = 0
    tune_skipped: str | None = None
    elapsed: float = 0.0
    # Set when the candidate never produced a usable profile. A skipped candidate
    # was never trained; a disqualified one was trained and then failed a
    # constraint. Both must be reportable, and they are not the same thing.
    skipped: str | None = None
    disqualified: str | None = None

    @property
    def eligible(self) -> bool:
        return self.skipped is None and self.disqualified is None and self.profile is not None

    @property
    def score(self) -> float:
        return self.profile.score if self.profile else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "eligible": self.eligible,
            "skipped": self.skipped,
            "disqualified": self.disqualified,
            "best_params": self.best_params,
            "n_trials": self.n_trials,
            "tune_skipped": self.tune_skipped,
            "elapsed_seconds": round(self.elapsed, 2),
            "profile": self.profile.to_dict() if self.profile else None,
        }


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """The winner, why it won, and every candidate that did not."""

    config: ExperimentConfig
    tuning: TuneResult
    winner: str = ""
    reason: str = ""
    objective: str = "tolerance"
    metric: str = ""
    direction: str = "max"
    reports: list[CandidateReport] = field(default_factory=list)
    elapsed: float = 0.0
    skipped: str | None = None

    @property
    def ran(self) -> bool:
        return self.skipped is None

    def to_dict(self) -> dict[str, Any]:
        """The ``selection.json`` payload — and ``manifest.selection``.

        Records every candidate, not just the winner. "We chose XGBoost" is not
        an answer to "why not the neural net?"; the table of what each one scored
        and cost is, and it is the artifact an architecture review actually wants.
        """
        return {
            "ran": self.ran,
            "skipped": self.skipped,
            "winner": self.winner,
            "reason": self.reason,
            "objective": self.objective,
            "metric": self.metric,
            "direction": self.direction,
            "elapsed_seconds": round(self.elapsed, 2),
            "n_candidates": len(self.reports),
            "n_eligible": sum(1 for r in self.reports if r.eligible),
            "candidates": [r.to_dict() for r in self.reports],
        }

    def table(self) -> str:
        """A fixed-width summary, for the CLI and the log.

        ASCII only — the same constraint the rest of the framework's user-facing
        strings carry, because a Windows console in cp1252 cannot render a box
        character and a crash in the *reporting* of a successful bake-off is an
        absurd way to lose one.
        """
        header = (
            f"{'model':<18} {'score':>9} {'+/-':>7} {'p95 ms':>8} {'MB':>7} {'expl':>5}  status"
        )
        lines = [header, "-" * len(header)]
        for r in sorted(self.reports, key=_report_sort_key):
            p = r.profile
            status = "WINNER" if r.model == self.winner else _status(r)
            if p is None:
                lines.append(
                    f"{r.model:<18} {'-':>9} {'-':>7} {'-':>8} {'-':>7} {'-':>5}  {status}"
                )
                continue
            lines.append(
                f"{r.model:<18} {_fmt(p.score):>9} {_fmt(p.score_std):>7} "
                f"{_fmt(p.latency.p95_ms) if p.latency.measured else '-':>8} "
                f"{_fmt(p.cost.artifact_mb):>7} {p.explainability:>5.2f}  {status}"
            )
        return "\n".join(lines)


def _status(report: CandidateReport) -> str:
    if report.skipped:
        return f"skipped: {report.skipped}"
    if report.disqualified:
        return f"rejected: {report.disqualified}"
    return "ok"


def _report_sort_key(report: CandidateReport) -> tuple[int, float, str]:
    # Eligible first, then by score descending, then by name for determinism.
    score = report.score
    return (0 if report.eligible else 1, -score if math.isfinite(score) else 0.0, report.model)


def _fmt(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.4f}"


# ── Candidate gating ──────────────────────────────────────
def candidate_models(config: ExperimentConfig, *, n_rows: int = 0) -> list[str]:
    """The families worth training, in priority order.

    Three filters, cheapest and most decisive first:

    1. **Compatibility and availability** — ``models_for`` already answers "can
       this model consume this task and data kind, and is it installed". Reusing
       it rather than re-deriving the rule is what keeps a third-party plugin
       automatically eligible for a bake-off.
    2. **Data-size rules** — ``MODEL_RULES`` in ``config/defaults.py`` knows that
       an LSTM on 80 rows is not a candidate, it is a mistake. Only consulted
       when ``n_rows`` is known.
    3. **Declared capability** — a hard ``min_explainability`` above the
       permutation tier drops models that could never have native importances,
       before spending a fit on them.

    An explicit ``select.candidates`` list overrides discovery entirely but still
    passes through validation, so a typo or an uninstalled extra is an error
    naming the model rather than a silently shorter bake-off.
    """
    from ..config.defaults import MODEL_RULES
    from ..core.registry import MODELS

    task, kind = config.task, config.data.kind

    if config.select.candidates:
        names = list(dict.fromkeys(config.select.candidates))
        for name in names:
            # Raises IncompatibleCombinationError or MissingExtraError, both of
            # which name the model and say what to do.
            validate_combination(task, kind, name)
        return names

    names = [spec.name for spec in models_for(task, kind)]
    if not names:
        raise SelectionError(
            f"no installed model supports task '{task}' on '{kind}' data. "
            f"Install an extra, or set select.candidates explicitly."
        )

    if n_rows > 0:
        rules = {r.model: r for r in MODEL_RULES.get((kind, task), ())}
        kept = []
        for name in names:
            rule = rules.get(name)
            if rule is not None and not rule.fits(n_rows):
                log.info("candidate '%s' skipped: %s (n_rows=%d)", name, rule.reason, n_rows)
                continue
            kept.append(name)
        names = kept or names

    floor = config.select.constraints.min_explainability
    if floor is not None and floor > 0.5:
        # Above the permutation tier, only a model with native importances can
        # qualify. Checking the declaration here saves a full fit per candidate;
        # the measured score still decides, so a wrong declaration costs a
        # candidate an opportunity but can never let an unexplainable model pass.
        eligible = []
        for name in names:
            spec = MODELS.get_spec(name)
            if spec.capabilities.native_feature_importance:
                eligible.append(name)
            else:
                log.info(
                    "candidate '%s' skipped: declares no native feature importance, and "
                    "min_explainability=%.2f is above the permutation tier",
                    name,
                    floor,
                )
        names = eligible

    if not names:
        raise SelectionError(
            "every candidate was gated out before training. Relax "
            "select.constraints, or name candidates explicitly with select.candidates."
        )
    return names[: config.select.max_candidates]


# ── Evaluating one candidate ──────────────────────────────
def evaluate_candidate(
    config: ExperimentConfig,
    model: str,
    *,
    output_dir: str | Path | None = None,
) -> CandidateReport:
    """Tune, cross-validate and profile one family. Never raises for a bad candidate.

    A bake-off over five families must survive one of them failing — an
    uninstallable extra, a shape the source cannot produce, a library that
    segfaults on this data. Every failure becomes a ``skipped`` report carrying
    the exception text, and the other four candidates still get compared.

    Called in a subprocess when ``select.max_workers > 1``, which is why it takes
    a config and a name rather than pre-built objects, and returns plain data.
    """
    started = time.perf_counter()
    out = Path(output_dir or config.runtime.output_dir) / "candidates" / _slug(model)

    try:
        candidate_cfg = _candidate_config(config, model)
    except Exception as exc:
        return CandidateReport(
            model=model, skipped=f"{type(exc).__name__}: {exc}", elapsed=_since(started)
        )

    try:
        spec = validate_combination(candidate_cfg.task, candidate_cfg.data.kind, model)
        backend = get_backend(spec.backend)
    except Exception as exc:
        return CandidateReport(
            model=model, skipped=f"{type(exc).__name__}: {exc}", elapsed=_since(started)
        )

    log.info("candidate '%s' (%s): tuning", model, spec.backend)
    try:
        tuned = tune(candidate_cfg, output_dir=out)
    except Exception as exc:
        log.warning("candidate '%s' failed during tuning: %s", model, exc)
        return CandidateReport(
            model=model,
            backend=spec.backend,
            skipped=f"tuning failed: {type(exc).__name__}: {exc}",
            elapsed=_since(started),
        )

    try:
        profile = profile_candidate(tuned.config, spec, backend, output_dir=out)
    except Exception as exc:
        log.warning("candidate '%s' failed during evaluation: %s", model, exc)
        return CandidateReport(
            model=model,
            backend=spec.backend,
            best_params=dict(tuned.best_params),
            n_trials=tuned.n_trials,
            tune_skipped=tuned.skipped,
            skipped=f"evaluation failed: {type(exc).__name__}: {exc}",
            elapsed=_since(started),
        )

    report = CandidateReport(
        model=model,
        backend=spec.backend,
        profile=profile,
        best_params=dict(tuned.best_params),
        n_trials=tuned.n_trials,
        tune_skipped=tuned.skipped,
        elapsed=_since(started),
    )
    log.info(
        "candidate '%s': %s=%.4f +/- %.4f, p95=%s, %.2f MB, explainability=%.2f (%s)",
        model,
        profile.primary_metric,
        profile.score,
        profile.score_std,
        f"{profile.latency.p95_ms:.2f} ms" if profile.latency.measured else "not measured",
        profile.cost.artifact_mb,
        profile.explainability,
        profile.explain_method,
    )
    return report


def profile_candidate(
    config: ExperimentConfig,
    spec: Any,
    backend: Any,
    *,
    output_dir: Path,
) -> ModelProfile:
    """Cross-validate the tuned config, then measure the other four criteria.

    Performance comes from folds rather than from a single holdout because the
    comparison between candidates is the whole point, and two candidates measured
    on one split differ partly by which one suited that split. The fold spread is
    kept, not just the mean: it *is* the tolerance the default decision rule uses
    and the stability term maintainability reads.

    The profiled estimator is the **last fold's**, deliberately. Latency and size
    are properties of the architecture and its hyperparameters, not of which rows
    it saw, so refitting once more on the full data to measure them would double
    the cost of a bake-off to change the numbers in the fourth decimal place.
    """
    task_spec = get_task_spec(config.task)
    metric = config.tune.metric or task_spec.primary_metric

    folds, estimator, fit_seconds, failures, last_bundle = _run_folds(
        config, spec, backend, output_dir
    )
    if not folds:
        raise SelectionError(f"no fold produced a score for '{spec.name}' — all {failures} failed")

    values = [f[metric] for f in folds if metric in f and math.isfinite(f[metric])]
    if not values:
        raise SelectionError(
            f"no fold reported '{metric}' for '{spec.name}'. Reported: {sorted(folds[0])}"
        )
    mean = float(statistics.fmean(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0

    aggregate: dict[str, float] = {}
    for name in folds[0]:
        series = [f[name] for f in folds if name in f and math.isfinite(f[name])]
        if series:
            aggregate[f"cv_{name}_mean"] = float(statistics.fmean(series))
            aggregate[f"cv_{name}_std"] = (
                float(statistics.pstdev(series)) if len(series) > 1 else 0.0
            )

    latency = LatencyProfile(error="profiling disabled")
    cost = CostProfile(error="profiling disabled")
    explain_score, explain_method = 0.0, "none"
    if config.select.profile and estimator is not None:
        latency = _measure_latency(
            estimator, last_bundle, max_samples=config.select.profile_samples
        )
        cost = measure_cost(backend, estimator)
        explain_score, explain_method = _measure_explainability(
            estimator, spec, last_bundle, output_dir, seed=config.runtime.seed
        )

    return ModelProfile(
        model=spec.name,
        backend=spec.backend,
        primary_metric=metric,
        score=mean,
        score_std=std,
        n_folds=len(values),
        metrics=aggregate,
        latency=latency,
        cost=cost,
        explainability=explain_score,
        explain_method=explain_method,
        maintainability=MaintainabilityProfile(
            fit_seconds=fit_seconds,
            fold_stability=fold_stability(mean, std),
            fold_failures=failures,
            n_folds=len(values) + failures,
        ),
    )


def _run_folds(
    config: ExperimentConfig, spec: Any, backend: Any, output_dir: Path
) -> tuple[list[dict[str, float]], Any, float, int, Any]:
    """Fit every fold; return per-fold scores, the last estimator and the cost.

    A fold that raises is counted rather than fatal: three good folds and one
    failure is a real and *reportable* property of a candidate — it is the
    maintainability signal — whereas aborting would silently prefer the fragile
    model that happened not to fail today.
    """
    from ..data.builders import build_cv_bundles

    task_spec = get_task_spec(config.task)
    folds = max(2, config.data.split.folds or 3)
    cv_config = config.with_overrides({"data.split.folds": folds})

    scores: list[dict[str, float]] = []
    estimator: Any = None
    bundle: Any = None
    failures = 0
    started = time.perf_counter()

    for i, fold in enumerate(build_cv_bundles(cv_config)):
        run = RunContext(
            output_dir=output_dir / "cv" / f"fold_{i}",
            seed=config.runtime.seed,
            budget=_budget(config),
            accelerator=config.runtime.accelerator,
            devices=config.runtime.devices,
            precision=config.runtime.precision,
            strategy=config.runtime.strategy,
            deterministic=config.runtime.deterministic,
        )
        try:
            result = backend.fit(spec, fold, cv_config, run=run)
            predictions = backend.predict_split(result.estimator, fold, "test")
            computed = task_spec.compute(predictions.y_true, predictions.y_pred, predictions.y_prob)
            scores.append({str(k): float(v) for k, v in computed.items()})
            estimator, bundle = result.estimator, fold
        except Exception as exc:
            failures += 1
            log.warning("fold %d of '%s' failed: %s", i, spec.name, exc)

    return scores, estimator, _since(started), failures, bundle


def _measure_latency(estimator: Any, bundle: Any, *, max_samples: int) -> LatencyProfile:
    """Time the estimator on the fold's test inputs, whatever shape they are."""
    inputs = _predict_inputs(bundle)
    if inputs is None:
        return LatencyProfile(error="this data kind exposes no row-indexable inputs")
    return measure_latency(estimator, inputs, max_samples=max_samples)


def _predict_inputs(bundle: Any) -> Any:
    """The test split's model inputs, or ``None`` for a payload with no rows.

    Forecasting is the ``None`` case and is not an oversight: a forecaster is
    called with a horizon, not with rows, so "milliseconds per row" is not a
    quantity it has. Reporting no measurement is the honest answer; inventing one
    would put a meaningless number into a constraint comparison.
    """
    split = getattr(bundle, "test", None)
    if split is None:
        return None
    return getattr(split, "x", None)


def _measure_explainability(
    estimator: Any, spec: Any, bundle: Any, output_dir: Path, *, seed: int
) -> tuple[float, str]:
    from ..core.explain import feature_importance

    split = getattr(bundle, "test", None)
    schema = getattr(bundle, "schema", None)
    features = list(getattr(schema, "feature_names", []) or []) or None
    importance = feature_importance(
        estimator,
        features=features,
        x=getattr(split, "x", None),
        y=getattr(split, "y", None),
        prefer_native=spec.capabilities.native_feature_importance,
        seed=seed,
    )
    if importance.method != "none":
        importance.write(output_dir)
    return importance.score, importance.method


# ── Constraints ───────────────────────────────────────────
def apply_constraints(
    reports: list[CandidateReport], config: ExperimentConfig
) -> list[CandidateReport]:
    """Disqualify candidates that miss a hard limit, recording by how much.

    Returns a new list — the reports are frozen, so a disqualified one is
    replaced rather than mutated. The violation text carries the measured value
    and the limit, because "too slow" is not actionable and "p95 24.10 ms exceeds
    the 20.00 ms budget" is.
    """
    from dataclasses import replace

    limits = config.select.constraints
    out: list[CandidateReport] = []
    for report in reports:
        if not report.eligible:
            out.append(report)
            continue
        violation = _violation(report.profile, limits, config)
        out.append(replace(report, disqualified=violation) if violation else report)
        if violation:
            log.info("candidate '%s' disqualified: %s", report.model, violation)
    return out


def _violation(profile: Any, limits: Any, config: ExperimentConfig) -> str | None:
    direction = get_task_spec(config.task).direction

    if limits.max_latency_p95_ms is not None:
        if not profile.latency.measured:
            return (
                f"latency could not be measured ({profile.latency.error or 'unknown'}), and "
                f"max_latency_p95_ms={limits.max_latency_p95_ms:g} was requested"
            )
        if profile.latency.p95_ms > limits.max_latency_p95_ms:
            return (
                f"p95 latency {profile.latency.p95_ms:.2f} ms exceeds the "
                f"{limits.max_latency_p95_ms:g} ms budget"
            )

    if limits.max_model_mb is not None:
        if profile.cost.error:
            return f"artifact size could not be measured ({profile.cost.error})"
        if profile.cost.artifact_mb > limits.max_model_mb:
            return (
                f"artifact {profile.cost.artifact_mb:.2f} MB exceeds the "
                f"{limits.max_model_mb:g} MB budget"
            )

    if limits.min_explainability is not None:
        if profile.explainability < limits.min_explainability:
            return (
                f"explainability {profile.explainability:.2f} ({profile.explain_method}) "
                f"is below the {limits.min_explainability:.2f} floor"
            )

    if limits.min_performance is not None:
        below = (
            profile.score < limits.min_performance
            if direction == "max"
            else profile.score > limits.min_performance
        )
        if below:
            return (
                f"{profile.primary_metric} {profile.score:.4f} does not clear the "
                f"{limits.min_performance:g} floor ({direction}imize)"
            )
    return None


# ── Decision rules ────────────────────────────────────────
def decide(reports: list[CandidateReport], config: ExperimentConfig) -> tuple[CandidateReport, str]:
    """The winner and the sentence explaining why it won."""
    eligible = [r for r in reports if r.eligible]
    if not eligible:
        raise SelectionError(
            "no candidate survived selection.\n"
            + "\n".join(f"  {r.model}: {_status(r)}" for r in reports)
        )
    direction = get_task_spec(config.task).direction
    if config.select.objective == "weighted":
        return decide_weighted(eligible, config, direction)
    return decide_tolerance(eligible, config, direction)


def decide_tolerance(
    reports: list[CandidateReport], config: ExperimentConfig, direction: str
) -> tuple[CandidateReport, str]:
    """Best score, then the cheapest model statistically tied with it.

    The rule practitioners actually use, made explicit: *take the simplest model
    that is not measurably worse than the best one.* A 0.9012 model that answers
    in 3 ms and a 0.9019 model that answers in 40 ms are the same model as far as
    the data can tell, and the 3 ms one is obviously the one to deploy.

    "Not measurably worse" defaults to one standard error of the best candidate's
    CV mean, which adapts to how variable the data actually is — a noisy dataset
    admits a wider tie, a clean one a narrower. ``select.tolerance`` replaces it
    with a fixed band in metric units when a domain has a view.

    The tie-break order is deliberate and fixed: **latency, then size, then
    explainability, then stability.** Latency leads because it is the constraint
    that turns into a user-visible failure; explainability sits below size
    because it is a tiered approximation while the first two are measurements.
    """
    better = max if direction == "max" else min
    best = better(reports, key=lambda r: r.score)
    tolerance = config.select.tolerance
    if tolerance is None:
        tolerance = _profile(best).score_std_error

    def tied(report: CandidateReport) -> bool:
        gap = best.score - report.score if direction == "max" else report.score - best.score
        return gap <= tolerance + 1e-12

    ties = [r for r in reports if tied(r)]
    winner = min(ties, key=_cheapness)

    if winner.model == best.model:
        reason = (
            f"best {_profile(best).primary_metric} ({best.score:.4f}) among "
            f"{len(reports)} eligible candidates"
        )
        if len(ties) > 1:
            reason += f"; also cheapest of the {len(ties)} within {tolerance:.4f}"
    else:
        band = "std error of the CV mean" if config.select.tolerance is None else "configured"
        reason = (
            f"{winner.model} scores {winner.score:.4f} against {best.model}'s "
            f"{best.score:.4f} — within the {tolerance:.4f} tolerance ({band}) — "
            f"and wins the tie-break on {_deciding_axis(winner, best)}"
        )
    return winner, reason


# The tie-break axes, in the order `_cheapness` applies them, each with the text
# that renders one candidate's value on it. Kept beside `_cheapness` because a
# reader must be able to see that the two agree; they are the same list twice,
# and a reason naming an axis the sort did not use would be worse than no reason.
_AXES: tuple[tuple[str, str, Any], ...] = (
    ("latency", "p95 latency", lambda p: p.latency.p95_ms if p.latency.measured else math.inf),
    ("size", "artifact size", lambda p: p.cost.artifact_mb if not p.cost.error else math.inf),
    ("explainability", "explainability", lambda p: -p.explainability),
    ("stability", "fold stability", lambda p: -p.maintainability.score),
)


def _deciding_axis(winner: CandidateReport, best: CandidateReport) -> str:
    """The first tie-break axis on which ``winner`` actually beat ``best``.

    Reporting the whole vector invites the reading that the winner is better on
    all of it, which it usually is not — a smaller model can win on latency while
    losing on size, and saying "is cheaper" while printing a larger number is how
    a report loses its reader's trust.
    """
    for _, label, key in _AXES:
        if key(_profile(winner)) < key(_profile(best)):
            return f"{label} ({_axis_value(label, winner)} vs {_axis_value(label, best)})"
    return "the candidate ordering"  # pragma: no cover - identical on every axis


def _axis_value(label: str, report: CandidateReport) -> str:
    p = _profile(report)
    if label == "p95 latency":
        return _ms(report)
    if label == "artifact size":
        return "not measured" if p.cost.error else f"{p.cost.artifact_mb:.2f} MB"
    if label == "explainability":
        return f"{p.explainability:.2f} ({p.explain_method})"
    return f"{p.maintainability.score:.2f}"


def _profile(report: CandidateReport) -> ModelProfile:
    """``report.profile``, narrowed.

    Every decision path below runs only on reports that passed ``eligible``,
    which already requires a profile. Stating that here rather than sprinkling
    ``if report.profile`` guards keeps the alternative — a fallback value for a
    profile that cannot be absent — from being written and then silently relied
    on.
    """
    if report.profile is None:  # pragma: no cover - guarded by `eligible`
        raise SelectionError(f"candidate '{report.model}' reached a decision with no profile")
    return report.profile


def _cheapness(report: CandidateReport) -> tuple[Any, ...]:
    """The tie-break key: lower is better on every axis, name last for determinism.

    Built from :data:`_AXES` rather than spelled out, so the sort order and the
    explanation :func:`_deciding_axis` prints cannot drift apart.

    An unmeasured latency sorts last rather than first. A model whose speed is
    unknown must not win a tie *because* it is unknown — that would make a
    measurement failure look like a measurement of zero.
    """
    p = _profile(report)
    return (*(key(p) for _, _, key in _AXES), report.model)


def _ms(report: CandidateReport) -> str:
    p = report.profile
    if p is None or not p.latency.measured:
        return "not measured"
    return f"{p.latency.p95_ms:.2f} ms"


def decide_weighted(
    reports: list[CandidateReport], config: ExperimentConfig, direction: str
) -> tuple[CandidateReport, str]:
    """Highest weighted sum of min-max normalized criteria.

    Every criterion is normalized to [0, 1] **across the candidates in this
    bake-off**, then combined with the configured weights. Normalizing within the
    run rather than against absolute scales is what makes 40 ms and 0.91 addable
    at all; the cost is that the composite has no meaning outside the run that
    produced it, and comparing two bake-offs' composites is not valid. The
    per-criterion normalized values are reported alongside so the arithmetic is
    auditable rather than a black box producing a rank.

    A criterion where every candidate is identical normalizes to 1.0 for all of
    them — it carries no information in this run, so it must not silently break
    the tie in favour of whichever floating-point value came out marginally
    larger.
    """
    weights = {k: float(v) for k, v in config.select.weights.items() if v > 0}
    total_weight = sum(weights.values())
    if total_weight <= 0:  # pragma: no cover - schema validation forbids it
        raise SelectionError("select.weights sum to zero")

    # Negated where smaller is better, so every series is larger-is-better before
    # normalization and the weights all mean the same thing.
    profiles = [_profile(r) for r in reports]
    raw = {
        "performance": [r.score if direction == "max" else -r.score for r in reports],
        "latency": [-(p.latency.p95_ms if p.latency.measured else math.inf) for p in profiles],
        "cost": [-(p.cost.artifact_mb if not p.cost.error else math.inf) for p in profiles],
        "explainability": [p.explainability for p in profiles],
        "maintainability": [p.maintainability.score for p in profiles],
    }
    normalized = {name: _min_max(values) for name, values in raw.items()}

    scores = [
        sum(weights.get(name, 0.0) * normalized[name][i] for name in CRITERIA) / total_weight
        for i in range(len(reports))
    ]
    # `max` returns the *first* maximal element, so visiting candidates in name
    # order makes an exact tie resolve to the lexicographically first name rather
    # than to whichever one the evaluation order happened to produce first.
    order = sorted(range(len(reports)), key=lambda i: reports[i].model)
    best_index = max(order, key=lambda i: scores[i])
    winner = reports[best_index]
    breakdown = ", ".join(
        f"{name}={normalized[name][best_index]:.2f}x{weights[name]:g}"
        for name in CRITERIA
        if name in weights
    )
    reason = (
        f"highest weighted score ({scores[best_index]:.4f}) over {len(reports)} "
        f"eligible candidates: {breakdown}"
    )
    return winner, reason


def _min_max(values: list[float]) -> list[float]:
    """Scale to [0, 1], larger-is-better, with non-finite values pinned to 0."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [0.0] * len(values)
    low, high = min(finite), max(finite)
    if high - low < 1e-12:
        # Every candidate is the same on this axis: it distinguishes nothing, so
        # it must contribute equally rather than amplifying float noise.
        return [1.0 if math.isfinite(v) else 0.0 for v in values]
    return [(v - low) / (high - low) if math.isfinite(v) else 0.0 for v in values]


# ── The driver ────────────────────────────────────────────
def select(
    config: ExperimentConfig,
    *,
    output_dir: str | Path | None = None,
) -> SelectionResult:
    """Run the bake-off and return the winning **config**.

    When ``select.enabled`` is False this is a pass-through that tunes the single
    configured model — same signature, same return type, so ``train()`` has one
    code path whether a bake-off happened or not. That symmetry is the reason
    this returns a :class:`SelectionResult` wrapping a :class:`TuneResult` rather
    than the caller branching on which of the two ran.
    """
    started = time.perf_counter()
    out = Path(output_dir or config.runtime.output_dir)

    if not config.select.enabled:
        tuning = tune(config, output_dir=out)
        return SelectionResult(
            # `tuning.config`, not `config`: the tuned values are carried by the
            # TuneResult, and returning the input here would throw the whole
            # search away one line after running it.
            config=tuning.config,
            tuning=tuning,
            winner=config.model.name,
            metric=config.tune.metric or get_task_spec(config.task).primary_metric,
            direction=get_task_spec(config.task).direction,
            skipped="select.enabled is false",
        )

    task_spec = get_task_spec(config.task)
    metric = config.tune.metric or task_spec.primary_metric
    names = candidate_models(config, n_rows=_n_rows(config))
    log.info(
        "model selection: %d candidates on '%s' (%s), objective=%s, workers=%d — %s",
        len(names),
        metric,
        task_spec.direction,
        config.select.objective,
        config.select.max_workers,
        ", ".join(names),
    )

    reports = _evaluate_all(config, names, out)
    reports = apply_constraints(reports, config)
    winner, reason = decide(reports, config)

    winning_config = _candidate_config(config, winner.model).with_overrides(winner.best_params)
    result = SelectionResult(
        config=winning_config,
        # The winner's search is replayed as a TuneResult so the bundle's
        # `hpo.json` and `manifest.hpo` describe the model that actually shipped,
        # not the search of whichever candidate the config happened to name.
        tuning=TuneResult(
            config=winning_config,
            best_params=dict(winner.best_params),
            best_value=winner.score,
            metric=metric,
            direction=task_spec.direction,
            n_trials=winner.n_trials,
            space={},
            skipped=winner.tune_skipped,
        ),
        winner=winner.model,
        reason=reason,
        objective=config.select.objective,
        metric=metric,
        direction=task_spec.direction,
        reports=reports,
        elapsed=_since(started),
    )
    write_selection(result, out)
    log.info("selected '%s': %s", winner.model, reason)
    log.info("selection summary:\n%s", result.table())
    return result


def _evaluate_all(config: ExperimentConfig, names: list[str], out: Path) -> list[CandidateReport]:
    """Every candidate's report, sequentially or across a process pool.

    Processes rather than threads: a fit is CPU-bound and holds the GIL for most
    of its life, and two Lightning trainers in one interpreter share global state
    (the seed, the logger, the accelerator registry) in ways that make results
    depend on interleaving. Separate interpreters have neither problem.

    Results are reordered back into candidate order after collection, so the
    report table and the tie-breaks are deterministic regardless of which worker
    finished first.
    """
    workers = min(config.select.max_workers, len(names))
    if workers <= 1:
        return [evaluate_candidate(config, name, output_dir=out) for name in names]

    from concurrent.futures import ProcessPoolExecutor, as_completed

    log.info("evaluating %d candidates across %d worker processes", len(names), workers)
    collected: dict[str, CandidateReport] = {}
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(evaluate_candidate, config, name, output_dir=out): name
                for name in names
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    collected[name] = future.result()
                except Exception as exc:
                    # A worker that dies (segfault, OOM kill, unpicklable result)
                    # takes its candidate down, not the bake-off.
                    log.warning("candidate '%s' worker failed: %s", name, exc)
                    collected[name] = CandidateReport(
                        model=name, skipped=f"worker failed: {type(exc).__name__}: {exc}"
                    )
    except Exception as exc:
        # Process pools are refused outright in some environments (restricted
        # sandboxes, frozen executables, a notebook without a __main__ guard).
        # Falling back is strictly better than failing a run over a scheduling
        # detail, and saying so is what stops it being a silent slowdown.
        log.warning("process pool unavailable (%s) — evaluating candidates sequentially", exc)
        return [evaluate_candidate(config, name, output_dir=out) for name in names]

    return [collected[name] for name in names if name in collected]


def write_selection(result: SelectionResult, out: Path) -> Path:
    """Write ``selection.json`` and return its path."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / SELECTION_FILE
    path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


# ── Helpers ───────────────────────────────────────────────
def _candidate_config(config: ExperimentConfig, model: str) -> ExperimentConfig:
    """``config`` retargeted at ``model``.

    ``with_overrides`` clears ``model.params`` when ``model.name`` changes, so the
    new plugin's own defaults are materialized by the config validator rather
    than the previous model's parameters being reinterpreted against a schema
    that has never seen them.

    ``select`` is switched off on the copy: a candidate evaluation must not
    recurse into another bake-off.
    """
    return config.with_overrides({"model.name": model, "select.enabled": False})


def _n_rows(config: ExperimentConfig) -> int:
    """Row count for the data-size gate, or 0 when it cannot be had cheaply.

    Best-effort by design: the gate is an optimization that avoids training a
    model the rule table already knows is wrong for this size, and a source that
    cannot be counted without materializing it should not pay that cost.
    """
    try:
        from ..data.builders import _cv_population

        return int(_cv_population(config)[0])
    except Exception as exc:
        log.debug("could not count rows for candidate gating (%s)", exc)
        return 0


def _budget(config: ExperimentConfig) -> Any:
    from ..backends.base import resolve_budget

    return resolve_budget(config)


def _slug(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def _since(started: float) -> float:
    return time.perf_counter() - started


__all__ = [
    "CRITERIA",
    "SELECTION_FILE",
    "CandidateReport",
    "SelectionError",
    "SelectionResult",
    "apply_constraints",
    "candidate_models",
    "decide",
    "decide_tolerance",
    "decide_weighted",
    "evaluate_candidate",
    "profile_candidate",
    "select",
    "write_selection",
]
