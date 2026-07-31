"""
backends/base.py
────────────────
The small amount of behaviour every fit-loop shape shares: budget resolution,
metric-name cleanup, and defaults for the optional half of the
:class:`~ml_framework.core.protocols.TrainingBackend` protocol.

A base *class* rather than a mixin because the defaults here are the answers a
backend gives when it has nothing interesting to say — ``trial_hooks`` returns
``TrialHooks.empty()`` for a loop with no pruning point, ``model_size`` returns
``{}`` for a model with no meaningful size. Making those abstract would force
every backend to write the same three stubs.

Nothing in this module imports torch, xgboost or any training library. It is the
part of a backend that can be reasoned about — and tested — without one.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from ..core.protocols import (
    ArtifactRef,
    Budget,
    Estimator,
    FitResult,
    Predictions,
    RunContext,
    SearchSpace,
    TrialHooks,
)
from ..core.types import Capabilities


def clean_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Coerce a metric mapping to plain floats, dropping what will not convert.

    Lightning's ``callback_metrics`` holds tensors; xgboost's eval history holds
    lists. ``FitResult.val_metrics`` is a plain ``dict[str, float]`` precisely so
    the HPO driver can read an objective from any backend without knowing which
    it has — this is where that promise is kept.
    """
    out: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def resolve_budget(config: Any) -> Budget:
    """The training budget from a config, without the backend reading the config.

    Kept here rather than in each backend so ``fit.budget`` grows one axis at a
    time in one place. ``max_seconds`` is carried but not yet enforced — the
    per-backend wall-clock caps that make tuning-by-default finish in minutes are
    the tuning driver's job.
    """
    fit = getattr(config, "fit", None)
    budget = getattr(fit, "budget", None)
    return Budget(
        max_epochs=getattr(budget, "max_epochs", None),
        max_seconds=getattr(budget, "max_seconds", None),
        patience=getattr(fit, "patience", None),
    )


class BaseBackend:
    """Defaults for the optional surface of a training backend."""

    name: ClassVar[str] = "base"
    capabilities: ClassVar[Capabilities] = Capabilities()

    # ── required of every backend ──
    def fit(self, spec: Any, bundle: Any, cfg: Any, *, run: RunContext) -> FitResult:
        raise NotImplementedError

    def save(self, est: Estimator, dest: Path) -> ArtifactRef:
        raise NotImplementedError

    def load(self, bundle_dir: Path, manifest: Any) -> Estimator:
        raise NotImplementedError

    def predict_split(self, est: Estimator, bundle: Any, split: str) -> Predictions:
        raise NotImplementedError

    # ── optional ──
    def search_space(self) -> SearchSpace:
        """Backend-level knobs, declared once here instead of in every plugin."""
        return {}

    def trial_hooks(self, trial: Any) -> TrialHooks:
        """Pruning plumbing, so the HPO driver never imports an integration package."""
        return TrialHooks.empty()

    def params_model(self) -> type | None:
        """Pydantic schema validating ``fit.params``. ``None`` means unvalidated."""
        return None

    def model_size(self, est: Estimator) -> dict[str, Any]:
        """Size, in whatever unit is meaningful for this loop shape.

        The generalization of ``BaseModel.count_parameters()``: parameter counts
        for a network, tree and leaf counts for a GBDT. Logged by the orchestrator
        and recorded in ``manifest.model.size``; informational, never load-bearing.
        """
        return {}
