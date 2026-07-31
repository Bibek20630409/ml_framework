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

import logging
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

log = logging.getLogger(__name__)


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


# The shorthand people type, and what Lightning actually wants.
_PRECISION_ALIASES: Mapping[str, str] = {
    "16": "16-mixed",
    "bf16": "bf16-mixed",
    "32": "32-true",
}
_MIXED_PRECISIONS = frozenset({"16-mixed", "bf16-mixed"})


def resolve_precision(
    requested: str, capabilities: Capabilities, *, accelerator: str = "auto", has_gpu: bool = False
) -> str:
    """The precision that will actually be used, having said so if it differs.

    This is the consumer of ``Capabilities.supports_mixed_precision``. Asking for
    fp16 and silently getting fp32 is the kind of thing you discover months later
    from a wall-clock number that never improved, so every downgrade is logged at
    WARNING with the reason.

    Two conditions force a downgrade: a backend that cannot do mixed precision at
    all (a GBDT has no autocast to enter), and fp16 on CPU — where torch's
    gradient scaler has nothing to scale on. ``bf16`` needs no scaler and is left
    alone, which is why it is the one that works on a laptop.
    """
    normalized = _PRECISION_ALIASES.get(str(requested), str(requested))
    if normalized not in _MIXED_PRECISIONS:
        return normalized

    if not capabilities.supports_mixed_precision:
        log.warning(
            "precision=%s requested but this backend does not support mixed precision "
            "— training at 32-true",
            requested,
        )
        return "32-true"

    if normalized == "16-mixed" and not has_gpu and accelerator in ("auto", "cpu"):
        # fp16 autocast on CPU has no gradient scaler behind it; Lightning will
        # construct the Trainer and then train badly or not at all.
        log.warning(
            "precision=%s needs a GPU (fp16 gradient scaling is GPU-only) "
            "— training at 32-true. Use bf16-mixed for mixed precision on CPU.",
            requested,
        )
        return "32-true"

    log.info("mixed precision: %s", normalized)
    return normalized


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
