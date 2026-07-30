"""Importing this package registers the built-in models (mlp, cnn).

Registration happens **twice** during the transition:

* the v1 class registry, via ``@register_model`` in each module (unchanged), and
* the v2 :data:`~ml_framework.core.registry.MODELS` registry, via the explicit
  builtin table below.

The v2 specs carry what the class registry could not: which tasks and data kinds
the model handles, what it needs installed, its capability flags, and its search
space. Their ``build`` callables import lazily, so a spec is registered — and
listable by ``mlf models`` — even when the model's optional dependency is absent.
That is why ``cnn`` shows up with ``available: False`` on a torchvision-less
install instead of vanishing.

This module becomes ``plugins/__init__.py`` in P2, at which point the ``try/except``
below is replaced by non-swallowing discovery.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from ..core.plugins import ModelSpec
from ..core.protocols import Float
from ..core.registry import register_model_spec
from ..core.types import Capabilities, Requirement
from .mlp import MLP

__all__ = ["MLP"]


def _register_optional() -> None:
    # CNN pulls in torchvision.models lazily; import guarded so the tabular path
    # works even if torchvision is unavailable.
    try:
        from .cnn import CNN  # noqa: F401

        globals()["CNN"] = CNN
        __all__.append("CNN")
    except Exception:  # pragma: no cover
        pass


_register_optional()


# ── v2 specs ──────────────────────────────────────────────
def _lazy_build(module: str, attr: str) -> Callable[..., Any]:
    """Defer both the import and the attribute lookup to call time.

    A spec must be registerable without importing the model's dependencies, and a
    genuinely broken module must fail loudly *when selected* — not be silently
    absent from the registry.
    """

    def _build(*args: Any, **kwargs: Any) -> Any:
        mod = importlib.import_module(module, package=__package__)
        return getattr(mod, attr)(*args, **kwargs)

    return _build


def _mlp_suggest(trial: Any, cfg: Any = None) -> dict[str, Any]:
    """Conditional search space: layer count decides how many width knobs exist.

    A declarative mapping cannot express "n_layers → n_units_l{i}", which is why
    ``ModelSpec.suggest`` exists as an escape hatch. Logic preserved from
    ``pipeline/hpo.py``; ``lr``/``weight_decay`` move to the Lightning backend's
    own space so every neural plugin stops repeating them.
    """
    n_layers = trial.suggest_int("n_layers", 1, 4)
    return {
        "model.params.hidden_dims": [
            trial.suggest_int(f"n_units_l{i}", 32, 512, log=True) for i in range(n_layers)
        ],
        "model.params.dropout": trial.suggest_float("dropout", 0.1, 0.5),
    }


_NEURAL_CAPS: dict[str, bool] = {
    "needs_scaling": True,
    "produces_proba": True,
    "supports_pruning": True,
    "supports_gpu": True,
    "supports_mixed_precision": True,
    "supports_lr_range_test": True,
    # SMOTE is the imbalance tool for neural tabular nets; sample weights are not
    # plumbed through the Lightning loop.
    "supports_sample_weight": False,
}

register_model_spec(
    ModelSpec(
        name="mlp",
        backend="lightning",
        build=_lazy_build(".mlp", "MLP"),
        tasks=frozenset({"binary", "multiclass", "regression"}),
        data_kinds=frozenset({"tabular"}),
        requires=(),  # torch is a base dependency
        capabilities=Capabilities(accepts=frozenset({"arrays"}), **_NEURAL_CAPS),
        # Declarative keys are dotted v2 config paths; `suggest` overrides them
        # because hidden_dims is conditional on the layer count.
        search_space={"model.params.dropout": Float(0.1, 0.5)},
        suggest=_mlp_suggest,
        # params_model lands in P2 with the v2 schema, where ModelConfig._check_dims
        # (positive hidden_dims) moves into it.
        params_model=None,
        auto_priority=10,
        description="Feed-forward network for tabular data (BatchNorm + dropout, Kaiming init).",
    )
)

register_model_spec(
    ModelSpec(
        name="cnn",
        backend="lightning",
        build=_lazy_build(".cnn", "CNN"),
        tasks=frozenset({"binary", "multiclass"}),
        data_kinds=frozenset({"image"}),
        requires=(
            Requirement("torchvision", extra="image", min_version="0.15"),
            Requirement("PIL", extra="image", min_version="9.0", dist="Pillow"),
        ),
        capabilities=Capabilities(accepts=frozenset({"dataset"}), **_NEURAL_CAPS),
        # Architecture knobs are deliberately not tuned: swapping the backbone or
        # discarding pretrained weights inside a 10-trial budget wastes the budget.
        # lr/batch_size come from the Lightning backend's space.
        search_space={},
        params_model=None,
        auto_priority=10,
        description="Transfer-learning CNN over a torchvision backbone (default resnet18).",
    )
)
