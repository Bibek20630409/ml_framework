"""Importing this package registers the built-in model plugins (mlp, cnn).

This is the ``models/`` package renamed, and the rename came with the thing that
made it worth doing: **discovery no longer swallows anything.**

v1 wrapped the ``cnn`` import in ``except Exception: pass`` so that a
torchvision-less install would not crash on import. The cost was that a genuinely
broken model module — a syntax error, a bad refactor — vanished from the registry
with no message, indistinguishable from "torchvision is not installed". The v2
design removes the exception rather than catching it better:

* **A plugin module must be importable with zero optional dependencies.** Heavy
  imports live inside ``build_network``/``fit``. Both builtins already obeyed
  this, which is why the ``try`` never had anything legitimate to catch.
* **Availability is answered by** ``importlib.util.find_spec`` **through**
  ``Requirement`` — no import, no exception. ``cnn`` therefore appears in
  ``MODELS`` with ``available: False`` on a bare install instead of disappearing.
* **Builtins are an explicit list** (:data:`BUILTINS`). A failure importing one is
  *our* bug, so it propagates with its real traceback.
* **Third-party plugins** come from the ``ml_framework.plugins`` entry-point
  group. A failure there is recorded, warned once, shown in ``mlf models --all``
  and re-raised chained if that plugin is actually selected — never silent.

Each :class:`~ml_framework.core.plugins.ModelSpec` carries what the v1 class
registry could not: the tasks and data kinds it handles, what it needs installed,
its capability flags, its search space, and its ``params_model`` — the Pydantic
schema the config validator runs against ``model.params``.
"""

from __future__ import annotations

from typing import Any

from ..core.plugins import ModelSpec
from ..core.protocols import Float
from ..core.registry import MODELS, register_model_spec
from ..core.types import Capabilities, Requirement

# The builtin imports are unconditional on purpose: both modules are
# dependency-free at module scope, so a failure here is a real defect and must
# not be hidden behind an optional-dependency excuse.
from .cnn import CNN, CNNParams
from .cnn import build as _build_cnn
from .mlp import MLP, MLPParams
from .mlp import build as _build_mlp

# The explicit builtin list, in registration order.
BUILTINS: tuple[str, ...] = ("mlp", "cnn")

__all__ = ["BUILTINS", "CNN", "CNNParams", "MLP", "MLPParams"]


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
        build=_build_mlp,
        tasks=frozenset({"binary", "multiclass", "regression"}),
        data_kinds=frozenset({"tabular"}),
        requires=(),  # torch is a base dependency
        capabilities=Capabilities(accepts=frozenset({"arrays"}), **_NEURAL_CAPS),
        # Declarative keys are dotted v2 config paths; `suggest` overrides them
        # because hidden_dims is conditional on the layer count.
        search_space={"model.params.dropout": Float(0.1, 0.5)},
        suggest=_mlp_suggest,
        params_model=MLPParams,
        auto_priority=10,
        description="Feed-forward network for tabular data (BatchNorm + dropout, Kaiming init).",
    )
)

register_model_spec(
    ModelSpec(
        name="cnn",
        backend="lightning",
        build=_build_cnn,
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
        params_model=CNNParams,
        auto_priority=10,
        description="Transfer-learning CNN over a torchvision backbone (default resnet18).",
    )
)

# Third-party plugins, after the builtins so a duplicate name is a deliberate
# `override=True` on their side rather than an accident of import order.
MODELS.discover()
