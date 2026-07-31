"""
core/registry.py
────────────────
Decorator-based registries for models and datamodules. This is the key
extensibility lever: new architectures / data sources register by name and are
selected via config (`model.name`, `data.kind`) without editing the pipeline.

    @register_model("mlp")
    class MLP(BaseModel): ...

    build_model("mlp", input_dim=..., output_dim=..., config=...)

**Two generations live here side by side.**

*v1* (``_MODEL_REGISTRY`` / ``_DATAMODULE_REGISTRY``, name → class) is what the
pipeline still runs on and is unchanged.

*v2* (:data:`MODELS` / :data:`BACKENDS` / :data:`SOURCES`) are spec-carrying
:class:`~ml_framework.core.plugins.PluginRegistry` instances holding capabilities,
optional-dependency requirements, params schemas and search spaces. They are what
the config validator, the orchestrator and the backends consume.

The v1 class registry survives for one reason: ``load_from_checkpoint`` is a
*classmethod*, so rebuilding a Lightning model from a bundle needs the class
itself, not a build callable. A plugin registers both, one line apart.

Who populates the v2 registries:
  * ``plugins/__init__.py``     → :data:`MODELS` (mlp, cnn) + entry-point discovery
  * ``data/builders.py``        → :data:`SOURCES` (tabular, image)
  * ``backends/__init__.py``    → :data:`BACKENDS` (lightning), via a lazy factory
    so registering never imports torch
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from .plugins import (
    BackendSpec,
    IncompatibleCombinationError,
    ModelSpec,
    PluginRegistry,
    SourceSpec,
    check_requirements,
)
from .protocols import DEFAULT_PAYLOAD, KIND_PAYLOADS
from .types import DataKind, Payload, Task

T = TypeVar("T")

_MODEL_REGISTRY: dict[str, type] = {}
_DATAMODULE_REGISTRY: dict[str, type] = {}


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    def _wrap(cls: type[T]) -> type[T]:
        key = name.lower()
        if key in _MODEL_REGISTRY:
            raise ValueError(f"Model '{key}' already registered")
        _MODEL_REGISTRY[key] = cls
        return cls

    return _wrap


def register_datamodule(name: str) -> Callable[[type[T]], type[T]]:
    def _wrap(cls: type[T]) -> type[T]:
        key = name.lower()
        if key in _DATAMODULE_REGISTRY:
            raise ValueError(f"DataModule '{key}' already registered")
        _DATAMODULE_REGISTRY[key] = cls
        return cls

    return _wrap


def get_model_class(name: str) -> type:
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[key]


def get_datamodule_class(name: str) -> type:
    key = name.lower()
    if key not in _DATAMODULE_REGISTRY:
        raise KeyError(f"Unknown datamodule '{name}'. Registered: {sorted(_DATAMODULE_REGISTRY)}")
    return _DATAMODULE_REGISTRY[key]


def instantiate_model(name: str, **kwargs: Any) -> Any:
    """Low-level: build a model by its registry name. Prefer
    ``ml_framework.data.build_model(config, ...)`` for the config-driven path."""
    return get_model_class(name)(**kwargs)


def instantiate_datamodule(name: str, **kwargs: Any) -> Any:
    """Low-level: build a datamodule by its registry name. Prefer
    ``ml_framework.data.build_datamodule(config)`` for the config-driven path."""
    return get_datamodule_class(name)(**kwargs)


def available_models() -> list[str]:
    """Every registered model, from the spec registry.

    A thin shim over :data:`MODELS` rather than over the v1 class registry, which
    only ever holds ``LightningModule`` subclasses — a GBDT plugin registers a
    build *function*, so it has no class to put there. Reading the class registry
    here would have made ``mlf``'s idea of "available models" quietly
    Lightning-only the moment a non-neural family landed.
    """
    return MODELS.names()


def available_datamodules() -> list[str]:
    """Every registered data source, from the spec registry.

    Named for the v1 concept it replaces — datamodules stopped being an extension
    point when there came to be exactly one (the Lightning adapter over a bundle);
    *sources* are what users plug in. Shims over :data:`SOURCES` for the same
    reason :func:`available_models` does: the v1 datamodule registry is populated
    by importing the Lightning adapter, which a torch-free process never does.
    """
    import ml_framework.data.builders  # noqa: F401  (registration side effect)

    return SOURCES.names()


# ══ v2: spec-carrying registries ═════════════════════════════════════
MODELS: PluginRegistry[ModelSpec] = PluginRegistry("model")
BACKENDS: PluginRegistry[BackendSpec] = PluginRegistry("backend")
SOURCES: PluginRegistry[SourceSpec] = PluginRegistry("source")


def register_model_spec(spec: ModelSpec, *, override: bool = False) -> ModelSpec:
    """Register a model's metadata in :data:`MODELS`.

    Separate from :func:`register_model` (which registers the *class* in the v1
    registry) for the length of the transition. A plugin registers both.
    """
    return MODELS.register(spec, override=override)


def register_backend(spec: BackendSpec, *, override: bool = False) -> BackendSpec:
    return BACKENDS.register(spec, override=override)


def register_source(spec: SourceSpec, *, override: bool = False) -> SourceSpec:
    return SOURCES.register(spec, override=override)


def get_backend(name: str) -> Any:
    """The instantiated :class:`TrainingBackend` for ``name``.

    Populates :data:`BACKENDS` first. Registration is an import side effect of
    ``ml_framework.backends``, and the training pipeline imports that package for
    other reasons — but the **serving** path does not, so without this a bundle
    would load in ``mlf train`` and fail with "Unknown backend" in the API. Doing
    it here rather than at every call site keeps the one rule in one place.

    Availability is checked after, so a missing extra produces a pip command
    rather than an ImportError from inside the factory.
    """
    import ml_framework.backends  # noqa: F401  (registration side effect)

    return BACKENDS.get(name).factory()


def validate_combination(
    task: Task | str,
    data_kind: DataKind | str,
    model_name: str,
    payload: Payload | str | None = None,
) -> ModelSpec:
    """Check that (task, data_kind, model) can actually work, and return the spec.

    Called from the config validator so an incompatible combination fails at load
    time — *"xgboost cannot consume an image folder"* — instead of surfacing as a
    shape error 200 lines into data loading.

    **Compatibility is checked before availability**, deliberately: telling a user
    to install 2 GB of torchvision for a combination that could never work would
    be the wrong instruction. Once the combination is sound, an uninstalled plugin
    raises :class:`~ml_framework.core.plugins.MissingExtraError`.
    """
    spec = MODELS.get_spec(model_name)
    if spec.tasks and task not in spec.tasks:
        raise IncompatibleCombinationError(
            f"model '{spec.name}' does not support task '{task}' "
            f"(supports: {sorted(spec.tasks)})"
        )
    if spec.data_kinds and data_kind not in spec.data_kinds:
        raise IncompatibleCombinationError(
            f"model '{spec.name}' cannot consume data kind '{data_kind}' "
            f"(supports: {sorted(spec.data_kinds)})"
        )
    if payload is not None:
        # An explicit payload is a question about one concrete shape.
        offered: frozenset = frozenset({payload})  # type: ignore[arg-type]
    else:
        # Otherwise: every shape this kind can be materialized as. A kind is a
        # statement about the *data*, not about the shape a model wants it in —
        # time series are handed to Prophet as an ordered series and to an LSTM as
        # sliding windows, from one source. Refuse only when the model can consume
        # none of them.
        offered = KIND_PAYLOADS.get(data_kind, frozenset())  # type: ignore[arg-type]
        if not offered:
            fallback = DEFAULT_PAYLOAD.get(data_kind)  # type: ignore[arg-type]
            offered = frozenset({fallback}) if fallback else frozenset()

    if offered and not (offered & spec.capabilities.accepts):
        wanted = sorted(offered)
        raise IncompatibleCombinationError(
            f"model '{spec.name}' cannot consume a "
            f"'{wanted[0] if len(wanted) == 1 else ' or '.join(wanted)}' payload "
            f"(accepts: {sorted(spec.capabilities.accepts)})"
        )
    check_requirements(spec.requires, what=f"model '{spec.name}'", dist=MODELS.dist)
    return spec


def models_for(task: str | None = None, data_kind: str | None = None) -> list[ModelSpec]:
    """Installed models compatible with ``task``/``data_kind``, best first.

    Ordered by ``auto_priority`` — the tie-break zero-config model selection uses.
    """
    matches = [s for s in MODELS.specs() if s.supports(task, data_kind)]
    matches = [s for s in matches if MODELS.is_available(s.name)]
    return sorted(matches, key=lambda s: (-s.auto_priority, s.name))
