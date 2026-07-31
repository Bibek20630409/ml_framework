"""Core: the framework-agnostic contracts, plus the Lightning-specific pieces.

**Torch is imported lazily, and that is load-bearing.** The v2 surface
(``types``, ``protocols``, ``plugins``, ``task``, ``bundle``, ``metrics``,
``inference``) imports no torch, no Lightning and no optional dependency — but a
package ``__init__`` that eagerly re-exported ``BaseModel`` would drag torch in
anyway, and every ``from ml_framework.core import …`` in the serving path would
pay for it. A GBDT serving image would then still install ~2 GB of torch to run a
50 KB booster.

So the torch-touching names (:class:`BaseModel`, the datamodules, the
``lit_data`` helpers) resolve through a module ``__getattr__`` (PEP 562). They
are importable exactly as before — ``from ml_framework.core import BaseModel``
works, and so does tab completion via ``__dir__`` — but the import happens on
first *use* rather than on ``import ml_framework``. There is a test asserting
that importing the inference path leaves torch out of ``sys.modules``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .bundle import (
    BUNDLE_VERSION,
    Manifest,
    Signature,
    read_manifest,
    write_bundle,
    write_manifest,
)
from .evaluate import evaluate
from .inference import Inferencer
from .plugins import (
    BackendSpec,
    MissingExtraError,
    ModelSpec,
    PluginLoadError,
    PluginRegistry,
    SourceSpec,
)
from .protocols import (
    Budget,
    BuildContext,
    Estimator,
    FitResult,
    Predictions,
    RunContext,
    TrainingBackend,
)
from .registry import (
    BACKENDS,
    MODELS,
    SOURCES,
    available_datamodules,
    available_models,
    get_backend,
    get_datamodule_class,
    get_model_class,
    instantiate_datamodule,
    instantiate_model,
    register_backend,
    register_datamodule,
    register_model,
    register_model_spec,
    register_source,
    validate_combination,
)
from .task import TaskSpec, available_tasks, get_task_spec
from .types import Capabilities, DataKind, FrameworkError, Requirement, Task, UnsupportedCapability

if TYPE_CHECKING:  # so type checkers and IDEs still see the lazy names
    from .lit_data import (
        FrameworkDataModule,
        ImageDataModule,
        TabularDataModule,
        apply_smote,
        compute_class_weights,
        detect_imbalance,
        read_table,
        split_dataset,
    )
    from .lit_model import BaseModel

# name → module it lives in. Every entry pulls torch or Lightning when resolved.
_LAZY: dict[str, str] = {
    "BaseModel": ".lit_model",
    "OptimSettings": ".lit_model",
    "FrameworkDataModule": ".lit_data",
    "TabularDataModule": ".lit_data",
    "ImageDataModule": ".lit_data",
    "read_table": ".lit_data",
    "detect_imbalance": ".lit_data",
    "apply_smote": ".lit_data",
    "compute_class_weights": ".lit_data",
    "split_dataset": ".lit_data",
}


def __getattr__(name: str) -> Any:
    """Resolve a torch-dependent export on first use (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    # v1 Lightning implementation (lazy — see __getattr__)
    "BaseModel",
    "FrameworkDataModule",
    "TabularDataModule",
    "ImageDataModule",
    "split_dataset",
    "detect_imbalance",
    "compute_class_weights",
    "apply_smote",
    "read_table",
    # eager
    "evaluate",
    "Inferencer",
    "register_model",
    "register_datamodule",
    "get_model_class",
    "get_datamodule_class",
    "get_backend",
    "instantiate_model",
    "instantiate_datamodule",
    "available_models",
    "available_datamodules",
    # v2 vocabulary
    "Task",
    "DataKind",
    "Capabilities",
    "Requirement",
    "FrameworkError",
    "UnsupportedCapability",
    # v2 contracts
    "Estimator",
    "TrainingBackend",
    "FitResult",
    "Predictions",
    "RunContext",
    "BuildContext",
    "Budget",
    # v2 task table
    "TaskSpec",
    "get_task_spec",
    "available_tasks",
    # v2 plugin system
    "PluginRegistry",
    "ModelSpec",
    "BackendSpec",
    "SourceSpec",
    "MissingExtraError",
    "PluginLoadError",
    "MODELS",
    "BACKENDS",
    "SOURCES",
    "register_model_spec",
    "register_backend",
    "register_source",
    "validate_combination",
    # v2 bundle
    "Manifest",
    "Signature",
    "BUNDLE_VERSION",
    "read_manifest",
    "write_manifest",
    "write_bundle",
]
