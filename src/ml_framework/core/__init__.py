"""Core: the framework-agnostic contracts plus the v1 Lightning implementation.

The v2 surface (``types``, ``protocols``, ``plugins``, ``task``, ``bundle``,
``metrics``) is imported here so ``from ml_framework.core import …`` reaches it,
but note the layering rule those modules obey: **none of them import torch,
Lightning or any optional dependency.** ``lit_model``/``lit_data``/``evaluate`` do,
and they are the ones being refactored away from in P1.
"""

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

__all__ = [
    # v1 Lightning implementation
    "BaseModel",
    "FrameworkDataModule",
    "TabularDataModule",
    "ImageDataModule",
    "evaluate",
    "Inferencer",
    "split_dataset",
    "detect_imbalance",
    "compute_class_weights",
    "apply_smote",
    "read_table",
    "register_model",
    "register_datamodule",
    "get_model_class",
    "get_datamodule_class",
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
