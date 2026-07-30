"""
core/task.py
────────────
The task table. One row per task, holding everything that currently gets
re-derived by ``task == "binary"`` branching in ``lit_model._shared_step``,
``evaluate.py`` and ``inference.py``.

A :class:`TaskSpec` is consumed by early stopping, checkpointing, HPO direction,
``evaluate()``, the serving response models and the estimator's head
postprocessing. Adding a task becomes "add a row (+ its metrics)" rather than a
six-file grep.

Rows exist only for tasks the framework can actually run today
(binary/multiclass/regression); ``forecasting`` arrives in P6 and the text tasks
in P7, each via :func:`register_task_spec`. ``get_task_spec`` on an unregistered
task says so rather than silently defaulting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .types import Direction, FrameworkError, OutputKind, Postprocess, Task


class UnknownTaskError(FrameworkError):
    """Raised for a task with no registered :class:`TaskSpec`."""


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Everything task-dependent, in one place.

    ``primary_metric``/``direction`` drive HPO and any quality gate.
    ``monitor``/``monitor_mode`` drive Lightning early stopping and
    checkpointing — kept separate because the thing you *monitor* during training
    (``val/loss``) is usually not the thing you *optimise for* (accuracy).
    """

    name: Task
    primary_metric: str
    direction: Direction
    output_kind: OutputKind
    postprocess: Postprocess
    metric_names: tuple[str, ...]
    monitor: str = "val/loss"
    monitor_mode: Direction = "min"
    # Human-facing label used in reports and API docs.
    description: str = ""
    # Extra per-task knobs a consumer may need without growing the dataclass:
    # kept empty by default so it cannot become a dumping ground silently.
    meta: Mapping[str, object] = field(default_factory=dict)

    @property
    def metric_fns(self) -> Mapping[str, object]:
        """Name → callable for this task's metrics.

        Resolved lazily so importing the task table does not import scikit-learn.
        """
        from .metrics import metric_fns

        return metric_fns(self.metric_names)

    @property
    def is_classification(self) -> bool:
        return self.output_kind in ("labels", "probabilities")

    def compute(self, y_true, y_pred, y_prob=None, *, prefix: str = "") -> dict[str, float]:
        """This task's metrics for one set of predictions."""
        from .metrics import compute

        return compute(self.metric_names, y_true, y_pred, y_prob, prefix=prefix)


_TASK_SPECS: dict[str, TaskSpec] = {}


def register_task_spec(spec: TaskSpec, *, override: bool = False) -> TaskSpec:
    if spec.name in _TASK_SPECS and not override:
        raise ValueError(f"TaskSpec '{spec.name}' already registered (pass override=True)")
    _TASK_SPECS[spec.name] = spec
    return spec


def get_task_spec(task: str) -> TaskSpec:
    try:
        return _TASK_SPECS[task]
    except KeyError:
        raise UnknownTaskError(
            f"No TaskSpec for task '{task}'. Registered: {available_tasks()}"
        ) from None


def has_task_spec(task: str) -> bool:
    return task in _TASK_SPECS


def available_tasks() -> list[str]:
    return sorted(_TASK_SPECS)


# ── Built-in rows ─────────────────────────────────────────
# `monitor="val/loss"` + mode="min" reproduces today's EarlyStopping /
# ModelCheckpoint configuration exactly; the metric names match the torchmetrics
# keys logged by BaseModel._shared_step.
register_task_spec(
    TaskSpec(
        name="binary",
        primary_metric="acc",
        direction="max",
        output_kind="probabilities",
        postprocess="sigmoid",
        metric_names=("acc", "f1_binary", "roc_auc"),
        description="Two-class classification with a single logit head.",
    )
)
register_task_spec(
    TaskSpec(
        name="multiclass",
        primary_metric="acc",
        direction="max",
        output_kind="probabilities",
        postprocess="softmax",
        metric_names=("acc", "f1", "roc_auc"),
        description="Single-label classification over n_classes logits.",
    )
)
register_task_spec(
    TaskSpec(
        name="regression",
        primary_metric="mae",
        direction="min",
        output_kind="values",
        postprocess="identity",
        metric_names=("mae", "rmse", "r2"),
        description="Continuous target, single output.",
    )
)
