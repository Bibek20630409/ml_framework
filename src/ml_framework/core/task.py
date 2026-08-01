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

**A row means the framework can run the task.** That is the rule this table is
kept to, and it is why ``token_classification`` and ``seq2seq`` were absent until
there was a source, a model and an evaluation path for each: a row that merely
lets a config validate would trade an honest refusal at load time for a confusing
failure somewhere in the fit loop. ``get_task_spec`` on an unregistered task says
so rather than silently defaulting.

Still unregistered, and for that reason: ``multilabel``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .types import IGNORE_INDEX, Direction, FrameworkError, OutputKind, Postprocess, Task


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
        """Whether one row yields one class.

        ``token_classification`` is deliberately **not** included even though it
        predicts classes: its predictions are one-per-token over a variable-length
        sequence, so every consumer that branches on this flag (the confusion
        matrix, ``predictions.csv``, the serving response) would produce the wrong
        shape. It gets its own branch instead of a flag that is true in name only.
        """
        return self.output_kind in ("labels", "probabilities")

    @property
    def ignore_index(self) -> int | None:
        """Label value the loss and the metrics must skip, or ``None``.

        Read by ``BaseModel._build_criterion`` and by ``predict_split``, so a task
        that pads its targets declares it once here instead of each of them
        hardcoding -100.
        """
        value = self.meta.get("ignore_index")
        # `meta` is deliberately typed `object` so it cannot become a typed
        # dumping ground; the narrowing belongs at the one accessor that reads it.
        return int(value) if isinstance(value, int) else None

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
register_task_spec(
    TaskSpec(
        name="forecasting",
        # MASE rather than MAE because it is scale-free: "MAE 4.2" says nothing
        # without knowing whether the series runs in single digits or millions.
        # It is *not* a pass mark — over a multi-step horizon values above 1 are
        # normal. See `metrics.mase`, and `evaluate._write_forecast`, which prints
        # no verdict for the same reason.
        primary_metric="mase",
        direction="min",
        output_kind="series",
        postprocess="identity",
        metric_names=("mase", "smape", "mae", "rmse"),
        description="Predict future values of a series over a horizon.",
        # Seasonality is a property of the data, not of the task, so the source
        # sets it. Recorded here as the default for a non-seasonal series.
        meta={"seasonality": 1},
    )
)
register_task_spec(
    TaskSpec(
        name="token_classification",
        # Macro-F1, not accuracy. Token tagging is overwhelmingly dominated by the
        # `O` class — a model that predicts "not an entity" for every token scores
        # around 90% accuracy on a typical NER corpus while being worth nothing.
        primary_metric="f1",
        direction="max",
        output_kind="token_labels",
        postprocess="softmax",
        metric_names=("acc", "f1", "precision", "recall"),
        description="One label per token (NER, POS tagging).",
        # Both the loss and the metrics skip these positions: sub-word
        # continuations, padding, and the special tokens the tokenizer adds. See
        # `data.preprocess.text.TokenTextPreprocessor` for why a *word*-level tag
        # cannot simply be repeated across the sub-words it became.
        meta={"ignore_index": IGNORE_INDEX, "scoring": "token-level, not entity-level"},
    )
)
register_task_spec(
    TaskSpec(
        name="seq2seq",
        # Reference-overlap, and shallow: see `metrics.rouge_l`. ROUGE-L leads
        # because it is the summarization convention and because it is the least
        # brittle of the three — exact_match is near-zero for anything longer than
        # a phrase, and token_f1 ignores word order entirely.
        primary_metric="rouge_l",
        direction="max",
        output_kind="text",
        # Not a function of the logits: generation is an autoregressive loop that
        # needs the inputs, which is why `Postprocess` had to grow a fourth value
        # rather than this task reusing `identity` and lying about it.
        postprocess="generate",
        metric_names=("rouge_l", "token_f1", "exact_match"),
        # `val/loss` stays the monitor. Generating on every validation epoch to
        # score ROUGE would multiply epoch time by the decode length, and
        # teacher-forced loss tracks quality closely enough to stop on.
        description="Generate a target string from a source string.",
    )
)
