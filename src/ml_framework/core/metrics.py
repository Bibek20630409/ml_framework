"""
core/metrics.py
───────────────
Per-task metrics computed from **arrays** — numpy and scikit-learn only, no
torch, no Lightning.

This is one half of the ``evaluate`` split: a backend produces
:class:`~ml_framework.core.protocols.Predictions`, and this module turns those
arrays into a metrics dict. That is what lets a GBDT or a Prophet model be
evaluated by the same code path as a LightningModule, and what lets the serving
image drop torch entirely.

Metric *names* deliberately match the torchmetrics log keys used inside the
Lightning backend (``acc``/``f1``/``mae``/``rmse``), so ``val/acc`` logged during
training and ``test_acc`` written to ``metrics.json`` refer to the same quantity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from .types import Task

# A metric takes (y_true, y_pred) or (y_true, y_prob) and returns one float.
MetricFn = Callable[..., float]


def _as_1d(a) -> np.ndarray:
    arr = np.asarray(a)
    return arr.reshape(-1) if arr.ndim > 1 and 1 in arr.shape else arr


# ── Classification ────────────────────────────────────────
def accuracy(y_true, y_pred) -> float:
    """Plain match rate. Identical to ``sklearn.accuracy_score`` but keeps the
    hot path free of a sklearn import."""
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(yp == yt))


def f1(y_true, y_pred, *, average: str = "macro") -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(_as_1d(y_true), _as_1d(y_pred), average=average, zero_division=0))


def precision(y_true, y_pred, *, average: str = "macro") -> float:
    from sklearn.metrics import precision_score

    return float(precision_score(_as_1d(y_true), _as_1d(y_pred), average=average, zero_division=0))


def recall(y_true, y_pred, *, average: str = "macro") -> float:
    from sklearn.metrics import recall_score

    return float(recall_score(_as_1d(y_true), _as_1d(y_pred), average=average, zero_division=0))


def roc_auc(y_true, y_prob) -> float:
    """ROC-AUC from probabilities. Returns NaN rather than raising when the
    metric is undefined (single class present, or degenerate probabilities) —
    a metrics dict must never be the thing that fails a training run."""
    from sklearn.metrics import roc_auc_score

    yt = _as_1d(y_true)
    prob = np.asarray(y_prob)
    try:
        if prob.ndim == 2 and prob.shape[1] == 2:
            return float(roc_auc_score(yt, prob[:, 1]))
        if prob.ndim == 2:
            return float(roc_auc_score(yt, prob, multi_class="ovr", average="macro"))
        return float(roc_auc_score(yt, _as_1d(prob)))
    except ValueError:
        return float("nan")


# ── Regression ────────────────────────────────────────────
def mae(y_true, y_pred) -> float:
    yt, yp = _as_1d(y_true).astype("float64"), _as_1d(y_pred).astype("float64")
    return float(np.abs(yp - yt).mean())


def rmse(y_true, y_pred) -> float:
    yt, yp = _as_1d(y_true).astype("float64"), _as_1d(y_pred).astype("float64")
    return float(np.sqrt(((yp - yt) ** 2).mean()))


def r2(y_true, y_pred) -> float:
    yt, yp = _as_1d(y_true).astype("float64"), _as_1d(y_pred).astype("float64")
    denom = float(((yt - yt.mean()) ** 2).sum())
    if denom == 0.0:
        return float("nan")  # constant target — R² is undefined, not 0.
    return float(1.0 - ((yt - yp) ** 2).sum() / denom)


# ── Name → function table ─────────────────────────────────
# `TaskSpec.metric_names` indexes into this. Forecasting metrics (MASE/sMAPE)
# land here in P6 alongside the forecasting TaskSpec row.
_LABEL_METRICS: dict[str, MetricFn] = {
    "acc": accuracy,
    "f1": f1,
    "f1_binary": lambda yt, yp: f1(yt, yp, average="binary"),
    "precision": precision,
    "recall": recall,
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
}
# Metrics that consume probabilities rather than hard predictions.
_PROBA_METRICS: dict[str, MetricFn] = {
    "roc_auc": roc_auc,
}


def metric_fn(name: str) -> MetricFn:
    if name in _LABEL_METRICS:
        return _LABEL_METRICS[name]
    if name in _PROBA_METRICS:
        return _PROBA_METRICS[name]
    raise KeyError(f"Unknown metric '{name}'. Known: {available_metrics()}")


def needs_proba(name: str) -> bool:
    return name in _PROBA_METRICS


def available_metrics() -> list[str]:
    return sorted({*_LABEL_METRICS, *_PROBA_METRICS})


def metric_fns(names: tuple[str, ...]) -> Mapping[str, MetricFn]:
    return {name: metric_fn(name) for name in names}


def compute(
    names: tuple[str, ...],
    y_true,
    y_pred,
    y_prob=None,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Evaluate the named metrics, skipping proba-only ones when ``y_prob`` is None.

    Skipping is silent by design: ``roc_auc`` is genuinely unavailable when a
    backend returns no probabilities, and a KeyError there would fail a training
    run over a reporting nicety.
    """
    out: dict[str, float] = {}
    for name in names:
        if needs_proba(name):
            if y_prob is None:
                continue
            out[f"{prefix}{name}"] = metric_fn(name)(y_true, y_prob)
        else:
            out[f"{prefix}{name}"] = metric_fn(name)(y_true, y_pred)
    return out


def compute_metrics(
    task: Task,
    y_true,
    y_pred,
    y_prob=None,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Metrics for ``task``, per its :class:`~ml_framework.core.task.TaskSpec`."""
    from .task import get_task_spec

    spec = get_task_spec(task)
    return compute(spec.metric_names, y_true, y_pred, y_prob, prefix=prefix)
