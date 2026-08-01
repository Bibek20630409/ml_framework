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


# ── Forecasting ───────────────────────────────────────────
def smape(y_true, y_pred) -> float:
    """Symmetric MAPE, as a percentage.

    Scale-free, so it compares across series whose magnitudes differ — which is
    the whole reason plain MAE is a poor primary metric for forecasting.

    Terms where both the actual and the prediction are zero contribute 0 rather
    than NaN: they are exactly right, and a division-by-zero that poisons the mean
    would make a *perfect* forecast unscoreable.
    """
    yt, yp = _as_1d(y_true).astype("float64"), _as_1d(y_pred).astype("float64")
    if yt.size == 0:
        return float("nan")
    denominator = np.abs(yt) + np.abs(yp)
    terms = np.where(
        denominator == 0.0,
        0.0,
        2.0 * np.abs(yp - yt) / np.where(denominator == 0.0, 1.0, denominator),
    )
    return float(100.0 * terms.mean())


def mase(y_true, y_pred, *, seasonality: int = 1) -> float:
    """Mean Absolute Scaled Error: MAE divided by the series' average step change.

    The denominator is ``mean|y_t - y_{t-m}|`` over the evaluation window, which
    makes the result **unit-free and comparable across series** — the reason it is
    the primary metric here rather than MAE, whose "4.2" means nothing without
    knowing whether the series runs in single digits or millions.

    **1.0 is not a pass mark.** It is often quoted that way, and for *one-step*
    forecasts the reading holds: an error equal to the average step change is what
    you would get by predicting no change. Over a multi-step horizon, values above
    1 are entirely normal — the model is being asked a harder question than the
    one the denominator measures. Use it to rank models on the same split, not as
    an absolute verdict. (A verdict needs a baseline forecast over the *same*
    horizon; ``ts.naive`` is that baseline, and comparing against it is a
    deliberate act rather than something a metric can do on its own.)

    Two further departures from the textbook definition, both deliberate:

    * The denominator is computed on the **evaluation** window rather than the
      training series, because ``compute()`` receives only ``(y_true, y_pred)`` and
      threading the training series through every metric call to serve one metric
      would distort the interface.
    * ``seasonality`` defaults to 1 (consecutive differences) since the metric
      table calls metrics positionally. A seasonal scaling is available by calling
      this directly.

    The consequence of both: comparable between models on one split, not with
    published MASE figures.
    """
    yt, yp = _as_1d(y_true).astype("float64"), _as_1d(y_pred).astype("float64")
    if yt.size <= seasonality:
        return float("nan")
    naive_error = np.abs(yt[seasonality:] - yt[:-seasonality]).mean()
    if naive_error == 0.0:
        # A constant series: the naive forecast is perfect, so the ratio is
        # undefined. NaN says so; 0 or inf would both be read as a result.
        return float("nan")
    return float(np.abs(yp - yt).mean() / naive_error)


# ── Generated text ────────────────────────────────────────
# These take **strings**, not numbers, which is why they sit apart. They are all
# reference-based n-gram overlap measures: cheap, deterministic, dependency-free,
# and blunt. None of them knows that "the cat sat" and "a feline was seated" mean
# the same thing. Reported because a number that is honest about being shallow
# beats no number at all, and because the alternatives (BERTScore, an LLM judge)
# are a model dependency this framework should not acquire by default.


def _norm_tokens(text: object) -> list[str]:
    """Lowercased whitespace tokens.

    Deliberately not the model's tokenizer: a metric that changed when you swapped
    checkpoints would be uncomparable between runs, which is the one thing a
    metric has to be.
    """
    return str(text).lower().split()


def _pairs(y_true, y_pred) -> list[tuple[str, str]]:
    yt = [str(v) for v in np.asarray(y_true, dtype=object).reshape(-1)]
    yp = [str(v) for v in np.asarray(y_pred, dtype=object).reshape(-1)]
    if len(yt) != len(yp):
        raise ValueError(f"reference/prediction counts differ: {len(yt)} vs {len(yp)}")
    return list(zip(yt, yp, strict=True))


def exact_match(y_true, y_pred) -> float:
    """Fraction of predictions equal to their reference, after stripping.

    The strictest and least forgiving of the three. Useful for short, constrained
    outputs (a normalized date, a SQL clause, a yes/no) and close to useless for
    summarization, where two correct summaries are almost never the same string.
    """
    pairs = _pairs(y_true, y_pred)
    if not pairs:
        return float("nan")
    return float(np.mean([t.strip() == p.strip() for t, p in pairs]))


def token_f1(y_true, y_pred) -> float:
    """Mean per-pair F1 over multiset token overlap (the SQuAD measure).

    Order-insensitive: a prediction with the right words in the wrong order scores
    the same as one in the right order. :func:`rouge_l` is the companion that
    cares about order, which is why both are reported rather than either alone.
    """
    pairs = _pairs(y_true, y_pred)
    if not pairs:
        return float("nan")

    scores = []
    for reference, prediction in pairs:
        ref, pred = _norm_tokens(reference), _norm_tokens(prediction)
        if not ref or not pred:
            # Both empty is a match; one empty is a miss. Computing precision
            # against an empty prediction would divide by zero.
            scores.append(float(not ref and not pred))
            continue
        from collections import Counter

        overlap = sum((Counter(ref) & Counter(pred)).values())
        if overlap == 0:
            scores.append(0.0)
            continue
        prec, rec = overlap / len(pred), overlap / len(ref)
        scores.append(2 * prec * rec / (prec + rec))
    return float(np.mean(scores))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length, the standard O(len(a)*len(b)) table."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b):
            current.append(previous[j] + 1 if token == other else max(current[j], previous[j + 1]))
        previous = current
    return previous[-1]


def rouge_l(y_true, y_pred) -> float:
    """Mean per-pair ROUGE-L F-measure: F1 over the longest common subsequence.

    A *subsequence*, not a substring, so it rewards getting the right words in the
    right order without demanding they be adjacent. This is the summarization
    convention.
    """
    pairs = _pairs(y_true, y_pred)
    if not pairs:
        return float("nan")

    scores = []
    for reference, prediction in pairs:
        ref, pred = _norm_tokens(reference), _norm_tokens(prediction)
        if not ref or not pred:
            scores.append(float(not ref and not pred))
            continue
        lcs = _lcs_length(ref, pred)
        if lcs == 0:
            scores.append(0.0)
            continue
        prec, rec = lcs / len(pred), lcs / len(ref)
        scores.append(2 * prec * rec / (prec + rec))
    return float(np.mean(scores))


# ── Name → function table ─────────────────────────────────
# `TaskSpec.metric_names` indexes into this.
_LABEL_METRICS: dict[str, MetricFn] = {
    "acc": accuracy,
    "f1": f1,
    "f1_binary": lambda yt, yp: f1(yt, yp, average="binary"),
    "precision": precision,
    "recall": recall,
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "mase": mase,
    "smape": smape,
    "exact_match": exact_match,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
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
