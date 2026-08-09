"""
core/explain.py
───────────────
Feature attribution, as a **tiered** capability rather than a yes/no one.

"Is this model explainable?" has no useful boolean answer. An XGBoost model
carries split-gain importances for free; an MLP carries nothing but can be
probed by shuffling one column at a time; a Prophet model has no feature matrix
to attribute anything to. Those are three different situations and a single flag
flattens them into a lie in one direction or the other.

So this module answers with a *method* and a *score*:

    native       1.0   the estimator's own importances (trees, linear models)
    shap         0.8   SHAP values, when the optional extra is installed
    permutation  0.5   sklearn's permutation_importance — works on any predict
    none         0.0   nothing attributable (no feature matrix, no predict)

The score is what ``select.constraints.min_explainability`` compares against and
what the weighted objective reads, which is the named consumer that keeps this
from being decoration. The *values* are written to ``feature_importance.json`` in
the bundle, because a regulator asking "why did it decline this application?"
wants the numbers, not the tier.

The ordering is not arbitrary. Native importances are exact statements about the
fitted model and cost nothing. SHAP is more informative than permutation
(per-prediction, signed, additive) but is an approximation for anything that is
not a tree. Permutation importance is model-agnostic and honest but expensive —
one full predict pass per feature — and measures *this dataset's* dependence
rather than the model's structure.

**Nothing here is imported at module scope beyond numpy.** ``sklearn`` is a base
dependency but a heavy import; ``shap`` is optional and must never be required.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .types import Requirement

log = logging.getLogger(__name__)

IMPORTANCE_FILE = "feature_importance.json"

# The tier → score table. One place, so a constraint threshold and a weighted
# score can never disagree about what "0.8" means.
METHOD_SCORES: dict[str, float] = {
    "native": 1.0,
    "shap": 0.8,
    "permutation": 0.5,
    "none": 0.0,
}

SHAP_REQUIREMENT = Requirement("shap", extra="explain", min_version="0.44")


@dataclass(frozen=True, slots=True)
class Importance:
    """Per-feature attribution, and how it was obtained.

    ``values`` are non-negative magnitudes normalized to sum to 1, so two
    candidates produced by different methods can be compared feature-for-feature.
    The raw scale of a split-gain and of a permutation delta have nothing to do
    with each other; the shares do.
    """

    method: str
    features: tuple[str, ...]
    values: tuple[float, ...]
    # Why the method above was chosen rather than a better one. Empty when the
    # best available method was used.
    reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """This attribution's place on the tiered scale."""
        return METHOD_SCORES.get(self.method, 0.0)

    def top(self, k: int = 10) -> list[tuple[str, float]]:
        """The ``k`` most important features, descending."""
        pairs = sorted(zip(self.features, self.values, strict=True), key=lambda p: -p[1])
        return pairs[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "score": self.score,
            "reason": self.reason,
            "features": list(self.features),
            "values": [float(v) for v in self.values],
            "top": [{"feature": f, "importance": float(v)} for f, v in self.top()],
            **({"extra": self.extra} if self.extra else {}),
        }

    def write(self, dest: Path) -> Path:
        """Write ``feature_importance.json`` under ``dest`` and return its path."""
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / IMPORTANCE_FILE
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def none_importance(reason: str) -> Importance:
    """The "nothing to attribute" answer, carrying why."""
    return Importance(method="none", features=(), values=(), reason=reason)


# ── The public entry point ────────────────────────────────
def feature_importance(
    estimator: Any,
    *,
    features: list[str] | tuple[str, ...] | None = None,
    x: Any = None,
    y: Any = None,
    prefer_native: bool = True,
    max_samples: int = 512,
    seed: int = 42,
) -> Importance:
    """Attribute this estimator's predictions to features, as well as it can be.

    Tries native importances, then SHAP, then permutation, and returns the first
    that works. Every fallback is logged with the reason, because "explainability
    silently degraded to none" is exactly the kind of thing that should not be
    discovered from a compliance review.

    ``x``/``y`` are only needed for the permutation and SHAP paths; a tree model
    is attributable without touching the data at all. Passing them is therefore
    optional, and omitting them narrows the reachable tiers rather than failing.

    Never raises for an unattributable model — that is a normal outcome and the
    caller (a bake-off scoring five candidates) must not have to guard every one.
    """
    if estimator is None:
        return none_importance("no estimator")

    if prefer_native:
        native = _native_importance(estimator, features)
        if native is not None:
            return native

    # Everything below needs data. `prefer_native=False` reorders the tiers, it
    # does not exclude one — so a tree asked for SHAP but handed no matrix still
    # falls back to the importances it was carrying all along.
    if x is not None:
        names = _feature_names(features, x)
        sample_x, sample_y = _subsample(x, y, max_samples=max_samples, seed=seed)

        shap_result = _shap_importance(estimator, sample_x, names)
        if shap_result is not None:
            return shap_result

        if sample_y is not None:
            perm = _permutation_importance(estimator, sample_x, sample_y, names, seed=seed)
            if perm is not None:
                return perm

    if not prefer_native:
        native = _native_importance(estimator, features)
        if native is not None:
            return native

    if x is None:
        return none_importance(
            "no native importances, and no feature matrix to probe the model with"
        )
    if y is None:
        return none_importance(
            "no native importances, SHAP unavailable, and permutation importance "
            "needs labels to score against"
        )
    return none_importance("no attribution method succeeded for this estimator")


# ── Tier 1: the estimator's own importances ───────────────
def _native_importance(estimator: Any, features: Any) -> Importance | None:
    """``feature_importances_`` or ``coef_``, through however many wrappers.

    The estimator handed in is a framework :class:`~ml_framework.core.protocols.
    Estimator`, not the library object — so the library object is looked for on
    the attributes a wrapper conventionally keeps it on before giving up.
    """
    target = _unwrap(estimator)
    if target is None:
        return None

    raw = getattr(target, "feature_importances_", None)
    source = "feature_importances_"
    if raw is None:
        coef = getattr(target, "coef_", None)
        if coef is None:
            return None
        # A multiclass linear model has one coefficient row per class. The
        # magnitude that matters for "does this feature drive the decision" is
        # the mean absolute weight across classes, not any single row.
        arr = np.asarray(coef, dtype="float64")
        raw = np.abs(arr).mean(axis=0) if arr.ndim > 1 else np.abs(arr)
        source = "coef_"

    values = np.asarray(raw, dtype="float64").ravel()
    if values.size == 0 or not np.isfinite(values).any():
        return None

    names = _feature_names(features, None, n=values.size)
    if len(names) != values.size:
        # A one-hot encoder upstream can make the model's feature count differ
        # from the schema's. Positional names beat a silently misaligned mapping.
        log.debug(
            "native importances have %d entries for %d feature names — using positional names",
            values.size,
            len(names),
        )
        names = tuple(f"f{i}" for i in range(values.size))

    return Importance(
        method="native",
        features=tuple(names),
        values=tuple(_normalize(values)),
        extra={"source": source},
    )


def _unwrap(estimator: Any) -> Any:
    """The underlying library estimator, or ``None``.

    Checks the object itself first, then the attribute names the framework's own
    wrappers use. Deliberately a short, explicit list rather than a recursive
    hunt through ``__dict__`` — guessing which attribute is "the real model"
    finds a preprocessor about as often as an estimator.
    """
    candidates = [estimator]
    for attr in ("model", "_model", "estimator", "_estimator", "booster", "_booster"):
        inner = getattr(estimator, attr, None)
        if inner is not None:
            candidates.append(inner)
    for candidate in candidates:
        if hasattr(candidate, "feature_importances_") or hasattr(candidate, "coef_"):
            return candidate
    return None


# ── Tier 2: SHAP ──────────────────────────────────────────
def _shap_importance(estimator: Any, x: Any, names: tuple[str, ...]) -> Importance | None:
    """Mean absolute SHAP value per feature, if ``shap`` is installed.

    Only attempted for tree models. ``shap.Explainer`` falls back to a sampling
    explainer for anything else, and a sampling explainer on a neural net takes
    minutes — inside a routine that is *also* measuring inference latency, which
    would make the bake-off both slow and wrong about why.
    """
    if not SHAP_REQUIREMENT.is_satisfied():
        return None
    target = _unwrap(estimator)
    if target is None or not hasattr(target, "feature_importances_"):
        # Trees only; see the docstring.
        return None

    try:
        import shap

        explainer = shap.TreeExplainer(target)
        values = explainer.shap_values(x)
        arr = np.asarray(values, dtype="float64")
        # Multiclass returns (n_classes, n, d) or (n, d, n_classes) depending on
        # the library and version. Collapse everything but the feature axis.
        if arr.ndim == 3:
            axis = 1 if arr.shape[0] < arr.shape[-1] else 0
            arr = np.abs(arr).mean(axis=axis if axis == 0 else 0)
            if arr.ndim == 2 and arr.shape[-1] != len(names) and arr.shape[0] == len(names):
                arr = arr.T
        magnitudes = np.abs(arr).reshape(-1, arr.shape[-1]).mean(axis=0)
    except Exception as exc:  # pragma: no cover - depends on an optional package
        # SHAP raises a wide variety of library-specific errors on unsupported
        # model/version pairs. A failed explanation must degrade to the next tier,
        # never fail the run that asked for it.
        log.debug("SHAP explanation failed (%s) — falling back", exc)
        return None

    if magnitudes.size != len(names):
        names = tuple(f"f{i}" for i in range(magnitudes.size))
    return Importance(
        method="shap",
        features=names,
        values=tuple(_normalize(magnitudes)),
        extra={"explainer": "TreeExplainer", "n_samples": int(np.asarray(x).shape[0])},
    )


# ── Tier 3: permutation ───────────────────────────────────
def _permutation_importance(
    estimator: Any, x: Any, y: Any, names: tuple[str, ...], *, seed: int
) -> Importance | None:
    """Drop in score when one column is shuffled, averaged over repeats.

    Model-agnostic: it needs only ``predict``. That is the reason it is the floor
    rather than the ceiling — it measures how much *this dataset's* score depends
    on a column, which is a fact about the pair, not about the model's structure.
    A feature that is important but duplicated by another shows as unimportant
    here, and that is a real property of the measurement, not a bug in it.
    """
    try:
        from sklearn.inspection import permutation_importance as sk_permutation
        from sklearn.metrics import accuracy_score, r2_score
    except ImportError:  # pragma: no cover - sklearn is a base dependency
        return None

    arr_x = np.asarray(x)
    arr_y = np.asarray(y)
    if arr_x.ndim != 2 or arr_x.shape[0] != arr_y.shape[0] or arr_x.shape[1] == 0:
        return None

    # `permutation_importance` wants an sklearn-shaped estimator with `.fit`,
    # `.predict` and a `score`. The framework's Estimator has only `predict`, so
    # it is adapted here rather than the protocol being widened for one caller.
    scorer = r2_score if _looks_continuous(arr_y) else accuracy_score

    class _Adapter:
        _estimator_type = "regressor" if scorer is r2_score else "classifier"

        def fit(self, *_: Any, **__: Any) -> _Adapter:  # pragma: no cover - never called
            return self

        def predict(self, data: Any) -> np.ndarray:
            return np.asarray(estimator.predict(data))

        def score(self, data: Any, target: Any) -> float:
            return float(scorer(target, self.predict(data)))

    try:
        result = sk_permutation(
            _Adapter(),
            arr_x,
            arr_y,
            n_repeats=5,
            random_state=seed,
            scoring=None,
        )
    except Exception as exc:
        log.debug("permutation importance failed (%s)", exc)
        return None

    # Negative means shuffling *helped*, which is noise around zero rather than
    # negative importance. Clipping keeps the normalized shares interpretable.
    magnitudes = np.clip(np.asarray(result.importances_mean, dtype="float64"), 0.0, None)
    if magnitudes.size != len(names):
        names = tuple(f"f{i}" for i in range(magnitudes.size))
    return Importance(
        method="permutation",
        features=names,
        values=tuple(_normalize(magnitudes)),
        reason="the model exposes no native importances",
        extra={
            "n_repeats": 5,
            "n_samples": int(arr_x.shape[0]),
            "scorer": "r2" if scorer is r2_score else "accuracy",
        },
    )


# ── Helpers ───────────────────────────────────────────────
def _normalize(values: np.ndarray) -> np.ndarray:
    """Non-negative magnitudes summing to 1, or all-zero if there is no signal."""
    arr = np.nan_to_num(np.abs(np.asarray(values, dtype="float64")), nan=0.0, posinf=0.0)
    total = arr.sum()
    return arr / total if total > 0 else arr


def _subsample(x: Any, y: Any, *, max_samples: int, seed: int) -> tuple[Any, Any]:
    """At most ``max_samples`` rows, drawn without replacement.

    Both SHAP and permutation importance cost time proportional to the row count
    — permutation multiplies it by the feature count and the repeat count — so
    attributing a 200k-row test split would dominate a bake-off that is *also*
    timing inference. A random sample rather than the first N rows, because the
    first N of a time-ordered or class-sorted table is not a sample of it.
    """
    n = _n_rows(x)
    if n == 0 or n <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(n, size=max_samples, replace=False))
    return _take(x, picked), (None if y is None else _take(y, picked))


def _n_rows(x: Any) -> int:
    if x is None:
        return 0
    shape = getattr(x, "shape", None)
    if shape:
        return int(shape[0])
    try:
        return len(x)
    except TypeError:
        return 0


def _take(data: Any, indices: np.ndarray) -> Any:
    if hasattr(data, "iloc"):  # DataFrame / Series
        return data.iloc[indices]
    if isinstance(data, (list, tuple)):
        return [data[i] for i in indices]
    return np.asarray(data)[indices]


def _feature_names(features: Any, x: Any, *, n: int | None = None) -> tuple[str, ...]:
    if features:
        return tuple(str(f) for f in features)
    width = n
    if width is None and x is not None:
        arr = np.asarray(x)
        width = int(arr.shape[1]) if arr.ndim == 2 else 0
    return tuple(f"f{i}" for i in range(width or 0))


def _looks_continuous(y: np.ndarray) -> bool:
    """Whether ``y`` should be scored with R² rather than accuracy.

    Float dtype with more distinct values than a plausible class count. Integer
    labels and small float label sets are treated as classification, which is the
    safe direction: scoring a classifier with R² gives a meaningless number,
    while scoring a regressor with accuracy gives approximately zero and shows up
    as "nothing is important" rather than as a wrong ranking.
    """
    if not np.issubdtype(y.dtype, np.floating):
        return False
    return len(np.unique(y)) > min(20, max(2, y.size // 10))


__all__ = [
    "IMPORTANCE_FILE",
    "METHOD_SCORES",
    "SHAP_REQUIREMENT",
    "Importance",
    "feature_importance",
    "none_importance",
]
