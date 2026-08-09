"""
core/profile.py
───────────────
The five production criteria, **measured** rather than assumed.

A trained model has one number attached to it by default: its score. Deploying on
that number alone is how a 40 ms ensemble ends up behind a 20 ms SLA and how an
uninterpretable model ends up in a credit decision. This module produces the
other four:

    predictive performance   the CV mean of the primary metric, and its spread
    inference latency        measured p50/p95/p99 of a warmed predict, in ms
    memory & compute cost    serialized artifact bytes + the backend's own count
    explainability           the tiered score from `core.explain`
    maintainability          fit wall-clock, fold stability, fold failures

Every one is a measurement taken on this machine with this data. That is a real
limitation and it is stated rather than papered over: a latency measured on a
laptop is not a production latency. What it *is* is a comparable number across
candidates measured under identical conditions in the same run, which is what
model selection needs — the ranking transfers even when the absolute value does
not.

**Latency is measured on the estimator, not through the HTTP layer.** Serving
adds request parsing, validation and network time that belong to the API, not to
the model, and including them would make every candidate look the same. The
serving-side numbers still exist, as Prometheus histograms from
``serving/metrics.py``; these two answer different questions and neither replaces
the other.

Nothing here imports torch, xgboost or any training library.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

PROFILE_FILE = "profile.json"

# Forward passes discarded before timing starts. The first call through a model
# pays for lazy kernel selection, cuDNN autotuning, JIT warmup and page faults —
# real costs, but one-time ones, and including them makes a fast model look slow
# in proportion to how few samples were measured.
WARMUP_ITERATIONS = 5
# Minimum timed repetitions. A single pass over 128 rows can be shorter than the
# clock's resolution on a fast tree model.
MIN_REPEATS = 20

BYTES_PER_MB = 1024.0 * 1024.0


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    """Single-row and batched inference timings, in milliseconds.

    Both, because they answer different production questions and a model can win
    one while losing the other. ``p95_ms`` is per *call* at batch size 1 — the
    number an online SLA is written against. ``batch_ms_per_row`` is throughput —
    the number an offline scoring job is costed against. A GPU model is often
    terrible at the first and excellent at the second.
    """

    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    batch_ms_per_row: float = 0.0
    batch_size: int = 0
    n_calls: int = 0
    # Set when timing could not be taken; the numbers above are then all zero and
    # must not be read as "infinitely fast".
    error: str = ""

    @property
    def measured(self) -> bool:
        return not self.error and self.n_calls > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CostProfile:
    """What the model costs to hold and to run."""

    # Serialized size on disk. The number that decides whether the model fits in
    # a serving image and in a memory budget — and unlike a parameter count, it
    # is comparable across a neural net, a tree ensemble and a Prophet pickle.
    artifact_bytes: int = 0
    # The backend's own notion of size (parameters, trees, nodes, series).
    # Informational and per-backend; never compared across backends.
    native: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def artifact_mb(self) -> float:
        return self.artifact_bytes / BYTES_PER_MB

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "artifact_mb": round(self.artifact_mb, 4)}


@dataclass(frozen=True, slots=True)
class MaintainabilityProfile:
    """How predictably this model can be retrained on fresh data.

    The image's framing — "how fast and reliably the model can be retrained on
    fresh daily data without breaking" — decomposes into three things that are
    all observable from a bake-off that has already happened:

    ``fit_seconds``   the retrain window this model needs, every day, forever.
    ``fold_stability``  1 - (CV std / |CV mean|), clamped to [0, 1]. A model whose
                      score swings between folds will swing between retrains too.
                      This is the closest observable proxy for "trains predictably
                      without exploding gradients or pipeline failures".
    ``fold_failures`` folds that raised. Nonzero means the training path is
                      fragile against ordinary data variation, which is the
                      failure mode that wakes people up.
    """

    fit_seconds: float = 0.0
    fold_stability: float = 1.0
    fold_failures: int = 0
    n_folds: int = 0

    @property
    def score(self) -> float:
        """A 0-1 composite, with any fold failure dominating.

        Deliberately harsh about failures: a pipeline that fails one fold in five
        is not 80% maintainable, it is a pager at 3am. Stability carries the rest.
        """
        if self.n_folds and self.fold_failures:
            return max(0.0, self.fold_stability * (1.0 - self.fold_failures / self.n_folds) * 0.5)
        return self.fold_stability

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "score": round(self.score, 4)}


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Everything measured about one candidate, on the five criteria."""

    model: str
    backend: str
    # Predictive performance.
    primary_metric: str = ""
    score: float = float("nan")
    score_std: float = 0.0
    n_folds: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    # The other four.
    latency: LatencyProfile = field(default_factory=LatencyProfile)
    cost: CostProfile = field(default_factory=CostProfile)
    explainability: float = 0.0
    explain_method: str = "none"
    maintainability: MaintainabilityProfile = field(default_factory=MaintainabilityProfile)

    @property
    def score_std_error(self) -> float:
        """Standard error of the CV mean — the natural "same score" tolerance.

        Two candidates whose means differ by less than this are not
        distinguishable by the data available, and picking the higher one is
        picking noise.
        """
        if self.n_folds < 2:
            return 0.0
        return float(self.score_std / np.sqrt(self.n_folds))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "primary_metric": self.primary_metric,
            "score": _jsonable(self.score),
            "score_std": _jsonable(self.score_std),
            "score_std_error": _jsonable(self.score_std_error),
            "n_folds": self.n_folds,
            "metrics": {k: _jsonable(v) for k, v in self.metrics.items()},
            "latency": self.latency.to_dict(),
            "cost": self.cost.to_dict(),
            "explainability": {"score": self.explainability, "method": self.explain_method},
            "maintainability": self.maintainability.to_dict(),
        }


# ── Latency ───────────────────────────────────────────────
def measure_latency(
    estimator: Any,
    inputs: Any,
    *,
    max_samples: int = 128,
    warmup: int = WARMUP_ITERATIONS,
    batch: bool = True,
) -> LatencyProfile:
    """Time ``estimator.predict`` on single rows and on one batch.

    Uses ``time.perf_counter``, the only clock in the standard library with a
    documented monotonic guarantee and sub-millisecond resolution on Windows —
    ``time.time()`` there has a ~16 ms granularity, which is most of a 20 ms
    budget.

    Returns a profile carrying ``error`` rather than raising. A candidate whose
    latency cannot be measured must still be reportable; the caller decides
    whether an unmeasurable model can win, and a raise here would take the whole
    bake-off down with one bad candidate.
    """
    rows = _as_rows(inputs, max_samples)
    if rows is None:
        return LatencyProfile(error="inputs are not row-indexable")
    n = len(rows)
    if n == 0:
        return LatencyProfile(error="no rows to time")

    predict = getattr(estimator, "predict", None)
    if predict is None:
        return LatencyProfile(error="estimator has no predict()")

    try:
        for i in range(max(0, warmup)):
            predict(rows[i % n : i % n + 1])
    except Exception as exc:
        return LatencyProfile(error=f"warmup failed: {type(exc).__name__}: {exc}")

    # Single-row timings. Repeated to at least MIN_REPEATS by cycling the sample,
    # so a 20-row dataset still produces a meaningful p95.
    repeats = max(MIN_REPEATS, n)
    timings: list[float] = []
    try:
        for i in range(repeats):
            row = rows[i % n : i % n + 1]
            started = time.perf_counter()
            predict(row)
            timings.append((time.perf_counter() - started) * 1000.0)
    except Exception as exc:
        return LatencyProfile(error=f"single-row predict failed: {type(exc).__name__}: {exc}")

    arr = np.asarray(timings, dtype="float64")
    batch_per_row = 0.0
    batch_size = 0
    if batch:
        try:
            started = time.perf_counter()
            predict(rows)
            elapsed = (time.perf_counter() - started) * 1000.0
            batch_per_row = elapsed / n
            batch_size = n
        except Exception as exc:
            # A model that cannot take a batch is a real finding, not a failure
            # of the measurement — the single-row numbers stand.
            log.debug("batch predict failed (%s) — reporting single-row latency only", exc)

    return LatencyProfile(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(arr.mean()),
        batch_ms_per_row=float(batch_per_row),
        batch_size=batch_size,
        n_calls=len(timings),
    )


def _as_rows(inputs: Any, max_samples: int) -> Any:
    """``inputs`` narrowed to at most ``max_samples`` row-indexable rows.

    Handles the payload shapes the framework actually produces: numpy arrays and
    DataFrames slice directly; a list of strings (text) slices directly; a
    forecast request is not row-indexable and returns ``None`` so the caller
    reports "not measured" rather than timing something meaningless.
    """
    if inputs is None:
        return None
    if hasattr(inputs, "iloc"):  # DataFrame
        return inputs.iloc[:max_samples]
    if isinstance(inputs, np.ndarray):
        return inputs[:max_samples]
    if isinstance(inputs, (list, tuple)):
        return list(inputs[:max_samples])
    arr = getattr(inputs, "__getitem__", None)
    if arr is None:
        return None
    try:
        return inputs[:max_samples]
    except (TypeError, KeyError):
        return None


# ── Cost ──────────────────────────────────────────────────
def measure_cost(backend: Any, estimator: Any, *, workdir: Path | None = None) -> CostProfile:
    """Serialize the model to a scratch directory and weigh what came out.

    Saving is the only honest way to get a comparable size: a parameter count
    does not know about dtype, a tree count does not know about node payloads,
    and neither knows about the tokenizer sitting in an HF model directory. The
    artifact is what ships, so the artifact is what is measured.

    Uses a temporary directory when ``workdir`` is not given, and cleans it up —
    a bake-off over five candidates must not leave five model copies behind.
    """
    native: dict[str, Any] = {}
    try:
        native = dict(backend.model_size(estimator) or {})
    except Exception as exc:
        log.debug("backend.model_size failed (%s)", exc)

    import shutil
    import tempfile

    tmp_created = workdir is None
    dest = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="mlf-profile-"))
    try:
        backend.save(estimator, dest)
        total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
        return CostProfile(artifact_bytes=int(total), native=native)
    except Exception as exc:
        return CostProfile(native=native, error=f"could not serialize: {type(exc).__name__}: {exc}")
    finally:
        if tmp_created:
            shutil.rmtree(dest, ignore_errors=True)


# ── Maintainability ───────────────────────────────────────
def fold_stability(mean: float, std: float) -> float:
    """``1 - coefficient of variation``, clamped to [0, 1].

    The coefficient of variation (std / |mean|) is the scale-free way to ask "how
    much does this score move between folds", which is what makes it comparable
    between an accuracy of 0.9 and an RMSE of 340. A mean of zero has no
    meaningful CV, and returns 0.0 — an unstable answer rather than a divide by
    zero, which is the conservative direction.
    """
    if not np.isfinite(mean) or not np.isfinite(std) or mean == 0:
        return 0.0
    return float(np.clip(1.0 - abs(std / mean), 0.0, 1.0))


# ── Serialization ─────────────────────────────────────────
def write_profile(profile: ModelProfile, dest: Path) -> Path:
    """Write ``profile.json`` under ``dest`` and return its path."""
    import json

    dest.mkdir(parents=True, exist_ok=True)
    path = dest / PROFILE_FILE
    path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return path


def _jsonable(value: float) -> float | None:
    """NaN/inf → ``None``, because ``json.dumps`` emits invalid JSON for them.

    ``NaN`` is a real outcome here — ROC-AUC on a single-class fold is undefined —
    so it must survive into the report as an explicit null rather than as the
    literal ``NaN`` token, which no strict JSON parser will read back.
    """
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


__all__ = [
    "PROFILE_FILE",
    "CostProfile",
    "LatencyProfile",
    "MaintainabilityProfile",
    "ModelProfile",
    "fold_stability",
    "measure_cost",
    "measure_latency",
    "write_profile",
]
