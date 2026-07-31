"""
data/preprocess/tabular.py
──────────────────────────
``StandardScaler`` plus the imbalance-handling logic, moved verbatim out of
``TabularDataModule.setup``.

Two things worth naming, both preserved from v1 because they were deliberate
fixes rather than accidents:

* **Imbalance is a single explicit strategy**, resolved once against the
  *pre-SMOTE* class distribution. Computing class weights *and* oversampling
  applies the correction twice.
* **``compute_class_weights`` returns a scalar for binary.** ``BCEWithLogitsLoss``
  takes a ``pos_weight`` for the positive class only; a 2-element vector does not
  broadcast the way it looks like it should. That fix has its own regression test.

The numpy function :func:`class_weights` is the single implementation; the
torch-returning ``compute_class_weights`` at the original import path wraps it.
That is what keeps the agnostic layer torch-free without duplicating the maths.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .base import BasePreprocessor, PreprocessorError

log = logging.getLogger(__name__)

SCALER_FILE = "scaler.pkl"


# ── Imbalance helpers ─────────────────────────────────────
def detect_imbalance(y: np.ndarray, threshold: float = 0.3) -> bool:
    """True if the minority class is < ``threshold`` fraction of the majority."""
    counts = Counter(np.asarray(y).tolist())
    if len(counts) < 2:
        return False
    mn, mx = min(counts.values()), max(counts.values())
    return (mn / mx) < threshold


def class_weights(y: np.ndarray, task: str) -> np.ndarray:
    """Balanced weights on the *pre-SMOTE* distribution, as numpy.

    binary     → 1-element array ``[n_neg / n_pos]`` (BCE ``pos_weight``)
    multiclass → per-class balanced weights ``n / (n_classes * count_c)``
    """
    counts = np.bincount(np.asarray(y).astype("int64"))
    counts = np.clip(counts, 1, None)
    if task == "binary":
        n_neg, n_pos = counts[0], counts[1] if len(counts) > 1 else 1
        return np.asarray([n_neg / n_pos], dtype="float32")
    n, nc = counts.sum(), len(counts)
    return np.asarray(n / (nc * counts), dtype="float32")


def apply_smote(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from imblearn.over_sampling import SMOTE  # local import: optional dep

    before = dict(Counter(np.asarray(y).tolist()))
    x_res, y_res = SMOTE(random_state=seed).fit_resample(x, y)
    log.info("SMOTE: %s → %s", before, dict(Counter(np.asarray(y_res).tolist())))
    return x_res, y_res


def balanced_sample_weights(y: np.ndarray, task: str, threshold: float = 0.3) -> np.ndarray | None:
    """Per-**row** weights for a model that consumes ``sample_weight`` natively.

    Distinct from :func:`class_weights`, which returns one weight per *class* for a
    loss function. Boosting libraries take a weight per training row instead, so
    this expands the balanced per-class weights back over the label array.

    This is the consumer of ``Capabilities.supports_sample_weight``: SMOTE
    synthesizes points by interpolating between neighbours, which is a poor fit for
    an axis-aligned splitter, and these libraries expose weighting natively. Returns
    ``None`` when the data is not imbalanced, so a balanced dataset carries no
    weights at all rather than a vector of ones.
    """
    if task not in ("binary", "multiclass"):
        return None
    if not detect_imbalance(y, threshold):
        log.info("sample weights: not applied (classes are balanced)")
        return None
    labels = np.asarray(y).astype("int64")
    counts = np.clip(np.bincount(labels), 1, None)
    per_class = counts.sum() / (len(counts) * counts)
    weights = np.asarray(per_class[labels], dtype="float64")
    log.info("sample weights: %d rows, %d classes", len(weights), len(counts))
    return weights


def resolve_imbalance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    strategy: str,
    threshold: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply the one chosen imbalance correction. Returns ``(x, y, class_weights)``.

    Only one of the two outputs is ever non-trivial: either the training rows were
    resampled, or weights were computed — never both.
    """
    if task not in ("binary", "multiclass"):
        return x, y, None
    imbalanced = detect_imbalance(y, threshold)
    if strategy == "class_weights" and imbalanced:
        weights = class_weights(y, task)
        log.info("class weights: %s", weights.tolist())
        return x, y, weights
    if strategy == "smote" and imbalanced:
        x_res, y_res = apply_smote(x, y, seed)
        return x_res, y_res, None
    log.info("imbalance strategy=%s applied=%s", strategy, False)
    return x, y, None


# ── Preprocessor ──────────────────────────────────────────
class TabularPreprocessor(BasePreprocessor):
    """Standardizes features, fitted on the training split only.

    ``needs_scaling=False`` produces a fitted-but-inert preprocessor rather than a
    different class, so the bundle layout does not depend on the backend: trees
    gain nothing from standardization and it destroys the interpretability of
    their split thresholds, which is what ``Capabilities.needs_scaling`` exists to
    say.
    """

    def __init__(self, *, needs_scaling: bool = True) -> None:
        self.needs_scaling = needs_scaling
        self.scaler: Any | None = None
        self._fitted = False

    # ── contract ──
    def fit(self, split: Any, schema: Any = None) -> None:
        x = split.x if hasattr(split, "x") else split
        if x is None:
            raise PreprocessorError("TabularPreprocessor.fit needs the training features")
        if self.needs_scaling:
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler().fit(np.asarray(x, dtype="float32"))
        self._fitted = True

    def transform(self, x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype="float32")
        if self.scaler is None:
            return arr
        return np.asarray(self.scaler.transform(arr), dtype="float32")

    def fit_transform(self, split: Any, schema: Any = None) -> np.ndarray:
        self.fit(split, schema)
        return self.transform(split.x if hasattr(split, "x") else split)

    @property
    def fitted(self) -> bool:
        return self._fitted

    # ── file hooks ──
    def params(self) -> dict[str, Any]:
        return {"needs_scaling": self.needs_scaling}

    def _write(self, dest: Path) -> list[str]:
        if self.scaler is None:
            return []
        import joblib

        joblib.dump(self.scaler, dest / SCALER_FILE)
        log.info("scaler saved → %s", dest / SCALER_FILE)
        return [SCALER_FILE]

    def _read(self, src: Path, spec: Any = None) -> None:
        path = Path(src) / SCALER_FILE
        if path.exists():
            import joblib

            self.scaler = joblib.load(path)
            log.info("scaler loaded ← %s", path)
        self._fitted = True

    # ── v1 bundles ──
    @classmethod
    def from_legacy_bundle(cls, bundle_dir: str | Path) -> TabularPreprocessor:
        """Wrap a v1 bundle's bare ``scaler.pkl`` (no ``preprocessor/`` dir).

        The clean break was authorized for configs and tests, **not** for bundles
        already deployed, so a loader that meets one keeps working.
        """
        obj = cls(needs_scaling=True)
        obj._read(Path(bundle_dir))
        return obj
