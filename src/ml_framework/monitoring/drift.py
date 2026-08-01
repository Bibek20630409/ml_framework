"""
monitoring/drift.py
───────────────────
Input-distribution drift detection: **PSI** (Population Stability Index) and the
**KS** statistic against a reference distribution captured at training time.

Rule of thumb for PSI: < 0.1 stable · 0.1–0.2 moderate shift · > 0.2 significant
drift (retrain-worthy). The reference is a compact per-feature summary (quantile
bin edges + fractions), so serving never has to store the training data.
"""

from __future__ import annotations

from collections import deque

import numpy as np

_EPS = 1e-6


def build_reference(x: np.ndarray, feature_cols: list[str], bins: int = 10) -> dict:
    """Compact per-feature reference from the (raw) training features."""
    ref: dict = {"bins": bins, "features": {}}
    for i, col in enumerate(feature_cols):
        c = x[:, i].astype("float64")
        edges = np.quantile(c, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)  # collapse duplicate quantiles (low-cardinality cols)
        comp = edges.copy()
        comp[0], comp[-1] = -np.inf, np.inf
        hist, _ = np.histogram(c, bins=comp)
        frac = hist / max(hist.sum(), 1)
        ref["features"][col] = {
            "edges": edges.tolist(),
            "ref_frac": frac.tolist(),
            "mean": float(c.mean()),
            "std": float(c.std()),
        }
    return ref


def psi_from_reference(ref_feature: dict, current: np.ndarray) -> float:
    """PSI of ``current`` values against a saved per-feature reference."""
    edges = np.array(ref_feature["edges"], dtype="float64")
    if edges.size < 2:
        return 0.0
    comp = edges.copy()
    comp[0], comp[-1] = -np.inf, np.inf
    cur_hist, _ = np.histogram(current.astype("float64"), bins=comp)
    cur_frac = np.clip(cur_hist / max(cur_hist.sum(), 1), _EPS, None)
    ref_frac = np.clip(np.array(ref_feature["ref_frac"], dtype="float64"), _EPS, None)
    n = min(len(cur_frac), len(ref_frac))
    return float(np.sum((cur_frac[:n] - ref_frac[:n]) * np.log(cur_frac[:n] / ref_frac[:n])))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Kolmogorov–Smirnov statistic (requires scipy)."""
    from scipy.stats import ks_2samp

    return float(ks_2samp(reference, current).statistic)


def compute_drift(reference: dict, x: np.ndarray, feature_cols: list[str]) -> dict[str, float]:
    """PSI per feature for a batch ``x`` (rows) vs the reference."""
    out: dict[str, float] = {}
    feats = reference.get("features", {})
    for i, col in enumerate(feature_cols):
        if col in feats and i < x.shape[1]:
            out[col] = psi_from_reference(feats[col], x[:, i])
    return out


class DriftTracker:
    """Keeps a rolling window of recent inputs and reports PSI per feature."""

    def __init__(
        self,
        reference: dict | None,
        feature_cols: list[str],
        window: int = 1000,
        min_samples: int = 30,
        gauge=None,
        labels: dict | None = None,
    ):
        self.reference = reference
        self.feature_cols = feature_cols
        self.min_samples = min_samples
        self.buf: deque = deque(maxlen=window)
        self.gauge = gauge
        # Identity labels (backend, model) applied to every point this tracker
        # reports. Passed in rather than read here so this module stays free of
        # the serving layer -- drift is computed the same way offline.
        self.labels = dict(labels or {})

    def observe(self, x: np.ndarray) -> None:
        for row in np.atleast_2d(x):
            self.buf.append(np.asarray(row, dtype="float64"))

    def compute(self) -> dict[str, float]:
        if not self.reference or len(self.buf) < self.min_samples:
            return {}
        drift = compute_drift(self.reference, np.array(self.buf), self.feature_cols)
        if self.gauge is not None:
            for feat, val in drift.items():
                self.gauge.labels(feature=feat, **self.labels).set(val)
        return drift
