import numpy as np
import pytest

from ml_framework.monitoring import (
    DriftTracker,
    build_reference,
    compute_drift,
    psi_from_reference,
)


@pytest.mark.unit
def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 2)).astype("float32")
    ref = build_reference(x, ["a", "b"])
    # same distribution → PSI near 0
    same = rng.normal(size=(2000, 2)).astype("float32")
    drift = compute_drift(ref, same, ["a", "b"])
    assert drift["a"] < 0.1
    assert drift["b"] < 0.1


@pytest.mark.unit
def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 1)).astype("float32")
    ref = build_reference(x, ["a"])
    shifted = (rng.normal(size=(2000, 1)) + 3.0).astype("float32")  # mean shift
    psi = psi_from_reference(ref["features"]["a"], shifted[:, 0])
    assert psi > 0.2  # significant drift


@pytest.mark.unit
def test_drift_tracker_window_and_compute():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1000, 3)).astype("float32")
    ref = build_reference(x, ["a", "b", "c"])
    tracker = DriftTracker(ref, ["a", "b", "c"], window=500, min_samples=50)
    assert tracker.compute() == {}  # not enough samples yet
    tracker.observe(rng.normal(size=(200, 3)).astype("float32"))
    drift = tracker.compute()
    assert set(drift) == {"a", "b", "c"}


@pytest.mark.unit
def test_drift_tracker_no_reference_is_safe():
    tracker = DriftTracker(None, ["a"], min_samples=1)
    tracker.observe(np.zeros((5, 1), dtype="float32"))
    assert tracker.compute() == {}
