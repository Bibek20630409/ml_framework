"""
The four non-accuracy criteria, and the shapes they are reported in.

The recurring theme: a measurement that *fails* must be distinguishable from a
measurement of zero. A model whose latency could not be timed must not sort
ahead of one that was timed at 3 ms, and a size that could not be read must not
appear as a 0 MB artifact. Most of these tests exist to pin that distinction.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from ml_framework.core.profile import (
    PROFILE_FILE,
    CostProfile,
    LatencyProfile,
    MaintainabilityProfile,
    ModelProfile,
    fold_stability,
    measure_cost,
    measure_latency,
    write_profile,
)

pytestmark = pytest.mark.unit


class _Estimator:
    """A predictor with a controllable per-call cost."""

    def __init__(self, delay: float = 0.0, fails: bool = False) -> None:
        self.delay = delay
        self.fails = fails
        self.calls = 0

    def predict(self, inputs):
        self.calls += 1
        if self.fails:
            raise RuntimeError("no")
        if self.delay:
            time.sleep(self.delay)
        return np.zeros(len(inputs))


class _Backend:
    """A backend whose save() writes a file of a known size."""

    def __init__(self, payload: bytes = b"x" * 2048, fails: bool = False) -> None:
        self.payload = payload
        self.fails = fails

    def model_size(self, est):
        return {"trees": 7}

    def save(self, est, dest: Path):
        if self.fails:
            raise OSError("disk on fire")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "model.bin").write_bytes(self.payload)
        (dest / "meta.json").write_text("{}", encoding="utf-8")


# ── Latency ───────────────────────────────────────────────
def test_measure_latency_reports_percentiles_and_a_call_count():
    profile = measure_latency(_Estimator(), np.zeros((32, 3)), max_samples=32, warmup=2)

    assert profile.measured
    assert profile.n_calls >= 20  # MIN_REPEATS, even from a 32-row sample
    assert profile.p50_ms <= profile.p95_ms <= profile.p99_ms
    assert profile.mean_ms > 0


def test_measure_latency_excludes_the_warmup_calls_from_the_timings():
    est = _Estimator()
    profile = measure_latency(est, np.zeros((10, 2)), max_samples=10, warmup=5)
    # 5 warmup + n_calls timed + 1 batch call.
    assert est.calls == 5 + profile.n_calls + 1


def test_measure_latency_separates_batch_throughput_from_single_row_latency():
    profile = measure_latency(_Estimator(), np.zeros((40, 3)), max_samples=40, warmup=1)
    assert profile.batch_size == 40
    assert profile.batch_ms_per_row > 0


def test_measure_latency_skips_the_batch_pass_when_asked():
    profile = measure_latency(
        _Estimator(), np.zeros((10, 2)), max_samples=10, warmup=1, batch=False
    )
    assert profile.batch_size == 0
    assert profile.batch_ms_per_row == 0.0


def test_measure_latency_records_a_failure_instead_of_raising():
    # A bake-off must survive one candidate whose predict is broken.
    profile = measure_latency(_Estimator(fails=True), np.zeros((10, 2)), warmup=1)
    assert not profile.measured
    assert "warmup failed" in profile.error
    assert profile.p95_ms == 0.0


def test_an_unmeasured_latency_is_not_a_fast_one():
    # The property the tie-break depends on: `measured` is False, so the sort key
    # can substitute infinity rather than reading the zeroed p95 as instant.
    assert not LatencyProfile(error="nope").measured
    assert not LatencyProfile().measured  # no calls, no error, still not measured


def test_measure_latency_refuses_inputs_it_cannot_index_by_row():
    profile = measure_latency(_Estimator(), object(), warmup=1)
    assert not profile.measured
    assert "row-indexable" in profile.error


def test_measure_latency_reports_a_missing_predict():
    profile = measure_latency(object(), np.zeros((5, 2)), warmup=1)
    assert "no predict()" in profile.error


def test_measure_latency_handles_a_dataframe():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(np.zeros((30, 3)), columns=list("abc"))
    assert measure_latency(_Estimator(), frame, max_samples=30, warmup=1).measured


def test_measure_latency_handles_a_list_of_strings():
    # The text payload: rows are strings, not array rows.
    profile = measure_latency(_Estimator(), ["a", "b", "c"] * 10, max_samples=30, warmup=1)
    assert profile.measured


def test_a_slower_model_measures_slower():
    fast = measure_latency(_Estimator(), np.zeros((20, 2)), max_samples=20, warmup=1)
    slow = measure_latency(_Estimator(delay=0.002), np.zeros((20, 2)), max_samples=20, warmup=1)
    assert slow.p50_ms > fast.p50_ms


# ── Cost ──────────────────────────────────────────────────
def test_measure_cost_weighs_every_file_the_backend_wrote():
    cost = measure_cost(_Backend(payload=b"z" * 4096), object())
    assert cost.artifact_bytes == 4096 + 2  # model.bin + "{}"
    assert cost.artifact_mb == pytest.approx(4098 / (1024 * 1024))
    assert not cost.error


def test_measure_cost_keeps_the_backends_own_size_counts():
    assert measure_cost(_Backend(), object()).native == {"trees": 7}


def test_measure_cost_cleans_up_its_scratch_directory():
    # Five candidates in a bake-off must not leave five model copies behind.
    import tempfile

    before = set(Path(tempfile.gettempdir()).glob("mlf-profile-*"))
    measure_cost(_Backend(), object())
    assert set(Path(tempfile.gettempdir()).glob("mlf-profile-*")) == before


def test_measure_cost_records_a_serialization_failure(tmp_path: Path):
    cost = measure_cost(_Backend(fails=True), object(), workdir=tmp_path)
    assert cost.artifact_bytes == 0
    assert "could not serialize" in cost.error
    # The native counts still came through — one failure does not erase the other
    # measurement taken beside it.
    assert cost.native == {"trees": 7}


def test_measure_cost_survives_a_backend_whose_size_call_raises(tmp_path: Path):
    class _Rude(_Backend):
        def model_size(self, est):
            raise ValueError("nope")

    cost = measure_cost(_Rude(), object(), workdir=tmp_path)
    assert cost.native == {}
    assert cost.artifact_bytes > 0


# ── Maintainability ───────────────────────────────────────
@pytest.mark.parametrize(
    ("mean", "std", "expected"),
    [
        (0.9, 0.0, 1.0),  # identical across folds
        (0.9, 0.09, 0.9),  # 10% coefficient of variation
        (0.9, 0.9, 0.0),  # swings as much as it scores
        (0.9, 2.0, 0.0),  # clamped, never negative
        (0.0, 0.1, 0.0),  # no meaningful CV around zero
    ],
)
def test_fold_stability_is_one_minus_the_coefficient_of_variation(mean, std, expected):
    assert fold_stability(mean, std) == pytest.approx(expected)


def test_fold_stability_is_scale_free():
    # The property that makes an accuracy and an RMSE comparable on this axis.
    assert fold_stability(0.9, 0.09) == pytest.approx(fold_stability(340.0, 34.0))


def test_fold_stability_of_a_nan_mean_is_zero():
    assert fold_stability(float("nan"), 0.1) == 0.0


def test_maintainability_score_is_dominated_by_a_fold_failure():
    clean = MaintainabilityProfile(fold_stability=0.9, fold_failures=0, n_folds=5)
    flaky = MaintainabilityProfile(fold_stability=0.9, fold_failures=1, n_folds=5)
    assert clean.score == pytest.approx(0.9)
    # A pipeline that fails one fold in five is not 80% maintainable.
    assert flaky.score < 0.4


# ── ModelProfile ──────────────────────────────────────────
def test_score_std_error_shrinks_with_more_folds():
    few = ModelProfile(model="m", backend="b", score=0.9, score_std=0.1, n_folds=4)
    many = ModelProfile(model="m", backend="b", score=0.9, score_std=0.1, n_folds=16)
    assert few.score_std_error == pytest.approx(0.05)
    assert many.score_std_error == pytest.approx(0.025)


def test_score_std_error_of_a_single_fold_is_zero():
    # One fold gives no spread to estimate from; claiming a tolerance would be
    # inventing one.
    assert ModelProfile(model="m", backend="b", score_std=0.3, n_folds=1).score_std_error == 0.0


def test_profile_serializes_nan_as_null_so_the_json_parses():
    # ROC-AUC on a single-class fold is genuinely NaN, and `json.dumps` emits the
    # bare token `NaN` for it, which no strict parser will read back.
    profile = ModelProfile(
        model="m", backend="b", score=float("nan"), metrics={"roc_auc": float("nan")}
    )
    text = json.dumps(profile.to_dict())
    assert "NaN" not in text
    assert json.loads(text)["score"] is None
    assert json.loads(text)["metrics"]["roc_auc"] is None


def test_profile_to_dict_carries_every_criterion():
    profile = ModelProfile(
        model="xgboost",
        backend="gbdt",
        primary_metric="acc",
        score=0.9,
        latency=LatencyProfile(p95_ms=3.0, n_calls=20),
        cost=CostProfile(artifact_bytes=1024),
        explainability=1.0,
        explain_method="native",
    )
    payload = profile.to_dict()
    assert payload["score"] == 0.9
    assert payload["latency"]["p95_ms"] == 3.0
    # `artifact_mb` is a rounded display convenience; `artifact_bytes` is the
    # exact number, and both ship so a reader never has to choose.
    assert payload["cost"]["artifact_bytes"] == 1024
    assert payload["cost"]["artifact_mb"] == pytest.approx(0.001, abs=1e-6)
    assert payload["explainability"] == {"score": 1.0, "method": "native"}
    assert "maintainability" in payload


def test_write_profile_lands_in_the_bundle(tmp_path: Path):
    path = write_profile(ModelProfile(model="m", backend="b", score=0.5), tmp_path / "out")
    assert path.name == PROFILE_FILE
    assert json.loads(path.read_text(encoding="utf-8"))["model"] == "m"


def test_infinite_scores_also_serialize_as_null():
    profile = ModelProfile(model="m", backend="b", score=math.inf)
    assert json.loads(json.dumps(profile.to_dict()))["score"] is None
