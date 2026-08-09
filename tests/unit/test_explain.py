"""
Tiered feature attribution.

The contract this file guards: :func:`feature_importance` returns the best tier
it can reach and **never raises**. A bake-off scoring five candidates calls it
five times, and a model with nothing to attribute is a normal outcome — it has to
come back as a score of 0.0 with a reason, not as an exception that takes the
comparison down.

The second contract is that the reported ``method`` is the truth. A score of 1.0
means native importances were actually read, not that the plugin declared it
could produce them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml_framework.core.explain import (
    IMPORTANCE_FILE,
    METHOD_SCORES,
    Importance,
    feature_importance,
    none_importance,
)

pytestmark = pytest.mark.unit


class _WithImportances:
    """A tree-shaped estimator: importances, no data needed."""

    def __init__(self, values=(0.5, 0.3, 0.2)):
        self.feature_importances_ = np.array(values)

    def predict(self, x):
        return np.zeros(len(x))


class _WithCoef:
    def __init__(self, coef):
        self.coef_ = np.array(coef)

    def predict(self, x):
        return np.zeros(len(x))


class _Wrapper:
    """The framework's own shape: an Estimator holding the library object."""

    def __init__(self, inner):
        self.model = inner

    def predict(self, x):
        return self.model.predict(x)


class _Opaque:
    """Predict and nothing else — the neural-net case."""

    def __init__(self, weights=(2.0, 0.0, 0.0)):
        self.w = np.array(weights)

    def predict(self, x):
        return (np.asarray(x) @ self.w > 0).astype(int)


# ── Tier scores ───────────────────────────────────────────
def test_the_tier_table_orders_native_above_shap_above_permutation():
    assert METHOD_SCORES["native"] > METHOD_SCORES["shap"] > METHOD_SCORES["permutation"] > 0
    assert METHOD_SCORES["none"] == 0.0


def test_none_importance_carries_its_reason():
    result = none_importance("no feature matrix")
    assert result.score == 0.0
    assert result.method == "none"
    assert "no feature matrix" in result.reason


# ── Native ────────────────────────────────────────────────
def test_native_importances_are_read_without_any_data():
    result = feature_importance(_WithImportances(), features=["a", "b", "c"])
    assert result.method == "native"
    assert result.score == 1.0
    assert result.features == ("a", "b", "c")


def test_importances_are_normalized_to_shares():
    # Comparing a split-gain against a permutation delta only means something
    # once both are shares of their own total.
    result = feature_importance(_WithImportances(values=(10.0, 30.0)), features=["a", "b"])
    assert sum(result.values) == pytest.approx(1.0)
    assert result.values == pytest.approx((0.25, 0.75))


def test_native_importances_are_found_through_a_wrapper():
    result = feature_importance(_Wrapper(_WithImportances()), features=["a", "b", "c"])
    assert result.method == "native"


def test_linear_coefficients_count_as_native():
    result = feature_importance(_WithCoef([1.0, -3.0]), features=["a", "b"])
    assert result.method == "native"
    assert result.extra["source"] == "coef_"
    # Magnitude, not sign: a strongly negative weight is a strong driver.
    assert result.values[1] > result.values[0]


def test_multiclass_coefficients_average_across_the_classes():
    # One row per class; the per-feature magnitude is the mean absolute weight.
    result = feature_importance(_WithCoef([[1.0, 0.0], [1.0, 0.0]]), features=["a", "b"])
    assert result.method == "native"
    assert result.values == pytest.approx((1.0, 0.0))


def test_positional_names_are_used_when_the_count_disagrees():
    # One-hot encoding upstream makes the model's feature count differ from the
    # schema's. Positional names beat a silently misaligned mapping.
    result = feature_importance(_WithImportances(values=(1.0, 1.0, 1.0)), features=["a", "b"])
    assert result.features == ("f0", "f1", "f2")


def test_all_zero_importances_do_not_divide_by_zero():
    result = feature_importance(_WithImportances(values=(0.0, 0.0)), features=["a", "b"])
    assert result.method == "native"
    assert result.values == (0.0, 0.0)


# ── Permutation ───────────────────────────────────────────
def test_permutation_is_the_fallback_for_a_model_with_no_importances():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 3))
    y = (x @ np.array([2.0, 0.0, 0.0]) > 0).astype(int)

    result = feature_importance(_Opaque(), features=["a", "b", "c"], x=x, y=y)

    assert result.method == "permutation"
    assert result.score == 0.5
    assert "no native importances" in result.reason


def test_permutation_finds_the_feature_that_actually_drives_the_prediction():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 3))
    y = (x @ np.array([2.0, 0.0, 0.0]) > 0).astype(int)

    result = feature_importance(_Opaque(), features=["driver", "noise1", "noise2"], x=x, y=y)

    assert result.top(1)[0][0] == "driver"


def test_permutation_needs_labels():
    result = feature_importance(_Opaque(), features=["a"], x=np.zeros((10, 3)), y=None)
    assert result.method == "none"
    assert "needs labels" in result.reason


def test_negative_permutation_deltas_are_clipped_rather_than_ranked():
    # Shuffling a useless column sometimes *improves* the score by luck. That is
    # noise around zero, not negative importance.
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 4))
    y = (x[:, 0] > 0).astype(int)
    result = feature_importance(_Opaque(weights=(1.0, 0, 0, 0)), x=x, y=y)
    assert all(v >= 0 for v in result.values)


def test_a_regression_target_is_scored_with_r2():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 2))

    class _Reg:
        def predict(self, data):
            return np.asarray(data)[:, 0] * 3.0

    result = feature_importance(_Reg(), features=["a", "b"], x=x, y=x[:, 0] * 3.0)
    assert result.method == "permutation"
    assert result.extra["scorer"] == "r2"


def test_integer_labels_are_scored_as_classification():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(80, 2))
    y = (x[:, 0] > 0).astype(int)
    result = feature_importance(_Opaque(weights=(1.0, 0.0)), x=x, y=y)
    assert result.extra["scorer"] == "accuracy"


# ── Sampling ──────────────────────────────────────────────
def test_a_large_matrix_is_subsampled_before_probing():
    # Permutation importance costs a predict pass per feature per repeat, so a
    # 200k-row split would dominate the bake-off that is also timing inference.
    rng = np.random.default_rng(5)
    x = rng.normal(size=(3000, 2))
    y = (x[:, 0] > 0).astype(int)

    result = feature_importance(_Opaque(weights=(1.0, 0.0)), x=x, y=y, max_samples=100)
    assert result.extra["n_samples"] == 100


def test_subsampling_is_deterministic_for_a_seed():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(500, 3))
    y = (x[:, 0] > 0).astype(int)
    kwargs = {"x": x, "y": y, "max_samples": 50, "seed": 7}
    first = feature_importance(_Opaque(weights=(1.0, 0, 0)), **kwargs)
    second = feature_importance(_Opaque(weights=(1.0, 0, 0)), **kwargs)
    assert first.values == pytest.approx(second.values)


# ── Refusal ───────────────────────────────────────────────
def test_a_model_with_neither_importances_nor_data_scores_zero():
    result = feature_importance(_Opaque())
    assert result.method == "none"
    assert result.score == 0.0
    assert "no feature matrix" in result.reason


def test_no_estimator_is_a_refusal_not_a_crash():
    assert feature_importance(None).method == "none"


def test_a_predict_that_raises_degrades_to_none_rather_than_propagating():
    class _Broken:
        def predict(self, x):
            raise RuntimeError("boom")

    rng = np.random.default_rng(8)
    x = rng.normal(size=(40, 2))
    result = feature_importance(_Broken(), x=x, y=(x[:, 0] > 0).astype(int))
    assert result.method == "none"


def test_prefer_native_false_still_reaches_native_as_a_last_resort():
    # `prefer_native=False` is a statement about *order*, not about exclusion: a
    # tree whose permutation pass fails should still report its own importances.
    result = feature_importance(_WithImportances(), features=["a", "b", "c"], prefer_native=False)
    assert result.method == "native"


# ── Serialization ─────────────────────────────────────────
def test_importance_writes_a_json_the_bundle_can_carry(tmp_path: Path):
    result = feature_importance(_WithImportances(), features=["a", "b", "c"])
    path = result.write(tmp_path)

    assert path.name == IMPORTANCE_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["method"] == "native"
    assert payload["score"] == 1.0
    assert payload["top"][0]["feature"] == "a"  # 0.5 is the largest share


def test_top_returns_features_in_descending_importance():
    result = Importance(method="native", features=("a", "b", "c"), values=(0.1, 0.7, 0.2))
    assert [f for f, _ in result.top()] == ["b", "c", "a"]
    assert result.top(1) == [("b", 0.7)]


# ── SHAP (optional extra) ─────────────────────────────────
def test_shap_is_used_for_a_tree_when_it_is_installed():
    shap = pytest.importorskip("shap", reason="the explain extra is not installed")
    xgb = pytest.importorskip("xgboost", reason="the gbdt extra is not installed")
    assert shap  # referenced so the import is not flagged as unused

    rng = np.random.default_rng(9)
    x = rng.normal(size=(80, 3))
    y = (x[:, 0] > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2).fit(x, y)

    # `prefer_native=False` is what makes SHAP reachable: native importances are
    # free and would otherwise always win.
    result = feature_importance(model, features=["a", "b", "c"], x=x, y=y, prefer_native=False)
    assert result.method == "shap"
    assert result.score == 0.8
    assert result.top(1)[0][0] == "a"


def test_shap_is_not_attempted_for_a_non_tree_model():
    pytest.importorskip("shap", reason="the explain extra is not installed")
    rng = np.random.default_rng(10)
    x = rng.normal(size=(60, 2))
    y = (x[:, 0] > 0).astype(int)

    # A sampling explainer on an opaque model takes minutes, inside a routine
    # that is also measuring latency. It must fall through to permutation.
    result = feature_importance(_Opaque(weights=(1.0, 0.0)), x=x, y=y, prefer_native=False)
    assert result.method == "permutation"
